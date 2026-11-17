package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Soft mixture of experts: instead of routing tokens, every expert slot receives a
 * weighted average of all tokens, and every token receives a weighted average of all slot
 * outputs:
 *
 * <pre>
 * logits = x Phi                  [b, tokens, experts * slots]
 * D = softmax over tokens         (dispatch: each slot's mix of tokens)
 * C = softmax over slots          (combine: each token's mix of slots)
 * y = C * expert(D^T x)
 * </pre>
 *
 * <p>Fully differentiable, no token dropping and no load balancing losses; the cost is
 * set by the number of slots, not of tokens. Mixing over the token axis makes it
 * non-causal, so it suits encoders and vision models rather than autoregressive decoding.
 * Input and output are {@code [batch, tokens, features]}.
 *
 * <p>Paper: Puigcerver et al., "From Sparse to Soft Mixtures of Experts",
 * <a href="https://arxiv.org/abs/2308.00951">arXiv:2308.00951</a>.
 */
public final class SoftMoe implements Layer {

    private final int features;
    private final int experts;
    private final int slots;
    private final Var phi;
    private final Var scale;
    private final List<Var[]> weights = new ArrayList<>();

    /**
     * Creates the layer with SiLU MLP experts.
     *
     * @param features the width
     * @param experts the number of experts
     * @param slots the slots per expert
     * @param hidden each expert's hidden width
     * @param random the random source
     */
    public SoftMoe(int features, int experts, int slots, int hidden, RandomGenerator random) {
        this.features = features;
        this.experts = experts;
        this.slots = slots;
        this.phi = MixerOps.weight(random, features, experts * slots);
        this.scale = MixerOps.filled(1f, 1);
        for (int i = 0; i < experts; i++) {
            weights.add(new Var[] {MixerOps.weight(random, features, hidden), MixerOps.weight(random, hidden, features)});
        }
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param experts the number of experts
     * @param slots the slots per expert
     * @param hidden each expert's hidden width
     * @return the spec
     */
    public static LayerSpec of(int experts, int slots, int hidden) {
        return MixerOps.sameShape((in, r) -> new SoftMoe(in.dim(-1), experts, slots, hidden, r));
    }

    /** Normalized logits: the paper L2-normalizes x and Phi and learns a scale. */
    private Var logits(Var x) {
        Var p = phi.div(phi.square().sum(0, true).add(1e-6f).sqrt());
        return MixerOps.unit(x).matmul(p).mul(scale);
    }

    /**
     * Returns the dispatch weights.
     *
     * @param input {@code [b, tokens, features]}
     * @return {@code [b, tokens, experts * slots]}; each slot's column sums to 1 over tokens
     */
    public Tensor dispatchWeights(Var input) {
        return MixerOps.softmax(logits(input).permute(0, 2, 1)).permute(0, 2, 1).value();
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "SoftMoe");
        int b = input.shape().dim(0);
        Var l = logits(input);                                         // [b, n, E*S]
        Var dispatch = MixerOps.softmax(l.permute(0, 2, 1));           // [b, E*S, n], over tokens
        Var combine = MixerOps.softmax(l);                             // [b, n, E*S], over slots
        Var slotsIn = dispatch.matmul(input);                          // [b, E*S, d]
        Var[] outs = new Var[experts];
        for (int e = 0; e < experts; e++) {
            Var part = slotsIn.slice(1, e * slots, (e + 1) * slots);
            Var[] w = weights.get(e);
            outs[e] = Activation.SILU.forward(part.matmul(w[0])).matmul(w[1]);
        }
        Var slotsOut = experts == 1 ? outs[0] : Var.concat(1, outs);    // [b, E*S, d]
        return combine.matmul(slotsOut).reshape(b, input.shape().dim(1), features);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(phi, scale));
        weights.forEach(w -> p.addAll(List.of(w)));
        return p;
    }

    @Override
    public String toString() {
        return "SoftMoe(" + experts + " experts x " + slots + " slots)";
    }
}
