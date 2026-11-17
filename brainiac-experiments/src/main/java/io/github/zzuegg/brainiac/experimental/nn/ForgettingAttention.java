package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Forgetting attention (FoX): causal softmax attention whose logits get a data-dependent
 * forget bias. Each head computes a forget gate {@code f_t = sigmoid(x_t w + b)} per step and
 * adds {@code D[i, j] = sum_{l = j+1..i} log f_l} to the score of query {@code i} and key
 * {@code j}; a gate near 0 makes the head forget everything before that step, a gate near 1
 * keeps it. The bias is computed as a difference of cumulative sums, so it costs one extra
 * {@code [steps, steps]} add per head.
 *
 * <p>The bias already encodes order and recency, so FoX needs no positional embedding (no
 * RoPE). This mirrors the forget gate of an LSTM or of gated linear attention in softmax
 * attention. The optional output gate gives the "Pro" variant of the paper.
 *
 * <p>Paper: Lin et al., "Forgetting Transformer: Softmax Attention with a Forget Gate",
 * <a href="https://arxiv.org/abs/2503.02130">arXiv:2503.02130</a>.
 */
public final class ForgettingAttention implements Layer {

    private final int features;
    private final int heads;
    private final boolean outputGate;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wf;
    private final Var bf;
    private final Var wg;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the model width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param outputGate whether to add a sigmoid output gate (FoX "Pro")
     * @param random the random source
     */
    public ForgettingAttention(int features, int heads, boolean outputGate, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.outputGate = outputGate;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wf = MixerOps.weight(random, features, heads);
        // start mostly remembering: sigmoid(3) ~ 0.95
        this.bf = MixerOps.filled(3f, heads);
        this.wg = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Returns the forget bias {@code D} added to the attention logits (including the causal mask).
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, heads, steps, steps]}
     */
    public Tensor forgetBias(Var input) {
        return bias(input).value();
    }

    /**
     * Returns the forget gates.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, heads, steps]}
     */
    public Tensor forgetGates(Var input) {
        return input.matmul(wf).add(bf).sigmoid().permute(0, 2, 1).value();
    }

    private Var bias(Var x) {
        int t = x.shape().dim(1);
        // log sigmoid(z) = -softplus(-z)
        Var logF = x.matmul(wf).add(bf).neg().softplus().neg().permute(0, 2, 1);   // [b, h, t]
        Var c = logF.cumsum(-1);   // c_i = sum_{l <= i} log f_l
        int b = x.shape().dim(0);
        Var ci = c.reshape(b, heads, t, 1);
        Var cj = c.reshape(b, heads, 1, t);
        return ci.sub(cj).add(MixerOps.causal(t));
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, reading the width from the
     * input {@code [steps, features]}. Always causal.
     *
     * @param heads the number of heads
     * @param outputGate whether to add the output gate (FoX "Pro")
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean outputGate) {
        return MixerOps.sameShape((in, r) -> new ForgettingAttention(in.dim(-1), heads, outputGate, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "ForgettingAttention");
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        float scale = (float) (1 / Math.sqrt(features / (double) heads));
        Var w = MixerOps.softmax(q.matmul(k.transpose()).mul(scale).add(bias(input)));
        Var o = MixerOps.merge(w.matmul(v));
        if (outputGate) {
            o = o.mul(input.matmul(wg).sigmoid());
        }
        return o.matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return outputGate ? List.of(wq, wk, wv, wf, bf, wg, wo) : List.of(wq, wk, wv, wf, bf, wo);
    }

    @Override
    public String toString() {
        return "ForgettingAttention(" + features + ", heads=" + heads + (outputGate ? ", pro" : "") + ")";
    }
}
