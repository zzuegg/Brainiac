package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Multi-head attention with contextual position encoding (CoPE) as a stand-alone, causal
 * drop-in layer. Positions are not token indices but counts: every query decides with
 * sigmoid gates {@code g_ij = sigmoid(q_i . k_j)} which earlier tokens "count" (for example
 * only sentence ends, or only a certain word), the contextual position of key {@code j} is
 * {@code p_ij = sum_{s=j..i} g_is}, and the score gets the bias {@code q_i . e[p_ij]},
 * interpolating learned position embeddings at the fractional count. So a head can attend
 * to "the 2nd last sentence" or "the 3rd word back" however long the words are.
 *
 * <p>The bias is {@link PositionEncodings#cope}. With {@code separateGateKeys} the gates use
 * their own key projection (the paper's variant that decouples "what counts" from "what
 * matches"); with a {@code rotaryBase} RoPE is applied to the content scores as well.
 *
 * <p>Paper: Golovneva et al., "Contextual Position Encoding: Learning to Count What's
 * Important", <a href="https://arxiv.org/abs/2405.18719">arXiv:2405.18719</a>.
 */
public final class ContextualPositionAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final float rotaryBase;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wGate;
    private final Var positions;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the heads
     * @param maxPositions the learned position embeddings (counts beyond are clipped)
     * @param separateGateKeys whether the gates get their own key projection
     * @param rotaryBase the RoPE base for content scores, or 0 for none
     * @param random the random source
     */
    public ContextualPositionAttention(int features, int heads, int maxPositions, boolean separateGateKeys,
            float rotaryBase, RandomGenerator random) {
        if (features % heads != 0 || maxPositions < 2) {
            throw new IllegalArgumentException("need heads dividing features and at least 2 positions");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.rotaryBase = rotaryBase;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wGate = separateGateKeys ? MixerOps.weight(random, features, features) : null;
        this.positions = TechniqueOps.normal(random, (float) (1 / Math.sqrt(headSize)), maxPositions, headSize);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}: 16 positions, shared gate
     * keys, no RoPE.
     *
     * @param heads the heads
     * @return the spec
     */
    public static LayerSpec of(int heads) {
        return of(heads, 16, false);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, without RoPE.
     *
     * @param heads the heads
     * @param maxPositions the learned position embeddings
     * @param separateGateKeys whether the gates get their own key projection
     * @return the spec
     */
    public static LayerSpec of(int heads, int maxPositions, boolean separateGateKeys) {
        return MixerOps.sameShape((in, r) -> new ContextualPositionAttention(in.dim(-1), heads, maxPositions,
                separateGateKeys, 0f, r));
    }

    /**
     * The contextual positions {@code p_ij} the layer assigns for an input, for inspection.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, heads, steps, steps]}, 0 above the diagonal
     */
    public Tensor contextualPositions(Var input) {
        int t = input.shape().dim(1);
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var g = MixerOps.split(input.matmul(wGate != null ? wGate : wk), heads);
        Tensor gates = q.matmul(g.transpose()).value().sigmoid().mul(MixerOps.lowerOnes(t));
        float[] upper = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j <= i; j++) {
                upper[i * t + j] = 1;
            }
        }
        return gates.matmul(Tensor.of(Shape.of(t, t), upper)).mul(MixerOps.lowerOnes(t));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "ContextualPositionAttention");
        int t = input.shape().dim(1);
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        Var gateKeys = wGate == null ? k : MixerOps.split(input.matmul(wGate), heads);
        Var bias = PositionEncodings.cope(q, gateKeys, positions, true);
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        Var scores = q.matmul(k.transpose()).mul((float) (1 / Math.sqrt(headSize))).add(bias);
        Tensor mask = RefinementOps.additive(RefinementOps.keep(t, t, 0, 0, true, 0, input.value().device()));
        return MixerOps.merge(MixerOps.softmax(scores.add(mask)).matmul(v)).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, positions, wo));
        if (wGate != null) {
            p.add(wGate);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "ContextualPositionAttention(" + features + ", heads=" + heads + (wGate != null ? ", sep-keys" : "")
                + ")";
    }
}
