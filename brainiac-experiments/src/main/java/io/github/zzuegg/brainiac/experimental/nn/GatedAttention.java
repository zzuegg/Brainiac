package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Multi-head attention with a sigmoid output gate: every head's output is multiplied
 * element-wise by {@code sigmoid(x Wg)}, computed from the same token, before the output
 * projection. The gate adds a cheap non-linearity between the value and output
 * projections and lets a head output nothing without parking its attention on a "sink"
 * token, which removes attention sinks and the large activations they cause and makes
 * training more stable.
 *
 * <p>Input and output are {@code [batch, steps, features]}. The gate is the
 * head-specific, element-wise variant after scaled dot-product attention (position "G1"
 * in the paper), which the paper found best.
 *
 * <p>Paper: Qiu et al., "Gated Attention for Large Language Models: Non-linearity,
 * Sparsity, and Attention-Sink-Free", <a href="https://arxiv.org/abs/2505.06708">arXiv:2505.06708</a>.
 */
public final class GatedAttention implements Layer {

    private final int features;
    private final int heads;
    private final boolean causal;
    private final float rotaryBase;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wg;
    private final Var bg;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param features the model width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param causal whether each step only sees earlier ones
     * @param rotaryBase the RoPE base for queries and keys, or 0 for no positions
     * @param random the random source
     */
    public GatedAttention(int features, int heads, boolean causal, float rotaryBase, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("features " + features + " not divisible by heads " + heads);
        }
        this.features = features;
        this.heads = heads;
        this.causal = causal;
        this.rotaryBase = rotaryBase;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wg = MixerOps.weight(random, features, features);
        this.bg = MixerOps.filled(0f, features);
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Returns the gate values.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, steps, features]}, each in {@code (0, 1)}
     */
    public Tensor gates(Var input) {
        return input.matmul(wg).add(bg).sigmoid().value();
    }

    /** @return the gate bias; a very negative bias closes every gate */
    public Var gateBias() {
        return bg;
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "GatedAttention");
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        int t = input.shape().dim(1);
        Tensor bias = causal ? MixerOps.causal(t) : null;
        Var o = Functional.attention(q, k, v, bias, (float) (1 / Math.sqrt(features / (double) heads)), 0, null);
        Var gate = input.matmul(wg).add(bg).sigmoid();
        return MixerOps.merge(o).mul(gate).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wg, bg, wo);
    }

    @Override
    public String toString() {
        return "GatedAttention(" + features + ", heads=" + heads + ")";
    }
}
