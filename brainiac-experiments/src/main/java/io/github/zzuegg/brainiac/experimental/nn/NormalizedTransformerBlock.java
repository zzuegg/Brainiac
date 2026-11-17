package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A normalized transformer block (nGPT): every hidden state lives on the unit hypersphere
 * and each sub-layer moves it along the sphere instead of adding to it:
 *
 * <pre>
 * h = norm(h + alphaA * (norm(attention(h)) - h))
 * h = norm(h + alphaM * (norm(mlp(h)) - h))
 * </pre>
 *
 * <p>The per-dimension "eigen learning rates" {@code alphaA, alphaM} are learned. Queries
 * and keys are normalized and rescaled by a learned {@code sQK}, so the softmax scale is
 * {@code sqrt(dk)} instead of {@code 1/sqrt(dk)}; the SwiGLU MLP has learned scales
 * {@code sU, sV}. There are no LayerNorms and no weight decay: after each optimizer step,
 * call {@link #renormalize()} so every weight vector along the embedding dimension has unit
 * norm again. Bounded values also help low-precision inference.
 *
 * <p>Paper: Loshchilov et al., "nGPT: Normalized Transformer with Representation Learning
 * on the Hypersphere", <a href="https://arxiv.org/abs/2410.01131">arXiv:2410.01131</a>.
 */
public final class NormalizedTransformerBlock implements Layer {

    private final int features;
    private final int heads;
    private final boolean causal;
    private final float rotaryBase;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var wu;
    private final Var wvMlp;
    private final Var wdown;
    private final Var alphaA;
    private final Var alphaM;
    private final Var sqk;
    private final Var su;
    private final Var sv;

    /**
     * Creates the block.
     *
     * @param features the model width, a multiple of {@code heads}
     * @param heads the number of heads
     * @param hidden the MLP width
     * @param causal whether each step only sees earlier ones
     * @param rotaryBase the RoPE base, or 0 for no positions
     * @param alphaInit the initial eigen step size, for example {@code 0.05} (paper: {@code 1/layers})
     * @param random the random source
     */
    public NormalizedTransformerBlock(int features, int heads, int hidden, boolean causal, float rotaryBase,
            float alphaInit, RandomGenerator random) {
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
        this.wo = MixerOps.weight(random, features, features);
        this.wu = MixerOps.weight(random, features, hidden);
        this.wvMlp = MixerOps.weight(random, features, hidden);
        this.wdown = MixerOps.weight(random, hidden, features);
        this.alphaA = MixerOps.filled(alphaInit, features);
        this.alphaM = MixerOps.filled(alphaInit, features);
        this.sqk = MixerOps.filled(1f, features);
        this.su = MixerOps.filled(1f, hidden);
        this.sv = MixerOps.filled(1f, hidden);
        renormalize();
    }

    /**
     * Projects every weight vector along the embedding dimension back to unit norm. Call it
     * after each optimizer step.
     */
    public void renormalize() {
        for (Var w : List.of(wq, wk, wv, wu, wvMlp)) {
            w.assign(normalize(w.value(), 0));   // columns: one per output, over the embedding
        }
        for (Var w : List.of(wo, wdown)) {
            w.assign(normalize(w.value(), 1));   // rows: one per input, over the embedding
        }
    }

    /**
     * Scales a matrix to unit-norm vectors along one axis.
     *
     * @param w the matrix
     * @param axis 0 to normalize columns, 1 to normalize rows
     * @return the normalized matrix
     */
    public static Tensor normalize(Tensor w, int axis) {
        return w.div(w.square().sum(axis, true).add(1e-12f).sqrt());
    }

    /** @return the attention eigen step sizes, {@code [features]} */
    public Var attentionStep() {
        return alphaA;
    }

    /** @return the MLP eigen step sizes, {@code [features]} */
    public Var mlpStep() {
        return alphaM;
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)}, reading the width from the
     * input {@code [steps, features]}; a drop-in for {@code TransformerBlock.of(heads)}.
     *
     * @param heads the number of heads
     * @param hidden the MLP width
     * @param causal whether each step only sees earlier ones
     * @param alphaInit the initial eigen step size
     * @return the spec
     */
    public static LayerSpec of(int heads, int hidden, boolean causal, float alphaInit) {
        return MixerOps.sameShape((in, r) -> new NormalizedTransformerBlock(in.dim(-1), heads, hidden, causal, 0f,
                alphaInit, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "NormalizedTransformerBlock");
        Var h = MixerOps.unit(input);
        Var a = MixerOps.unit(attention(h));
        h = MixerOps.unit(h.add(alphaA.mul(a.sub(h))));
        Var u = h.matmul(wu).mul(su);
        Var v = h.matmul(wvMlp).mul(sv).mul((float) Math.sqrt(features));
        Var m = MixerOps.unit(u.mul(Activation.SILU.forward(v)).matmul(wdown));
        return MixerOps.unit(h.add(alphaM.mul(m.sub(h))));
    }

    private Var attention(Var h) {
        int dk = features / heads;
        Var sq = MixerOps.split(sqk.reshape(1, 1, features), heads);   // [1, h, 1, dk]
        Var q = MixerOps.split(h.matmul(wq), heads);
        Var k = MixerOps.split(h.matmul(wk), heads);
        if (rotaryBase > 0) {
            q = Functional.rotary(q, rotaryBase, 0);
            k = Functional.rotary(k, rotaryBase, 0);
        }
        q = MixerOps.unit(q).mul(sq);
        k = MixerOps.unit(k).mul(sq);
        Var v = MixerOps.split(h.matmul(wv), heads);
        int t = h.shape().dim(1);
        Tensor bias = causal ? MixerOps.causal(t) : null;
        Var o = Functional.attention(q, k, v, bias, (float) Math.sqrt(dk), 0, null);
        return MixerOps.merge(o).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo, wu, wvMlp, wdown, alphaA, alphaM, sqk, su, sv);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "NormalizedTransformerBlock(" + features + ", heads=" + heads + ")";
    }
}
