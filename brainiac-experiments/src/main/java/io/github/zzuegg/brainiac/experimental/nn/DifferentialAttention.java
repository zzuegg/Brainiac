package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Differential attention: each head computes two softmax attention maps and subtracts
 * them, {@code (softmax(Q1 K1^T) - lambda softmax(Q2 K2^T)) V}, so attention noise that both
 * maps share (weight smeared over irrelevant context) cancels, like a differential
 * amplifier. The scalar {@code lambda} is re-parameterised per layer as
 * {@code exp(lq1 . lk1) - exp(lq2 . lk2) + lambdaInit}; every head output passes through
 * an RMS norm and is scaled by {@code 1 - lambdaInit}.
 *
 * <p>Input and output are {@code [batch, steps, features]}. Reported to reduce
 * hallucination and improve long-context retrieval.
 *
 * <p>Paper: Ye et al., "Differential Transformer", <a href="https://arxiv.org/abs/2410.05258">arXiv:2410.05258</a>.
 */
public final class DifferentialAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final boolean causal;
    private final float lambdaInit;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var lq1;
    private final Var lk1;
    private final Var lq2;
    private final Var lk2;

    /**
     * Creates the layer.
     *
     * @param features the model width
     * @param heads the number of heads; {@code features} must split into {@code 2 * heads} parts
     * @param causal whether each step only sees earlier ones
     * @param depth the 1-based layer index, which sets {@code lambdaInit = 0.8 - 0.6 exp(-0.3 (depth - 1))}
     * @param random the random source
     */
    public DifferentialAttention(int features, int heads, boolean causal, int depth, RandomGenerator random) {
        if (features % (2 * heads) != 0) {
            throw new IllegalArgumentException("features " + features + " must split into 2 * heads = " + 2 * heads);
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads / 2;
        this.causal = causal;
        this.lambdaInit = (float) (0.8 - 0.6 * Math.exp(-0.3 * (depth - 1)));
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.wo = MixerOps.weight(random, features, features);
        this.lq1 = MixerOps.uniform(random, 0.1f, headSize);
        this.lk1 = MixerOps.uniform(random, 0.1f, headSize);
        this.lq2 = MixerOps.uniform(random, 0.1f, headSize);
        this.lk2 = MixerOps.uniform(random, 0.1f, headSize);
    }

    /** @return the current {@code lambda} */
    public float lambda() {
        return lambdaVar().value().item();
    }

    private Var lambdaVar() {
        return lq1.mul(lk1).sum().exp().sub(lq2.mul(lk2).sum().exp()).add(lambdaInit);
    }

    /**
     * Returns the differential attention map of every head.
     *
     * @param input {@code [batch, steps, features]}
     * @return {@code [batch, heads, steps, steps]}; rows sum to {@code 1 - lambda}
     */
    public Tensor weights(Var input) {
        return maps(input)[2].value();
    }

    /** Returns {map1, map2, diff}. */
    private Var[] maps(Var x) {
        int t = x.shape().dim(1);
        Var q = MixerOps.split(x.matmul(wq), 2 * heads);   // [b, 2h, t, d]
        Var k = MixerOps.split(x.matmul(wk), 2 * heads);
        float scale = (float) (1 / Math.sqrt(headSize));
        Var scores = q.matmul(k.transpose()).mul(scale);
        if (causal) {
            scores = scores.add(MixerOps.causal(t));
        }
        Var a = MixerOps.softmax(scores);   // [b, 2h, t, t]
        int b = x.shape().dim(0);
        Var pair = a.reshape(b, heads, 2, t, t);
        Var a1 = pair.slice(2, 0, 1).reshape(b, heads, t, t);
        Var a2 = pair.slice(2, 1, 2).reshape(b, heads, t, t);
        return new Var[] {a1, a2, a1.sub(a2.mul(lambdaVar()))};
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, reading the width from the
     * input {@code [steps, features]}.
     *
     * @param heads the number of heads
     * @param causal whether each step only sees earlier ones
     * @param depth the 1-based layer index, which sets {@code lambdaInit}
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean causal, int depth) {
        return MixerOps.sameShape((in, r) -> new DifferentialAttention(in.dim(-1), heads, causal, depth, r));
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "DifferentialAttention");
        Var diff = maps(input)[2];
        Var v = MixerOps.split(input.matmul(wv), heads);   // [b, h, t, 2d]
        Var o = MixerOps.rmsNorm(diff.matmul(v)).mul(1 - lambdaInit);
        return MixerOps.merge(o).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo, lq1, lk1, lq2, lk2);
    }

    @Override
    public String toString() {
        return "DifferentialAttention(" + features + ", heads=" + heads + ")";
    }
}
