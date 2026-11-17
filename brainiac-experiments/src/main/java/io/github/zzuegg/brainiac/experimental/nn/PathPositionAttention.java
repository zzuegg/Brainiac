package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * PaTH attention: a data-dependent position encoding. RoPE relates query {@code i} and key
 * {@code j} by a fixed rotation {@code R^{i-j}}; PaTH uses instead the cumulative product of
 * input-dependent Householder-like matrices of the tokens in between,
 *
 * <pre>
 * score(i, j) = q_i^T H_i H_{i-1} ... H_{j+1} k_j,    H_s = I - beta_s w_s w_s^T,
 * </pre>
 *
 * <p>with {@code w_s} an L2-normalized projection of token {@code s} and
 * {@code beta_s = 2 sigmoid(.)} in {@code (0, 2)} (so {@code H_s} can be a reflection). The
 * transform between two positions thus depends on what lies between them, which lets one
 * layer track state (flip-flops, counters, permutations) that a fixed rotation cannot. With
 * {@code forget} a learned forget gate adds the FoX bias {@code sum_{s=j+1..i} log f_s}
 * (PaTH-FoX).
 *
 * <p>Computed in {@code O(T^2 d)} without materializing the products: going backwards over
 * keys {@code j}, the running queries {@code r_{i,j} = H_{j+1} ... H_i q_i} of every later
 * query are updated by one rank-one Householder step per key; the paper's blockwise UT
 * transform is the hardware-efficient form of the same.
 *
 * <p>Paper: Yang et al., "PaTH Attention: Position Encoding via Accumulating Householder
 * Transformations", <a href="https://arxiv.org/abs/2505.16381">arXiv:2505.16381</a>.
 */
public final class PathPositionAttention implements Layer {

    private final int features;
    private final int heads;
    private final int headSize;
    private final boolean forget;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var ww;
    private final Var wBeta;
    private final Var betaBias;
    private final Var wForget;
    private final Var forgetBias;
    private final Var wo;

    /**
     * Creates the layer (causal).
     *
     * @param features the width, a multiple of {@code heads}
     * @param heads the heads
     * @param forget whether to add a FoX forget gate
     * @param random the random source
     */
    public PathPositionAttention(int features, int heads, boolean forget, RandomGenerator random) {
        if (features % heads != 0) {
            throw new IllegalArgumentException("heads must divide features");
        }
        this.features = features;
        this.heads = heads;
        this.headSize = features / heads;
        this.forget = forget;
        this.wq = MixerOps.weight(random, features, features);
        this.wk = MixerOps.weight(random, features, features);
        this.wv = MixerOps.weight(random, features, features);
        this.ww = MixerOps.weight(random, features, features);
        this.wBeta = MixerOps.uniform(random, 0.01f, features, heads);
        this.betaBias = MixerOps.filled(0f, heads);
        this.wForget = forget ? MixerOps.uniform(random, 0.01f, features, heads) : null;
        this.forgetBias = forget ? MixerOps.filled(3f, heads) : null;
        this.wo = MixerOps.weight(random, features, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param heads the heads
     * @param forget whether to add a FoX forget gate
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean forget) {
        return MixerOps.sameShape((in, r) -> new PathPositionAttention(in.dim(-1), heads, forget, r));
    }

    /**
     * The PaTH scores {@code q_i^T H_i ... H_{j+1} k_j} for {@code j <= i} (0 above the
     * diagonal).
     *
     * @param q queries {@code [b, h, t, d]}
     * @param k keys {@code [b, h, t, d]}
     * @param w unit Householder directions {@code [b, h, t, d]}
     * @param beta step sizes {@code [b, h, t]}
     * @return {@code [b, h, t, t]}
     */
    public static Var scores(Var q, Var k, Var w, Var beta) {
        int b = q.shape().dim(0);
        int h = q.shape().dim(1);
        int t = q.shape().dim(2);
        int d = q.shape().dim(3);
        Var running = Var.of(Tensor.zeros(Shape.of(b, h, t, d)));
        Var[] columns = new Var[t];
        for (int j = t - 1; j >= 0; j--) {
            float[] row = new float[t];
            row[j] = 1;
            running = running.add(q.mul(Tensor.of(Shape.of(t, 1), row)));        // activate query j
            columns[j] = running.matmul(k.slice(2, j, j + 1).transpose());       // [b, h, t, 1]
            if (j > 0) {
                Var wj = w.slice(2, j, j + 1);                                    // [b, h, 1, d]
                Var bj = beta.slice(2, j, j + 1).reshape(b, h, 1, 1);
                running = running.sub(running.matmul(wj.transpose()).mul(bj).matmul(wj));   // H_j R
            }
        }
        return Var.concat(3, columns);
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, features, "PathPositionAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = MixerOps.split(input.matmul(wq), heads);
        Var k = MixerOps.split(input.matmul(wk), heads);
        Var v = MixerOps.split(input.matmul(wv), heads);
        Var w = MixerOps.unit(MixerOps.split(input.matmul(ww), heads));
        Var beta = input.matmul(wBeta).add(betaBias).sigmoid().mul(2f).permute(0, 2, 1);   // [b, h, t]
        Var s = scores(q, k, w, beta).mul((float) (1 / Math.sqrt(headSize)));
        if (forget) {
            Var logF = input.matmul(wForget).add(forgetBias).sigmoid().add(1e-6f).log().permute(0, 2, 1);
            Var c = logF.cumsum(-1);                                              // [b, h, t]
            s = s.add(c.reshape(b, heads, t, 1).sub(c.reshape(b, heads, 1, t)).mul(MixerOps.lowerOnes(t)));
        }
        Tensor mask = RefinementOps.additive(RefinementOps.keep(t, t, 0, 0, true, 0, input.value().device()));
        Var a = MixerOps.softmax(s.add(mask));
        return MixerOps.merge(a.matmul(v)).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, ww, wBeta, betaBias, wo));
        if (forget) {
            p.add(wForget);
            p.add(forgetBias);
        }
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "PathPositionAttention(" + features + ", heads=" + heads + (forget ? ", FoX" : "") + ")";
    }
}
