package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * APOLLO (Zhu et al. 2025, "APOLLO: SGD-like Memory, AdamW-level Performance",
 * <a href="https://arxiv.org/abs/2412.05270">arXiv:2412.05270</a>): Adam's element-wise
 * learning-rate adaptation is mostly redundant, and a coarse channel-wise (or even
 * tensor-wise) scaling of the raw gradient does as well. APOLLO estimates those scales
 * cheaply: the gradient {@code G} ({@code [m, n]}, {@code m} the shorter side) is projected
 * by a random Gaussian {@code P} ({@code [r, m]}) to {@code R = P G}, Adam's moments are kept
 * for {@code R} only, and each channel {@code j} (a column of length {@code m}) is scaled by
 * {@code s_j = ||adam(R)_j|| / ||R_j||}. The update is {@code alpha * G diag(s)}. The
 * projection is regenerated from a seed every {@code updateEvery} steps, so it costs no
 * memory (the moments carry over, as in the paper); unlike {@link GaLore} it needs no SVD.
 *
 * <p>A norm-growth limiter caps each step's update norm at {@code 1.01} times the previous
 * one, which replaces gradient clipping. {@link #mini} is APOLLO-Mini: rank 1 and a single
 * tensor-wise scale (the paper sets {@code alpha = sqrt(128)} for LLM-sized matrices), at
 * SGD's memory cost. Vectors and matrices whose shorter side is at most the rank use plain
 * Adam.
 */
public final class Apollo extends SlotOptimizer {

    private static final float BETA1 = 0.9f;
    private static final float BETA2 = 0.999f;
    private static final float GROWTH = 1.01f;

    private final int rank;
    private final int updateEvery;
    private final float scale;
    private final boolean tensorWise;
    private final float weightDecay;
    private final long seed;
    private final Map<Var, Integer> ids = new IdentityHashMap<>();

    /**
     * @param learningRate the step size, as for AdamW
     * @param rank the projection rank
     * @param updateEvery steps between new random projections, for example 200
     * @param scale the factor {@code alpha} on the scaled gradient, or 0 for {@code sqrt(m / r)},
     *     which gives the update Adam's size (a rank-{@code r} Adam step has norm about
     *     {@code sqrt(r / m)} of a full one); the paper uses 1, and {@code sqrt(128)} for Mini
     * @param tensorWise whether to use one scale per matrix instead of one per channel
     * @param weightDecay the decoupled weight decay
     * @param seed the seed of the projections
     */
    public Apollo(float learningRate, int rank, int updateEvery, float scale, boolean tensorWise, float weightDecay,
            long seed) {
        super(learningRate, "m", "v", "lastNorm");
        if (rank < 1 || updateEvery < 1 || scale < 0 || weightDecay < 0) {
            throw new IllegalArgumentException("bad APOLLO settings");
        }
        this.rank = rank;
        this.updateEvery = updateEvery;
        this.scale = scale;
        this.tensorWise = tensorWise;
        this.weightDecay = weightDecay;
        this.seed = seed;
    }

    /**
     * APOLLO with channel-wise scaling and {@code alpha = sqrt(m / r)}.
     *
     * @param learningRate the step size
     * @param rank the projection rank, for example a quarter of the shorter side
     * @return the optimizer
     */
    public static Apollo of(float learningRate, int rank) {
        return new Apollo(learningRate, rank, 200, 0, false, 0, 7);
    }

    /**
     * APOLLO-Mini: rank 1, tensor-wise scaling, {@code alpha = sqrt(m)}.
     *
     * @param learningRate the step size
     * @return the optimizer
     */
    public static Apollo mini(float learningRate) {
        return new Apollo(learningRate, 1, 200, 0, true, 0, 7);
    }

    private boolean projected(Tensor g) {
        return g.rank() == 2 && Math.min(g.shape().dim(0), g.shape().dim(1)) > rank;
    }

    private Tensor projection(Var p, int m, int t) {
        int id = ids.computeIfAbsent(p, k -> ids.size());
        long period = (t - 1) / updateEvery;
        SplittableRandom r = new SplittableRandom(seed * 1_000_003L + id * 7919L + period);
        return Tensor.normal(Shape.of(rank, m), 0, (float) (1 / Math.sqrt(rank)), r);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        float c1 = 1 - (float) Math.pow(BETA1, t);
        float c2 = 1 - (float) Math.pow(BETA2, t);
        float lr = learningRate();
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        if (!projected(g)) {
            Tensor m = s[0] == null ? g.mul(1 - BETA1) : s[0].mul(BETA1).add(g.mul(1 - BETA1));
            Tensor v = s[1] == null ? g.square().mul(1 - BETA2) : s[1].mul(BETA2).add(g.square().mul(1 - BETA2));
            p.assign(w.sub(m.div(c1).div(v.div(c2).sqrt().add(1e-8f)).mul(lr)));
            return new Tensor[] {m, v, null};
        }
        boolean tall = g.shape().dim(0) > g.shape().dim(1);
        Tensor gg = tall ? g.transpose() : g;   // [m, n], m the shorter side; channels are columns
        Tensor proj = projection(p, gg.shape().dim(0), t);
        Tensor r = proj.matmul(gg);   // [r, n]
        Tensor m = s[0] == null ? r.mul(1 - BETA1) : s[0].mul(BETA1).add(r.mul(1 - BETA1));
        Tensor v = s[1] == null ? r.square().mul(1 - BETA2) : s[1].mul(BETA2).add(r.square().mul(1 - BETA2));
        Tensor adam = m.div(c1).div(v.div(c2).sqrt().add(1e-8f));
        Tensor eps = Tensor.full(Shape.of(), 1e-12f);
        Tensor factor = tensorWise
                ? adam.square().sum().sqrt().div(r.square().sum().sqrt().maximum(eps))
                : adam.square().sum(0, true).sqrt().div(r.square().sum(0, true).sqrt().maximum(eps));   // [1, n]
        float alpha = scale > 0 ? scale : (float) Math.sqrt(gg.shape().dim(0) / (double) rank);
        Tensor u = gg.mul(factor).mul(alpha);
        Tensor norm = u.square().sum().sqrt().reshape(1);
        if (s[2] != null) {   // the norm-growth limiter
            Tensor cap = s[2].mul(GROWTH);
            Tensor shrink = cap.div(norm.maximum(eps)).minimum(Tensor.full(Shape.of(), 1f));
            u = u.mul(shrink);
            norm = norm.mul(shrink);
        }
        p.assign(w.sub((tall ? u.transpose() : u).mul(lr)));
        return new Tensor[] {m, v, norm};
    }

    @Override
    public String toString() {
        return "APOLLO(lr=" + learningRate() + ", rank=" + rank + (tensorWise ? ", tensor-wise" : "") + ")";
    }
}
