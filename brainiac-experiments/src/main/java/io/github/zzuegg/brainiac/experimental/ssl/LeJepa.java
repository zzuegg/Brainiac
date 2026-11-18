package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The LeJEPA objective (Balestriero and LeCun, 2025): a joint-embedding predictive loss with
 * {@link Sigreg} as the only anti-collapse term, and none of the usual heuristics: no stop
 * gradient, no momentum teacher, no predictor network, no negatives and no schedules.
 *
 * <p>Every view of a sample (two or more "global" views and any number of local crops) is
 * embedded by the same network. The prediction term pulls each view's embedding to the
 * <em>center</em>, the mean embedding of the sample's global views, by mean squared error; the
 * regulariser makes each view's batch of embeddings an isotropic Gaussian. The total is
 * {@code (1 - lambda) * prediction + lambda * SIGReg}, with {@code lambda = 0.05} the paper's
 * robust default (Algorithm 2).
 *
 * <p>For the ablation the paper discusses, {@link Regularizer#VARIANCE_COVARIANCE} swaps SIGReg
 * for VICReg's variance and covariance terms (Bardes et al., 2022), the best-known explicit
 * moment-matching alternative.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2511.08544">arXiv:2511.08544</a>.
 */
public final class LeJepa {

    /** The anti-collapse term. */
    public enum Regularizer {
        /** Sketched isotropic Gaussian regularisation (the paper). */
        SIGREG,
        /** VICReg's hinge on the standard deviation plus the off-diagonal covariance. */
        VARIANCE_COVARIANCE
    }

    private final float lambda;
    private final int slices;
    private final Regularizer regularizer;

    /**
     * Creates the objective.
     *
     * @param lambda the regulariser's share in {@code [0, 1]}, 0.05 in the paper
     * @param slices SIGReg's random directions per call
     * @param regularizer the anti-collapse term
     */
    public LeJepa(float lambda, int slices, Regularizer regularizer) {
        if (!(lambda >= 0 && lambda <= 1)) {
            throw new IllegalArgumentException("lambda must be in [0, 1]: " + lambda);
        }
        this.lambda = lambda;
        this.slices = slices;
        this.regularizer = regularizer;
    }

    /**
     * The paper's default: {@code lambda = 0.05}, 256 slices, SIGReg.
     *
     * @return the objective
     */
    public static LeJepa defaults() {
        return new LeJepa(0.05f, Sigreg.DEFAULT_SLICES, Regularizer.SIGREG);
    }

    /** @return the regulariser's share */
    public float lambda() {
        return lambda;
    }

    /**
     * The LeJEPA loss of one batch.
     *
     * @param globalViews the global views' embeddings, each {@code [n, d]}, at least one
     * @param allViews every view's embeddings (the global views included), each {@code [n, d]}
     * @param random the source of SIGReg's directions
     * @return {@code (1 - lambda) prediction + lambda regulariser}
     */
    public Var loss(List<Var> globalViews, List<Var> allViews, RandomGenerator random) {
        Var reg = null;
        for (Var v : allViews) {
            Var r = regularizer == Regularizer.SIGREG ? Sigreg.loss(v, slices, Sigreg.DEFAULT_POINTS, random)
                    : varianceCovariance(v);
            reg = reg == null ? r : reg.add(r);
        }
        reg = reg.div(allViews.size());
        return prediction(globalViews, allViews).mul(1 - lambda).add(reg.mul(lambda));
    }

    /**
     * The prediction term: mean squared distance of every view to the mean of the global
     * views, gradients flowing into both.
     *
     * @param globalViews the global views' embeddings
     * @param allViews every view's embeddings
     * @return the mean over views, rows and dimensions
     */
    public static Var prediction(List<Var> globalViews, List<Var> allViews) {
        Var center = globalViews.get(0);
        for (int i = 1; i < globalViews.size(); i++) {
            center = center.add(globalViews.get(i));
        }
        center = center.div(globalViews.size());
        Var total = null;
        for (Var v : allViews) {
            Var l = center.sub(v).square().mean();
            total = total == null ? l : total.add(l);
        }
        return total.div(allViews.size());
    }

    /**
     * VICReg's variance and covariance terms on one view: {@code mean relu(1 - std_j)} plus
     * the sum of squared off-diagonal covariances divided by {@code d}.
     *
     * @param z the embeddings {@code [n, d]}
     * @return the regulariser
     */
    public static Var varianceCovariance(Var z) {
        int n = z.shape().dim(0);
        int d = z.shape().dim(1);
        Var c = z.sub(z.mean(0, true));
        Var std = c.square().mean(0, false).add(1e-4f).sqrt();
        Var variance = std.neg().add(1).relu().mean();
        Var cov = c.transpose().matmul(c).div(n - 1);
        float[] off = new float[d * d];
        for (int i = 0; i < d * d; i++) {
            off[i] = i / d == i % d ? 0 : 1;
        }
        Var covariance = cov.square().mul(Tensor.of(Shape.of(d, d), off, z.value().device())).sum().div(d);
        return variance.add(covariance);
    }
}
