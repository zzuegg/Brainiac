package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;

/**
 * SimSiam (Chen and He, CVPR 2021): a Siamese network that learns without negatives, without a
 * momentum encoder and with small batches. Both views go through the same encoder and
 * projector ({@code z}); a predictor MLP maps one view's projection to the other's
 * ({@code p}). The loss is the symmetrised negative cosine
 * {@code -(cos(p1, sg(z2)) + cos(p2, sg(z1))) / 2} with a stop-gradient {@code sg} on the
 * target branch. The stop-gradient is essential: the authors read the method as an EM-like
 * alternation, and without it (or without the predictor) the representation collapses to a
 * constant. Both ablations are switches here, so the collapse can be measured.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2011.10566">arXiv:2011.10566</a>.
 */
public final class SimSiam {

    private SimSiam() {
    }

    /**
     * The symmetrised SimSiam loss, in {@code [-1, 1]}; {@code -1} is perfect agreement.
     *
     * @param p1 the predictor output of view 1 {@code [n, d]}
     * @param z1 the projection of view 1 {@code [n, d]}
     * @param p2 the predictor output of view 2
     * @param z2 the projection of view 2
     * @param stopGradient whether the targets are detached (the method); false is the
     *     collapsing ablation
     * @return the loss
     */
    public static Var loss(Var p1, Var z1, Var p2, Var z2, boolean stopGradient) {
        Var t1 = stopGradient ? z1.detach() : z1;
        Var t2 = stopGradient ? z2.detach() : z2;
        return cosine(p1, t2).add(cosine(p2, t1)).mul(-0.5f);
    }

    /**
     * The mean cosine similarity of matching rows.
     *
     * @param a {@code [n, d]}
     * @param b {@code [n, d]}
     * @return the mean of {@code cos(a_i, b_i)}
     */
    public static Var cosine(Var a, Var b) {
        return SelfSupervisedLosses.normalize(a).mul(SelfSupervisedLosses.normalize(b)).sum(-1, false).mean();
    }
}
