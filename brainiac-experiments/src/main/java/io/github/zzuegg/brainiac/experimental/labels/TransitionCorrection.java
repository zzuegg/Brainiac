package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Loss correction with a class-conditional noise transition matrix {@code T[i][j] = P(noisy j |
 * clean i)} (Patrini et al., CVPR 2017, <a href="https://arxiv.org/abs/1609.03683">arXiv:1609.03683</a>;
 * Xia et al., NeurIPS 2019, <a href="https://arxiv.org/abs/1906.00189">arXiv:1906.00189</a>).
 *
 * <ul>
 *   <li>{@link #forwardLoss}: cross-entropy of the noisy label against {@code T^T p}, the
 *       predicted noisy-label distribution; consistent for the clean posterior {@code p}.</li>
 *   <li>{@link #backwardLoss}: the per-class losses multiplied by {@code T^-1}, an unbiased
 *       estimate of the clean loss (high variance when {@code T} is ill conditioned).</li>
 *   <li>{@link #estimateAnchors}: {@code T} estimated from anchor points, for each class the
 *       training example whose predicted probability of that class is at a high percentile (97%,
 *       robust against outliers), its predicted distribution becoming that row of {@code T}.</li>
 *   <li>{@link Revision} (T-revision): a learnable slack {@code dT} added to the anchor estimate
 *       and trained with the classifier through importance reweighting {@code beta = p_y / (T^T
 *       p)_y}, which fixes errors of the anchor estimate.</li>
 * </ul>
 */
public final class TransitionCorrection {

    private TransitionCorrection() {
    }

    /**
     * Forward correction.
     *
     * @param logits the classifier's logits (clean-class scores), {@code [batch, classes]}
     * @param noisyIds the noisy labels {@code [batch]}
     * @param transition {@code T}
     * @return {@code mean -log (T^T softmax(logits))_noisy}
     */
    public static Var forwardLoss(Var logits, Tensor noisyIds, double[][] transition) {
        Var q = SoftTargets.softmax(logits).matmul(Var.of(matrix(transition)));
        return q.add(1e-7f).log().mul(RobustLosses.oneHot(logits, noisyIds)).sum(-1, false).neg().mean();
    }

    /**
     * Backward correction.
     *
     * @param logits the classifier's logits
     * @param noisyIds the noisy labels
     * @param transition {@code T}
     * @return {@code mean (T^-1 l(p))_noisy} with {@code l_k = -log p_k}
     */
    public static Var backwardLoss(Var logits, Tensor noisyIds, double[][] transition) {
        Tensor inv = Linalg.inverse(matrix(transition));
        Var perClass = SoftTargets.logSoftmax(logits).neg();
        Var corrected = perClass.matmul(Var.of(inv.transpose()));
        return corrected.mul(RobustLosses.oneHot(logits, noisyIds)).sum(-1, false).mean();
    }

    /**
     * Estimates {@code T} from anchor points.
     *
     * @param probabilities a noisily trained classifier's softmax rows on the training set
     * @param percentile which percentile of each class's probability marks its anchor, such as 0.97
     * @return the estimated transition matrix
     */
    public static double[][] estimateAnchors(float[][] probabilities, double percentile) {
        int classes = probabilities[0].length;
        double[][] t = new double[classes][];
        for (int i = 0; i < classes; i++) {
            final int c = i;
            Integer[] order = new Integer[probabilities.length];
            for (int k = 0; k < order.length; k++) {
                order[k] = k;
            }
            Arrays.sort(order, (a, b) -> Float.compare(probabilities[a][c], probabilities[b][c]));
            int anchor = order[(int) Math.min(order.length - 1, Math.floor(percentile * order.length))];
            t[i] = new double[classes];
            for (int j = 0; j < classes; j++) {
                t[i][j] = probabilities[anchor][j];
            }
        }
        return t;
    }

    /**
     * A steadier anchor estimate: row {@code i} of {@code T} is the mean predicted distribution of
     * the examples whose most likely noisy label is {@code i}. With pair-flip or other diagonally
     * dominant noise, those are mostly clean examples of class {@code i}, and averaging avoids
     * picking a single over-confident outlier as the anchor.
     *
     * @param probabilities a noisily trained classifier's softmax rows on held-out inputs
     * @return the estimated transition matrix (a row with no examples falls back to identity)
     */
    public static double[][] estimateMeanAnchors(float[][] probabilities) {
        int classes = probabilities[0].length;
        double[][] t = new double[classes][classes];
        int[] count = new int[classes];
        int[] arg = SoftTargets.argmax(probabilities);
        for (int k = 0; k < probabilities.length; k++) {
            count[arg[k]]++;
            for (int j = 0; j < classes; j++) {
                t[arg[k]][j] += probabilities[k][j];
            }
        }
        for (int i = 0; i < classes; i++) {
            for (int j = 0; j < classes; j++) {
                t[i][j] = count[i] == 0 ? (i == j ? 1 : 0) : t[i][j] / count[i];
            }
        }
        return t;
    }

    /**
     * @param a a matrix
     * @param b another of the same shape
     * @return the mean absolute difference of their entries
     */
    public static double meanAbsoluteError(double[][] a, double[][] b) {
        double s = 0;
        int n = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++) {
                s += Math.abs(a[i][j] - b[i][j]);
                n++;
            }
        }
        return s / n;
    }

    /** T-revision: the anchor estimate plus a learned slack, trained by importance reweighting. */
    public static final class Revision {

        private final Tensor base;
        private final Var slack;

        /**
         * Creates the slack, zero at first.
         *
         * @param estimate the anchor estimate of {@code T}
         */
        public Revision(double[][] estimate) {
            this.base = matrix(estimate);
            int k = estimate.length;
            this.slack = Var.param(Tensor.of(Shape.of(k, k), new float[k * k]));
        }

        /** @return the slack, to add to the optimised parameters */
        public List<Var> parameters() {
            return List.of(slack);
        }

        /** @return the revised, row-normalised {@code T} */
        public Var transition() {
            Var t = Var.of(base).add(slack).clamp(1e-6f, 1f);
            return t.div(t.sum(1, true));
        }

        /** @return the revised {@code T} as numbers */
        public double[][] current() {
            float[] v = transition().value().toArray();
            int k = base.shape().dim(0);
            double[][] out = new double[k][k];
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    out[i][j] = v[i * k + j];
                }
            }
            return out;
        }

        /**
         * The importance-reweighted loss.
         *
         * @param logits the classifier's logits
         * @param noisyIds the noisy labels
         * @return {@code mean beta_i * -log (T^T p_i)_noisy}, {@code beta = p_y / (T^T p)_y} held constant
         */
        public Var loss(Var logits, Tensor noisyIds) {
            Tensor oh = RobustLosses.oneHot(logits, noisyIds);
            Var p = SoftTargets.softmax(logits);
            Var q = p.matmul(transition());
            Tensor py = p.value().mul(oh).sum(-1, false);
            Tensor qy = q.value().mul(oh).sum(-1, false);
            Tensor beta = py.div(qy.add(1e-7f));
            return q.add(1e-7f).log().mul(oh).sum(-1, false).mul(beta).neg().mean();
        }
    }

    static Tensor matrix(double[][] m) {
        int r = m.length;
        int c = m[0].length;
        float[] v = new float[r * c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                v[i * c + j] = (float) m[i][j];
            }
        }
        return Tensor.of(Shape.of(r, c), v);
    }
}
