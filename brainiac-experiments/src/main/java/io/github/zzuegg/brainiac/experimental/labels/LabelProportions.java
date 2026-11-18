package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Learning from label proportions: only the class proportions of groups (bags) are known.
 *
 * <ul>
 *   <li>{@link #dllpLoss} (DLLP; Ardehaly and Culotta, ICDM 2017,
 *       <a href="https://arxiv.org/abs/1709.04108">arXiv:1709.04108</a>): the model's predicted
 *       class distribution averaged over the bag, {@code p_bar}, against the known proportions
 *       {@code pi}, by cross-entropy {@code -sum_c pi_c log p_bar_c}.</li>
 *   <li>{@link #transportLabels}: proportion-constrained pseudo-labels, an entropic
 *       optimal-transport plan ({@link EntropicTransport#sinkhorn}) between the bag's instances
 *       (uniform mass) and its classes (mass {@code pi}) with cost {@code -log p}; each
 *       instance takes its row's largest entry, so the bag's pseudo-labels respect its
 *       proportions. Retraining on them is the EM-style refinement.</li>
 *   <li>{@link MeanMap} (Quadrianto, Smola, Caetano, Le, JMLR 2009,
 *       <a href="https://jmlr.org/papers/v10/quadrianto09a.html">JMLR 10</a>): the class means
 *       are recovered from the bag means by least squares, since {@code m_b = sum_c pi_bc mu_c},
 *       and new rows are classified by the nearest recovered mean, a linear baseline.</li>
 * </ul>
 */
public final class LabelProportions {

    private LabelProportions() {
    }

    /**
     * The DLLP proportion loss of one bag.
     *
     * @param logits the bag's logits, {@code [instances, classes]}
     * @param proportions the bag's class proportions
     * @return {@code -sum_c pi_c log mean_i p_ic}
     */
    public static Var dllpLoss(Var logits, double[] proportions) {
        int classes = proportions.length;
        Var p = logits.sub(logits.logsumexp(-1, true)).exp().mean(0, false);
        float[] pi = new float[classes];
        for (int c = 0; c < classes; c++) {
            pi[c] = (float) proportions[c];
        }
        return p.add(1e-7f).log().mul(Tensor.of(Shape.of(classes), pi)).sum().neg();
    }

    /**
     * Proportion-constrained pseudo-labels of one bag by entropic optimal transport.
     *
     * @param probabilities the model's probability rows for the bag's instances
     * @param proportions the bag's class proportions
     * @param epsilon the entropic regularisation, such as 0.05
     * @return a class per instance
     */
    public static int[] transportLabels(float[][] probabilities, double[] proportions, double epsilon) {
        int n = probabilities.length;
        int classes = proportions.length;
        double[][] cost = new double[n][classes];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < classes; c++) {
                cost[i][c] = -Math.log(Math.max(probabilities[i][c], 1e-9));
            }
        }
        double[] b = new double[classes];
        double sum = 0;
        for (int c = 0; c < classes; c++) {
            b[c] = proportions[c] + 1e-6;
            sum += b[c];
        }
        for (int c = 0; c < classes; c++) {
            b[c] /= sum;
        }
        double[][] plan = EntropicTransport.sinkhorn(EntropicTransport.uniform(n), b, cost, epsilon, 200, 1e-6)
                .coupling();
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (plan[i][c] > plan[i][best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    /** The mean-map estimator. */
    public static final class MeanMap {

        private final double[][] means;

        /**
         * Recovers the class means from bags.
         *
         * @param x the rows
         * @param bags each bag's row indices
         * @param proportions each bag's class proportions
         */
        public MeanMap(float[][] x, int[][] bags, double[][] proportions) {
            int d = x[0].length;
            int classes = proportions[0].length;
            float[] m = new float[bags.length * d];
            float[] p = new float[bags.length * classes];
            for (int b = 0; b < bags.length; b++) {
                for (int i : bags[b]) {
                    for (int f = 0; f < d; f++) {
                        m[b * d + f] += x[i][f] / bags[b].length;
                    }
                }
                for (int c = 0; c < classes; c++) {
                    p[b * classes + c] = (float) proportions[b][c];
                }
            }
            Tensor pi = Tensor.of(Shape.of(bags.length, classes), p);
            Tensor mb = Tensor.of(Shape.of(bags.length, d), m);
            Tensor gram = pi.transpose().matmul(pi).add(Tensor.of(Shape.of(classes, classes), ridge(classes, 1e-3f)));
            Tensor mu = Linalg.solve(gram, pi.transpose().matmul(mb));
            float[] v = mu.toArray();
            means = new double[classes][d];
            for (int c = 0; c < classes; c++) {
                for (int f = 0; f < d; f++) {
                    means[c][f] = v[c * d + f];
                }
            }
        }

        private static float[] ridge(int k, float lambda) {
            float[] out = new float[k * k];
            for (int i = 0; i < k; i++) {
                out[i * k + i] = lambda;
            }
            return out;
        }

        /**
         * @param row a row
         * @return the class of the nearest recovered mean
         */
        public int predict(float[] row) {
            int best = 0;
            double bestD = Double.POSITIVE_INFINITY;
            for (int c = 0; c < means.length; c++) {
                double s = 0;
                for (int f = 0; f < row.length; f++) {
                    double v = row[f] - means[c][f];
                    s += v * v;
                }
                if (s < bestD) {
                    bestD = s;
                    best = c;
                }
            }
            return best;
        }
    }
}
