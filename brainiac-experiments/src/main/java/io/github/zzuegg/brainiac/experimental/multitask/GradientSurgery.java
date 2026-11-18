package io.github.zzuegg.brainiac.experimental.multitask;

import java.util.random.RandomGenerator;

/**
 * Gradient surgery for multi-task learning: when task gradients conflict (negative cosine),
 * their sum can hurt one task, so the per-task gradients ({@link TaskGradients#of}) are edited
 * before they are added.
 *
 * <ul>
 *   <li>{@link #pcGrad} (Yu et al., NeurIPS 2020): each task gradient is projected onto the
 *       normal plane of every other task gradient it conflicts with, in random order.</li>
 *   <li>{@link GradVac} (Wang et al., ICLR 2021): instead of only removing conflict, each pair
 *       is pushed towards an adaptive cosine target (an EMA of the observed cosines).</li>
 *   <li>{@link #gradOps} (GradOPS, WSDM 2025): a gradient that conflicts with any other is
 *       projected onto the orthogonal complement of the span of <em>all</em> other task
 *       gradients (Gram-Schmidt), so every modified gradient is non-conflicting with every
 *       original one regardless of order. The modified gradients are then reweighted by
 *       {@code w_i = r_i^alpha / mean(r^alpha)}, with {@code r_i} the scalar projection of
 *       their sum on {@code g_i} relative to the mean: {@code alpha > 0} favours dominating
 *       tasks, {@code alpha < 0} dominated ones, {@code alpha = 0} gives the plain sum. If every
 *       gradient projects to zero, MGDA's min-norm point is used instead, as the paper
 *       suggests.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2001.06782">arXiv:2001.06782</a>,
 * <a href="https://arxiv.org/abs/2010.05874">arXiv:2010.05874</a>,
 * <a href="https://arxiv.org/abs/2503.03438">arXiv:2503.03438</a>.
 */
public final class GradientSurgery {

    private GradientSurgery() {
    }

    /**
     * PCGrad's combined direction.
     *
     * @param g per-task gradients {@code [tasks][params]}
     * @param random the source of the projection order
     * @return the sum of the projected gradients
     */
    public static double[] pcGrad(double[][] g, RandomGenerator random) {
        int t = g.length;
        double[] out = new double[g[0].length];
        for (int i = 0; i < t; i++) {
            double[] gi = g[i].clone();
            int[] order = new int[t];
            for (int k = 0; k < t; k++) {
                order[k] = k;
            }
            for (int k = t - 1; k > 0; k--) {
                int j = random.nextInt(k + 1);
                int tmp = order[k];
                order[k] = order[j];
                order[j] = tmp;
            }
            for (int j : order) {
                if (j == i) {
                    continue;
                }
                double d = TaskGradients.dot(gi, g[j]);
                if (d < 0) {
                    double nn = TaskGradients.dot(g[j], g[j]);
                    if (nn > 0) {
                        for (int k = 0; k < gi.length; k++) {
                            gi[k] -= d / nn * g[j][k];
                        }
                    }
                }
            }
            for (int k = 0; k < out.length; k++) {
                out[k] += gi[k];
            }
        }
        return out;
    }

    /**
     * GradOPS's combined direction.
     *
     * @param g per-task gradients {@code [tasks][params]}
     * @param alpha the trade-off exponent (0: plain sum of the projected gradients)
     * @return {@code sum_i w_i g_i'}
     */
    public static double[] gradOps(double[][] g, double alpha) {
        int t = g.length;
        double[][] projected = new double[t][];
        boolean allZero = true;
        for (int i = 0; i < t; i++) {
            boolean conflict = false;
            for (int j = 0; j < t; j++) {
                conflict |= j != i && TaskGradients.dot(g[i], g[j]) < 0;
            }
            double[] gi = g[i].clone();
            if (conflict) {
                // Gram-Schmidt basis of the other gradients, then remove those components
                double[][] basis = new double[t - 1][];
                int b = 0;
                for (int j = 0; j < t; j++) {
                    if (j == i) {
                        continue;
                    }
                    double[] u = g[j].clone();
                    for (int k = 0; k < b; k++) {
                        if (basis[k] == null) {
                            continue;
                        }
                        double c = TaskGradients.dot(u, basis[k]);
                        for (int m = 0; m < u.length; m++) {
                            u[m] -= c * basis[k][m];
                        }
                    }
                    double nu = TaskGradients.norm(u);
                    if (nu > 1e-12 * (TaskGradients.norm(g[j]) + 1e-30)) {
                        for (int m = 0; m < u.length; m++) {
                            u[m] /= nu;
                        }
                        basis[b] = u;
                    }
                    b++;
                }
                for (double[] u : basis) {
                    if (u == null) {
                        continue;
                    }
                    double c = TaskGradients.dot(gi, u);
                    for (int m = 0; m < gi.length; m++) {
                        gi[m] -= c * u[m];
                    }
                }
            }
            projected[i] = gi;
            allZero &= TaskGradients.norm(gi) <= 1e-12 * (TaskGradients.norm(g[i]) + 1e-30);
        }
        if (allZero) {
            double[] w = ParetoDescent.mgdaWeights(g);
            double[] d = TaskGradients.combine(g, w);
            for (int k = 0; k < d.length; k++) {
                d[k] *= t;
            }
            return d;
        }
        double[] sum = TaskGradients.combine(projected, ones(t));
        if (alpha == 0) {
            return sum;
        }
        double[] r = new double[t];
        double mean = 0;
        for (int i = 0; i < t; i++) {
            double ng = TaskGradients.norm(g[i]);
            r[i] = ng == 0 ? 0 : Math.max(0, TaskGradients.dot(sum, g[i]) / ng);
            mean += r[i] / t;
        }
        double[] w = new double[t];
        double wm = 0;
        for (int i = 0; i < t; i++) {
            w[i] = mean > 0 && r[i] > 0 ? Math.pow(r[i] / mean, alpha) : 0;
            wm += w[i] / t;
        }
        for (int i = 0; i < t; i++) {
            w[i] = wm > 0 ? w[i] / wm : 1;
        }
        return TaskGradients.combine(projected, w);
    }

    private static double[] ones(int t) {
        double[] w = new double[t];
        java.util.Arrays.fill(w, 1);
        return w;
    }

    /**
     * GradVac: gradient vaccine with adaptive pairwise cosine targets, kept between steps.
     */
    public static final class GradVac {

        private final double beta;
        private final double[][] target;

        /**
         * Creates zero targets (the first steps only remove conflicts, like PCGrad).
         *
         * @param tasks the number of tasks
         * @param beta the EMA rate of the targets, 0.01 in the paper
         */
        public GradVac(int tasks, double beta) {
            this.beta = beta;
            this.target = new double[tasks][tasks];
        }

        /**
         * The combined direction of one step; updates the targets.
         *
         * @param g per-task gradients
         * @param random the source of the pair order
         * @return the sum of the adjusted gradients
         */
        public double[] combine(double[][] g, RandomGenerator random) {
            int t = g.length;
            double[] out = new double[g[0].length];
            for (int i = 0; i < t; i++) {
                double[] gi = g[i].clone();
                for (int step = 0; step < t; step++) {
                    int j = random.nextInt(t);
                    if (j == i) {
                        continue;
                    }
                    double phi = TaskGradients.cosine(gi, g[j]);
                    double hat = target[i][j];
                    if (phi < hat) {
                        double ngi = TaskGradients.norm(gi);
                        double ngj = TaskGradients.norm(g[j]);
                        double scale = ngi * (hat * Math.sqrt(1 - phi * phi) - phi * Math.sqrt(1 - hat * hat))
                                / (ngj * Math.sqrt(1 - hat * hat) + 1e-12);
                        for (int k = 0; k < gi.length; k++) {
                            gi[k] += scale * g[j][k];
                        }
                    }
                    target[i][j] = (1 - beta) * hat + beta * phi;
                }
                for (int k = 0; k < out.length; k++) {
                    out[k] += gi[k];
                }
            }
            return out;
        }
    }
}
