package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * NOTEARS: structure learning of a linear structural equation model {@code X = X W + noise}
 * as continuous optimization. The weighted adjacency {@code W} minimizes the least-squares
 * loss {@code |X - X W|^2 / 2n} plus an L1 penalty, subject to a smooth equality constraint
 * that holds exactly when {@code W} is acyclic, solved by the augmented Lagrangian method.
 * Small weights are then cut at a threshold.
 *
 * <p>The constraint used is the polynomial form {@code h(W) = tr((I + W o W / d)^d) - d}
 * (Yu et al. 2019, DAG-GNN), which is zero exactly for DAGs like the paper's
 * {@code tr(exp(W o W)) - d} and needs only a matrix power ({@link Linalg#matrixPower});
 * its gradient is {@code ((I + W o W / d)^(d-1))^T o 2W}. {@code W} is split into
 * non-negative parts {@code W+ - W-} so the L1 term is smooth, and the inner problems are
 * solved by projected Adam with a learning rate decaying to zero (an approximate inner
 * solve, so the penalty stops growing at {@code rho = 1e6} and the constraint at
 * {@code h <= 1e-6}; the threshold removes what is left of any cycle).
 *
 * <p>Paper: Zheng, Aragam, Ravikumar, Xing, "DAGs with NO TEARS: Continuous Optimization
 * for Structure Learning" (2018), <a href="https://arxiv.org/abs/1803.01422">arXiv:1803.01422</a>.
 */
public final class Notears {

    private final double lambda;
    private final double threshold;
    private final int innerSteps;
    private final int outerSteps;

    /**
     * Creates the learner.
     *
     * @param lambda the L1 weight, such as 0.05
     * @param threshold the absolute weight below which an edge is dropped, such as 0.3
     * @param innerSteps Adam steps per augmented-Lagrangian round
     * @param outerSteps the most augmented-Lagrangian rounds
     */
    public Notears(double lambda, double threshold, int innerSteps, int outerSteps) {
        this.lambda = lambda;
        this.threshold = threshold;
        this.innerSteps = innerSteps;
        this.outerSteps = outerSteps;
    }

    /**
     * Learns the weighted adjacency.
     *
     * @param x samples {@code [n][d]}; centered here
     * @return {@code W[i][j]} the weight of edge {@code i -> j}, 0 where there is none
     */
    public double[][] fit(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        for (double[] r : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += r[j] / n;
            }
        }
        double[][] cov = new double[d][d];
        for (double[] r : x) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    cov[i][j] += (r[i] - mean[i]) * (r[j] - mean[j]) / n;
                }
            }
        }
        double[] wp = new double[d * d];
        double[] wn = new double[d * d];
        double rho = 1;
        double alpha = 0;
        double h = Double.POSITIVE_INFINITY;
        for (int outer = 0; outer < outerSteps && h > 1e-6; outer++) {
            double hNew = h;
            while (true) {
                inner(cov, wp, wn, rho, alpha, d);
                hNew = constraint(weights(wp, wn, d), d)[0];
                if (hNew > 0.25 * h && rho < 1e6) {
                    rho *= 10;
                } else {
                    break;
                }
            }
            h = hNew;
            alpha += rho * h;
        }
        double[][] w = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double v = wp[i * d + j] - wn[i * d + j];
                w[i][j] = i != j && Math.abs(v) >= threshold ? v : 0;
            }
        }
        return w;
    }

    private static double[] weights(double[] wp, double[] wn, int d) {
        double[] w = new double[d * d];
        for (int k = 0; k < d * d; k++) {
            w[k] = wp[k] - wn[k];
        }
        return w;
    }

    /** {@code {h, grad h...}} for a flattened {@code W}. */
    private static double[] constraint(double[] w, int d) {
        double[] m = new double[d * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                m[i * d + j] = (i == j ? 1 : 0) + w[i * d + j] * w[i * d + j] / d;
            }
        }
        Tensor mt = Tensor.of(Shape.of(d, d), m, DType.FLOAT64, Device.cpu());
        double[] pm1 = Linalg.matrixPower(mt, d - 1).toDoubleArray();
        double tr = 0;
        for (int i = 0; i < d; i++) {
            for (int k = 0; k < d; k++) {
                tr += pm1[i * d + k] * m[k * d + i];
            }
        }
        double[] out = new double[1 + d * d];
        out[0] = tr - d;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                out[1 + i * d + j] = pm1[j * d + i] * 2 * w[i * d + j];
            }
        }
        return out;
    }

    private void inner(double[][] cov, double[] wp, double[] wn, double rho, double alpha, int d) {
        double[] mp = new double[d * d];
        double[] vp = new double[d * d];
        double[] mn = new double[d * d];
        double[] vn = new double[d * d];
        for (int s = 1; s <= innerSteps; s++) {
            double lr = 0.02 * (1 - (s - 1.0) / innerSteps);   // decays to 0, so each round settles
            double[] w = weights(wp, wn, d);
            double[] hc = constraint(w, d);
            double h = hc[0];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    int k = i * d + j;
                    // d/dW of tr((I - W)^T S (I - W)) / 2 is S W - S
                    double g = -cov[i][j];
                    for (int m = 0; m < d; m++) {
                        g += cov[i][m] * w[m * d + j];
                    }
                    g += (rho * h + alpha) * hc[1 + k];
                    double gp = g + lambda;
                    double gn = -g + lambda;
                    mp[k] = 0.9 * mp[k] + 0.1 * gp;
                    vp[k] = 0.999 * vp[k] + 0.001 * gp * gp;
                    mn[k] = 0.9 * mn[k] + 0.1 * gn;
                    vn[k] = 0.999 * vn[k] + 0.001 * gn * gn;
                    double c1 = 1 - Math.pow(0.9, s);
                    double c2 = 1 - Math.pow(0.999, s);
                    wp[k] = i == j ? 0 : Math.max(0, wp[k] - lr * mp[k] / c1 / (Math.sqrt(vp[k] / c2) + 1e-8));
                    wn[k] = i == j ? 0 : Math.max(0, wn[k] - lr * mn[k] / c1 / (Math.sqrt(vn[k] / c2) + 1e-8));
                }
            }
        }
    }

    /**
     * The structural Hamming distance between two graphs: missing plus extra edges, a
     * reversed edge counting once.
     *
     * @param truth the true adjacency (non-zero is an edge {@code i -> j})
     * @param estimate the estimated adjacency
     * @return the distance
     */
    public static int shd(double[][] truth, double[][] estimate) {
        int d = truth.length;
        int dist = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i + 1; j < d; j++) {
                boolean tij = truth[i][j] != 0;
                boolean tji = truth[j][i] != 0;
                boolean eij = estimate[i][j] != 0;
                boolean eji = estimate[j][i] != 0;
                if (tij != eij || tji != eji) {
                    dist++;
                }
            }
        }
        return dist;
    }

    /**
     * The skeleton distance: pairs connected in one graph and not the other, directions ignored.
     *
     * @param truth the true adjacency
     * @param estimate an estimated adjacency or undirected skeleton
     * @return the number of differing pairs
     */
    public static int skeletonDistance(double[][] truth, double[][] estimate) {
        int d = truth.length;
        int dist = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i + 1; j < d; j++) {
                boolean t = truth[i][j] != 0 || truth[j][i] != 0;
                boolean e = estimate[i][j] != 0 || estimate[j][i] != 0;
                dist += t != e ? 1 : 0;
            }
        }
        return dist;
    }
}
