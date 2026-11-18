package io.github.zzuegg.brainiac.experimental.signal;

import io.github.zzuegg.brainiac.experimental.panel.TriplyRobustPanel;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.Arrays;

/**
 * Robust PCA: split a matrix {@code M = L + S} into a low-rank part and a sparse part.
 *
 * <ul>
 *   <li>{@link #pcp}: principal component pursuit, {@code min ||L||_* + lambda ||S||_1} subject
 *       to {@code L + S = M}, by the inexact augmented Lagrangian method: singular-value
 *       thresholding for {@code L} (reusing {@link TriplyRobustPanel#shrink}), soft
 *       thresholding for {@code S}, and a growing penalty {@code mu}; with
 *       {@code lambda = 1 / sqrt(max(m, n))} it recovers both parts exactly under incoherence
 *       (Candes, Li, Ma and Wright, JACM 2011, <a href="https://arxiv.org/abs/0912.3599">arXiv:0912.3599</a>;
 *       Lin, Chen and Ma 2010).</li>
 *   <li>{@link #alternatingProjections}: the non-convex variant for a known rank: alternate the
 *       best rank-r approximation of {@code M - S} with hard thresholding of the residual at a
 *       robust (median absolute deviation) level.</li>
 *   <li>{@link #truncatedPca}: the classical baseline, rank-r truncation of {@code M}.</li>
 * </ul>
 */
public final class RobustPca {

    private RobustPca() {
    }

    /**
     * A decomposition.
     *
     * @param low the low-rank part
     * @param sparse the sparse part
     * @param iterations iterations used
     */
    public record Decomposition(double[][] low, double[][] sparse, int iterations) {
    }

    private static double frob(double[][] a) {
        double s = 0;
        for (double[] row : a) {
            for (double v : row) {
                s += v * v;
            }
        }
        return Math.sqrt(s);
    }

    /**
     * Principal component pursuit by the inexact ALM.
     *
     * @param m the matrix
     * @param lambda the sparsity weight ({@code NaN} for {@code 1 / sqrt(max(rows, cols))})
     * @param maxIterations the iteration cap
     * @param tolerance the stop on {@code ||M - L - S||_F / ||M||_F}
     * @return the decomposition
     */
    public static Decomposition pcp(double[][] m, double lambda, int maxIterations, double tolerance) {
        int rows = m.length;
        int cols = m[0].length;
        double lam = Double.isNaN(lambda) ? 1 / Math.sqrt(Math.max(rows, cols)) : lambda;
        double norm2 = ThinSvd.of(m).s()[0];
        double maxAbs = 0;
        for (double[] row : m) {
            for (double v : row) {
                maxAbs = Math.max(maxAbs, Math.abs(v));
            }
        }
        double dual = Math.max(norm2, maxAbs / lam);
        double[][] y = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y[i][j] = m[i][j] / dual;
            }
        }
        double mu = 1.25 / norm2;
        double rho = 1.5;
        double mNorm = frob(m);
        double[][] l = new double[rows][cols];
        double[][] s = new double[rows][cols];
        int it = 0;
        for (; it < maxIterations; it++) {
            double[][] t = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    t[i][j] = m[i][j] - s[i][j] + y[i][j] / mu;
                }
            }
            l = TriplyRobustPanel.shrink(t, 1 / mu);
            double res = 0;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    s[i][j] = SparseRecovery.soft(m[i][j] - l[i][j] + y[i][j] / mu, lam / mu);
                    double z = m[i][j] - l[i][j] - s[i][j];
                    y[i][j] += mu * z;
                    res += z * z;
                }
            }
            mu = Math.min(mu * rho, 1e7);
            if (Math.sqrt(res) / mNorm < tolerance) {
                it++;
                break;
            }
        }
        return new Decomposition(l, s, it);
    }

    /**
     * The best rank-{@code r} approximation.
     *
     * @param m the matrix
     * @param rank the rank
     * @return {@code U_r S_r V_r^T}
     */
    public static double[][] truncatedPca(double[][] m, int rank) {
        ThinSvd svd = ThinSvd.of(m);
        int rows = m.length;
        int cols = m[0].length;
        double[][] out = new double[rows][cols];
        for (int k = 0; k < Math.min(rank, svd.s().length); k++) {
            for (int i = 0; i < rows; i++) {
                double u = svd.u()[i][k] * svd.s()[k];
                for (int j = 0; j < cols; j++) {
                    out[i][j] += u * svd.v()[j][k];
                }
            }
        }
        return out;
    }

    /**
     * Non-convex robust PCA by alternating projections for a known rank.
     *
     * @param m the matrix
     * @param rank the rank of {@code L}
     * @param iterations alternations
     * @return the decomposition
     */
    public static Decomposition alternatingProjections(double[][] m, int rank, int iterations) {
        int rows = m.length;
        int cols = m[0].length;
        double[][] s = new double[rows][cols];
        double[][] l = new double[rows][cols];
        for (int it = 0; it < iterations; it++) {
            double[][] t = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    t[i][j] = m[i][j] - s[i][j];
                }
            }
            l = truncatedPca(t, rank);
            double[] resid = new double[rows * cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    resid[i * cols + j] = Math.abs(m[i][j] - l[i][j]);
                }
            }
            double[] sorted = resid.clone();
            Arrays.sort(sorted);
            double mad = 1.4826 * sorted[sorted.length / 2];
            // the threshold starts high and decays towards 3 robust standard deviations
            double zeta = Math.max(3 * mad, sorted[sorted.length - 1] * Math.pow(0.5, it + 1));
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double r = m[i][j] - l[i][j];
                    s[i][j] = Math.abs(r) > zeta ? r : 0;
                }
            }
        }
        return new Decomposition(l, s, iterations);
    }

    /**
     * The relative error {@code ||est - truth||_F / ||truth||_F}.
     *
     * @param est the estimate
     * @param truth the truth
     * @return the relative error
     */
    public static double relativeError(double[][] est, double[][] truth) {
        double e = 0;
        double t = 0;
        for (int i = 0; i < truth.length; i++) {
            for (int j = 0; j < truth[0].length; j++) {
                e += (est[i][j] - truth[i][j]) * (est[i][j] - truth[i][j]);
                t += truth[i][j] * truth[i][j];
            }
        }
        return Math.sqrt(e / Math.max(t, 1e-30));
    }

    /**
     * The F1 score of a support (entries above {@code threshold} in absolute value) against the
     * true support.
     *
     * @param est the estimated sparse part
     * @param truth the true sparse part
     * @param threshold the magnitude counted as non-zero
     * @return the F1
     */
    public static double supportF1(double[][] est, double[][] truth, double threshold) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < truth.length; i++) {
            for (int j = 0; j < truth[0].length; j++) {
                boolean p = Math.abs(est[i][j]) > threshold;
                boolean t = truth[i][j] != 0;
                tp += p && t ? 1 : 0;
                fp += p && !t ? 1 : 0;
                fn += !p && t ? 1 : 0;
            }
        }
        return tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
    }
}
