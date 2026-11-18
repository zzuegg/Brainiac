package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Linear non-Gaussian acyclic models (LiNGAM): with non-Gaussian noise a linear SEM
 * {@code x = B x + e} is identifiable beyond its Markov equivalence class.
 *
 * <ul>
 *   <li>{@link #ica}: ICA-LiNGAM (Shimizu et al., JMLR 2006,
 *       <a href="https://www.jmlr.org/papers/v7/shimizu06a.html">v7/shimizu06a</a>). FastICA
 *       ({@link ComponentAnalysis#fastIca}) gives an unmixing matrix {@code W}; its rows are
 *       permuted by a linear assignment that keeps the diagonal large, scaled to a unit diagonal,
 *       {@code B = I - W'}, and the causal order is read off after zeroing the smallest entries
 *       until {@code B} is permutable to strictly lower-triangular form.</li>
 *   <li>{@link #direct}: DirectLiNGAM (Shimizu et al., JMLR 2011,
 *       <a href="https://arxiv.org/abs/1101.2489">arXiv:1101.2489</a>) with the pairwise
 *       likelihood-ratio measure of Hyvärinen and Smith (JMLR 2013): repeatedly pick the variable
 *       whose regression residuals are most independent of it (maximum-entropy approximations of
 *       differential entropy), regress it out of the rest, and continue.</li>
 *   <li>Weights on the found order are estimated by an adaptive lasso (Zou 2006): OLS on the
 *       predecessors, then an L1 fit with penalties {@code 1 / |b_ols|}, the level chosen by
 *       BIC.</li>
 * </ul>
 */
public final class Lingam {

    private Lingam() {
    }

    /**
     * A fitted model.
     *
     * @param order the causal order, most upstream first
     * @param weights {@code weights[i][j]}: the effect of {@code i} on {@code j}
     */
    public record Result(int[] order, double[][] weights) {

        /**
         * @param threshold an absolute weight threshold
         * @return the 0/1 DAG of weights above it
         */
        public double[][] dag(double threshold) {
            int d = weights.length;
            double[][] g = new double[d][d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    g[i][j] = Math.abs(weights[i][j]) > threshold ? 1 : 0;
                }
            }
            return g;
        }
    }

    // ------------------------------------------------------------------ DirectLiNGAM

    private static double entropy(double[] u) {
        double k1 = 79.047;
        double k2 = 7.4129;
        double gamma = 0.37457;
        double a = 0;
        double b = 0;
        for (double v : u) {
            a += Math.log(Math.cosh(v)) / u.length;
            b += v * Math.exp(-v * v / 2) / u.length;
        }
        return (1 + Math.log(2 * Math.PI)) / 2 - k1 * (a - gamma) * (a - gamma) - k2 * b * b;
    }

    private static double[] residual(double[] xi, double[] xj) {
        double c = 0;
        double v = 0;
        for (int k = 0; k < xi.length; k++) {
            c += xi[k] * xj[k];
            v += xj[k] * xj[k];
        }
        double beta = v == 0 ? 0 : c / v;
        double[] r = new double[xi.length];
        for (int k = 0; k < r.length; k++) {
            r[k] = xi[k] - beta * xj[k];
        }
        return r;
    }

    private static double[] standard(double[] x) {
        double m = 0;
        for (double v : x) {
            m += v / x.length;
        }
        double s = 0;
        for (double v : x) {
            s += (v - m) * (v - m) / x.length;
        }
        double sd = Math.sqrt(Math.max(s, 1e-300));
        double[] out = new double[x.length];
        for (int k = 0; k < x.length; k++) {
            out[k] = (x[k] - m) / sd;
        }
        return out;
    }

    /**
     * The pairwise likelihood-ratio measure: positive when {@code x_i -> x_j} is the more likely
     * direction.
     *
     * @param xi one standardised variable
     * @param xj the other, standardised
     * @return {@code H(x_j) + H(r_i|j) - H(x_i) - H(r_j|i)}
     */
    public static double pairwiseMeasure(double[] xi, double[] xj) {
        double[] rij = standard(residual(xi, xj));
        double[] rji = standard(residual(xj, xi));
        return entropy(xj) + entropy(rij) - entropy(xi) - entropy(rji);
    }

    /**
     * DirectLiNGAM.
     *
     * @param x the data {@code [n][d]}
     * @return the order and adaptive-lasso weights
     */
    public static Result direct(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[][] cols = new double[d][];
        for (int j = 0; j < d; j++) {
            cols[j] = LeastSquares.column(x, j);
        }
        List<Integer> remaining = new ArrayList<>();
        for (int j = 0; j < d; j++) {
            remaining.add(j);
        }
        int[] order = new int[d];
        for (int step = 0; step < d; step++) {
            int best = remaining.get(0);
            if (remaining.size() > 1) {
                double bestScore = Double.NEGATIVE_INFINITY;
                double[][] std = new double[d][];
                for (int j : remaining) {
                    std[j] = standard(cols[j]);
                }
                for (int i : remaining) {
                    double m = 0;
                    for (int j : remaining) {
                        if (j != i) {
                            double v = Math.min(0, pairwiseMeasure(std[i], std[j]));
                            m += v * v;
                        }
                    }
                    if (-m > bestScore) {
                        bestScore = -m;
                        best = i;
                    }
                }
            }
            order[step] = best;
            remaining.remove(Integer.valueOf(best));
            for (int j : remaining) {
                cols[j] = residual(cols[j], cols[best]);
            }
        }
        return new Result(order, adaptiveLasso(x, order, n));
    }

    // ------------------------------------------------------------------ ICA-LiNGAM

    /**
     * ICA-LiNGAM.
     *
     * @param x the data {@code [n][d]}
     * @param iterations FastICA iterations
     * @param seed the ICA seed
     * @return the order and adaptive-lasso weights
     */
    public static Result ica(double[][] x, int iterations, long seed) {
        int d = x[0].length;
        ComponentAnalysis ica = ComponentAnalysis.fastIca(x, d, iterations, seed);
        // recover the affine unmixing map s = W (x - m) column by column
        double[] mean = new double[d];
        for (double[] r : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += r[j] / x.length;
            }
        }
        double[][] probe = new double[d + 1][];
        probe[0] = mean.clone();
        for (int j = 0; j < d; j++) {
            probe[j + 1] = mean.clone();
            probe[j + 1][j] += 1;
        }
        double[][] s = ica.transform(probe);
        double[][] w = new double[d][d];
        for (int r = 0; r < d; r++) {
            for (int j = 0; j < d; j++) {
                w[r][j] = s[j + 1][r] - s[0][r];
            }
        }
        // permute rows so that no diagonal entry is small: assign row r to variable j at cost 1/|w|
        double[] cost = new double[d * d];
        for (int r = 0; r < d; r++) {
            for (int j = 0; j < d; j++) {
                cost[r * d + j] = 1 / Math.max(Math.abs(w[r][j]), 1e-12);
            }
        }
        int[] assign = Linalg.linearAssignment(Tensor.of(Shape.of(d, d), cost, DType.FLOAT64, Device.cpu()));
        double[][] wp = new double[d][];
        for (int r = 0; r < d; r++) {
            wp[assign[r]] = w[r];
        }
        double[][] b = new double[d][d];   // b[j][k]: effect of k on j
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                b[j][k] = (j == k ? 1 : 0) - wp[j][k] / wp[j][j];
            }
        }
        int[] order = orderFrom(b);
        return new Result(order, adaptiveLasso(x, order, x.length));
    }

    /** Zeroes the smallest entries of {@code b} until a strictly lower-triangular order exists. */
    private static int[] orderFrom(double[][] b) {
        int d = b.length;
        List<double[]> entries = new ArrayList<>();
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                if (j != k) {
                    entries.add(new double[] {Math.abs(b[j][k]), j, k});
                }
            }
        }
        entries.sort((p, q) -> Double.compare(p[0], q[0]));
        boolean[][] nz = new boolean[d][d];
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                nz[j][k] = j != k;
            }
        }
        int start = d * (d + 1) / 2 - d;
        for (int e = 0; e < entries.size(); e++) {
            double[] en = entries.get(e);
            nz[(int) en[1]][(int) en[2]] = false;
            if (e + 1 >= start) {
                int[] order = topological(nz);
                if (order != null) {
                    return order;
                }
            }
        }
        int[] id = new int[d];
        Arrays.setAll(id, i -> i);
        return id;
    }

    private static int[] topological(boolean[][] parentOf) {
        int d = parentOf.length;
        boolean[] done = new boolean[d];
        int[] order = new int[d];
        for (int step = 0; step < d; step++) {
            int pick = -1;
            for (int j = 0; j < d && pick < 0; j++) {
                if (done[j]) {
                    continue;
                }
                boolean root = true;
                for (int k = 0; k < d; k++) {
                    root &= done[k] || !parentOf[j][k];
                }
                if (root) {
                    pick = j;
                }
            }
            if (pick < 0) {
                return null;
            }
            done[pick] = true;
            order[step] = pick;
        }
        return order;
    }

    // ------------------------------------------------------------------ weights

    /**
     * Adaptive-lasso weights for every node regressed on its predecessors in an order.
     *
     * @param x the data
     * @param order the causal order
     * @param n the sample size used in BIC
     * @return {@code w[i][j]}: the effect of {@code i} on {@code j}
     */
    public static double[][] adaptiveLasso(double[][] x, int[] order, int n) {
        int d = order.length;
        double[][] w = new double[d][d];
        for (int k = 1; k < d; k++) {
            int target = order[k];
            int[] pred = Arrays.copyOf(order, k);
            double[] beta = lassoBic(LeastSquares.columns(x, pred), LeastSquares.column(x, target), n, true);
            for (int a = 0; a < k; a++) {
                w[pred[a]][target] = beta[a];
            }
        }
        return w;
    }

    /**
     * An (adaptive) lasso regression with the level chosen by BIC over a geometric grid, by
     * coordinate descent on centred data.
     *
     * @param x predictors {@code [n][p]}
     * @param y the response
     * @param n the sample size in BIC
     * @param adaptive whether to weight penalties by {@code 1/|b_ols|}
     * @return the coefficients (no intercept: data are centred internally)
     */
    public static double[] lassoBic(double[][] x, double[] y, int n, boolean adaptive) {
        int p = x[0].length;
        int m = y.length;
        double[][] xc = new double[m][p];
        double[] yc = new double[m];
        double my = 0;
        double[] mx = new double[p];
        for (int i = 0; i < m; i++) {
            my += y[i] / m;
            for (int a = 0; a < p; a++) {
                mx[a] += x[i][a] / m;
            }
        }
        for (int i = 0; i < m; i++) {
            yc[i] = y[i] - my;
            for (int a = 0; a < p; a++) {
                xc[i][a] = x[i][a] - mx[a];
            }
        }
        double[] ols = LeastSquares.fit(xc, yc, false, 1e-8).beta();
        double[] pen = new double[p];
        for (int a = 0; a < p; a++) {
            pen[a] = adaptive ? 1 / Math.max(Math.abs(ols[a]), 1e-6) : 1;
        }
        double[] sq = new double[p];
        double lmax = 0;
        for (int a = 0; a < p; a++) {
            double c = 0;
            for (int i = 0; i < m; i++) {
                sq[a] += xc[i][a] * xc[i][a];
                c += xc[i][a] * yc[i];
            }
            lmax = Math.max(lmax, Math.abs(c) / pen[a]);
        }
        double[] best = new double[p];
        double bestBic = Double.POSITIVE_INFINITY;
        double[] beta = new double[p];
        for (int g = 0; g < 20; g++) {
            double lambda = lmax * Math.pow(1e-4, g / 19.0);
            beta = coordinateDescent(xc, yc, sq, pen, lambda, beta);
            double rss = 0;
            for (int i = 0; i < m; i++) {
                double f = 0;
                for (int a = 0; a < p; a++) {
                    f += xc[i][a] * beta[a];
                }
                rss += (yc[i] - f) * (yc[i] - f);
            }
            int nz = 0;
            for (double v : beta) {
                nz += v != 0 ? 1 : 0;
            }
            double bic = n * Math.log(Math.max(rss / m, 1e-300)) + nz * Math.log(n);
            if (bic < bestBic) {
                bestBic = bic;
                best = beta.clone();
            }
        }
        return best;
    }

    /**
     * Coordinate descent for {@code 1/2 ||y - X b||^2 + lambda sum pen_a |b_a|}.
     *
     * @param x centred predictors
     * @param y centred response
     * @param sq column sums of squares
     * @param pen per-coefficient penalty factors
     * @param lambda the level
     * @param start a warm start
     * @return the solution
     */
    public static double[] coordinateDescent(double[][] x, double[] y, double[] sq, double[] pen, double lambda,
            double[] start) {
        int p = sq.length;
        int m = y.length;
        double[] b = start.clone();
        double[] r = y.clone();
        for (int i = 0; i < m; i++) {
            for (int a = 0; a < p; a++) {
                r[i] -= x[i][a] * b[a];
            }
        }
        for (int sweep = 0; sweep < 200; sweep++) {
            double change = 0;
            for (int a = 0; a < p; a++) {
                if (sq[a] == 0) {
                    continue;
                }
                double rho = 0;
                for (int i = 0; i < m; i++) {
                    rho += x[i][a] * (r[i] + x[i][a] * b[a]);
                }
                double t = lambda * pen[a];
                double nb = Math.signum(rho) * Math.max(0, Math.abs(rho) - t) / sq[a];
                double delta = nb - b[a];
                if (delta != 0) {
                    for (int i = 0; i < m; i++) {
                        r[i] -= x[i][a] * delta;
                    }
                    change = Math.max(change, Math.abs(delta));
                    b[a] = nb;
                }
            }
            if (change < 1e-7) {
                break;
            }
        }
        return b;
    }
}
