package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * SCORE: causal ordering by score matching (Rolland et al., ICML 2022,
 * <a href="https://arxiv.org/abs/2203.04413">arXiv:2203.04413</a>). In a nonlinear additive-noise
 * model with Gaussian noise, {@code x_j} is a leaf exactly when the diagonal entry
 * {@code d^2 log p(x) / dx_j^2} of the score's Jacobian is constant, so its variance over the
 * samples is zero.
 *
 * <p>The Jacobian diagonal is estimated by the Stein identity with a Gaussian kernel of median
 * bandwidth {@code s}, exactly as in the authors' {@code Stein_hess}:
 * {@code G = (K + eta_G I)^-1 nabla K} (the score) and
 * {@code H = -G^2 + (K + eta_H I)^-1 nabla^2 K}. The leaf with the smallest variance of its
 * column of {@code H} is removed and the procedure repeats on the rest, giving a topological
 * order. Edges are then pruned CAM-style: each node is regressed on its predecessors with an
 * additive cubic-spline basis (truncated powers at the quartiles) and a predecessor is kept when
 * dropping its basis block fails an F-test at level {@code alpha}.
 */
public final class ScoreOrdering {

    private final double etaG;
    private final double etaH;
    private final double alpha;

    /**
     * Creates the learner.
     *
     * @param etaG the ridge of the score estimator, for example {@code 1e-3}
     * @param etaH the ridge of the Hessian estimator, for example {@code 1e-3}
     * @param alpha the pruning level, for example {@code 0.001}
     */
    public ScoreOrdering(double etaG, double etaH, double alpha) {
        this.etaG = etaG;
        this.etaH = etaH;
        this.alpha = alpha;
    }

    /**
     * A fitted graph.
     *
     * @param order the topological order, most upstream first
     * @param dag the pruned 0/1 DAG
     */
    public record Result(int[] order, double[][] dag) {
    }

    /**
     * The Stein estimate of the score Jacobian's diagonal at every sample.
     *
     * @param x the data {@code [n][d]}
     * @param etaG the score ridge
     * @param etaH the Hessian ridge
     * @return {@code [n][d]}
     */
    public static double[][] steinHessianDiagonal(double[][] x, double etaG, double etaH) {
        int n = x.length;
        int d = x[0].length;
        double[] dist = new double[n * (n - 1) / 2];
        int c = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                dist[c++] = Math.sqrt(MlMatrices.squaredDistance(x[i], x[j]));
            }
        }
        Arrays.sort(dist);
        double s = Math.max(dist[dist.length / 2], 1e-6);
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                k[i][j] = Math.exp(-MlMatrices.squaredDistance(x[i], x[j]) / (2 * s * s)) / s;
            }
        }
        double[][] nablaK = new double[n][d];
        double[][] nabla2K = new double[n][d];
        for (int a = 0; a < n; a++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    double diff = x[a][j] - x[i][j];
                    nablaK[a][j] -= diff * k[i][a] / (s * s);
                    nabla2K[a][j] += (-1 / (s * s) + diff * diff / (s * s * s * s)) * k[i][a];
                }
            }
        }
        double[][] g = solveRidge(k, etaG, nablaK);
        double[][] h = solveRidge(k, etaH, nabla2K);
        for (int a = 0; a < n; a++) {
            for (int j = 0; j < d; j++) {
                h[a][j] -= g[a][j] * g[a][j];
            }
        }
        return h;
    }

    private static double[][] solveRidge(double[][] k, double eta, double[][] b) {
        int n = k.length;
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            m[i] = k[i].clone();
            m[i][i] += eta;
        }
        return MlMatrices.choleskySolve(MlMatrices.cholesky(m), b);
    }

    /**
     * The topological order by repeated leaf removal.
     *
     * @param x the data
     * @return the order, most upstream first
     */
    public int[] order(double[][] x) {
        int d = x[0].length;
        double[][] z = LeastSquares.standardise(x);
        List<Integer> remaining = new ArrayList<>();
        for (int j = 0; j < d; j++) {
            remaining.add(j);
        }
        int[] order = new int[d];
        for (int pos = d - 1; pos >= 0; pos--) {
            if (remaining.size() == 1) {
                order[pos] = remaining.get(0);
                break;
            }
            int[] cols = remaining.stream().mapToInt(Integer::intValue).toArray();
            double[][] h = steinHessianDiagonal(LeastSquares.columns(z, cols), etaG, etaH);
            int leaf = 0;
            double best = Double.POSITIVE_INFINITY;
            for (int j = 0; j < cols.length; j++) {
                double m = 0;
                for (double[] row : h) {
                    m += row[j] / h.length;
                }
                double v = 0;
                for (double[] row : h) {
                    v += (row[j] - m) * (row[j] - m) / h.length;
                }
                if (v < best) {
                    best = v;
                    leaf = j;
                }
            }
            order[pos] = cols[leaf];
            remaining.remove(Integer.valueOf(cols[leaf]));
        }
        return order;
    }

    /**
     * Orders and prunes.
     *
     * @param x the data
     * @return the order and DAG
     */
    public Result fit(double[][] x) {
        int[] order = order(x);
        return new Result(order, prune(x, order, alpha));
    }

    /**
     * CAM-style pruning of the full DAG implied by an order.
     *
     * @param x the data
     * @param order the order
     * @param alpha the F-test level
     * @return the pruned DAG
     */
    public static double[][] prune(double[][] x, int[] order, double alpha) {
        int d = order.length;
        int n = x.length;
        double[][] z = LeastSquares.standardise(x);
        double[][][] basis = new double[d][][];
        for (int j = 0; j < d; j++) {
            basis[j] = splineBasis(LeastSquares.column(z, j));
        }
        double[][] g = new double[d][d];
        for (int k = 1; k < d; k++) {
            int target = order[k];
            double[] y = LeastSquares.column(z, target);
            int[] pred = Arrays.copyOf(order, k);
            double[][] full = design(basis, pred, -1, n);
            LeastSquares.Fit f = LeastSquares.fit(full, y, true, 1e-6);
            int pFull = full[0].length + 1;
            for (int a = 0; a < k; a++) {
                double[][] reduced = design(basis, pred, a, n);
                double rssR = LeastSquares.fit(reduced, y, true, 1e-6).rss();
                int q = basis[pred[a]][0].length;
                double fStat = ((rssR - f.rss()) / q) / (f.rss() / Math.max(1, n - pFull));
                if (Distributions.fUpper(fStat, q, n - pFull) < alpha) {
                    g[pred[a]][target] = 1;
                }
            }
        }
        return g;
    }

    private static double[][] design(double[][][] basis, int[] pred, int skip, int n) {
        int cols = 0;
        for (int a = 0; a < pred.length; a++) {
            cols += a == skip ? 0 : basis[pred[a]][0].length;
        }
        double[][] out = new double[n][cols];
        for (int i = 0; i < n; i++) {
            int o = 0;
            for (int a = 0; a < pred.length; a++) {
                if (a == skip) {
                    continue;
                }
                double[] b = basis[pred[a]][i];
                System.arraycopy(b, 0, out[i], o, b.length);
                o += b.length;
            }
        }
        return out;
    }

    /**
     * A cubic regression-spline basis (truncated powers at the quartiles).
     *
     * @param v a standardised variable
     * @return {@code [n][6]}
     */
    public static double[][] splineBasis(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        double[] knots = {s[s.length / 4], s[s.length / 2], s[3 * s.length / 4]};
        double[][] out = new double[v.length][6];
        for (int i = 0; i < v.length; i++) {
            double t = v[i];
            out[i][0] = t;
            out[i][1] = t * t;
            out[i][2] = t * t * t;
            for (int k = 0; k < 3; k++) {
                double u = Math.max(0, t - knots[k]);
                out[i][3 + k] = u * u * u;
            }
        }
        return out;
    }
}
