package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * Hierarchical forecast reconciliation: base forecasts {@code y_hat} of every node of a
 * hierarchy with summing matrix {@code S} ({@code [nodes][leaves]}) are mapped to coherent
 * forecasts {@code S b}.
 *
 * <ul>
 *   <li>{@link Method#BOTTOM_UP}: {@code b} is the leaves' base forecasts;</li>
 *   <li>the projections {@code b = (S^T W^-1 S)^-1 S^T W^-1 y_hat} (Hyndman et al. 2011,
 *       <a href="https://doi.org/10.1016/j.csda.2011.03.006">doi</a>; Wickramasuriya,
 *       Athanasopoulos and Hyndman 2019,
 *       <a href="https://doi.org/10.1080/01621459.2018.1448825">doi</a>) with {@code W = I}
 *       ({@link Method#OLS}), the number of leaves under each node ({@link Method#WLS_STRUCT}),
 *       the base residual variances ({@link Method#WLS_VAR}) or the shrunk residual covariance
 *       ({@link Method#MINT_SHRINK}: Schafer-Strimmer shrinkage of the correlations towards 0
 *       with the data-driven intensity);</li>
 *   <li>robust M-estimation ({@link Method#ROBUST_HUBER}, {@link Method#ROBUST_LAD}; after
 *       <a href="https://arxiv.org/abs/2602.22694">arXiv:2602.22694</a>): the weighted
 *       least-squares loss is replaced by a robust loss of the base-forecast errors, with the Huber ({@code k = 1.345})
 *       or absolute loss, solved by iteratively reweighted least squares (the paper's modified
 *       Newton steps on a local quadratic approximation) from the {@link Method#WLS_VAR}
 *       solution. The weights act on each node's error standardised by its own variance, with the
 *       diagonal of the shrunk covariance as the base weights: with the full MinT precision the
 *       nearly collinear hierarchy spreads one bad forecast over many whitened errors and the
 *       reweighting no longer isolates it (measured in wave 11).</li>
 * </ul>
 */
public final class Reconciliation {

    private Reconciliation() {
    }

    /** A reconciliation method. */
    public enum Method {
        /** Leaves only. */
        BOTTOM_UP,
        /** Orthogonal projection. */
        OLS,
        /** Weights by the number of leaves under each node. */
        WLS_STRUCT,
        /** Weights by the base residual variances. */
        WLS_VAR,
        /** The shrunk residual covariance. */
        MINT_SHRINK,
        /** The Huber loss on variance-standardised errors. */
        ROBUST_HUBER,
        /** The absolute loss on variance-standardised errors. */
        ROBUST_LAD
    }

    /**
     * The summing matrix of a balanced hierarchy, nodes ordered level by level from the total.
     *
     * @param branching children per node at each level, for example {@code {4, 4}} for 1 total,
     *     4 regions and 16 leaves
     * @return {@code S}, {@code [nodes][leaves]}
     */
    public static double[][] hierarchy(int... branching) {
        int leaves = 1;
        for (int b : branching) {
            leaves *= b;
        }
        int nodes = 0;
        int width = 1;
        for (int l = 0; l <= branching.length; l++) {
            nodes += width;
            if (l < branching.length) {
                width *= branching[l];
            }
        }
        double[][] s = new double[nodes][leaves];
        int row = 0;
        width = 1;
        for (int l = 0; l <= branching.length; l++) {
            int span = leaves / width;
            for (int k = 0; k < width; k++) {
                for (int j = k * span; j < (k + 1) * span; j++) {
                    s[row][j] = 1;
                }
                row++;
            }
            if (l < branching.length) {
                width *= branching[l];
            }
        }
        return s;
    }

    /**
     * The level of every node of {@link #hierarchy}.
     *
     * @param branching the branching
     * @return {@code [nodes]}, 0 for the total
     */
    public static int[] levels(int... branching) {
        int width = 1;
        int nodes = 0;
        for (int l = 0; l <= branching.length; l++) {
            nodes += width;
            if (l < branching.length) {
                width *= branching[l];
            }
        }
        int[] out = new int[nodes];
        int row = 0;
        width = 1;
        for (int l = 0; l <= branching.length; l++) {
            for (int k = 0; k < width; k++) {
                out[row++] = l;
            }
            if (l < branching.length) {
                width *= branching[l];
            }
        }
        return out;
    }

    /**
     * The Schafer-Strimmer shrinkage covariance of residuals.
     *
     * @param residuals in-sample residuals {@code [T][nodes]}
     * @return the shrunk covariance
     */
    public static double[][] shrinkCovariance(double[][] residuals) {
        int t = residuals.length;
        int n = residuals[0].length;
        double[] mean = new double[n];
        double[] sd = new double[n];
        for (double[] r : residuals) {
            for (int i = 0; i < n; i++) {
                mean[i] += r[i] / t;
            }
        }
        for (double[] r : residuals) {
            for (int i = 0; i < n; i++) {
                sd[i] += (r[i] - mean[i]) * (r[i] - mean[i]) / t;
            }
        }
        for (int i = 0; i < n; i++) {
            sd[i] = Math.sqrt(Math.max(sd[i], 1e-12));
        }
        double[][] z = new double[t][n];
        for (int k = 0; k < t; k++) {
            for (int i = 0; i < n; i++) {
                z[k][i] = (residuals[k][i] - mean[i]) / sd[i];
            }
        }
        double num = 0;
        double den = 0;
        double[][] corr = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    corr[i][j] = 1;
                    continue;
                }
                double m = 0;
                for (int k = 0; k < t; k++) {
                    m += z[k][i] * z[k][j] / t;
                }
                double v = 0;
                for (int k = 0; k < t; k++) {
                    double w = z[k][i] * z[k][j] - m;
                    v += w * w;
                }
                v *= t / Math.pow(t - 1.0, 3);
                double rij = m * t / (t - 1.0);
                corr[i][j] = rij;
                num += v;
                den += rij * rij;
            }
        }
        double lambda = den == 0 ? 1 : Math.max(0, Math.min(1, num / den));
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double c = i == j ? 1 : (1 - lambda) * corr[i][j];
                w[i][j] = c * sd[i] * sd[j];
            }
        }
        return w;
    }

    /** {@code (S^T A S)^-1 S^T A y} for a precision matrix {@code A}. */
    private static double[] gls(double[][] s, double[][] a, double[] y) {
        int n = s.length;
        int m = s[0].length;
        double[][] as = MlMatrices.multiply(a, s);
        double[][] g = new double[m][m];
        double[] rhs = new double[m];
        for (int p = 0; p < m; p++) {
            for (int q = 0; q < m; q++) {
                double v = 0;
                for (int i = 0; i < n; i++) {
                    v += s[i][p] * as[i][q];
                }
                g[p][q] = v;
            }
            for (int i = 0; i < n; i++) {
                rhs[p] += as[i][p] * y[i];
            }
        }
        return MlMatrices.choleskySolve(MlMatrices.cholesky(g), rhs);
    }

    private static double[][] diag(double[] v) {
        double[][] a = new double[v.length][v.length];
        for (int i = 0; i < v.length; i++) {
            a[i][i] = v[i];
        }
        return a;
    }

    /**
     * Reconciles one vector of base forecasts.
     *
     * @param s the summing matrix
     * @param base base forecasts of every node
     * @param method the method
     * @param covariance the base-error covariance, normally {@link #shrinkCovariance} of the
     *     in-sample residuals (used by the variance-based methods; may be null for the others)
     * @return coherent forecasts of every node
     */
    public static double[] reconcile(double[][] s, double[] base, Method method, double[][] covariance) {
        int n = s.length;
        int m = s[0].length;
        double[] b;
        switch (method) {
            case BOTTOM_UP -> b = java.util.Arrays.copyOfRange(base, n - m, n);
            case OLS -> b = gls(s, diag(ones(n)), base);
            case WLS_STRUCT -> {
                double[] p = new double[n];
                for (int i = 0; i < n; i++) {
                    double c = 0;
                    for (double v : s[i]) {
                        c += v;
                    }
                    p[i] = 1 / c;
                }
                b = gls(s, diag(p), base);
            }
            case WLS_VAR -> {
                double[][] w = covariance;
                double[] p = new double[n];
                for (int i = 0; i < n; i++) {
                    p[i] = 1 / w[i][i];
                }
                b = gls(s, diag(p), base);
            }
            case MINT_SHRINK -> b = gls(s, MlMatrices.choleskyInverse(MlMatrices.cholesky(covariance)), base);
            default -> {
                double[] p = new double[n];
                for (int i = 0; i < n; i++) {
                    p[i] = 1 / covariance[i][i];
                }
                double[][] precision = diag(p);
                b = robust(s, covariance, precision, base, gls(s, precision, base), method == Method.ROBUST_HUBER);
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = MlMatrices.dot(s[i], b);
        }
        return out;
    }

    private static double[] ones(int n) {
        double[] o = new double[n];
        java.util.Arrays.fill(o, 1);
        return o;
    }

    /**
     * IRLS: node weights {@code w_i = psi(u_i) / u_i} of the standardised node errors
     * {@code u_i = (y_hat_i - (S b)_i) / (sigma_i * scale)} enter the precision as
     * {@code D^1/2 W^-1 D^1/2}, so an outlying base forecast loses its weight without its error
     * being spread by the whitening; the scale is the MAD of {@code u} floored at 1.
     */
    private static double[] robust(double[][] s, double[][] covariance, double[][] precision, double[] y,
            double[] start, boolean huber) {
        int n = s.length;
        int m = s[0].length;
        double[] b = start.clone();
        double k = 1.345;
        for (int it = 0; it < 30; it++) {
            double[] u = new double[n];
            for (int i = 0; i < n; i++) {
                u[i] = Math.abs(y[i] - MlMatrices.dot(s[i], b)) / Math.sqrt(covariance[i][i]);
            }
            double[] sorted = u.clone();
            java.util.Arrays.sort(sorted);
            double scale = Math.max(1, 1.4826 * sorted[n / 2]);
            double[] root = new double[n];
            for (int i = 0; i < n; i++) {
                double e = u[i] / scale;
                double w = huber ? (e <= k ? 1 : k / e) : 1 / Math.max(e, 1e-3);
                root[i] = Math.sqrt(w);
            }
            double[][] a = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    a[i][j] = root[i] * precision[i][j] * root[j];
                }
            }
            double[] next = gls(s, a, y);
            double change = 0;
            for (int p = 0; p < m; p++) {
                change = Math.max(change, Math.abs(next[p] - b[p]));
            }
            b = next;
            if (change < 1e-9) {
                break;
            }
        }
        return b;
    }
}
