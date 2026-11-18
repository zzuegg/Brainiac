package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * Ordinary (optionally ridge) least squares by the normal equations and a jittered Cholesky
 * factor, the workhorse of the regression-based tests and estimators in this package: residuals
 * for partial correlations, VAR fits for Granger tests, and pruning regressions.
 */
public final class LeastSquares {

    private LeastSquares() {
    }

    /**
     * A fit.
     *
     * @param beta the coefficients, the intercept first when one was fitted
     * @param residuals {@code y - X beta}
     * @param rss the residual sum of squares
     */
    public record Fit(double[] beta, double[] residuals, double rss) {
    }

    /**
     * Regresses {@code y} on the columns of {@code x}.
     *
     * @param x the design {@code [n][p]}, may have zero columns
     * @param y the response
     * @param intercept whether to add a constant column
     * @param ridge a ridge added to the Gram diagonal (not on the intercept)
     * @return the fit
     */
    public static Fit fit(double[][] x, double[] y, boolean intercept, double ridge) {
        int n = y.length;
        int p = (x.length == 0 ? 0 : x[0].length) + (intercept ? 1 : 0);
        if (p == 0) {
            double rss = 0;
            for (double v : y) {
                rss += v * v;
            }
            return new Fit(new double[0], y.clone(), rss);
        }
        double[][] g = new double[p][p];
        double[] b = new double[p];
        double[] row = new double[p];
        for (int i = 0; i < n; i++) {
            fill(row, x.length == 0 ? null : x[i], intercept);
            for (int a = 0; a < p; a++) {
                b[a] += row[a] * y[i];
                for (int c = a; c < p; c++) {
                    g[a][c] += row[a] * row[c];
                }
            }
        }
        for (int a = 0; a < p; a++) {
            for (int c = 0; c < a; c++) {
                g[a][c] = g[c][a];
            }
            if (!(intercept && a == 0)) {
                g[a][a] += ridge;
            }
        }
        double[] beta = MlMatrices.choleskySolve(MlMatrices.cholesky(g), b);
        double[] res = new double[n];
        double rss = 0;
        for (int i = 0; i < n; i++) {
            fill(row, x.length == 0 ? null : x[i], intercept);
            res[i] = y[i] - MlMatrices.dot(row, beta);
            rss += res[i] * res[i];
        }
        return new Fit(beta, res, rss);
    }

    private static void fill(double[] row, double[] x, boolean intercept) {
        int o = 0;
        if (intercept) {
            row[o++] = 1;
        }
        if (x != null) {
            System.arraycopy(x, 0, row, o, x.length);
        }
    }

    /**
     * Selects columns of a data matrix.
     *
     * @param data {@code [n][d]}
     * @param columns the columns to keep
     * @return {@code [n][columns.length]}
     */
    public static double[][] columns(double[][] data, int[] columns) {
        double[][] out = new double[data.length][columns.length];
        for (int i = 0; i < data.length; i++) {
            for (int k = 0; k < columns.length; k++) {
                out[i][k] = data[i][columns[k]];
            }
        }
        return out;
    }

    /**
     * One column of a data matrix.
     *
     * @param data {@code [n][d]}
     * @param column the column
     * @return its values
     */
    public static double[] column(double[][] data, int column) {
        double[] out = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i][column];
        }
        return out;
    }

    /**
     * Standardises every column to zero mean and unit variance.
     *
     * @param data {@code [n][d]}
     * @return a standardised copy
     */
    public static double[][] standardise(double[][] data) {
        int n = data.length;
        int d = data[0].length;
        double[][] out = new double[n][d];
        for (int j = 0; j < d; j++) {
            double m = 0;
            for (double[] r : data) {
                m += r[j] / n;
            }
            double v = 0;
            for (double[] r : data) {
                v += (r[j] - m) * (r[j] - m) / n;
            }
            double s = Math.sqrt(Math.max(v, 1e-300));
            for (int i = 0; i < n; i++) {
                out[i][j] = (data[i][j] - m) / s;
            }
        }
        return out;
    }

    /**
     * The Pearson correlation of two vectors.
     *
     * @param a one
     * @param b the other
     * @return the correlation, 0 if either is constant
     */
    public static double correlation(double[] a, double[] b) {
        int n = a.length;
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < n; i++) {
            ma += a[i] / n;
            mb += b[i] / n;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < n; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return saa <= 0 || sbb <= 0 ? 0 : sab / Math.sqrt(saa * sbb);
    }
}
