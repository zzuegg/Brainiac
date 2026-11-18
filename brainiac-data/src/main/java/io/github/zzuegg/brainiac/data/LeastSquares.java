package io.github.zzuegg.brainiac.data;

import io.github.zzuegg.brainiac.tensor.Distributions;

/**
 * Ordinary least squares with the usual inference, as {@code statsmodels.OLS}: coefficients,
 * standard errors, t statistics and their two-sided p-values, R squared and the overall F test.
 */
public final class LeastSquares {

    private LeastSquares() {
    }

    /**
     * A fitted linear model. With an intercept it comes first.
     *
     * @param coefficients the estimates
     * @param standardErrors their standard errors
     * @param tValues estimates over standard errors
     * @param pValues the two-sided p-values of the t tests
     * @param rSquared the share of variance explained
     * @param adjustedRSquared R squared corrected for the number of predictors
     * @param residualStandardError the estimated noise level
     * @param fStatistic the F statistic of all predictors (but the intercept) being zero
     * @param fPValue its p-value
     * @param residualDegreesOfFreedom rows minus coefficients
     */
    public record Fit(double[] coefficients, double[] standardErrors, double[] tValues, double[] pValues,
            double rSquared, double adjustedRSquared, double residualStandardError, double fStatistic, double fPValue,
            int residualDegreesOfFreedom) {

        /**
         * Predicts one row.
         *
         * @param x the row's predictors, without the intercept's 1
         * @param intercept whether the model was fitted with an intercept
         * @return the prediction
         */
        public double predict(double[] x, boolean intercept) {
            int off = intercept ? 1 : 0;
            double y = intercept ? coefficients[0] : 0;
            for (int j = 0; j < x.length; j++) {
                y += coefficients[j + off] * x[j];
            }
            return y;
        }
    }

    /**
     * Fits {@code y = b0 + x b} (or {@code x b} without an intercept) by least squares.
     *
     * @param x rows of predictors
     * @param y one target per row
     * @param intercept whether to fit an intercept
     * @return the fit
     * @throws IllegalArgumentException if there are too few rows or the predictors are collinear
     */
    public static Fit ols(double[][] x, double[] y, boolean intercept) {
        return wls(x, y, null, intercept);
    }

    /**
     * Fits by weighted least squares, as {@code statsmodels.WLS}: minimizes
     * {@code sum w_i (y_i - x_i b)^2}, such as with weights {@code 1 / variance_i}. R squared is
     * about the weighted mean.
     *
     * @param x rows of predictors
     * @param y one target per row
     * @param weights one positive weight per row, or null for all ones
     * @param intercept whether to fit an intercept
     * @return the fit
     * @throws IllegalArgumentException if there are too few rows or the predictors are collinear
     */
    public static Fit wls(double[][] x, double[] y, double[] weights, boolean intercept) {
        int n = x.length;
        if (weights != null && weights.length != n) {
            throw new IllegalArgumentException(n + " rows and " + weights.length + " weights");
        }
        if (n == 0 || y.length != n) {
            throw new IllegalArgumentException(n + " rows and " + y.length + " targets");
        }
        int k = x[0].length + (intercept ? 1 : 0);
        if (n <= k) {
            throw new IllegalArgumentException("need more rows (" + n + ") than coefficients (" + k + ")");
        }
        double[][] design = new double[n][k];
        for (int i = 0; i < n; i++) {
            if (x[i].length != k - (intercept ? 1 : 0)) {
                throw new IllegalArgumentException("row " + i + " has " + x[i].length + " predictors");
            }
            int c = 0;
            if (intercept) {
                design[i][c++] = 1;
            }
            for (double v : x[i]) {
                design[i][c++] = v;
            }
        }
        double[][] xtx = new double[k][k];
        double[] xty = new double[k];
        for (int i = 0; i < n; i++) {
            double w = weight(weights, i);
            for (int a = 0; a < k; a++) {
                xty[a] += w * design[i][a] * y[i];
                for (int b = 0; b < k; b++) {
                    xtx[a][b] += w * design[i][a] * design[i][b];
                }
            }
        }
        double[][] inverse = invert(xtx);
        double[] beta = new double[k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                beta[a] += inverse[a][b] * xty[b];
            }
        }
        double mean = 0;
        double wsum = 0;
        for (int i = 0; i < n; i++) {
            mean += weight(weights, i) * y[i];
            wsum += weight(weights, i);
        }
        mean /= wsum;
        double rss = 0;
        double tss = 0;
        for (int i = 0; i < n; i++) {
            double fit = 0;
            for (int a = 0; a < k; a++) {
                fit += design[i][a] * beta[a];
            }
            double w = weight(weights, i);
            rss += w * (y[i] - fit) * (y[i] - fit);
            double centred = intercept ? y[i] - mean : y[i];
            tss += w * centred * centred;
        }
        int df = n - k;
        double sigma2 = rss / df;
        double[] se = new double[k];
        double[] t = new double[k];
        double[] p = new double[k];
        for (int a = 0; a < k; a++) {
            se[a] = Math.sqrt(sigma2 * inverse[a][a]);
            t[a] = beta[a] / se[a];
            p[a] = Distributions.studentTTwoSided(t[a], df);
        }
        double r2 = tss == 0 ? 0 : 1 - rss / tss;
        int predictors = k - (intercept ? 1 : 0);
        double adjusted = 1 - (1 - r2) * (intercept ? n - 1 : n) / df;
        double f = predictors == 0 ? Double.NaN : (tss - rss) / predictors / sigma2;
        double fp = predictors == 0 ? Double.NaN : Distributions.fSurvival(f, predictors, df);
        return new Fit(beta, se, t, p, r2, adjusted, Math.sqrt(sigma2), f, fp, df);
    }

    private static double weight(double[] weights, int i) {
        if (weights == null) {
            return 1;
        }
        if (!(weights[i] > 0)) {
            throw new IllegalArgumentException("weight " + i + " is not positive: " + weights[i]);
        }
        return weights[i];
    }

    /**
     * Fits ridge regression, as {@code sklearn.linear_model.Ridge}: minimizes
     * {@code sum w_i (y_i - b0 - x_i b)^2 + lambda |b|^2}; the intercept is not penalized.
     *
     * @param x rows of predictors
     * @param y one target per row
     * @param weights one positive weight per row, or null for all ones
     * @param lambda the penalty, at least 0
     * @param intercept whether to fit an intercept
     * @return the coefficients, the intercept first when fitted
     */
    public static double[] ridge(double[][] x, double[] y, double[] weights, double lambda, boolean intercept) {
        int n = x.length;
        if (n == 0 || y.length != n || (weights != null && weights.length != n)) {
            throw new IllegalArgumentException(n + " rows, " + y.length + " targets"
                    + (weights == null ? "" : ", " + weights.length + " weights"));
        }
        if (!(lambda >= 0)) {
            throw new IllegalArgumentException("lambda must be at least 0");
        }
        int p = x[0].length;
        double[] xm = new double[p];
        double ym = 0;
        double wsum = 0;
        if (intercept) {
            for (int i = 0; i < n; i++) {
                double w = weight(weights, i);
                wsum += w;
                ym += w * y[i];
                for (int j = 0; j < p; j++) {
                    xm[j] += w * x[i][j];
                }
            }
            ym /= wsum;
            for (int j = 0; j < p; j++) {
                xm[j] /= wsum;
            }
        }
        double[][] a = new double[p][p];
        double[] b = new double[p];
        for (int i = 0; i < n; i++) {
            if (x[i].length != p) {
                throw new IllegalArgumentException("row " + i + " has " + x[i].length + " predictors");
            }
            double w = weight(weights, i);
            for (int j = 0; j < p; j++) {
                double xj = x[i][j] - xm[j];
                b[j] += w * xj * (y[i] - ym);
                for (int l = 0; l < p; l++) {
                    a[j][l] += w * xj * (x[i][l] - xm[l]);
                }
            }
        }
        for (int j = 0; j < p; j++) {
            a[j][j] += lambda;
        }
        double[][] inv = invert(a);
        double[] beta = new double[p];
        for (int j = 0; j < p; j++) {
            for (int l = 0; l < p; l++) {
                beta[j] += inv[j][l] * b[l];
            }
        }
        if (!intercept) {
            return beta;
        }
        double[] out = new double[p + 1];
        double b0 = ym;
        for (int j = 0; j < p; j++) {
            b0 -= xm[j] * beta[j];
            out[j + 1] = beta[j];
        }
        out[0] = b0;
        return out;
    }

    // Gauss-Jordan with partial pivoting
    private static double[][] invert(double[][] m) {
        int k = m.length;
        double[][] a = new double[k][2 * k];
        double scale = 0;
        for (int i = 0; i < k; i++) {
            System.arraycopy(m[i], 0, a[i], 0, k);
            a[i][k + i] = 1;
            scale = Math.max(scale, Math.abs(m[i][i]));
        }
        for (int c = 0; c < k; c++) {
            int pivot = c;
            for (int r = c + 1; r < k; r++) {
                if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) {
                    pivot = r;
                }
            }
            if (Math.abs(a[pivot][c]) <= 1e-12 * Math.max(scale, 1e-300)) {
                throw new IllegalArgumentException("the predictors are collinear");
            }
            double[] tmp = a[c];
            a[c] = a[pivot];
            a[pivot] = tmp;
            double d = a[c][c];
            for (int j = 0; j < 2 * k; j++) {
                a[c][j] /= d;
            }
            for (int r = 0; r < k; r++) {
                if (r != c && a[r][c] != 0) {
                    double f = a[r][c];
                    for (int j = 0; j < 2 * k; j++) {
                        a[r][j] -= f * a[c][j];
                    }
                }
            }
        }
        double[][] out = new double[k][k];
        for (int i = 0; i < k; i++) {
            System.arraycopy(a[i], k, out[i], 0, k);
        }
        return out;
    }
}
