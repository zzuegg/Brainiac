package io.github.zzuegg.brainiac.experimental.discovery;

/**
 * Granger causality (Granger, Econometrica 1969,
 * <a href="https://doi.org/10.2307/1912791">doi:10.2307/1912791</a>): {@code X} Granger-causes
 * {@code Y} when the past of {@code X} improves the prediction of {@code Y} beyond {@code Y}'s own
 * past. A VAR({@code p}) is fitted by least squares, with {@code p} chosen by BIC, and for every
 * source-target pair the restricted model without the source's lags is compared by an F-test.
 * {@link #conditional} uses the full multivariate VAR (conditioning on every other series);
 * {@link #pairwise} uses a bivariate VAR per pair, the version that inflates false positives
 * when a common driver or strong autocorrelation is present.
 */
public final class GrangerCausality {

    private GrangerCausality() {
    }

    /**
     * The lagged design for a set of series.
     *
     * @param series {@code [T][d]}
     * @param sources the series whose lags enter
     * @param lags the number of lags
     * @return {@code [T - lags][sources.length * lags]}, row {@code t} predicting {@code t + lags}
     */
    public static double[][] laggedDesign(double[][] series, int[] sources, int lags) {
        int rows = series.length - lags;
        double[][] x = new double[rows][sources.length * lags];
        for (int t = 0; t < rows; t++) {
            int c = 0;
            for (int s : sources) {
                for (int l = 1; l <= lags; l++) {
                    x[t][c++] = series[t + lags - l][s];
                }
            }
        }
        return x;
    }

    private static double[] target(double[][] series, int j, int lags) {
        double[] y = new double[series.length - lags];
        for (int t = 0; t < y.length; t++) {
            y[t] = series[t + lags][j];
        }
        return y;
    }

    /**
     * The VAR order with the smallest BIC.
     *
     * @param series {@code [T][d]}
     * @param maxLag the largest order tried
     * @return the order
     */
    public static int selectLag(double[][] series, int maxLag) {
        int d = series[0].length;
        int[] all = new int[d];
        for (int j = 0; j < d; j++) {
            all[j] = j;
        }
        int best = 1;
        double bestBic = Double.POSITIVE_INFINITY;
        for (int p = 1; p <= maxLag; p++) {
            double[][] x = laggedDesign(series, all, p);
            int n = x.length;
            double logDet = 0;
            for (int j = 0; j < d; j++) {
                logDet += Math.log(LeastSquares.fit(x, target(series, j, p), true, 1e-8).rss() / n);
            }
            double bic = n * logDet + Math.log(n) * d * (d * p + 1);
            if (bic < bestBic) {
                bestBic = bic;
                best = p;
            }
        }
        return best;
    }

    private static double fTest(double[][] series, int[] sources, int source, int j, int lags) {
        int[] reduced = new int[sources.length - 1];
        int c = 0;
        for (int s : sources) {
            if (s != source) {
                reduced[c++] = s;
            }
        }
        double[] y = target(series, j, lags);
        double[][] full = laggedDesign(series, sources, lags);
        double rssU = LeastSquares.fit(full, y, true, 1e-8).rss();
        double rssR = LeastSquares.fit(laggedDesign(series, reduced, lags), y, true, 1e-8).rss();
        int n = y.length;
        int df2 = n - full[0].length - 1;
        double f = ((rssR - rssU) / lags) / (rssU / df2);
        return Distributions.fUpper(f, lags, df2);
    }

    /**
     * Conditional Granger tests in the full VAR.
     *
     * @param series {@code [T][d]}
     * @param lags the VAR order
     * @return {@code p[i][j]}: the p-value of "{@code i} does not Granger-cause {@code j}" (NaN
     *     on the diagonal)
     */
    public static double[][] conditional(double[][] series, int lags) {
        int d = series[0].length;
        int[] all = new int[d];
        for (int j = 0; j < d; j++) {
            all[j] = j;
        }
        double[][] p = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                p[i][j] = i == j ? Double.NaN : fTest(series, all, i, j, lags);
            }
        }
        return p;
    }

    /**
     * Pairwise (bivariate) Granger tests.
     *
     * @param series {@code [T][d]}
     * @param lags the VAR order
     * @return {@code p[i][j]} as in {@link #conditional}
     */
    public static double[][] pairwise(double[][] series, int lags) {
        int d = series[0].length;
        double[][] p = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                p[i][j] = i == j ? Double.NaN : fTest(series, new int[] {i, j}, i, j, lags);
            }
        }
        return p;
    }
}
