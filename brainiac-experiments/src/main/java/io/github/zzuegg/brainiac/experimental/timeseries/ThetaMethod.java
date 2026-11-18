package io.github.zzuegg.brainiac.experimental.timeseries;

/**
 * The Theta method (Assimakopoulos and Nikolopoulos, IJF 2000), the M3 competition's winner, in
 * the form Hyndman and Billah showed it to be ("Unmasking the Theta method", IJF 2003): simple
 * exponential smoothing with a drift equal to half the slope {@code b0} of a linear trend fitted
 * to the series,
 * {@code ŷ_n+h = l_n + (b0 / 2) ((h - 1) + (1 - (1 - α)^n) / α)} (as in R's {@code thetaf}).
 * Seasonal series are first deseasonalised by classical multiplicative decomposition (additive
 * for non-positive series) when the autocorrelation at the season lag is significant at 90%, and
 * reseasonalised after forecasting. Intervals use SES's variance
 * {@code σ^2 (1 + α^2 (h - 1))}.
 */
public final class ThetaMethod {

    private final double level;
    private final double alpha;
    private final double drift;
    private final double sigma;
    private final int n;
    private final double[] seasonal;
    private final boolean multiplicative;
    private final int period;

    private ThetaMethod(double level, double alpha, double drift, double sigma, int n, double[] seasonal,
            boolean multiplicative, int period) {
        this.level = level;
        this.alpha = alpha;
        this.drift = drift;
        this.sigma = sigma;
        this.n = n;
        this.seasonal = seasonal;
        this.multiplicative = multiplicative;
        this.period = period;
    }

    /**
     * Fits the method.
     *
     * @param y the series
     * @param period the season length (1 for none)
     * @return the fitted model
     */
    public static ThetaMethod fit(double[] y, int period) {
        int n = y.length;
        double[] season = null;
        boolean mult = true;
        double[] x = y.clone();
        if (period > 1 && n >= 2 * period && seasonalTest(y, period)) {
            mult = java.util.Arrays.stream(y).min().orElse(0) > 0;
            season = decompose(y, period, mult);
            for (int t = 0; t < n; t++) {
                x[t] = mult ? y[t] / season[t % period] : y[t] - season[t % period];
            }
        }
        // linear trend slope by least squares on t
        double tm = (n - 1) / 2.0;
        double ym = 0;
        for (double v : x) {
            ym += v / n;
        }
        double sxy = 0;
        double sxx = 0;
        for (int t = 0; t < n; t++) {
            sxy += (t - tm) * (x[t] - ym);
            sxx += (t - tm) * (t - tm);
        }
        double b0 = sxx > 0 ? sxy / sxx : 0;
        // SES by a fine grid on the one-step squared error
        double bestA = 0.5;
        double bestSse = Double.POSITIVE_INFINITY;
        double bestLevel = x[0];
        for (int i = 1; i <= 99; i++) {
            double a = i / 100.0;
            double l = x[0];
            double sse = 0;
            for (int t = 1; t < n; t++) {
                double e = x[t] - l;
                sse += e * e;
                l += a * e;
            }
            if (sse < bestSse) {
                bestSse = sse;
                bestA = a;
                bestLevel = l;
            }
        }
        return new ThetaMethod(bestLevel, bestA, b0 / 2, Math.sqrt(bestSse / Math.max(1, n - 1)), n, season, mult,
                period);
    }

    /** Whether the autocorrelation at the season lag exceeds its 90% band. */
    private static boolean seasonalTest(double[] y, int m) {
        int n = y.length;
        double mean = 0;
        for (double v : y) {
            mean += v / n;
        }
        double c0 = 0;
        for (double v : y) {
            c0 += (v - mean) * (v - mean);
        }
        double[] r = new double[m + 1];
        for (int k = 1; k <= m; k++) {
            double s = 0;
            for (int t = k; t < n; t++) {
                s += (y[t] - mean) * (y[t - k] - mean);
            }
            r[k] = s / c0;
        }
        double sum = 0;
        for (int k = 1; k < m; k++) {
            sum += r[k] * r[k];
        }
        double limit = 1.645 * Math.sqrt((1 + 2 * sum) / n);
        return Math.abs(r[m]) > limit;
    }

    /** Classical decomposition: seasonal indices from a centred moving average. */
    private static double[] decompose(double[] y, int m, boolean mult) {
        int n = y.length;
        double[] sum = new double[m];
        int[] count = new int[m];
        int half = m / 2;
        for (int t = half; t < n - half; t++) {
            double ma;
            if (m % 2 == 0) {
                double s = 0.5 * y[t - half] + 0.5 * y[t + half];
                for (int k = -half + 1; k < half; k++) {
                    s += y[t + k];
                }
                ma = s / m;
            } else {
                double s = 0;
                for (int k = -half; k <= half; k++) {
                    s += y[t + k];
                }
                ma = s / m;
            }
            sum[t % m] += mult ? y[t] / ma : y[t] - ma;
            count[t % m]++;
        }
        double[] idx = new double[m];
        double norm = 0;
        for (int k = 0; k < m; k++) {
            idx[k] = count[k] > 0 ? sum[k] / count[k] : (mult ? 1 : 0);
            norm += idx[k] / m;
        }
        for (int k = 0; k < m; k++) {
            idx[k] = mult ? idx[k] / norm : idx[k] - norm;
        }
        return idx;
    }

    /**
     * Point forecasts.
     *
     * @param horizon steps ahead
     * @return the forecasts
     */
    public double[] forecast(int horizon) {
        double[] out = new double[horizon];
        double tail = (1 - Math.pow(1 - alpha, n)) / alpha;
        for (int h = 1; h <= horizon; h++) {
            double f = level + drift * ((h - 1) + tail);
            if (seasonal != null) {
                double s = seasonal[(n + h - 1) % period];
                f = multiplicative ? f * s : f + s;
            }
            out[h - 1] = f;
        }
        return out;
    }

    /**
     * Gaussian prediction intervals around {@link #forecast}.
     *
     * @param horizon steps ahead
     * @param z the normal quantile, for example 1.2816 for 80%
     * @return {@code [2][horizon]} lower and upper bounds
     */
    public double[][] intervals(int horizon, double z) {
        double[] f = forecast(horizon);
        double[][] out = new double[2][horizon];
        for (int h = 1; h <= horizon; h++) {
            double sd = sigma * Math.sqrt(1 + alpha * alpha * (h - 1));
            if (seasonal != null && multiplicative) {
                sd *= seasonal[(n + h - 1) % period];
            }
            out[0][h - 1] = f[h - 1] - z * sd;
            out[1][h - 1] = f[h - 1] + z * sd;
        }
        return out;
    }

    /** @return the SES smoothing parameter */
    public double alpha() {
        return alpha;
    }

    /** @return whether the series was deseasonalised */
    public boolean seasonal() {
        return seasonal != null;
    }
}
