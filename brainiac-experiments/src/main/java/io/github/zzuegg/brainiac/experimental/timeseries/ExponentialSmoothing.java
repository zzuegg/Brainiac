package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;

import java.util.Arrays;

/**
 * Exponential smoothing (Holt-Winters) with additive errors, written in innovations
 * state-space form (the ETS framework of Hyndman, Koehler, Ord and Snyder, <i>Forecasting with
 * Exponential Smoothing</i>, Springer 2008, <a href="https://doi.org/10.1007/978-3-540-71918-2">doi</a>):
 * <pre>
 *   y_t = l_t-1 + phi b_t-1 + s_t-m + e_t
 *   l_t = l_t-1 + phi b_t-1 + alpha e_t
 *   b_t = phi b_t-1 + beta e_t
 *   s_t = s_t-m + gamma e_t
 * </pre>
 * with no, additive or damped trend and an optional additive season of length {@code m}. The
 * single source of error is folded into the state ({@code [x_t-1, e_t]}), which turns the model
 * into the linear-Gaussian form of {@link StateSpaceFilter#kalman}; the filter gives the exact
 * likelihood (with a vague prior on the initial state, so the start-up values need not be
 * fitted), {@code alpha, beta, gamma, phi, sigma} are fitted by L-BFGS, {@link #select} chooses
 * among trend and season variants by AICc, and forecasts carry exact Gaussian prediction
 * intervals. Multiplicative error and season variants are not linear-Gaussian and are left out.
 */
public final class ExponentialSmoothing {

    /** The trend component. */
    public enum Trend {
        /** No trend (simple exponential smoothing, or Holt-Winters without trend). */
        NONE,
        /** An additive (Holt) trend. */
        ADDITIVE,
        /** An additive trend damped by {@code phi} in {@code [0.8, 0.98]}. */
        DAMPED
    }

    private final Trend trend;
    private final int period;
    private final double alpha;
    private final double beta;
    private final double gamma;
    private final double phi;
    private final double sigma;
    private final double logLikelihood;
    private final int observations;

    private ExponentialSmoothing(Trend trend, int period, double[] p, double logLikelihood, int observations) {
        this.trend = trend;
        this.period = period;
        this.alpha = p[0];
        this.beta = p[1];
        this.gamma = p[2];
        this.phi = p[3];
        this.sigma = p[4];
        this.logLikelihood = logLikelihood;
        this.observations = observations;
    }

    /** Maps unconstrained values to {alpha, beta, gamma, phi, sigma}. */
    private static double[] constrain(double[] u, Trend trend, int period) {
        double a = StateSpaceForecast.sigmoid(u[0]);
        double b = trend == Trend.NONE ? 0 : a * StateSpaceForecast.sigmoid(u[1]);
        double g = period > 1 ? (1 - a) * StateSpaceForecast.sigmoid(u[2]) : 0;
        double ph = switch (trend) {
            case NONE -> 0;
            case ADDITIVE -> 1;
            case DAMPED -> 0.8 + 0.18 * StateSpaceForecast.sigmoid(u[3]);
        };
        return new double[] {a, b, g, ph, Math.exp(u[4])};
    }

    /**
     * Fits one variant.
     *
     * @param y the series
     * @param trend the trend
     * @param period the season length, or 1 for none
     * @param iterations L-BFGS iterations, for example 15
     * @return the fitted model
     */
    public static ExponentialSmoothing fit(double[] y, Trend trend, int period, int iterations) {
        double sd = Math.sqrt(variance(y));
        double[] u0 = {StateSpaceForecast.logit(0.3), StateSpaceForecast.logit(0.1), StateSpaceForecast.logit(0.1), 0,
            Math.log(0.5 * sd + 1e-6)};
        double[] u = StateSpaceForecast.maximise(v -> filter(y, constrain(v, trend, period), trend, period)
                .logLikelihood(), u0, iterations);
        double[] p = constrain(u, trend, period);
        return new ExponentialSmoothing(trend, period, p, filter(y, p, trend, period).logLikelihood(), y.length);
    }

    /** How {@link #fit(double[], Trend, int, Method, int)} chooses the smoothing parameters. */
    public enum Method {
        /** L-BFGS on the exact Kalman likelihood (the default of {@link #fit(double[], Trend, int, int)}). */
        LBFGS,
        /**
         * A coarse-to-fine grid over {@code alpha, beta, gamma, phi} minimising the sum of squared
         * one-step residuals of the innovations recursion ({@link #residuals}); each level
         * refines a grid of half the spacing around the previous best (a warm start), and the
         * reported likelihood is the Kalman likelihood at the chosen parameters.
         */
        GRID_WARM
    }

    /**
     * Fits one variant with the chosen method.
     *
     * @param y the series
     * @param trend the trend
     * @param period the season length, or 1 for none
     * @param method the fitting method
     * @param iterations L-BFGS iterations, or the number of grid refinements for
     *     {@link Method#GRID_WARM} (for example 3)
     * @return the fitted model
     */
    public static ExponentialSmoothing fit(double[] y, Trend trend, int period, Method method, int iterations) {
        if (method == Method.LBFGS) {
            return fit(y, trend, period, iterations);
        }
        // coordinates: alpha, beta / alpha, gamma / (1 - alpha), phi
        boolean hasTrend = trend != Trend.NONE;
        boolean damped = trend == Trend.DAMPED;
        boolean seasonal = period > 1;
        double[] best = {0.3, hasTrend ? 0.2 : 0, seasonal ? 0.1 : 0, damped ? 0.9 : hasTrend ? 1 : 0};
        double[] lo = {0.02, 0, 0, 0.8};
        double[] hi = {0.98, hasTrend ? 1 : 0, seasonal ? 0.98 : 0, damped ? 0.98 : best[3]};
        double[] width = {0.24, hasTrend ? 0.25 : 0, seasonal ? 0.24 : 0, damped ? 0.045 : 0};
        double bestSse = Double.POSITIVE_INFINITY;
        for (int level = 0; level <= Math.max(0, iterations); level++) {
            double[] centre = level == 0 ? new double[] {0.5, 0.5, 0.5, 0.89} : best.clone();
            int half = level == 0 ? 2 : 1;
            double[] step = width.clone();
            if (level > 0) {
                for (int k = 0; k < 4; k++) {
                    step[k] = width[k] / (1 << level);
                }
            }
            for (int i = -half; i <= half; i++) {
                for (int j = hasTrend ? -half : 0; j <= (hasTrend ? half : 0); j++) {
                    for (int k = seasonal ? -half : 0; k <= (seasonal ? half : 0); k++) {
                        for (int l = damped ? -half : 0; l <= (damped ? half : 0); l++) {
                            double[] c = {clip(centre[0] + i * step[0], lo[0], hi[0]),
                                hasTrend ? clip(centre[1] + j * step[1], lo[1], hi[1]) : 0,
                                seasonal ? clip(centre[2] + k * step[2], lo[2], hi[2]) : 0,
                                damped ? clip(centre[3] + l * step[3], lo[3], hi[3]) : hasTrend ? 1 : 0};
                            double sse = sumSquares(residuals(y, natural(c), trend, period));
                            if (sse < bestSse) {
                                bestSse = sse;
                                best = c;
                            }
                        }
                    }
                }
            }
        }
        double[] p = natural(best);
        p[4] = Math.sqrt(bestSse / y.length) + 1e-9;
        return new ExponentialSmoothing(trend, period, p, filter(y, p, trend, period).logLikelihood(), y.length);
    }

    private static double clip(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    /** {alpha, beta/alpha, gamma/(1-alpha), phi} to {alpha, beta, gamma, phi, sigma = 1}. */
    private static double[] natural(double[] c) {
        return new double[] {c[0], c[0] * c[1], (1 - c[0]) * c[2], c[3], 1};
    }

    private static double sumSquares(double[] e) {
        double s = 0;
        for (double v : e) {
            s += v * v;
        }
        return s;
    }

    /**
     * The in-sample one-step residuals {@code e_t = y_t - (l_t-1 + phi b_t-1 + s_t-m)} of the
     * innovations recursion with this model's parameters, started from the same heuristic
     * initial states as the Kalman prior.
     *
     * @param y the series
     * @return one residual per observation
     */
    public double[] residuals(double[] y) {
        return residuals(y, new double[] {alpha, beta, gamma, phi, sigma}, trend, period);
    }

    private static double[] residuals(double[] y, double[] p, Trend trend, int period) {
        StateSpaceFilter.LinearModel init = model(y, p, trend, period);
        double[] x0 = init.x0();
        double level = x0[0];
        double slope = x0[1];
        int m = Math.max(1, period);
        double[] season = new double[m];
        if (period > 1) {
            for (int k = 0; k < period; k++) {
                season[(period - 1 - k) % m] = x0[2 + k];
            }
        }
        double a = p[0];
        double b = p[1];
        double g = p[2];
        double ph = p[3];
        double[] e = new double[y.length];
        for (int t = 0; t < y.length; t++) {
            int si = t % m;
            double s = period > 1 ? season[si] : 0;
            double err = y[t] - (level + ph * slope + s);
            e[t] = err;
            level = level + ph * slope + a * err;
            slope = ph * slope + b * err;
            if (period > 1) {
                season[si] = s + g * err;
            }
        }
        return e;
    }

    /**
     * Fits every trend variant, with and without the season, and keeps the lowest AICc.
     *
     * @param y the series
     * @param period the season length (1 to consider no season only)
     * @param iterations L-BFGS iterations per fit
     * @return the selected model
     */
    public static ExponentialSmoothing select(double[] y, int period, int iterations) {
        ExponentialSmoothing best = null;
        for (int m : period > 1 ? new int[] {1, period} : new int[] {1}) {
            for (Trend t : Trend.values()) {
                ExponentialSmoothing e = fit(y, t, m, iterations);
                if (best == null || e.aicc() < best.aicc()) {
                    best = e;
                }
            }
        }
        return best;
    }

    private static int stateSize(int period) {
        return 2 + (period > 1 ? period : 0);
    }

    /** The augmented linear-Gaussian model, prior centred on heuristic start values from {@code y}. */
    private static StateSpaceFilter.LinearModel model(double[] y, double[] p, Trend trend, int period) {
        int n = stateSize(period);
        int d = n + 1;
        double a = p[0];
        double b = p[1];
        double g = p[2];
        double ph = p[3];
        double s2 = p[4] * p[4];
        double[][] f = new double[d][d];
        f[0][0] = 1;
        f[0][1] = ph;
        f[1][1] = ph;
        f[0][n] = a;
        f[1][n] = b;
        if (period > 1) {
            f[2][n - 1] = 1;           // s_t = s_t-m + gamma e
            f[2][n] = g;
            for (int k = 3; k < n; k++) {
                f[k][k - 1] = 1;       // shift the seasonal states
            }
        }
        double[][] q = new double[d][d];
        q[n][n] = s2;
        double[][] h = new double[1][d];
        h[0][0] = 1;
        h[0][1] = ph;
        if (period > 1) {
            h[0][n - 1] = 1;
        }
        h[0][n] = 1;
        double[][] r = {{1e-6 * s2}};
        double[] x0 = new double[d];
        int m = Math.max(1, period);
        int seasons = Math.min(2, y.length / m);
        double l1 = mean(y, 0, m);
        x0[0] = l1;
        x0[1] = trend == Trend.NONE || seasons < 2 ? 0 : (mean(y, m, 2 * m) - l1) / m;
        if (period > 1) {
            for (int k = 0; k < period; k++) {
                x0[2 + k] = y[period - 1 - k] - l1;
            }
        }
        double v = variance(y);
        double[][] p0 = new double[d][d];
        for (int i = 0; i < n; i++) {
            p0[i][i] = (i == 1 ? 0.01 : 0.5) * v + 1e-6;
        }
        if (trend == Trend.NONE) {
            p0[1][1] = 1e-9;
        }
        p0[n][n] = s2;
        return new StateSpaceFilter.LinearModel(f, q, h, r, x0, p0);
    }

    private static StateSpaceFilter.Estimate filter(double[] y, double[] p, Trend trend, int period) {
        return StateSpaceFilter.kalman(model(y, p, trend, period), StateSpaceForecast.column(y), false);
    }

    /**
     * Filters a history with the fitted parameters and forecasts past its end.
     *
     * @param history the observed values (at least one season)
     * @param horizon steps ahead
     * @return the forecast with exact Gaussian variances
     */
    public StateSpaceForecast forecast(double[] history, int horizon) {
        double[] p = {alpha, beta, gamma, phi, sigma};
        StateSpaceFilter.LinearModel model = model(history, p, trend, period);
        StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(model, StateSpaceForecast.column(history), false);
        int last = history.length - 1;
        return StateSpaceForecast.propagate(model, e.means()[last], e.covariances()[last], horizon);
    }

    /** @return the maximised log-likelihood on the training series */
    public double logLikelihood() {
        return logLikelihood;
    }

    /** @return the small-sample corrected AIC */
    public double aicc() {
        int k = 2 + (trend == Trend.NONE ? 0 : 1) + (trend == Trend.DAMPED ? 1 : 0) + (period > 1 ? 1 : 0);
        return StateSpaceForecast.aicc(logLikelihood, k, observations);
    }

    /** @return {alpha, beta, gamma, phi, sigma} */
    public double[] parameters() {
        return new double[] {alpha, beta, gamma, phi, sigma};
    }

    /** @return the trend variant */
    public Trend trend() {
        return trend;
    }

    /** @return the season length, 1 for none */
    public int period() {
        return period;
    }

    private static double mean(double[] y, int from, int to) {
        return Arrays.stream(y, from, Math.min(to, y.length)).average().orElse(0);
    }

    private static double variance(double[] y) {
        double m = mean(y, 0, y.length);
        double s = 0;
        for (double v : y) {
            s += (v - m) * (v - m);
        }
        return s / Math.max(1, y.length - 1);
    }

    @Override
    public String toString() {
        return String.format("ETS(A,%s,%s) alpha=%.3f beta=%.3f gamma=%.3f phi=%.3f sigma=%.3f", switch (trend) {
            case NONE -> "N";
            case ADDITIVE -> "A";
            case DAMPED -> "Ad";
        }, period > 1 ? "A" : "N", alpha, beta, gamma, phi, sigma);
    }
}
