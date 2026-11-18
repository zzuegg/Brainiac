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
