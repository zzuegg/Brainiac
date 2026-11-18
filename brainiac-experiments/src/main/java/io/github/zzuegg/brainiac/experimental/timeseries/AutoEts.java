package io.github.zzuegg.brainiac.experimental.timeseries;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Automatic exponential smoothing over the ETS family of Hyndman, Koehler, Snyder and Grose
 * (IJF 2002) and Hyndman and Khandakar ("Automatic time series forecasting: the forecast package
 * for R", J. Stat. Softw. 2008): error additive or multiplicative, trend none, additive or damped,
 * season none, additive or multiplicative. Each model runs its innovations recursion
 *
 * <pre>
 *   μ_t = (l + φ b) (+|×) s_t-m,   e_t = y_t - μ_t
 *   l_t = l + φ b + α e_t (/ s_t-m when the season is multiplicative)
 *   b_t = φ b + β e_t (/ s_t-m)
 *   s_t = s_t-m + γ e_t (/ (l + φ b) when multiplicative)
 * </pre>
 *
 * which gives the same point forecasts for both error types; the error type changes the
 * likelihood: Gaussian on {@code e_t} (additive) or on {@code e_t / μ_t} (multiplicative, with
 * the Jacobian {@code Σ log |μ_t|}). Parameters are fitted by a coarse-to-fine grid on the
 * likelihood (the fast fit wave 11 asked for, instead of ~1 s of L-BFGS per series), the
 * starting states come from the first seasons, and the model with the lowest AICc wins.
 * Prediction intervals are simulated from the fitted model with Gaussian innovations of the
 * in-sample residual variance. Multiplicative components are skipped for non-positive series.
 */
public final class AutoEts {

    /** The error type. */
    public enum Error {
        /** additive errors */
        ADDITIVE,
        /** multiplicative (relative) errors */
        MULTIPLICATIVE
    }

    /** The trend type. */
    public enum Trend {
        /** no trend */
        NONE,
        /** additive trend */
        ADDITIVE,
        /** damped additive trend */
        DAMPED
    }

    /** The season type. */
    public enum Season {
        /** no season */
        NONE,
        /** additive season */
        ADDITIVE,
        /** multiplicative season */
        MULTIPLICATIVE
    }

    private final Error error;
    private final Trend trend;
    private final Season season;
    private final int period;
    private final double[] params;
    private final double[] finalState;
    private final double sigma;
    private final double logLikelihood;
    private final int observations;

    private AutoEts(Error error, Trend trend, Season season, int period, double[] params, double[] finalState,
            double sigma, double logLikelihood, int observations) {
        this.error = error;
        this.trend = trend;
        this.season = season;
        this.period = period;
        this.params = params;
        this.finalState = finalState;
        this.sigma = sigma;
        this.logLikelihood = logLikelihood;
        this.observations = observations;
    }

    /**
     * Fits every admissible model and keeps the lowest AICc.
     *
     * @param y the series (at least two seasons for seasonal models)
     * @param period the season length (1 for none)
     * @return the selected model
     */
    public static AutoEts select(double[] y, int period) {
        boolean positive = Arrays.stream(y).min().orElse(0) > 0;
        AutoEts best = null;
        for (Error e : Error.values()) {
            if (e == Error.MULTIPLICATIVE && !positive) {
                continue;
            }
            for (Trend t : Trend.values()) {
                for (Season s : Season.values()) {
                    if (s != Season.NONE && (period < 2 || y.length < 2 * period)) {
                        continue;
                    }
                    if (s == Season.MULTIPLICATIVE && !positive) {
                        continue;
                    }
                    AutoEts m = fit(y, e, t, s, period);
                    if (best == null || m.aicc() < best.aicc()) {
                        best = m;
                    }
                }
            }
        }
        return best;
    }

    /**
     * Fits one model by a coarse-to-fine grid.
     *
     * @param y the series
     * @param error the error type
     * @param trend the trend type
     * @param season the season type
     * @param period the season length
     * @return the fitted model
     */
    public static AutoEts fit(double[] y, Error error, Trend trend, Season season, int period) {
        int m = season == Season.NONE ? 1 : period;
        boolean hasTrend = trend != Trend.NONE;
        boolean damped = trend == Trend.DAMPED;
        boolean seasonal = season != Season.NONE;
        // coordinates alpha, beta/alpha, gamma/(1-alpha), phi
        double[] best = {0.5, hasTrend ? 0.3 : 0, seasonal ? 0.3 : 0, damped ? 0.9 : 1};
        double bestLl = Double.NEGATIVE_INFINITY;
        double[] width = {0.24, 0.24, 0.24, 0.045};
        for (int level = 0; level < 4; level++) {
            double[] centre = level == 0 ? new double[] {0.5, 0.5, 0.5, 0.89} : best.clone();
            int half = level == 0 ? 2 : 1;
            double scale = level == 0 ? 1 : 1.0 / (1 << level);
            for (int i = -half; i <= half; i++) {
                for (int j = hasTrend ? -half : 0; j <= (hasTrend ? half : 0); j++) {
                    for (int k = seasonal ? -half : 0; k <= (seasonal ? half : 0); k++) {
                        for (int l = damped ? -half : 0; l <= (damped ? half : 0); l++) {
                            double[] c = {clip(centre[0] + i * width[0] * scale, 0.02, 0.98),
                                hasTrend ? clip(centre[1] + j * width[1] * scale, 0, 1) : 0,
                                seasonal ? clip(centre[2] + k * width[2] * scale, 0, 0.98) : 0,
                                damped ? clip(centre[3] + l * width[3] * scale, 0.8, 0.98) : 1};
                            double ll = run(y, natural(c), error, trend, season, m).logLikelihood;
                            if (ll > bestLl) {
                                bestLl = ll;
                                best = c;
                            }
                        }
                    }
                }
            }
        }
        double[] p = natural(best);
        Run r = run(y, p, error, trend, season, m);
        return new AutoEts(error, trend, season, m, p, r.state, r.sigma, r.logLikelihood, y.length);
    }

    private static double clip(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double[] natural(double[] c) {
        return new double[] {c[0], c[0] * c[1], (1 - c[0]) * c[2], c[3]};
    }

    /** The result of one pass of the recursion. */
    private record Run(double logLikelihood, double sigma, double[] state, double[] residuals) {
    }

    /** State layout: {level, slope, s_t-1, ..., s_t-m} (the last entry is the oldest season). */
    private static Run run(double[] y, double[] p, Error error, Trend trend, Season season, int m) {
        int n = y.length;
        double[] state = initial(y, trend, season, m);
        double a = p[0];
        double b = p[1];
        double g = p[2];
        double phi = trend == Trend.DAMPED ? p[3] : trend == Trend.ADDITIVE ? 1 : 0;
        double sse = 0;
        double logMu = 0;
        double[] res = new double[n];
        for (int t = 0; t < n; t++) {
            double lb = state[0] + phi * state[1];
            double s = season == Season.NONE ? 0 : state[1 + m];
            double mu = switch (season) {
                case NONE -> lb;
                case ADDITIVE -> lb + s;
                case MULTIPLICATIVE -> lb * s;
            };
            double e = y[t] - mu;
            if (error == Error.MULTIPLICATIVE) {
                if (!(Math.abs(mu) > 1e-9)) {
                    return new Run(Double.NEGATIVE_INFINITY, Double.NaN, state, res);
                }
                double rel = e / mu;
                sse += rel * rel;
                logMu += Math.log(Math.abs(mu));
                res[t] = rel;
            } else {
                sse += e * e;
                res[t] = e;
            }
            double level;
            double slope;
            double ns;
            if (season == Season.MULTIPLICATIVE) {
                double safe = Math.abs(s) < 1e-9 ? 1e-9 : s;
                level = lb + a * e / safe;
                slope = phi * state[1] + b * e / safe;
                ns = s + g * e / (Math.abs(lb) < 1e-9 ? 1e-9 : lb);
            } else {
                level = lb + a * e;
                slope = phi * state[1] + b * e;
                ns = s + g * e;
            }
            state[0] = level;
            state[1] = trend == Trend.NONE ? 0 : slope;
            if (season != Season.NONE) {
                System.arraycopy(state, 2, state, 3, m - 1);
                state[2] = ns;
            }
            if (!Double.isFinite(level)) {
                return new Run(Double.NEGATIVE_INFINITY, Double.NaN, state, res);
            }
        }
        double s2 = Math.max(sse / n, 1e-300);
        double ll = -0.5 * n * (Math.log(2 * Math.PI * s2) + 1) - logMu;
        return new Run(ll, Math.sqrt(s2), state, res);
    }

    private static double[] initial(double[] y, Trend trend, Season season, int m) {
        double[] state = new double[2 + (season == Season.NONE ? 0 : m)];
        int span = season == Season.NONE ? Math.min(y.length, Math.max(2, Math.min(10, y.length / 2))) : m;
        double l0 = mean(y, 0, span);
        state[0] = l0;
        if (trend != Trend.NONE) {
            state[1] = y.length >= 2 * span ? (mean(y, span, 2 * span) - l0) / span : 0;
        }
        if (season != Season.NONE) {
            for (int k = 0; k < m; k++) {
                // state[2 + j] is s_t-1-j; the first observation uses state[1 + m] = s_t-m, i.e. y[0]
                double v = season == Season.ADDITIVE ? y[k] - l0 : y[k] / l0;
                state[1 + m - k] = v;
            }
        }
        return state;
    }

    private static double mean(double[] y, int from, int to) {
        double s = 0;
        for (int i = from; i < to; i++) {
            s += y[i];
        }
        return s / Math.max(1, to - from);
    }

    /**
     * Point forecasts.
     *
     * @param horizon steps ahead
     * @return the forecasts
     */
    public double[] forecast(int horizon) {
        double[] out = new double[horizon];
        double phi = trend == Trend.DAMPED ? params[3] : trend == Trend.ADDITIVE ? 1 : 0;
        double damp = 0;
        for (int h = 1; h <= horizon; h++) {
            damp += Math.pow(phi, h);
            double lb = finalState[0] + damp * finalState[1];
            double s = season == Season.NONE ? 0 : finalState[2 + Math.floorMod(-h, period)];
            out[h - 1] = switch (season) {
                case NONE -> lb;
                case ADDITIVE -> lb + s;
                case MULTIPLICATIVE -> lb * s;
            };
        }
        return out;
    }

    /**
     * Simulated prediction intervals.
     *
     * @param horizon steps ahead
     * @param level the coverage, for example 0.8
     * @param paths simulated paths, for example 500
     * @param seed the seed
     * @return {@code [2][horizon]} lower and upper bounds
     */
    public double[][] intervals(int horizon, double level, int paths, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] sims = new double[horizon][paths];
        double phi = trend == Trend.DAMPED ? params[3] : trend == Trend.ADDITIVE ? 1 : 0;
        for (int p = 0; p < paths; p++) {
            double[] st = finalState.clone();
            for (int h = 0; h < horizon; h++) {
                double lb = st[0] + phi * st[1];
                double s = season == Season.NONE ? 0 : st[1 + period];
                double mu = switch (season) {
                    case NONE -> lb;
                    case ADDITIVE -> lb + s;
                    case MULTIPLICATIVE -> lb * s;
                };
                double eps = r.nextGaussian() * sigma;
                double e = error == Error.MULTIPLICATIVE ? mu * eps : eps;
                sims[h][p] = mu + e;
                if (season == Season.MULTIPLICATIVE) {
                    double safe = Math.abs(s) < 1e-9 ? 1e-9 : s;
                    st[0] = lb + params[0] * e / safe;
                    st[1] = trend == Trend.NONE ? 0 : phi * st[1] + params[1] * e / safe;
                    double ns = s + params[2] * e / (Math.abs(lb) < 1e-9 ? 1e-9 : lb);
                    System.arraycopy(st, 2, st, 3, period - 1);
                    st[2] = ns;
                } else {
                    st[0] = lb + params[0] * e;
                    st[1] = trend == Trend.NONE ? 0 : phi * st[1] + params[1] * e;
                    if (season != Season.NONE) {
                        double ns = s + params[2] * e;
                        System.arraycopy(st, 2, st, 3, period - 1);
                        st[2] = ns;
                    }
                }
            }
        }
        double[][] out = new double[2][horizon];
        for (int h = 0; h < horizon; h++) {
            Arrays.sort(sims[h]);
            out[0][h] = sims[h][(int) Math.floor((1 - level) / 2 * (paths - 1))];
            out[1][h] = sims[h][(int) Math.ceil((1 + level) / 2 * (paths - 1))];
        }
        return out;
    }

    /** @return the small-sample corrected AIC, counting smoothing parameters, initial states and the variance */
    public double aicc() {
        int k = 1 + 1 + (trend == Trend.NONE ? 0 : 2) + (trend == Trend.DAMPED ? 1 : 0)
                + (season == Season.NONE ? 0 : period);
        int n = observations;
        double aic = -2 * logLikelihood + 2 * k;
        return n - k - 1 > 0 ? aic + 2.0 * k * (k + 1) / (n - k - 1) : Double.POSITIVE_INFINITY;
    }

    /** @return the model name, for example {@code ETS(M,Ad,A)} */
    public String name() {
        return "ETS(" + (error == Error.ADDITIVE ? "A" : "M") + "," + switch (trend) {
            case NONE -> "N";
            case ADDITIVE -> "A";
            case DAMPED -> "Ad";
        } + "," + switch (season) {
            case NONE -> "N";
            case ADDITIVE -> "A";
            case MULTIPLICATIVE -> "M";
        } + ")";
    }

    /** @return {alpha, beta, gamma, phi} */
    public double[] parameters() {
        return params.clone();
    }

    /** @return the innovation standard deviation (relative for multiplicative errors) */
    public double sigma() {
        return sigma;
    }

    @Override
    public String toString() {
        return name();
    }
}
