package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;

/**
 * ARIMA(p, d, q) with optional seasonal differencing {@code (1 - B^m)^D} (Box and Jenkins,
 * <i>Time Series Analysis: Forecasting and Control</i>, 1970;
 * <a href="https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method">Box-Jenkins method</a>),
 * in state-space form: the differenced series follows ARMA(p, q) in Harvey's representation
 * (state {@code r = max(p, q + 1)}, transition with the AR coefficients in the first column,
 * noise loading {@code [1, theta_1, ..., theta_r-1]}), and the differencing is undone inside
 * the state, which also carries the last {@code d + D m} observations, so the
 * {@link StateSpaceFilter#kalman} likelihood, filter and forecasts are on the original scale
 * and the prediction intervals include the integration. AR and MA coefficients are kept
 * stationary and invertible by mapping unconstrained values through {@code tanh} partial
 * autocorrelations (Jones 1980); they are fitted by L-BFGS, and {@link #select} picks the
 * orders by AICc.
 */
public final class Arima {

    private final int p;
    private final int d;
    private final int q;
    private final int season;
    private final int seasonalD;
    private final double[] ar;
    private final double[] ma;
    private final double mean;
    private final double sigma;
    private final double logLikelihood;
    private final int observations;

    private Arima(int p, int d, int q, int season, int seasonalD, double[] u, double[] y) {
        this.p = p;
        this.d = d;
        this.q = q;
        this.season = season;
        this.seasonalD = seasonalD;
        this.ar = pacfToCoefficients(u, 0, p);
        this.ma = maCoefficients(u, p, q);
        this.mean = d + seasonalD == 0 ? u[p + q] : 0;
        this.sigma = Math.exp(u[p + q + 1]);
        this.logLikelihood = loglik(y, ar, ma, mean, sigma);
        this.observations = y.length;
    }

    /**
     * Maps unconstrained values to the coefficients of a stationary polynomial: {@code tanh}
     * gives partial autocorrelations in (-1, 1), and the Durbin-Levinson recursion turns them
     * into coefficients.
     */
    private static double[] pacfToCoefficients(double[] u, int from, int count) {
        double[] phi = new double[count];
        for (int k = 0; k < count; k++) {
            double r = Math.tanh(u[from + k]);
            double[] next = phi.clone();
            next[k] = r;
            for (int j = 0; j < k; j++) {
                next[j] = phi[j] - r * phi[k - 1 - j];
            }
            phi = next;
        }
        return phi;
    }

    /** MA coefficients {@code theta = -c}, so {@code 1 + sum theta_i z^i} is invertible. */
    private static double[] maCoefficients(double[] u, int from, int count) {
        double[] c = pacfToCoefficients(u, from, count);
        for (int i = 0; i < c.length; i++) {
            c[i] = -c[i];
        }
        return c;
    }

    /** The differencing polynomial: {@code y_t = sum_i delta_i y_t-i + w_t}. */
    private double[] delta() {
        double[] poly = {1};
        for (int i = 0; i < d; i++) {
            poly = multiply(poly, new double[] {1, -1});
        }
        for (int i = 0; i < seasonalD; i++) {
            double[] s = new double[season + 1];
            s[0] = 1;
            s[season] = -1;
            poly = multiply(poly, s);
        }
        double[] delta = new double[poly.length - 1];
        for (int i = 1; i < poly.length; i++) {
            delta[i - 1] = -poly[i];
        }
        return delta;
    }

    private static double[] multiply(double[] a, double[] b) {
        double[] c = new double[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                c[i + j] += a[i] * b[j];
            }
        }
        return c;
    }

    /**
     * The state-space model on {@code y - mean}; state {@code [a_t (r), y_t-1 .. y_t-k]}, prior
     * on the ARMA part at its stationary scale and on the lags at the given history values.
     */
    private StateSpaceFilter.LinearModel model(double[] phi, double[] theta, double s, double[] lags) {
        double[] delta = delta();
        int k = delta.length;
        int r = Math.max(phi.length, theta.length + 1);
        int n = r + k;
        double[][] f = new double[n][n];
        for (int i = 0; i < r; i++) {
            if (i < phi.length) {
                f[i][0] = phi[i];
            }
            if (i + 1 < r) {
                f[i][i + 1] = 1;
            }
        }
        if (k > 0) {
            // y_t = a_t[0] + sum delta_i y_t-i becomes the newest lag
            f[r][0] = 1;
            for (int i = 0; i < k; i++) {
                f[r][r + i] = delta[i];
            }
            for (int i = 1; i < k; i++) {
                f[r + i][r + i - 1] = 1;
            }
        }
        double[] g = new double[r];
        g[0] = 1;
        for (int i = 0; i < theta.length; i++) {
            g[i + 1] = theta[i];
        }
        double[][] qm = new double[n][n];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < r; j++) {
                qm[i][j] = s * s * g[i] * g[j];
            }
        }
        double[][] h = new double[1][n];
        h[0][0] = 1;
        for (int i = 0; i < k; i++) {
            h[0][r + i] = delta[i];
        }
        double[][] rm = {{1e-6 * s * s}};
        double[] x0 = new double[n];
        double[][] p0 = new double[n][n];
        // a crude stationary scale for the ARMA block: sigma^2 (1 + |theta|^2) / (1 - |phi|^2)
        double ps = 0;
        for (double v : phi) {
            ps += v * v;
        }
        double ts = 1;
        for (double v : theta) {
            ts += v * v;
        }
        double scale = s * s * ts / Math.max(0.05, 1 - Math.min(0.95, ps));
        for (int i = 0; i < r; i++) {
            p0[i][i] = scale;
        }
        for (int i = 0; i < k; i++) {
            x0[r + i] = lags[i];
            p0[r + i][r + i] = 1e-8;
        }
        return new StateSpaceFilter.LinearModel(f, qm, h, rm, x0, p0);
    }

    private int lagCount() {
        return d + seasonalD * season;
    }

    private double loglik(double[] y, double[] phi, double[] theta, double mu, double s) {
        int k = lagCount();
        double[] lags = new double[k];
        for (int i = 0; i < k; i++) {
            lags[i] = y[k - 1 - i] - mu;
        }
        double[] rest = new double[y.length - k];
        for (int t = k; t < y.length; t++) {
            rest[t - k] = y[t] - mu;
        }
        return StateSpaceFilter.kalman(model(phi, theta, s, lags), StateSpaceForecast.column(rest), false)
                .logLikelihood();
    }

    /**
     * Fits ARIMA(p, d, q) x (0, D, 0)_m.
     *
     * @param y the series
     * @param p the AR order
     * @param d the differencing order (0 to 2)
     * @param q the MA order
     * @param season the seasonal period {@code m} (ignored when {@code seasonalD} is 0)
     * @param seasonalD the seasonal differencing order (0 or 1)
     * @param iterations L-BFGS iterations
     * @return the fitted model
     */
    public static Arima fit(double[] y, int p, int d, int q, int season, int seasonalD, int iterations) {
        double mu = 0;
        for (double v : y) {
            mu += v / y.length;
        }
        double var = 0;
        for (double v : y) {
            var += (v - mu) * (v - mu) / y.length;
        }
        double[] u0 = new double[p + q + 2];
        u0[p + q] = mu;
        u0[p + q + 1] = Math.log(Math.sqrt(var) * 0.5 + 1e-6);
        Arima shape = new Arima(p, d, q, season, seasonalD, u0, y);
        boolean hasMean = d + seasonalD == 0;
        double[] u = StateSpaceForecast.maximise(v -> shape.loglik(y, pacfToCoefficients(v, 0, p),
                maCoefficients(v, p, q), hasMean ? v[p + q] : 0, Math.exp(v[p + q + 1])), u0, iterations);
        return new Arima(p, d, q, season, seasonalD, u, y);
    }

    /**
     * Searches {@code p <= maxP}, {@code q <= maxQ} and {@code d in {0, 1}} (with the given
     * seasonal differencing) and keeps the lowest AICc. Models with different differencing are
     * compared on the same observations, as is usual for a fixed {@code d} search.
     *
     * @param y the series
     * @param maxP the largest AR order
     * @param maxQ the largest MA order
     * @param season the seasonal period
     * @param seasonalD the seasonal differencing order (0 or 1)
     * @param iterations L-BFGS iterations per fit
     * @return the selected model
     */
    public static Arima select(double[] y, int maxP, int maxQ, int season, int seasonalD, int iterations) {
        Arima best = null;
        for (int dd = 0; dd <= 1; dd++) {
            for (int pp = 0; pp <= maxP; pp++) {
                for (int qq = 0; qq <= maxQ; qq++) {
                    Arima a = fit(y, pp, dd, qq, season, seasonalD, iterations);
                    if (best == null || a.aicc() < best.aicc()) {
                        best = a;
                    }
                }
            }
        }
        return best;
    }

    /**
     * Filters a history with the fitted parameters and forecasts past its end.
     *
     * @param history observed values, longer than {@code d + D m}
     * @param horizon steps ahead
     * @return the forecast with exact Gaussian variances (including the integration)
     */
    public StateSpaceForecast forecast(double[] history, int horizon) {
        int k = lagCount();
        double[] lags = new double[k];
        for (int i = 0; i < k; i++) {
            lags[i] = history[k - 1 - i] - mean;
        }
        double[] rest = new double[history.length - k];
        for (int t = k; t < history.length; t++) {
            rest[t - k] = history[t] - mean;
        }
        StateSpaceFilter.LinearModel model = model(ar, ma, sigma, lags);
        StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(model, StateSpaceForecast.column(rest), false);
        int last = rest.length - 1;
        StateSpaceForecast f = StateSpaceForecast.propagate(model, e.means()[last], e.covariances()[last], horizon);
        double[] m = f.mean().clone();
        for (int h = 0; h < horizon; h++) {
            m[h] += mean;
        }
        return new StateSpaceForecast(m, f.variance());
    }

    /** @return the maximised log-likelihood */
    public double logLikelihood() {
        return logLikelihood;
    }

    /** @return the small-sample corrected AIC */
    public double aicc() {
        int k = p + q + 1 + (d + seasonalD == 0 ? 1 : 0);
        return StateSpaceForecast.aicc(logLikelihood, k, observations - lagCount());
    }

    /** @return the AR coefficients */
    public double[] ar() {
        return ar.clone();
    }

    /** @return the MA coefficients */
    public double[] ma() {
        return ma.clone();
    }

    /** @return the innovation standard deviation */
    public double sigma() {
        return sigma;
    }

    @Override
    public String toString() {
        return "ARIMA(" + p + "," + d + "," + q + ")" + (seasonalD > 0 ? "(0," + seasonalD + ",0)_" + season : "")
                + String.format(" sigma=%.3f", sigma);
    }
}
