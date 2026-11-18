package io.github.zzuegg.brainiac.experimental.online;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Conformal PID control for online prediction intervals (Angelopoulos, Candès and Tibshirani,
 * "Conformal PID control for time series prediction", NeurIPS 2023,
 * <a href="https://arxiv.org/abs/2307.16895">arXiv:2307.16895</a>). The interval is
 * {@code ŷ ± q_t} for the absolute residual score {@code s = |y - ŷ|}; with the miscoverage
 * indicator {@code err_t = 1{s_t > q_t}} and its running sum {@code E_t = Σ (err_i - α)}:
 *
 * <ul>
 *   <li>P, quantile tracking: {@code q_t+1 = q_t + η (err_t - α)}, with {@code η} a
 *       learning rate times the largest recent score, so the quantile itself (not the
 *       miscoverage level as in ACI) moves;</li>
 *   <li>I, the saturating integrator {@code r_t(E_t) = K_I tan(E_t log t / (t C_sat))}, clipped
 *       to {@code ±∞} past {@code π/2}, which guarantees long-run coverage for any sequence;</li>
 *   <li>D-like, the scorecaster: a model forecasting the next score, here an online
 *       least-squares autoregression on the last {@code p} scores, whose forecast is added so
 *       that trends and bursts are anticipated rather than reacted to.</li>
 * </ul>
 *
 * <p>Long-run coverage is guaranteed for {@link Kind#PI} and {@link Kind#PID} whatever the base
 * forecaster; {@link Kind#QUANTILE_TRACKING} keeps it whenever the scores are bounded.
 */
public final class ConformalPid {

    /** Which terms are on. */
    public enum Kind {
        /** the P term alone */
        QUANTILE_TRACKING,
        /** P plus the saturating integrator */
        PI,
        /** P, I and the linear scorecaster */
        PID
    }

    private final Kind kind;
    private final double alpha;
    private final double rate;
    private final double ki;
    private final double cSat;
    private final int order;
    private final Deque<Double> recent = new ArrayDeque<>();
    private final int window;
    private double qp;
    private double scale;
    private double errorSum;
    private int t;
    private boolean started;
    private final double[][] gram;
    private final double[] moment;
    private final double[] lags;

    private ConformalPid(Kind kind, double alpha, double rate, double ki, double cSat, int order, int window) {
        this.kind = kind;
        this.alpha = alpha;
        this.rate = rate;
        this.ki = ki;
        this.cSat = cSat;
        this.order = order;
        this.window = window;
        this.gram = new double[order + 1][order + 1];
        this.moment = new double[order + 1];
        this.lags = new double[order];
        for (int i = 0; i <= order; i++) {
            gram[i][i] = 1e-3;
        }
    }

    /**
     * A controller with default settings: learning rate 0.1 and {@code K_I = 0.5}, both relative to the
     * largest recent score,
     * {@code C_sat = 2}, an AR(3) scorecaster, a window of 50 scores.
     *
     * @param kind the terms
     * @param alpha the target miscoverage, for example 0.1
     * @return the controller
     */
    public static ConformalPid of(Kind kind, double alpha) {
        return of(kind, alpha, 0.1, 0.5, 2, 3, 50);
    }

    /**
     * A controller.
     *
     * @param kind the terms
     * @param alpha the target miscoverage
     * @param rate the P learning rate, relative to the largest of the recent scores
     * @param ki the integrator gain {@code K_I}, relative to the largest of the recent scores
     * @param cSat the integrator saturation {@code C_sat}
     * @param order the scorecaster's autoregressive order
     * @param window the recent scores used for the rate scale
     * @return the controller
     */
    public static ConformalPid of(Kind kind, double alpha, double rate, double ki, double cSat, int order,
            int window) {
        return new ConformalPid(kind, alpha, rate, ki, cSat, order, window);
    }

    private double integrator() {
        if (kind == Kind.QUANTILE_TRACKING || t < 2) {
            return 0;
        }
        double arg = errorSum * Math.log(t) / (t * cSat);
        if (Math.abs(arg) >= Math.PI / 2) {
            return Math.signum(arg) * 1e12;
        }
        return ki * scale * Math.tan(arg);
    }

    private double scorecast() {
        if (kind != Kind.PID || t <= order) {
            return 0;
        }
        double[] beta = solve(gram, moment);
        double f = beta[order];
        for (int i = 0; i < order; i++) {
            f += beta[i] * lags[i];
        }
        return Math.max(0, f);
    }

    /** @return the current half-width {@code q_t} */
    public double quantile() {
        return Math.max(0, qp + integrator() + scorecast());
    }

    /**
     * The interval around a point forecast.
     *
     * @param forecast {@code ŷ}
     * @return {@code {lower, upper}}
     */
    public double[] interval(double forecast) {
        double q = quantile();
        return new double[] {forecast - q, forecast + q};
    }

    /**
     * Observes the outcome.
     *
     * @param forecast the point forecast {@code ŷ} that was used
     * @param y the observed value
     */
    public void update(double forecast, double y) {
        double s = Math.abs(y - forecast);
        if (!started) {
            started = true;
            qp = kind == Kind.PID ? 0 : s;
        }
        double err = s > quantile() ? 1 : 0;
        recent.addLast(s);
        while (recent.size() > window) {
            recent.removeFirst();
        }
        scale = 0;
        for (double v : recent) {
            scale = Math.max(scale, v);
        }
        qp += rate * scale * (err - alpha);
        errorSum += err - alpha;
        if (kind == Kind.PID) {
            if (t >= order) {
                double[] row = new double[order + 1];
                System.arraycopy(lags, 0, row, 0, order);
                row[order] = 1;
                for (int i = 0; i <= order; i++) {
                    for (int j = 0; j <= order; j++) {
                        gram[i][j] = 0.99 * gram[i][j] + row[i] * row[j];
                    }
                    moment[i] = 0.99 * moment[i] + row[i] * s;
                }
            }
            System.arraycopy(lags, 0, lags, 1, order - 1);
            lags[0] = s;
        }
        t++;
    }

    private static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int c = 0; c < n; c++) {
            int p = c;
            for (int r = c + 1; r < n; r++) {
                if (Math.abs(m[r][c]) > Math.abs(m[p][c])) {
                    p = r;
                }
            }
            double[] tmp = m[c];
            m[c] = m[p];
            m[p] = tmp;
            if (Math.abs(m[c][c]) < 1e-12) {
                continue;
            }
            for (int r = 0; r < n; r++) {
                if (r != c) {
                    double f = m[r][c] / m[c][c];
                    for (int k = c; k <= n; k++) {
                        m[r][k] -= f * m[c][k];
                    }
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = Math.abs(m[i][i]) < 1e-12 ? 0 : m[i][n] / m[i][i];
        }
        return x;
    }
}
