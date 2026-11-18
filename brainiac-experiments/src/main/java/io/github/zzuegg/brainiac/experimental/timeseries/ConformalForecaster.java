package io.github.zzuegg.brainiac.experimental.timeseries;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * Conformal prediction intervals for forecasts, including under drift. A base interval
 * {@code [lo, hi]} (a quantile model's, or a point forecast with {@code lo = hi}) is widened
 * by a quantile {@code q} of the nonconformity scores {@code max(lo - y, y - hi)} of past
 * forecasts:
 * <ul>
 *   <li>{@link #split}: split conformal on absolute residuals of a point forecaster, the
 *       {@code ConformalClassifier} recipe for regression;</li>
 *   <li>{@link #cqr}: conformalised quantile regression (Romano, Patterson and Candès, NeurIPS
 *       2019, <a href="https://arxiv.org/abs/1905.03222">arXiv:1905.03222</a>), which keeps
 *       the quantile model's adaptive width and has finite-sample coverage under
 *       exchangeability;</li>
 *   <li>{@link #aci}: adaptive conformal inference (Gibbs and Candès, NeurIPS 2021,
 *       <a href="https://arxiv.org/abs/2106.00170">arXiv:2106.00170</a>): after each
 *       observation the working miscoverage moves as
 *       {@code alpha_t+1 = alpha_t + gamma (alpha - err_t)}, which guarantees the long-run
 *       miscoverage is {@code alpha} under arbitrary drift; scores also slide over a window;</li>
 *   <li>{@link #enbpi}: ensemble batch prediction intervals (Xu and Xie, ICML 2021,
 *       <a href="https://arxiv.org/abs/2010.09107">arXiv:2010.09107</a>): scores are the
 *       leave-one-out residuals of a bootstrap ensemble (no calibration split), refreshed with
 *       each new residual over a sliding window.</li>
 * </ul>
 * Use: {@link #interval} before each observation, {@link #update} after it.
 */
public final class ConformalForecaster {

    private final double alpha;
    private final double gamma;
    private final int window;
    private final Deque<Double> scores = new ArrayDeque<>();
    private double alphaT;

    private ConformalForecaster(double[] initialScores, double alpha, double gamma, int window) {
        this.alpha = alpha;
        this.gamma = gamma;
        this.window = window;
        this.alphaT = alpha;
        for (double s : initialScores) {
            scores.addLast(s);
        }
        trim();
    }

    /**
     * Split conformal on absolute residuals, static.
     *
     * @param residuals calibration residuals {@code y - forecast}
     * @param alpha the target miscoverage, for example 0.2
     * @return the forecaster (use {@code lo = hi = forecast})
     */
    public static ConformalForecaster split(double[] residuals, double alpha) {
        double[] s = new double[residuals.length];
        for (int i = 0; i < s.length; i++) {
            s[i] = Math.abs(residuals[i]);
        }
        return new ConformalForecaster(s, alpha, 0, 0);
    }

    /**
     * Conformalised quantile regression, static.
     *
     * @param lo the quantile model's lower calibration outputs
     * @param hi its upper calibration outputs
     * @param y the calibration targets
     * @param alpha the target miscoverage
     * @return the forecaster
     */
    public static ConformalForecaster cqr(double[] lo, double[] hi, double[] y, double alpha) {
        return new ConformalForecaster(scores(lo, hi, y), alpha, 0, 0);
    }

    /**
     * Adaptive conformal inference on CQR scores.
     *
     * @param lo the quantile model's lower calibration outputs
     * @param hi its upper calibration outputs
     * @param y the calibration targets
     * @param alpha the target miscoverage
     * @param gamma the step size of the miscoverage update, for example 0.01-0.05
     * @param window the scores kept (a sliding window over the most recent)
     * @return the forecaster
     */
    public static ConformalForecaster aci(double[] lo, double[] hi, double[] y, double alpha, double gamma,
            int window) {
        return new ConformalForecaster(scores(lo, hi, y), alpha, gamma, window);
    }

    /**
     * EnbPI from leave-one-out residuals of a bootstrap ensemble.
     *
     * @param looResiduals residuals {@code y_i - mean of the members not trained on i}
     * @param alpha the target miscoverage
     * @param window the residuals kept (sliding)
     * @return the forecaster (use the full ensemble's mean as {@code lo = hi})
     */
    public static ConformalForecaster enbpi(double[] looResiduals, double alpha, int window) {
        double[] s = new double[looResiduals.length];
        for (int i = 0; i < s.length; i++) {
            s[i] = Math.abs(looResiduals[i]);
        }
        return new ConformalForecaster(s, alpha, 0, window);
    }

    private static double[] scores(double[] lo, double[] hi, double[] y) {
        double[] s = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            s[i] = Math.max(lo[i] - y[i], y[i] - hi[i]);
        }
        return s;
    }

    private void trim() {
        while (window > 0 && scores.size() > window) {
            scores.removeFirst();
        }
    }

    /**
     * The finite-sample conformal quantile: the {@code ceil((n + 1)(1 - a))}-th smallest score.
     *
     * @param s scores
     * @param a the miscoverage
     * @return the quantile ({@code +inf} when {@code a} is too small for {@code n}, {@code -inf} when {@code a >= 1})
     */
    public static double quantile(double[] s, double a) {
        if (a >= 1) {
            return Double.NEGATIVE_INFINITY;
        }
        int n = s.length;
        int k = (int) Math.ceil((n + 1) * (1 - a));
        if (k > n || a <= 0) {
            return Double.POSITIVE_INFINITY;
        }
        double[] sorted = s.clone();
        Arrays.sort(sorted);
        return sorted[Math.max(0, k - 1)];
    }

    /**
     * The conformal interval around a base interval.
     *
     * @param lo the base lower bound
     * @param hi the base upper bound
     * @return {@code {lower, upper}} (infinite or empty bounds become the largest score's
     *     width or the base interval's midpoint)
     */
    public double[] interval(double lo, double hi) {
        double[] s = new double[scores.size()];
        int i = 0;
        for (double v : scores) {
            s[i++] = v;
        }
        double q = quantile(s, alphaT);
        if (q == Double.POSITIVE_INFINITY) {
            q = Arrays.stream(s).max().orElse(0) * 2;
        }
        if (q == Double.NEGATIVE_INFINITY) {
            double mid = (lo + hi) / 2;
            return new double[] {mid, mid};
        }
        double a = lo - q;
        double b = hi + q;
        return a <= b ? new double[] {a, b} : new double[] {(lo + hi) / 2, (lo + hi) / 2};
    }

    /**
     * Records an observation: updates the working miscoverage (ACI) and the sliding scores
     * (ACI, EnbPI).
     *
     * @param lo the base lower bound used for it
     * @param hi the base upper bound
     * @param y the observed value
     */
    public void update(double lo, double hi, double y) {
        double[] iv = interval(lo, hi);
        double err = y < iv[0] || y > iv[1] ? 1 : 0;
        alphaT += gamma * (alpha - err);
        if (window > 0) {
            scores.addLast(Math.max(lo - y, y - hi));
            trim();
        }
    }

    /** @return the current working miscoverage {@code alpha_t} */
    public double workingAlpha() {
        return alphaT;
    }
}
