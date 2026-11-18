package io.github.zzuegg.brainiac.experimental.panel;

import io.github.zzuegg.brainiac.experimental.discovery.Distributions;
import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Arrays;

/**
 * Sensitivity analysis for unobserved confounding: how strong would a hidden confounder have to
 * be to explain an observational estimate away?
 *
 * <ul>
 *   <li>{@link #eValue}: the E-value (VanderWeele and Ding, Annals of Internal Medicine 2017,
 *       <a href="https://doi.org/10.7326/M16-2607">doi:10.7326/M16-2607</a>),
 *       {@code RR + sqrt(RR (RR - 1))}, the minimum risk ratio a confounder needs with both
 *       treatment and outcome; continuous outcomes go through {@code RR = exp(0.91 d)} for a
 *       standardised difference {@code d}.</li>
 *   <li>{@link #regression}: omitted-variable bias in partial R² terms (Cinelli and Hazlett,
 *       JRSS-B 2020, <a href="https://doi.org/10.1111/rssb.12348">doi:10.1111/rssb.12348</a>):
 *       the treatment's partial R² with the outcome, the robustness value
 *       {@code RV_q = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2} with {@code f_q = q |t| / sqrt(df)} (and
 *       its version that only removes significance), and the bias bound of a confounder
 *       {@code k} times as strong as an observed covariate.</li>
 *   <li>{@link #rosenbaum}: Rosenbaum bounds for matched pairs (Rosenbaum 2002): the upper bound
 *       of the Wilcoxon signed-rank p-value when hidden bias can change the odds of treatment
 *       within a pair by a factor {@code Gamma}, and {@link #criticalGamma}, the largest
 *       {@code Gamma} at which the effect stays significant.</li>
 * </ul>
 */
public final class SensitivityAnalysis {

    private SensitivityAnalysis() {
    }

    /**
     * The E-value of a risk ratio.
     *
     * @param riskRatio the observed risk ratio (inverted when below 1)
     * @return the E-value, 1 when the ratio is 1
     */
    public static double eValue(double riskRatio) {
        double rr = riskRatio < 1 ? 1 / riskRatio : riskRatio;
        return rr + Math.sqrt(rr * (rr - 1));
    }

    /**
     * The E-value of a standardised mean difference, through {@code RR = exp(0.91 d)}.
     *
     * @param d the effect divided by the outcome's standard deviation
     * @return the E-value
     */
    public static double eValueOfDifference(double d) {
        return eValue(Math.exp(0.91 * Math.abs(d)));
    }

    /**
     * A regression-based sensitivity summary.
     *
     * @param estimate the treatment coefficient
     * @param se its standard error
     * @param df residual degrees of freedom
     * @param partialR2 the treatment's partial R² with the outcome
     * @param robustness the robustness value {@code RV_1} (a confounder explaining this share of
     *     the residual variance of both treatment and outcome brings the estimate to 0)
     * @param robustnessSignificance {@code RV_{1, 0.05}}, the same for losing significance
     */
    public record Regression(double estimate, double se, double df, double partialR2, double robustness,
            double robustnessSignificance) {

        /**
         * The bias bound for a hidden confounder with given partial R² values.
         *
         * @param r2dz its partial R² with treatment (given covariates)
         * @param r2yz its partial R² with the outcome (given treatment and covariates)
         * @return the largest absolute bias
         */
        public double bias(double r2dz, double r2yz) {
            return se * Math.sqrt(df) * Math.sqrt(r2yz * r2dz / (1 - r2dz));
        }

        /**
         * The estimate adjusted for a confounder {@code k} times as strong as an observed
         * covariate (the bound of Cinelli and Hazlett, section 4.4, with {@code k_D = k_Y = k}).
         *
         * @param r2dx the covariate's partial R² with treatment
         * @param r2yx the covariate's partial R² with the outcome
         * @param k the multiple
         * @return {@code {r2dz, r2yz, |estimate| - bias}}
         */
        public double[] benchmark(double r2dx, double r2yx, double k) {
            double r2dz = Math.min(0.999, k * r2dx / (1 - r2dx));
            double r2zx = k * r2dx * r2dx / ((1 - k * r2dx) * (1 - r2dx));
            double r2yz = Math.min(0.999, Math.pow((Math.sqrt(k) + Math.sqrt(Math.max(0, r2zx)))
                    / Math.sqrt(Math.max(1e-9, 1 - r2zx)), 2) * r2yx / (1 - r2yx));
            return new double[] {r2dz, r2yz, Math.abs(estimate) - bias(r2dz, r2yz)};
        }
    }

    private static double rv(double f) {
        return 0.5 * (Math.sqrt(f * f * f * f + 4 * f * f) - f * f);
    }

    /**
     * Fits {@code y ~ 1 + t + x} by OLS and computes the sensitivity summary of the coefficient
     * on {@code t}.
     *
     * @param x covariates {@code [n][d]}
     * @param t treatment (any coding)
     * @param y outcome
     * @return the summary
     */
    public static Regression regression(double[][] x, double[] t, double[] y) {
        int n = y.length;
        int d = x[0].length;
        double[][] design = new double[n][d + 1];
        for (int i = 0; i < n; i++) {
            design[i][0] = t[i];
            System.arraycopy(x[i], 0, design[i], 1, d);
        }
        LeastSquares.Fit full = LeastSquares.fit(design, y, true, 1e-10);
        double df = n - d - 2;
        double sigma2 = full.rss() / df;
        double[] tRes = LeastSquares.fit(x, t, true, 1e-10).residuals();
        double stt = 0;
        for (double v : tRes) {
            stt += v * v;
        }
        double est = full.beta()[1];
        double se = Math.sqrt(sigma2 / stt);
        double tStat = est / se;
        double partial = tStat * tStat / (tStat * tStat + df);
        double f = Math.abs(tStat) / Math.sqrt(df);
        double crit = tQuantile(df);
        double fSig = Math.max(0, f - crit / Math.sqrt(df - 1));
        return new Regression(est, se, df, partial, rv(f), rv(fSig));
    }

    /** The two-sided 5% critical value of Student's t by bisection. */
    private static double tQuantile(double df) {
        double lo = 0;
        double hi = 20;
        for (int it = 0; it < 80; it++) {
            double mid = (lo + hi) / 2;
            if (Distributions.studentTwoSided(mid, df) > 0.05) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) / 2;
    }

    /**
     * Partial R² values of one covariate with treatment and outcome, for benchmarking.
     *
     * @param x covariates
     * @param t treatment
     * @param y outcome
     * @param j the covariate
     * @return {@code {R²_{D~Xj|X-j}, R²_{Y~Xj|D,X-j}}}
     */
    public static double[] covariateR2(double[][] x, double[] t, double[] y, int j) {
        int n = y.length;
        int d = x[0].length;
        double[][] others = new double[n][d - 1];
        double[][] withT = new double[n][d];
        double[] xj = new double[n];
        for (int i = 0; i < n; i++) {
            int c = 0;
            withT[i][0] = t[i];
            for (int a = 0; a < d; a++) {
                if (a != j) {
                    others[i][c] = x[i][a];
                    withT[i][1 + c] = x[i][a];
                    c++;
                }
            }
            xj[i] = x[i][j];
        }
        double rD = partialR2(others, t, xj);
        double rY = partialR2(withT, y, xj);
        return new double[] {rD, rY};
    }

    /** The partial R² of {@code z} in the regression of {@code v} on {@code [base, z]}. */
    private static double partialR2(double[][] base, double[] v, double[] z) {
        double[] rv = LeastSquares.fit(base, v, true, 1e-10).residuals();
        double[] rz = LeastSquares.fit(base, z, true, 1e-10).residuals();
        double c = LeastSquares.correlation(rv, rz);
        return c * c;
    }

    /**
     * The upper bound of the one-sided Wilcoxon signed-rank p-value under hidden bias
     * {@code Gamma} (normal approximation).
     *
     * @param differences treated-minus-control outcome differences of matched pairs
     * @param gamma the bias factor, at least 1
     * @return the worst-case p-value
     */
    public static double rosenbaum(double[] differences, double gamma) {
        double[] nz = Arrays.stream(differences).filter(v -> v != 0).toArray();
        int s = nz.length;
        Integer[] idx = new Integer[s];
        for (int i = 0; i < s; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(Math.abs(nz[a]), Math.abs(nz[b])));
        double t = 0;
        for (int r = 0; r < s; r++) {
            if (nz[idx[r]] > 0) {
                t += r + 1;
            }
        }
        double p = gamma / (1 + gamma);
        double mean = p * s * (s + 1) / 2.0;
        double var = p * (1 - p) * s * (s + 1) * (2 * s + 1) / 6.0;
        return 1 - NormalTails.cdf((t - mean) / Math.sqrt(var));
    }

    /**
     * The largest {@code Gamma} at which the matched-pair effect stays significant.
     *
     * @param differences pair differences
     * @param alpha the level
     * @return the critical {@code Gamma} (1 when not significant even without bias)
     */
    public static double criticalGamma(double[] differences, double alpha) {
        if (rosenbaum(differences, 1) > alpha) {
            return 1;
        }
        double lo = 1;
        double hi = 2;
        while (rosenbaum(differences, hi) <= alpha && hi < 1e4) {
            lo = hi;
            hi *= 2;
        }
        for (int it = 0; it < 60; it++) {
            double mid = (lo + hi) / 2;
            if (rosenbaum(differences, mid) <= alpha) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
}
