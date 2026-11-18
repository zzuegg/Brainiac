package io.github.zzuegg.brainiac.experiments.econometrics;

import io.github.zzuegg.brainiac.experimental.causal.DoubleMl;
import io.github.zzuegg.brainiac.experimental.causal.Regressor;
import io.github.zzuegg.brainiac.experimental.data.PanelStudies;
import io.github.zzuegg.brainiac.experimental.ml.CausalEstimators;
import io.github.zzuegg.brainiac.experimental.panel.BalancingWeights;
import io.github.zzuegg.brainiac.experimental.panel.SensitivityAnalysis;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Covariate balancing and sensitivity analysis on Kang-Schafer studies (wave 11, items 11.3.9
 * and 11.3.10).
 *
 * <ul>
 *   <li>{@link #weighting}: entropy balancing, CBPS and stable balancing weights against logistic
 *       IPW odds weights, normalised IPW and AIPW from {@link CausalEstimators}, cross-fitted
 *       {@link DoubleMl} (ridge nuisances), 1:1 nearest-neighbour matching with replacement and
 *       the naive difference. With {@code misspecified} only the Kang-Schafer transforms of the
 *       covariates are observed, so every linear model (propensity and outcome) is wrong. Scores:
 *       bias and RMSE of the ATT (the effect is constant), the largest standardised mean
 *       difference after weighting and the effective number of controls (NaN where a method has
 *       no control weights).</li>
 *   <li>{@link #sensitivity}: correctly specified studies with a hidden confounder {@code s}
 *       times as strong as the observed covariate {@code z2} (same log-odds slope on treatment
 *       and same outcome slope, times {@code s}). Per study: the OLS estimate and whether its
 *       95% interval excludes 0, the entropy-balancing estimate, the E-value of the standardised
 *       estimate, the robustness value {@code RV_1}, the Cinelli-Hazlett estimate adjusted for a
 *       confounder as strong as {@code z2} ({@code k = 1}), the benchmark multiple {@code k*} at
 *       which the adjusted estimate reaches 0, and the critical Rosenbaum {@code Gamma} of the
 *       matched pairs.</li>
 * </ul>
 */
public final class ObservationalAdjustment {

    private ObservationalAdjustment() {
    }

    /**
     * 1:1 nearest-neighbour matching of every treated unit to a control (with replacement) on
     * standardised covariates.
     *
     * @param x covariates
     * @param t treatment
     * @param y outcome
     * @return the treated-minus-matched-control differences
     */
    static double[] matchedDifferences(double[][] x, int[] t, double[] y) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                sd[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        int treated = 0;
        for (int v : t) {
            treated += v;
        }
        double[] out = new double[treated];
        int k = 0;
        for (int i = 0; i < n; i++) {
            if (t[i] != 1) {
                continue;
            }
            int best = -1;
            double bd = Double.POSITIVE_INFINITY;
            for (int j = 0; j < n; j++) {
                if (t[j] != 0) {
                    continue;
                }
                double dist = 0;
                for (int a = 0; a < d; a++) {
                    double v = (x[i][a] - x[j][a]) / Math.sqrt(sd[a] + 1e-12);
                    dist += v * v;
                }
                if (dist < bd) {
                    bd = dist;
                    best = j;
                }
            }
            out[k++] = y[i] - y[best];
        }
        return out;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double a : v) {
            s += a / v.length;
        }
        return s;
    }

    private static void add(Map<String, double[]> out, String key, double est, double truth, double smd,
            double ess, int sims) {
        double[] acc = out.computeIfAbsent(key, s -> new double[4]);
        acc[0] += (est - truth) / sims;
        acc[1] += (est - truth) * (est - truth) / sims;
        acc[2] += smd / sims;
        acc[3] += ess / sims;
    }

    /**
     * Weighting estimators on Kang-Schafer studies without hidden confounding.
     *
     * @param misspecified whether only the transformed covariates are observed
     * @param overlap the propensity slope (1 is the original design)
     * @param n units
     * @param sims simulations
     * @param seed the seed
     * @return method to {@code {bias, RMSE, max |SMD|, effective controls}}
     */
    public static Map<String, double[]> weighting(boolean misspecified, double overlap, int n, int sims,
            long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double effect = 10;
        for (int s = 0; s < sims; s++) {
            PanelStudies.Observational o = PanelStudies.observational(n, misspecified, overlap, effect, 0, 0,
                    seed + s);
            double[][] x = o.x();
            int[] t = o.t();
            double[] y = o.y();
            add(out, "naive difference", CausalEstimators.naive(t, y), effect, Double.NaN, Double.NaN, sims);
            double[] e = CausalEstimators.propensity(x, t, 1e-3);
            add(out, "IPW (normalised, ATE)", CausalEstimators.ipw(t, y, e, 0.01), effect, Double.NaN, Double.NaN,
                    sims);
            double[][] mu = CausalEstimators.outcomes(x, t, y, 1e-3);
            add(out, "AIPW", CausalEstimators.aipw(t, y, e, mu[0], mu[1], 0.01), effect, Double.NaN, Double.NaN,
                    sims);
            add(out, "DoubleML (interactive, ridge)", DoubleMl.interactive(x, t, y, Regressor.ridge(1e-3),
                    Regressor.ridge(1e-3), 2, true, 0.01, seed + s).theta(), effect, Double.NaN, Double.NaN, sims);
            add(out, "1:1 NN matching", mean(matchedDifferences(x, t, y)), effect, Double.NaN, Double.NaN, sims);
            for (BalancingWeights.Kind k : BalancingWeights.Kind.values()) {
                double[] w = BalancingWeights.weights(x, t, k, 0.02);
                add(out, k.name().toLowerCase().replace('_', ' ') + " weights", BalancingWeights.att(y, t, w),
                        effect, BalancingWeights.maxSmd(x, t, w), BalancingWeights.effectiveControls(t, w), sims);
            }
        }
        out.values().forEach(v -> v[1] = Math.sqrt(v[1]));
        return out;
    }

    /**
     * Sensitivity summaries of single studies.
     *
     * @param effect the true effect
     * @param strength the hidden confounder's strength {@code s} relative to {@code z2}
     * @param n units
     * @param studies studies averaged
     * @param seed the seed
     * @return {@code {OLS estimate, share of 95% CIs excluding 0, entropy-balancing estimate,
     *     E-value, RV_1, adjusted estimate at k = 1, k*, critical Gamma}}, averaged over studies
     */
    public static double[] sensitivity(double effect, double strength, int n, int studies, long seed) {
        double[] acc = new double[8];
        for (int s = 0; s < studies; s++) {
            PanelStudies.Observational o = PanelStudies.observational(n, false, 1, effect, 0.5 * strength,
                    13.7 * strength, seed + s);
            double[][] x = o.x();
            int[] t = o.t();
            double[] y = o.y();
            double[] td = new double[n];
            for (int i = 0; i < n; i++) {
                td[i] = t[i];
            }
            SensitivityAnalysis.Regression reg = SensitivityAnalysis.regression(x, td, y);
            double[] r2 = SensitivityAnalysis.covariateR2(x, td, y, 1);
            double adjusted = reg.benchmark(r2[0], r2[1], 1)[2];
            double lo = 0;
            double hi = 1;
            while (reg.benchmark(r2[0], r2[1], hi)[2] > 0 && hi < 1e3) {
                lo = hi;
                hi *= 2;
            }
            for (int it = 0; it < 50; it++) {
                double mid = (lo + hi) / 2;
                if (reg.benchmark(r2[0], r2[1], mid)[2] > 0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            double my = mean(y);
            double sy = 0;
            for (double v : y) {
                sy += (v - my) * (v - my) / n;
            }
            double eb = BalancingWeights.att(y, t,
                    BalancingWeights.weights(x, t, BalancingWeights.Kind.ENTROPY_BALANCING, 0));
            double[] diffs = matchedDifferences(x, t, y);
            double[] row = {reg.estimate(), Math.abs(reg.estimate()) > 1.96 * reg.se() ? 1 : 0, eb,
                SensitivityAnalysis.eValueOfDifference(reg.estimate() / Math.sqrt(sy)), reg.robustness(), adjusted,
                (lo + hi) / 2, SensitivityAnalysis.criticalGamma(diffs, 0.05)};
            for (int k = 0; k < row.length; k++) {
                acc[k] += row[k] / studies;
            }
        }
        return acc;
    }
}
