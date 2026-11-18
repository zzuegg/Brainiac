package io.github.zzuegg.brainiac.experimental.stats;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Predictive model checking and averaging from posterior draws.
 * <ul>
 *   <li>{@link #psis}: Pareto-smoothed importance sampling (Vehtari, Gelman and Gabry, Stat.
 *       Comput. 2017, <a href="https://arxiv.org/abs/1507.04544">arXiv:1507.04544</a>;
 *       Vehtari et al., JMLR 2024): the largest {@code M = min(S/5, 3 sqrt(S))} importance
 *       ratios are replaced by expected order statistics of a generalised Pareto fitted to them
 *       (the GPD fit is {@link ExtremeValues#fitGpd}) and capped at the largest raw ratio; the
 *       fitted shape {@code k-hat} diagnoses reliability ({@code > 0.7}: do not trust);</li>
 *   <li>{@link #psisLoo} and {@link #isLoo}: leave-one-out predictive density with ratios
 *       {@code 1 / p(y_i | theta_s)}, Pareto-smoothed or raw;</li>
 *   <li>{@link #waic}: Watanabe's widely applicable information criterion (JMLR 2010);</li>
 *   <li>{@link #stackingWeights}: Bayesian stacking (Yao et al., Bayesian Analysis 2018,
 *       <a href="https://arxiv.org/abs/1704.02030">arXiv:1704.02030</a>): simplex weights
 *       maximising the LOO log score of the mixture {@code sum_i log sum_k w_k p_k(y_i |
 *       y_-i)}, a concave problem solved by EM-style fixed-point updates;</li>
 *   <li>{@link #pseudoBmaPlus}: elpd-based weights stabilised by the Bayesian bootstrap;
 *       {@link #bmaWeights}: classical Bayesian model averaging from log evidences.</li>
 * </ul>
 */
public final class ModelComparison {

    private ModelComparison() {
    }

    /**
     * Smoothed log weights and the Pareto diagnostic.
     *
     * @param logWeights smoothed, normalised log weights
     * @param khat the fitted GPD shape of the tail
     */
    public record Smoothed(double[] logWeights, double khat) {
    }

    /**
     * A leave-one-out estimate.
     *
     * @param elpd the expected log predictive density summed over points
     * @param pointwise per-point LOO log predictive densities
     * @param khat per-point Pareto diagnostics (zeros for plain IS)
     * @param se the standard error of {@code elpd}
     */
    public record Loo(double elpd, double[] pointwise, double[] khat, double se) {
    }

    private static double logSumExp(double[] v) {
        double m = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            m = Math.max(m, x);
        }
        double s = 0;
        for (double x : v) {
            s += Math.exp(x - m);
        }
        return m + Math.log(s);
    }

    /**
     * Pareto-smoothed importance sampling of log ratios.
     *
     * @param logRatios unnormalised log importance ratios, one per draw
     * @return smoothed normalised log weights and {@code k-hat}
     */
    public static Smoothed psis(double[] logRatios) {
        int s = logRatios.length;
        int m = (int) Math.ceil(Math.min(s / 5.0, 3 * Math.sqrt(s)));
        double max = Arrays.stream(logRatios).max().orElse(0);
        Integer[] order = new Integer[s];
        for (int i = 0; i < s; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(logRatios[a], logRatios[b]));
        double[] lw = logRatios.clone();
        double khat = Double.NaN;
        if (m >= 5) {
            double cut = Math.exp(logRatios[order[s - m - 1]] - max);
            double[] y = new double[m];
            for (int i = 0; i < m; i++) {
                y[i] = Math.exp(logRatios[order[s - m + i]] - max) - cut;
            }
            if (y[m - 1] > 0) {
                ExtremeValues.Gpd g = ExtremeValues.fitGpd(Arrays.stream(y).map(v -> Math.max(v, 1e-300)).toArray());
                // Vehtari et al. regularise k-hat towards 0.5 with a weak prior of 10 pseudo-points
                khat = (m * g.shape() + 10 * 0.5) / (m + 10);
                ExtremeValues.Gpd smooth = new ExtremeValues.Gpd(khat, g.scale());
                for (int i = 0; i < m; i++) {
                    double v = cut + smooth.quantile((i + 0.5) / m);
                    lw[order[s - m + i]] = Math.min(Math.log(Math.max(v, 1e-300)), 0) + max;
                }
            }
        }
        double z = logSumExp(lw);
        for (int i = 0; i < s; i++) {
            lw[i] -= z;
        }
        return new Smoothed(lw, khat);
    }

    private static Loo loo(double[][] logLik, boolean smooth) {
        int s = logLik.length;
        int n = logLik[0].length;
        double[] point = new double[n];
        double[] k = new double[n];
        for (int i = 0; i < n; i++) {
            double[] lr = new double[s];
            double[] ll = new double[s];
            for (int d = 0; d < s; d++) {
                ll[d] = logLik[d][i];
                lr[d] = -ll[d];
            }
            double[] lw;
            if (smooth) {
                Smoothed ps = psis(lr);
                lw = ps.logWeights();
                k[i] = ps.khat();
            } else {
                double z = logSumExp(lr);
                lw = new double[s];
                for (int d = 0; d < s; d++) {
                    lw[d] = lr[d] - z;
                }
            }
            double[] t = new double[s];
            for (int d = 0; d < s; d++) {
                t[d] = lw[d] + ll[d];
            }
            point[i] = logSumExp(t);
        }
        return summarise(point, k);
    }

    private static Loo summarise(double[] point, double[] k) {
        int n = point.length;
        double sum = 0;
        for (double v : point) {
            sum += v;
        }
        double mean = sum / n;
        double var = 0;
        for (double v : point) {
            var += (v - mean) * (v - mean) / (n - 1);
        }
        return new Loo(sum, point, k, Math.sqrt(n * var));
    }

    /**
     * PSIS-LOO.
     *
     * @param logLik pointwise log likelihoods {@code [draws][points]}
     * @return the estimate with per-point {@code k-hat}
     */
    public static Loo psisLoo(double[][] logLik) {
        return loo(logLik, true);
    }

    /**
     * Plain importance-sampling LOO (no smoothing).
     *
     * @param logLik pointwise log likelihoods {@code [draws][points]}
     * @return the estimate
     */
    public static Loo isLoo(double[][] logLik) {
        return loo(logLik, false);
    }

    /**
     * WAIC on the elpd scale: {@code lppd - p_waic}.
     *
     * @param logLik pointwise log likelihoods {@code [draws][points]}
     * @return {@code {elpd_waic, p_waic}}
     */
    public static double[] waic(double[][] logLik) {
        int s = logLik.length;
        int n = logLik[0].length;
        double lppd = 0;
        double p = 0;
        for (int i = 0; i < n; i++) {
            double[] ll = new double[s];
            double m = 0;
            for (int d = 0; d < s; d++) {
                ll[d] = logLik[d][i];
                m += ll[d] / s;
            }
            lppd += logSumExp(ll) - Math.log(s);
            double v = 0;
            for (double x : ll) {
                v += (x - m) * (x - m) / (s - 1);
            }
            p += v;
        }
        return new double[] {lppd - p, p};
    }

    /**
     * Stacking weights from pointwise LOO log predictive densities.
     *
     * @param pointwise {@code [models][points]}
     * @return simplex weights
     */
    public static double[] stackingWeights(double[][] pointwise) {
        int k = pointwise.length;
        int n = pointwise[0].length;
        double[] w = new double[k];
        Arrays.fill(w, 1.0 / k);
        double[] shift = new double[n];
        for (int i = 0; i < n; i++) {
            double m = Double.NEGATIVE_INFINITY;
            for (double[] row : pointwise) {
                m = Math.max(m, row[i]);
            }
            shift[i] = m;
        }
        for (int it = 0; it < 2000; it++) {
            double[] next = new double[k];
            for (int i = 0; i < n; i++) {
                double tot = 0;
                double[] p = new double[k];
                for (int c = 0; c < k; c++) {
                    p[c] = w[c] * Math.exp(pointwise[c][i] - shift[i]);
                    tot += p[c];
                }
                for (int c = 0; c < k; c++) {
                    next[c] += p[c] / tot / n;
                }
            }
            double change = 0;
            for (int c = 0; c < k; c++) {
                change += Math.abs(next[c] - w[c]);
            }
            w = next;
            if (change < 1e-10) {
                break;
            }
        }
        return w;
    }

    /**
     * The LOO log score of a weighted mixture.
     *
     * @param pointwise {@code [models][points]} log predictive densities
     * @param weights the weights
     * @return {@code sum_i log sum_k w_k exp(pointwise[k][i])}
     */
    public static double mixtureLogScore(double[][] pointwise, double[] weights) {
        double s = 0;
        for (int i = 0; i < pointwise[0].length; i++) {
            double[] t = new double[weights.length];
            for (int c = 0; c < weights.length; c++) {
                t[c] = Math.log(Math.max(weights[c], 1e-300)) + pointwise[c][i];
            }
            s += logSumExp(t);
        }
        return s;
    }

    /**
     * Pseudo-BMA+ weights: {@code exp(elpd_k)} normalised, averaged over Bayesian-bootstrap
     * (Dirichlet(1)) reweightings of the points.
     *
     * @param pointwise {@code [models][points]}
     * @param bootstraps bootstrap replicates
     * @param seed the seed
     * @return simplex weights
     */
    public static double[] pseudoBmaPlus(double[][] pointwise, int bootstraps, long seed) {
        int k = pointwise.length;
        int n = pointwise[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[k];
        for (int b = 0; b < bootstraps; b++) {
            double[] a = new double[n];
            double tot = 0;
            for (int i = 0; i < n; i++) {
                a[i] = -Math.log(1 - r.nextDouble());
                tot += a[i];
            }
            double[] e = new double[k];
            for (int c = 0; c < k; c++) {
                for (int i = 0; i < n; i++) {
                    e[c] += n * a[i] / tot * pointwise[c][i];
                }
            }
            double z = logSumExp(e);
            for (int c = 0; c < k; c++) {
                w[c] += Math.exp(e[c] - z) / bootstraps;
            }
        }
        return w;
    }

    /**
     * Bayesian model averaging weights from log evidences (equal model priors).
     *
     * @param logEvidences {@code log p(y | M_k)}
     * @return simplex weights
     */
    public static double[] bmaWeights(double[] logEvidences) {
        double z = logSumExp(logEvidences);
        double[] w = new double[logEvidences.length];
        for (int c = 0; c < w.length; c++) {
            w[c] = Math.exp(logEvidences[c] - z);
        }
        return w;
    }
}
