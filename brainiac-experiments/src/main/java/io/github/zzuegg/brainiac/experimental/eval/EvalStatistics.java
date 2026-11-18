package io.github.zzuegg.brainiac.experimental.eval;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Error bars for evaluations, after Miller ("Adding Error Bars to Evals", 2024,
 * <a href="https://arxiv.org/abs/2411.00640">arXiv:2411.00640</a>) and Dror et al. ("The
 * Hitchhiker's Guide to Testing Statistical Significance in NLP", ACL 2018,
 * <a href="https://aclanthology.org/P18-1128/">P18-1128</a>). Scores are per question (0/1
 * correctness or a continuous score); questions may share a cluster (a passage, a context).
 * <ul>
 *   <li>{@link #iid}: the central-limit standard error {@code s / √n};</li>
 *   <li>{@link #clustered}: the cluster-robust standard error
 *       {@code √(Σ_c (Σ_{i∈c} (x_i - x̄))²) / n}, which can be several times wider;</li>
 *   <li>{@link #paired} and {@link #unpaired}: comparing two models on the same questions
 *       by the per-question difference, against treating the two means as independent;</li>
 *   <li>{@link #mcNemar}: the exact McNemar test on discordant pairs of 0/1 outcomes;</li>
 *   <li>{@link #clusterBootstrap}: a percentile bootstrap that resamples whole clusters;</li>
 *   <li>{@link #questionsNeeded}: power analysis, the questions a difference needs.</li>
 * </ul>
 */
public final class EvalStatistics {

    private EvalStatistics() {
    }

    /**
     * A mean with its standard error.
     *
     * @param mean the mean
     * @param se the standard error
     */
    public record Interval(double mean, double se) {
        /**
         * @param z the normal quantile, 1.96 for 95%
         * @return the lower end
         */
        public double low(double z) {
            return mean - z * se;
        }

        /**
         * @param z the normal quantile
         * @return the upper end
         */
        public double high(double z) {
            return mean + z * se;
        }

        /**
         * @param truth the true value
         * @param z the normal quantile
         * @return whether the interval covers it
         */
        public boolean covers(double truth, double z) {
            return low(z) <= truth && truth <= high(z);
        }

        /** @return the two-sided p-value of "the mean is zero" */
        public double pValue() {
            return se == 0 ? (mean == 0 ? 1 : 0) : 2 * (1 - normalCdf(Math.abs(mean / se)));
        }
    }

    private static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    /**
     * The i.i.d. standard error.
     *
     * @param x per-question scores
     * @return mean and {@code s / √n}
     */
    public static Interval iid(double[] x) {
        double m = mean(x);
        double s = 0;
        for (double v : x) {
            s += (v - m) * (v - m);
        }
        return new Interval(m, Math.sqrt(s / Math.max(1, x.length - 1) / x.length));
    }

    /**
     * The cluster-robust standard error.
     *
     * @param x per-question scores
     * @param cluster the cluster of each question
     * @return mean and the clustered standard error
     */
    public static Interval clustered(double[] x, int[] cluster) {
        double m = mean(x);
        Map<Integer, Double> sums = new LinkedHashMap<>();
        for (int i = 0; i < x.length; i++) {
            sums.merge(cluster[i], x[i] - m, Double::sum);
        }
        double s = 0;
        for (double v : sums.values()) {
            s += v * v;
        }
        int c = sums.size();
        double correction = c > 1 ? c / (c - 1.0) : 1;
        return new Interval(m, Math.sqrt(correction * s) / x.length);
    }

    /**
     * The paired difference {@code a - b} on the same questions, optionally clustered.
     *
     * @param a model A's per-question scores
     * @param b model B's per-question scores on the same questions
     * @param cluster clusters, or null for i.i.d. questions
     * @return the mean difference and its standard error
     */
    public static Interval paired(double[] a, double[] b, int[] cluster) {
        double[] d = new double[a.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = a[i] - b[i];
        }
        return cluster == null ? iid(d) : clustered(d, cluster);
    }

    /**
     * The unpaired difference: {@code SE = √(SE_a² + SE_b²)}.
     *
     * @param a model A's scores
     * @param b model B's scores
     * @param cluster clusters, or null
     * @return the mean difference and its standard error
     */
    public static Interval unpaired(double[] a, double[] b, int[] cluster) {
        Interval ia = cluster == null ? iid(a) : clustered(a, cluster);
        Interval ib = cluster == null ? iid(b) : clustered(b, cluster);
        return new Interval(ia.mean() - ib.mean(), Math.sqrt(ia.se() * ia.se() + ib.se() * ib.se()));
    }

    /**
     * The exact two-sided McNemar test: under the null, each discordant pair favours A with
     * probability 1/2.
     *
     * @param a model A's 0/1 correctness
     * @param b model B's 0/1 correctness on the same questions
     * @return the p-value
     */
    public static double mcNemar(double[] a, double[] b) {
        int onlyA = 0;
        int onlyB = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] > 0.5 && b[i] <= 0.5) {
                onlyA++;
            } else if (b[i] > 0.5 && a[i] <= 0.5) {
                onlyB++;
            }
        }
        int n = onlyA + onlyB;
        if (n == 0) {
            return 1;
        }
        int k = Math.min(onlyA, onlyB);
        double logHalf = n * Math.log(0.5);
        double tail = 0;
        double logC = 0;
        for (int i = 0; i <= k; i++) {
            if (i > 0) {
                logC += Math.log(n - i + 1) - Math.log(i);
            }
            tail += Math.exp(logC + logHalf);
        }
        return Math.min(1, 2 * tail);
    }

    /**
     * A percentile bootstrap interval of the mean that resamples whole clusters.
     *
     * @param x per-question scores
     * @param cluster clusters (each question its own cluster for an i.i.d. bootstrap)
     * @param resamples bootstrap resamples
     * @param level the coverage, for example 0.95
     * @param seed the seed
     * @return {@code {low, high}}
     */
    public static double[] clusterBootstrap(double[] x, int[] cluster, int resamples, double level, long seed) {
        Map<Integer, double[]> byCluster = new LinkedHashMap<>();
        for (int i = 0; i < x.length; i++) {
            double[] s = byCluster.computeIfAbsent(cluster[i], k -> new double[2]);
            s[0] += x[i];
            s[1]++;
        }
        double[][] cs = byCluster.values().toArray(new double[0][]);
        SplittableRandom r = new SplittableRandom(seed);
        double[] means = new double[resamples];
        for (int k = 0; k < resamples; k++) {
            double sum = 0;
            double n = 0;
            for (int j = 0; j < cs.length; j++) {
                double[] c = cs[r.nextInt(cs.length)];
                sum += c[0];
                n += c[1];
            }
            means[k] = sum / n;
        }
        Arrays.sort(means);
        double tail = (1 - level) / 2;
        return new double[] {means[(int) Math.floor(tail * (resamples - 1))],
            means[(int) Math.ceil((1 - tail) * (resamples - 1))]};
    }

    /**
     * Questions needed to detect a difference with a two-sided z-test:
     * {@code n = (z_{α/2} + z_β)² σ² / δ²}, with {@code σ²} the per-question variance of the
     * difference (paired) or the sum of both models' variances (unpaired), times a design
     * effect for clustering.
     *
     * @param delta the difference to detect
     * @param varianceOfDifference the per-question variance of the compared quantity
     * @param alpha the significance level, for example 0.05
     * @param power the power, for example 0.8
     * @return questions needed, rounded up
     */
    public static long questionsNeeded(double delta, double varianceOfDifference, double alpha, double power) {
        double z = normalQuantile(1 - alpha / 2) + normalQuantile(power);
        return (long) Math.ceil(z * z * varianceOfDifference / (delta * delta));
    }

    /**
     * The standard normal CDF (Abramowitz–Stegun 7.1.26 via erf, error below 1.5e-7).
     *
     * @param x the point
     * @return {@code Φ(x)}
     */
    public static double normalCdf(double x) {
        double t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.sqrt(2));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592) * t * Math.exp(-x * x / 2);
        return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
    }

    /**
     * The standard normal quantile (Acklam's rational approximation, relative error 1e-9).
     *
     * @param p the probability in (0, 1)
     * @return {@code Φ⁻¹(p)}
     */
    public static double normalQuantile(double p) {
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
            3.754408661907416e+00};
        double low = 0.02425;
        if (p < low) {
            double q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        if (p > 1 - low) {
            return -normalQuantile(1 - p);
        }
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
}
