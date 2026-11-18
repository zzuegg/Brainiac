package io.github.zzuegg.brainiac.experimental.causal;

/**
 * Statistics for online A/B tests.
 * <ul>
 *   <li>{@link #tTest}: the fixed-horizon two-sample (Welch) z-test on the difference in means;
 *       valid only when looked at once, at the planned sample size.</li>
 *   <li>{@link #cuped} (Deng et al., WSDM 2013, <a href="https://doi.org/10.1145/2433396.2433413">
 *       doi:10.1145/2433396.2433413</a>): subtract {@code theta (X - mean X)} from each user's
 *       metric, with a pre-experiment covariate {@code X} and
 *       {@code theta = cov(Y, X) / var(X)} pooled over both arms. The difference in means stays
 *       unbiased (X is independent of the assignment) and its variance falls by {@code 1 - rho^2}.
 *       {@link #cupac} is the same adjustment with an ML prediction of {@code Y} as the
 *       covariate (CUPAC, DoorDash 2020).</li>
 *   <li>{@link Msprt} (Johari et al. 2017, <a href="https://arxiv.org/abs/1512.04922">
 *       arXiv:1512.04922</a>): the mixture sequential probability ratio test with a normal
 *       mixing distribution {@code N(0, tau^2)} over the effect, whose running p-value
 *       {@code p_n = min(p_{n-1}, 1 / Lambda_n)} is <em>always valid</em>: it can be monitored after
 *       every user and the test stopped whenever {@code p_n < alpha}, keeping the false-positive
 *       rate at most {@code alpha}.</li>
 * </ul>
 */
public final class OnlineExperiments {

    private OnlineExperiments() {
    }

    /**
     * A difference-in-means test result.
     *
     * @param effect treated mean minus control mean
     * @param se its standard error
     * @param pValue the two-sided normal p-value
     */
    public record Test(double effect, double se, double pValue) {
    }

    /**
     * The two-sample z-test with unpooled variances, over the first {@code n} users.
     *
     * @param y metrics
     * @param t arms
     * @param n users to use (a prefix, for peeking)
     * @return the test
     */
    public static Test tTest(double[] y, int[] t, int n) {
        double[] s = new double[2];
        double[] ss = new double[2];
        int[] c = new int[2];
        for (int i = 0; i < n; i++) {
            s[t[i]] += y[i];
            ss[t[i]] += y[i] * y[i];
            c[t[i]]++;
        }
        double m1 = s[1] / c[1];
        double m0 = s[0] / c[0];
        double v1 = (ss[1] - c[1] * m1 * m1) / (c[1] - 1);
        double v0 = (ss[0] - c[0] * m0 * m0) / (c[0] - 1);
        double se = Math.sqrt(v1 / c[1] + v0 / c[0]);
        double z = (m1 - m0) / se;
        return new Test(m1 - m0, se, 2 * normalTail(Math.abs(z)));
    }

    /**
     * CUPED-adjusted metrics over the first {@code n} users.
     *
     * @param y metrics
     * @param covariate the pre-experiment covariate
     * @param n users to use
     * @return {@code y - theta (x - mean x)} for those users (the rest copied unchanged)
     */
    public static double[] cuped(double[] y, double[] covariate, int n) {
        double mx = 0;
        double my = 0;
        for (int i = 0; i < n; i++) {
            mx += covariate[i] / n;
            my += y[i] / n;
        }
        double cov = 0;
        double var = 0;
        for (int i = 0; i < n; i++) {
            cov += (y[i] - my) * (covariate[i] - mx);
            var += (covariate[i] - mx) * (covariate[i] - mx);
        }
        double theta = cov / var;
        double[] out = y.clone();
        for (int i = 0; i < n; i++) {
            out[i] = y[i] - theta * (covariate[i] - mx);
        }
        return out;
    }

    /**
     * CUPAC: CUPED with an ML prediction of the metric from pre-experiment features as the
     * covariate; the predictor must be trained on data that excludes the experiment.
     *
     * @param y metrics
     * @param features pre-experiment features per user
     * @param predictor a model fitted on pre-period data
     * @param n users to use
     * @return the adjusted metrics
     */
    public static double[] cupac(double[] y, double[][] features, Regressor.Fitted predictor, int n) {
        return cuped(y, predictor.predict(features), n);
    }

    /**
     * The fraction of variance CUPED removes, {@code rho^2}.
     *
     * @param y metrics
     * @param covariate the covariate
     * @return the squared correlation
     */
    public static double varianceReduction(double[] y, double[] covariate) {
        int n = y.length;
        double mx = 0;
        double my = 0;
        for (int i = 0; i < n; i++) {
            mx += covariate[i] / n;
            my += y[i] / n;
        }
        double cov = 0;
        double vx = 0;
        double vy = 0;
        for (int i = 0; i < n; i++) {
            cov += (y[i] - my) * (covariate[i] - mx);
            vx += (covariate[i] - mx) * (covariate[i] - mx);
            vy += (y[i] - my) * (y[i] - my);
        }
        return cov * cov / (vx * vy);
    }

    /**
     * A streaming two-sample mixture SPRT for a difference in means. Observations arrive one user
     * at a time; the running estimate uses {@code delta_n = mean_T - mean_C}, whose variance is
     * {@code sigma^2 (1 / n_T + 1 / n_C)}, and the normal-mixture likelihood ratio
     * {@code Lambda = sqrt(V / (V + tau^2)) exp(tau^2 delta^2 / (2 V (V + tau^2)))}. The variance
     * {@code sigma^2} is the pooled running variance (a plug-in, as in practice).
     */
    public static final class Msprt {
        private final double tau2;
        private final int burnIn;
        private final double[] sum = new double[2];
        private final double[] sumSq = new double[2];
        private final long[] count = new long[2];
        private double p = 1;

        /**
         * Creates the test.
         *
         * @param tau the standard deviation of the mixing distribution over the effect, in metric
         *     units (match it to the effects of interest)
         * @param burnIn users per arm before the variance estimate is trusted
         */
        public Msprt(double tau, int burnIn) {
            this.tau2 = tau * tau;
            this.burnIn = burnIn;
        }

        /**
         * Adds one user.
         *
         * @param value the metric
         * @param arm 0 control, 1 treatment
         * @return the always-valid p-value so far
         */
        public double add(double value, int arm) {
            sum[arm] += value;
            sumSq[arm] += value * value;
            count[arm]++;
            if (count[0] < burnIn || count[1] < burnIn) {
                return p;
            }
            double m1 = sum[1] / count[1];
            double m0 = sum[0] / count[0];
            double var = (sumSq[1] - count[1] * m1 * m1 + sumSq[0] - count[0] * m0 * m0)
                    / (count[0] + count[1] - 2);
            double v = var * (1.0 / count[1] + 1.0 / count[0]);
            double delta = m1 - m0;
            double logLambda = 0.5 * Math.log(v / (v + tau2)) + tau2 * delta * delta / (2 * v * (v + tau2));
            p = Math.min(p, Math.exp(-logLambda));
            return p;
        }

        /** @return the always-valid p-value so far */
        public double pValue() {
            return p;
        }
    }

    /**
     * The upper tail of the standard normal, {@code P(Z > z)}.
     *
     * @param z the point
     * @return the tail probability
     */
    public static double normalTail(double z) {
        return 0.5 * erfc(z / Math.sqrt(2));
    }

    private static double erfc(double x) {
        double t = 1 / (1 + 0.5 * Math.abs(x));
        double y = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? y : 2 - y;
    }
}
