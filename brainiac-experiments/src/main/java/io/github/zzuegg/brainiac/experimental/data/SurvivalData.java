package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Synthetic time-to-event data with known hazards. Every unit has, per cause {@code c}, a Weibull
 * cause-specific hazard {@code h_c(t) = lambda_c k_c t^(k_c - 1)}; the event time is the first of
 * the latent cause times, and an independent exponential censoring time, tuned by bisection to a
 * target censoring share, may cut it short (right censoring). Because the hazards are known, the
 * true survival {@code S(t | x) = exp(-sum_c lambda_c t^k_c)} and cumulative incidence per cause
 * are available for scoring.
 * <ul>
 *   <li>{@link #weibull}: one cause, proportional hazards with a linear or a nonlinear risk
 *       {@code log lambda = f(x)} (Cox's model holds, DeepSurv's MLP is needed for the nonlinear
 *       risk);</li>
 *   <li>{@link #crossingHazards}: one subgroup fails early (decreasing hazard), the other late
 *       (increasing hazard), so the hazards cross and proportional hazards fails;</li>
 *   <li>{@link #competingRisks}: two causes driven by different covariates.</li>
 * </ul>
 */
public final class SurvivalData {

    private SurvivalData() {
    }

    /**
     * Right-censored survival data.
     *
     * @param x covariates {@code [n][d]}
     * @param time the observed time (event or censoring)
     * @param event 0 if censored, else the cause {@code 1..causes}
     * @param scale {@code [n][causes]} the Weibull rates {@code lambda_c}
     * @param shape {@code [n][causes]} the Weibull shapes {@code k_c}
     * @param risk the true log-risk {@code f(x)} of cause 1 (for proportional-hazards scenarios)
     */
    public record Survival(double[][] x, double[] time, int[] event, double[][] scale, double[][] shape,
            double[] risk) {

        /** @return the number of units */
        public int size() {
            return time.length;
        }

        /** @return the number of causes */
        public int causes() {
            return scale[0].length;
        }

        /** @return the share of censored units */
        public double censoring() {
            int c = 0;
            for (int e : event) {
                c += e == 0 ? 1 : 0;
            }
            return c / (double) event.length;
        }

        /**
         * @param i a unit
         * @param t a time
         * @return the true survival {@code S(t | x_i)}
         */
        public double survival(int i, double t) {
            double h = 0;
            for (int c = 0; c < scale[i].length; c++) {
                h += scale[i][c] * Math.pow(t, shape[i][c]);
            }
            return Math.exp(-h);
        }

        /**
         * The true cumulative incidence of a cause, by numerical integration of
         * {@code h_c(s) S(s)} over {@code [0, t]}.
         *
         * @param i a unit
         * @param cause the cause, {@code 1..causes}
         * @param t the time
         * @return {@code P(T <= t, cause)}
         */
        public double incidence(int i, int cause, double t) {
            int steps = 200;
            double dt = t / steps;
            double s = 0;
            int c = cause - 1;
            for (int k = 0; k < steps; k++) {
                double u = (k + 0.5) * dt;
                double h = scale[i][c] * shape[i][c] * Math.pow(u, shape[i][c] - 1);
                s += h * survival(i, u) * dt;
            }
            return s;
        }

        /**
         * @param from first unit
         * @param to one past the last
         * @return those units
         */
        public Survival rows(int from, int to) {
            return new Survival(java.util.Arrays.copyOfRange(x, from, to),
                    java.util.Arrays.copyOfRange(time, from, to), java.util.Arrays.copyOfRange(event, from, to),
                    java.util.Arrays.copyOfRange(scale, from, to), java.util.Arrays.copyOfRange(shape, from, to),
                    java.util.Arrays.copyOfRange(risk, from, to));
        }
    }

    /** The risk function of {@link #weibull}. */
    public enum Risk {
        /** {@code f(x) = beta . x}. */
        LINEAR,
        /** {@code f(x) = 1.5 sin(2 x0) + x1 x2 - 0.8 x3^2 + 0.5 x4}. */
        NONLINEAR
    }

    /**
     * Proportional-hazards Weibull data (shape 1.5) over {@code d} Gaussian covariates.
     *
     * @param n units
     * @param d covariates, at least 5
     * @param risk the risk function
     * @param censoring the target censoring share, in {@code [0, 0.9]}
     * @param seed the seed
     * @return the data
     */
    public static Survival weibull(int n, int d, Risk risk, double censoring, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] beta = {0.8, -0.6, 0.5, 0.3, -0.4};
        double[][] x = new double[n][d];
        double[][] scale = new double[n][1];
        double[][] shape = new double[n][1];
        double[] f = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextGaussian();
            }
            double[] xi = x[i];
            f[i] = switch (risk) {
                case LINEAR -> beta[0] * xi[0] + beta[1] * xi[1] + beta[2] * xi[2] + beta[3] * xi[3]
                        + beta[4] * xi[4];
                case NONLINEAR -> 1.5 * Math.sin(2 * xi[0]) + xi[1] * xi[2] - 0.8 * xi[3] * xi[3] + 0.5 * xi[4];
            };
            scale[i][0] = 0.1 * Math.exp(f[i]);
            shape[i][0] = 1.5;
        }
        return draw(x, scale, shape, f, censoring, r);
    }

    /**
     * Crossing hazards: units with {@code x0 > 0} have a decreasing hazard (shape 0.6, early
     * failures, but a long tail), the others an increasing hazard (shape 3, late failures); the
     * remaining covariates scale both mildly.
     *
     * @param n units
     * @param d covariates, at least 3
     * @param censoring the target censoring share
     * @param seed the seed
     * @return the data
     */
    public static Survival crossingHazards(int n, int d, double censoring, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][d];
        double[][] scale = new double[n][1];
        double[][] shape = new double[n][1];
        double[] f = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextGaussian();
            }
            boolean early = x[i][0] > 0;
            f[i] = 0.4 * x[i][1] - 0.3 * x[i][2];
            shape[i][0] = early ? 0.6 : 3;
            scale[i][0] = (early ? 0.5 : 0.02) * Math.exp(f[i]);
        }
        return draw(x, scale, shape, f, censoring, r);
    }

    /**
     * Two competing causes: cause 1 driven by {@code x0, x1} (shape 1.2), cause 2 by
     * {@code x2, x3} (shape 2).
     *
     * @param n units
     * @param d covariates, at least 4
     * @param censoring the target censoring share
     * @param seed the seed
     * @return the data
     */
    public static Survival competingRisks(int n, int d, double censoring, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][d];
        double[][] scale = new double[n][2];
        double[][] shape = new double[n][2];
        double[] f = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextGaussian();
            }
            f[i] = 0.9 * x[i][0] - 0.5 * x[i][1];
            scale[i][0] = 0.06 * Math.exp(f[i]);
            shape[i][0] = 1.2;
            scale[i][1] = 0.03 * Math.exp(0.8 * x[i][2] + 0.6 * x[i][3] * x[i][3] - 0.6);
            shape[i][1] = 2;
        }
        return draw(x, scale, shape, f, censoring, r);
    }

    private static Survival draw(double[][] x, double[][] scale, double[][] shape, double[] f, double censoring,
            SplittableRandom r) {
        int n = x.length;
        int causes = scale[0].length;
        double[] latent = new double[n];
        int[] cause = new int[n];
        double[] u = new double[n];
        for (int i = 0; i < n; i++) {
            latent[i] = Double.POSITIVE_INFINITY;
            for (int c = 0; c < causes; c++) {
                double tc = Math.pow(-Math.log(1 - r.nextDouble()) / scale[i][c], 1 / shape[i][c]);
                if (tc < latent[i]) {
                    latent[i] = tc;
                    cause[i] = c + 1;
                }
            }
            u[i] = -Math.log(1 - r.nextDouble());
        }
        double rate = 0;
        if (censoring > 0) {
            double lo = 1e-6;
            double hi = 1e3;
            for (int it = 0; it < 60; it++) {
                double mid = Math.sqrt(lo * hi);
                int censored = 0;
                for (int i = 0; i < n; i++) {
                    censored += u[i] / mid < latent[i] ? 1 : 0;
                }
                if (censored / (double) n < censoring) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            rate = Math.sqrt(lo * hi);
        }
        double[] time = new double[n];
        int[] event = new int[n];
        for (int i = 0; i < n; i++) {
            double c = rate > 0 ? u[i] / rate : Double.POSITIVE_INFINITY;
            if (c < latent[i]) {
                time[i] = c;
                event[i] = 0;
            } else {
                time[i] = latent[i];
                event[i] = cause[i];
            }
        }
        return new Survival(x, time, event, scale, shape, f);
    }
}
