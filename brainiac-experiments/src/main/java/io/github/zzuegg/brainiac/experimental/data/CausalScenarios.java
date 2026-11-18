package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Synthetic causal-inference scenarios with a known ground truth, so that every estimator can be
 * scored against the true per-unit effect rather than against held-out outcomes:
 * <ul>
 *   <li>{@link #confounded}: the confounded structural model of the wave 6 {@code CausalAndFairness}
 *       experiment ({@code tau(x) = 1 + x0}), reproduced exactly so its rows stay comparable;</li>
 *   <li>{@link #kunzel}: the simulation settings of Künzel et al. (PNAS 2019,
 *       <a href="https://arxiv.org/abs/1706.03461">arXiv:1706.03461</a>) where the meta-learners
 *       differ: unbalanced arms, a complex outcome with a simple effect, confounding, and balanced
 *       arms with a complex effect;</li>
 *   <li>{@link #nonlinearConfounding}: a constant effect under strongly nonlinear confounding, the
 *       setting where plug-in estimators carry regularisation bias (double machine learning);</li>
 *   <li>{@link #stepEffect}: the smooth-step heterogeneous effect of Wager and Athey (JASA 2018) in
 *       two of ten uniform covariates (causal forests);</li>
 *   <li>{@link #campaign}: a randomised marketing campaign over four latent customer types
 *       (persuadables, sure things, lost causes, do-not-disturbs) with a cost per contact (uplift
 *       modelling);</li>
 *   <li>{@link #airline}: the airline-demand simulation of Hartford et al. (ICML 2017,
 *       <a href="https://arxiv.org/abs/1612.09596">arXiv:1612.09596</a>): price confounded by an
 *       unobserved demand shock, fuel cost as the instrument (instrumental variables);</li>
 *   <li>{@link #abTest}: an A/B test of heavy-tailed revenue with a correlated pre-period covariate
 *       and users arriving over days (CUPED and sequential testing).</li>
 * </ul>
 */
public final class CausalScenarios {

    private CausalScenarios() {
    }

    /**
     * Observational data with a binary treatment and the true potential outcomes.
     *
     * @param x covariates {@code [n][d]}
     * @param t treatments, 0 or 1
     * @param y observed outcomes
     * @param tau the true conditional effect {@code E[y1 - y0 | x]} per unit
     * @param propensity the true {@code P(t = 1 | x)}
     * @param mu0 the true {@code E[y0 | x]}
     */
    public record Observational(double[][] x, int[] t, double[] y, double[] tau, double[] propensity,
            double[] mu0) {

        /** @return the number of units */
        public int size() {
            return t.length;
        }

        /** @return the true average treatment effect over these units */
        public double ate() {
            double s = 0;
            for (double v : tau) {
                s += v;
            }
            return s / tau.length;
        }

        /**
         * @param from first unit
         * @param to one past the last unit
         * @return those units
         */
        public Observational rows(int from, int to) {
            return new Observational(java.util.Arrays.copyOfRange(x, from, to),
                    java.util.Arrays.copyOfRange(t, from, to), java.util.Arrays.copyOfRange(y, from, to),
                    java.util.Arrays.copyOfRange(tau, from, to), java.util.Arrays.copyOfRange(propensity, from, to),
                    java.util.Arrays.copyOfRange(mu0, from, to));
        }
    }

    /**
     * The wave 6 confounded model: five Gaussian covariates,
     * {@code e(x) = sigmoid(1.2 x0 - 0.6 x1 + 0.4 x2)},
     * {@code y = 1.5 x0 + x1^2 + 0.5 x3 + t (1 + x0) + 0.5 eps}.
     *
     * @param n units
     * @param seed the seed
     * @return the data (true ATE 1)
     */
    public static Observational confounded(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][5];
        int[] t = new int[n];
        double[] y = new double[n];
        double[] tau = new double[n];
        double[] e = new double[n];
        double[] mu0 = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 5; j++) {
                x[i][j] = r.nextGaussian();
            }
            e[i] = 1 / (1 + Math.exp(-(1.2 * x[i][0] - 0.6 * x[i][1] + 0.4 * x[i][2])));
            t[i] = r.nextDouble() < e[i] ? 1 : 0;
            tau[i] = 1 + x[i][0];
            mu0[i] = 1.5 * x[i][0] + x[i][1] * x[i][1] + 0.5 * x[i][3];
            y[i] = mu0[i] + t[i] * tau[i] + 0.5 * r.nextGaussian();
        }
        return new Observational(x, t, y, tau, e, mu0);
    }

    /** The Künzel et al. simulation settings. */
    public enum Setting {
        /** 10 % treated, a piecewise-linear outcome and a step effect (the X-learner's case). */
        UNBALANCED,
        /** Balanced arms, a complex nonlinear outcome and a simple linear effect (R/DR case). */
        SIMPLE_EFFECT,
        /** Propensity driven by the same covariates as the outcome, simple effect. */
        CONFOUNDED,
        /** Balanced arms and a complex effect sharing structure with the outcome (TARNet case). */
        COMPLEX_EFFECT
    }

    /**
     * One of the Künzel et al. settings over {@code d = 10} Gaussian covariates.
     *
     * @param setting the setting
     * @param n units
     * @param seed the seed
     * @return the data
     */
    public static Observational kunzel(Setting setting, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = 10;
        double[] beta = new double[d];
        for (int j = 0; j < d; j++) {
            beta[j] = r.nextDouble() * 2 - 1;
        }
        double[][] x = new double[n][d];
        int[] t = new int[n];
        double[] y = new double[n];
        double[] tau = new double[n];
        double[] e = new double[n];
        double[] mu0 = new double[n];
        for (int i = 0; i < n; i++) {
            double[] xi = x[i];
            for (int j = 0; j < d; j++) {
                xi[j] = r.nextGaussian();
            }
            double lin = 0;
            for (int j = 0; j < d; j++) {
                lin += beta[j] * xi[j];
            }
            switch (setting) {
                case UNBALANCED -> {
                    e[i] = 0.1;
                    mu0[i] = lin + 3 * (xi[0] > 0.5 ? 1 : 0);
                    tau[i] = 2 * (xi[1] > 0.1 ? 1 : 0);
                }
                case SIMPLE_EFFECT -> {
                    e[i] = 0.5;
                    mu0[i] = 2 * Math.sin(2 * xi[0]) * xi[1] + xi[2] * xi[2] + Math.abs(xi[3]) * 1.5
                            + Math.cos(3 * xi[4]);
                    tau[i] = 1 + 0.5 * xi[0];
                }
                case CONFOUNDED -> {
                    e[i] = 0.1 + 0.8 / (1 + Math.exp(-1.5 * (xi[0] + xi[1])));
                    mu0[i] = 2 * xi[0] + 2 * xi[1] + xi[2] * xi[2] + Math.sin(2 * xi[3]);
                    tau[i] = 1 + 0.5 * xi[1];
                }
                case COMPLEX_EFFECT -> {
                    e[i] = 0.5;
                    double f = Math.sin(2 * xi[0]) * xi[1] + 0.5 * xi[2] * xi[2];
                    mu0[i] = f + lin * 0.5;
                    tau[i] = 2 * f + Math.abs(xi[3]);
                }
                default -> throw new IllegalArgumentException(setting.toString());
            }
            t[i] = r.nextDouble() < e[i] ? 1 : 0;
            y[i] = mu0[i] + t[i] * tau[i] + r.nextGaussian();
        }
        return new Observational(x, t, y, tau, e, mu0);
    }

    /**
     * A constant effect of 1 under nonlinear confounding:
     * {@code e(x) = sigmoid(sin(2 x0) + x1 x2 - 0.5 |x3|)} and
     * {@code y = t + 2 sin(2 x0) + x1 x2 + |x3| + x4^2 / 2 + eps}, over 6 Gaussian covariates.
     *
     * @param n units
     * @param seed the seed
     * @return the data (true ATE 1)
     */
    public static Observational nonlinearConfounding(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = 6;
        double[][] x = new double[n][d];
        int[] t = new int[n];
        double[] y = new double[n];
        double[] tau = new double[n];
        double[] e = new double[n];
        double[] mu0 = new double[n];
        for (int i = 0; i < n; i++) {
            double[] xi = x[i];
            for (int j = 0; j < d; j++) {
                xi[j] = r.nextGaussian();
            }
            double g = 2 * Math.sin(2 * xi[0]) + xi[1] * xi[2];
            e[i] = Math.clamp(1 / (1 + Math.exp(-(g - 0.5 * Math.abs(xi[3])))), 0.05, 0.95);
            t[i] = r.nextDouble() < e[i] ? 1 : 0;
            tau[i] = 1;
            mu0[i] = g + Math.abs(xi[3]) + 0.5 * xi[4] * xi[4];
            y[i] = mu0[i] + t[i] + r.nextGaussian();
        }
        return new Observational(x, t, y, tau, e, mu0);
    }

    /**
     * Wager and Athey's step effect over 10 uniform covariates:
     * {@code tau(x) = s(x0) s(x1)} with {@code s(u) = 1 + 1 / (1 + exp(-20 (u - 1/3)))},
     * {@code e(x) = (1 + beta(x2; 2, 4)) / 4} (mild confounding) and
     * {@code y = 2 x2 - 1 + x3 + (t - 1/2) tau(x) + eps}.
     *
     * @param n units
     * @param seed the seed
     * @return the data
     */
    public static Observational stepEffect(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = 10;
        double[][] x = new double[n][d];
        int[] t = new int[n];
        double[] y = new double[n];
        double[] tau = new double[n];
        double[] e = new double[n];
        double[] mu0 = new double[n];
        for (int i = 0; i < n; i++) {
            double[] xi = x[i];
            for (int j = 0; j < d; j++) {
                xi[j] = r.nextDouble();
            }
            tau[i] = step(xi[0]) * step(xi[1]);
            double u = xi[2];
            double betaDensity = 20 * u * Math.pow(1 - u, 3);
            e[i] = (1 + betaDensity) / 4;
            t[i] = r.nextDouble() < e[i] ? 1 : 0;
            double base = 2 * u - 1 + xi[3];
            mu0[i] = base - 0.5 * tau[i];
            y[i] = base + (t[i] - 0.5) * tau[i] + r.nextGaussian();
        }
        return new Observational(x, t, y, tau, e, mu0);
    }

    private static double step(double u) {
        return 1 + 1 / (1 + Math.exp(-20 * (u - 1.0 / 3)));
    }

    /**
     * A randomised campaign.
     *
     * @param x covariates {@code [n][d]}
     * @param t 1 if contacted (randomised 50/50)
     * @param y 1 if the customer bought
     * @param type the latent type: 0 persuadable, 1 sure thing, 2 lost cause, 3 do-not-disturb
     * @param uplift the true {@code P(buy | contacted, x) - P(buy | not contacted, x)}
     * @param response the true {@code P(buy | not contacted, x)}
     */
    public record Campaign(double[][] x, int[] t, int[] y, int[] type, double[] uplift, double[] response) {

        /** @return the number of customers */
        public int size() {
            return t.length;
        }
    }

    /**
     * A campaign over four latent customer types whose shares depend on six Gaussian covariates
     * through a softmax: sure things are driven by {@code x0} (they look like the best buyers),
     * persuadables by {@code x1 - x0 / 2}, do-not-disturbs by {@code x2 + x0 / 2}, lost causes are
     * the rest. A persuadable buys only when contacted, a do-not-disturb only when not; each type
     * also buys at random with probability 0.05.
     *
     * @param n customers
     * @param seed the seed
     * @return the campaign
     */
    public static Campaign campaign(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = 6;
        double[][] x = new double[n][d];
        int[] t = new int[n];
        int[] y = new int[n];
        int[] type = new int[n];
        double[] uplift = new double[n];
        double[] response = new double[n];
        double noise = 0.05;
        for (int i = 0; i < n; i++) {
            double[] xi = x[i];
            for (int j = 0; j < d; j++) {
                xi[j] = r.nextGaussian();
            }
            double[] logit = {1.5 * (xi[1] - 0.5 * xi[0]) - 0.5, 1.8 * xi[0] - 0.3, 0.5,
                1.5 * (xi[2] + 0.5 * xi[0]) - 1.0};
            double[] p = softmax(logit);
            double c = r.nextDouble();
            int k = 0;
            double acc = p[0];
            while (c > acc && k < 3) {
                acc += p[++k];
            }
            type[i] = k;
            t[i] = r.nextBoolean() ? 1 : 0;
            boolean buys = switch (k) {
                case 0 -> t[i] == 1;
                case 1 -> true;
                case 2 -> false;
                default -> t[i] == 0;
            };
            if (r.nextDouble() < noise) {
                buys = r.nextBoolean();
            }
            y[i] = buys ? 1 : 0;
            double treated = (1 - noise) * (p[0] + p[1]) + noise / 2;
            double control = (1 - noise) * (p[1] + p[3]) + noise / 2;
            uplift[i] = treated - control;
            response[i] = control;
        }
        return new Campaign(x, t, y, type, uplift, response);
    }

    private static double[] softmax(double[] logit) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logit) {
            max = Math.max(max, v);
        }
        double z = 0;
        double[] p = new double[logit.length];
        for (int k = 0; k < p.length; k++) {
            p[k] = Math.exp(logit[k] - max);
            z += p[k];
        }
        for (int k = 0; k < p.length; k++) {
            p[k] /= z;
        }
        return p;
    }

    /**
     * Instrumental-variable data.
     *
     * @param x observed covariates {@code [n][2]}: time of year {@code t in [0, 10]} and customer
     *     type {@code s in 1..7}
     * @param z the instrument (fuel cost)
     * @param price the treatment
     * @param sales the outcome
     */
    public record IvData(double[][] x, double[] z, double[] price, double[] sales) {

        /** @return the number of rows */
        public int size() {
            return z.length;
        }
    }

    /**
     * Hartford et al.'s demand simulation with confounding strength {@code rho}:
     * {@code p = 25 + (z + 3) psi(t) + v} and {@code y = h(p, t, s) + e} with
     * {@code e ~ N(rho v, 1 - rho^2)}, where {@code v} is the unobserved demand shock.
     *
     * @param n rows
     * @param rho the correlation between the price and outcome noise, such as 0.5
     * @param seed the seed
     * @return the data
     */
    public static IvData airline(int n, double rho, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][2];
        double[] z = new double[n];
        double[] p = new double[n];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double time = r.nextDouble() * 10;
            int s = 1 + r.nextInt(7);
            x[i][0] = time;
            x[i][1] = s;
            z[i] = r.nextGaussian();
            double v = r.nextGaussian();
            double noise = rho * v + Math.sqrt(1 - rho * rho) * r.nextGaussian();
            p[i] = 25 + (z[i] + 3) * psi(time) + v;
            y[i] = demand(p[i], time, s) + noise;
        }
        return new IvData(x, z, p, y);
    }

    /**
     * The true structural demand {@code h(p, t, s) = 100 + (10 + p) s psi(t) - 2 p}, the target of
     * counterfactual price interventions.
     *
     * @param price the price
     * @param time the time of year
     * @param type the customer type
     * @return expected sales
     */
    public static double demand(double price, double time, double type) {
        return 100 + (10 + price) * type * psi(time) - 2 * price;
    }

    /**
     * Hartford et al.'s seasonal curve
     * {@code psi(t) = 2 ((t - 5)^4 / 600 + exp(-4 (t - 5)^2) + t / 10 - 2)}.
     *
     * @param time in {@code [0, 10]}
     * @return its value
     */
    public static double psi(double time) {
        double c = time - 5;
        return 2 * (c * c * c * c / 600 + Math.exp(-4 * c * c) + time / 10 - 2);
    }

    /**
     * An A/B test.
     *
     * @param pre each user's pre-period metric (the CUPED covariate)
     * @param y the in-experiment metric
     * @param t 1 for the treatment arm
     * @param day the day each user entered the test
     */
    public record AbTest(double[] pre, double[] y, int[] t, int[] day) {

        /** @return the number of users */
        public int size() {
            return t.length;
        }
    }

    /**
     * Heavy-tailed revenue with a pre-period covariate. The pre-period metric is lognormal; the
     * in-experiment metric is {@code 10 + rho * a * z(pre) + sqrt(1 - rho^2) * a * eta} with a
     * standardised lognormal {@code eta}, so that its correlation with the covariate is
     * {@code rho}; the treatment multiplies it by {@code 1 + lift}. Users are split 50/50 and
     * spread evenly over {@code days}.
     *
     * @param n users
     * @param rho the pre-period correlation
     * @param lift the true relative lift, such as 0.02
     * @param days the days users arrive over
     * @param seed the seed
     * @return the test
     */
    public static AbTest abTest(int n, double rho, double lift, int days, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double sigma = 0.8;
        double lnMean = Math.exp(sigma * sigma / 2);
        double lnSd = Math.sqrt((Math.exp(sigma * sigma) - 1) * Math.exp(sigma * sigma));
        double scale = 6;
        double[] pre = new double[n];
        double[] y = new double[n];
        int[] t = new int[n];
        int[] day = new int[n];
        for (int i = 0; i < n; i++) {
            double xi = Math.exp(sigma * r.nextGaussian());
            double eta = (Math.exp(sigma * r.nextGaussian()) - lnMean) / lnSd;
            pre[i] = scale * xi;
            double y0 = 10 + scale * (rho * (xi - lnMean) / lnSd + Math.sqrt(1 - rho * rho) * eta);
            t[i] = r.nextBoolean() ? 1 : 0;
            y[i] = y0 * (t[i] == 1 ? 1 + lift : 1);
            day[i] = (int) ((long) i * days / n);
        }
        return new AbTest(pre, y, t, day);
    }
}
