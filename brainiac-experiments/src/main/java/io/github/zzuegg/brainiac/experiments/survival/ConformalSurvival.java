package io.github.zzuegg.brainiac.experiments.survival;

import io.github.zzuegg.brainiac.experimental.data.SurvivalData;
import io.github.zzuegg.brainiac.experimental.survival.CompetingRisks;
import io.github.zzuegg.brainiac.experimental.survival.CureModel;
import io.github.zzuegg.brainiac.experimental.survival.DeepHit;
import io.github.zzuegg.brainiac.experimental.survival.SurvivalForest;
import io.github.zzuegg.brainiac.experimental.survival.SurvivalMetrics;
import io.github.zzuegg.brainiac.experimental.survival.SurvivalModels;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Conformal survival bounds (13.6.11) and cure and competing-risks models (13.6.12).
 * <ul>
 *   <li>{@link #conformal}: lower predictive bounds from
 *       {@link io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival} around a Cox model
 *       or a {@link SurvivalForest}, on simulated data whose true event and censoring times are
 *       known (so coverage {@code P(T >= L(X))} is exact), with covariate-dependent censoring and an
 *       optionally misspecified (nonlinear) risk, against the uncalibrated model quantiles and naive
 *       split conformal that treats censoring times as event times;</li>
 *   <li>{@link #cure}: the {@link CureModel} (Weibull and Cox latency) against Cox and Kaplan-Meier
 *       on data with a planted cured fraction;</li>
 *   <li>{@link #competing}: Aalen-Johansen and Fine-Gray ({@link CompetingRisks}) against
 *       {@code 1 - KM}, cause-specific Cox and {@link DeepHit}, on data simulated from a Fine-Gray
 *       model (so the subdistribution coefficients are planted), plus the existing
 *       {@link SurvivalData#competingRisks} data for the nonparametric curves.</li>
 * </ul>
 * {@link SurvivalData} does not expose the latent event and censoring times, a cure fraction or
 * subdistribution effects, so this class simulates those settings itself with the same Weibull
 * conventions.
 */
public final class ConformalSurvival {

    private ConformalSurvival() {
    }

    /**
     * Simulated right-censored data with the latent times kept.
     *
     * @param x covariates
     * @param t true event times
     * @param c censoring times
     */
    public record Latent(double[][] x, double[] t, double[] c) {

        /** @return {@code min(T, C)} */
        public double[] observed() {
            double[] o = new double[t.length];
            for (int i = 0; i < o.length; i++) {
                o[i] = Math.min(t[i], c[i]);
            }
            return o;
        }

        /** @return 1 where the event is observed */
        public int[] event() {
            int[] e = new int[t.length];
            for (int i = 0; i < e.length; i++) {
                e[i] = t[i] <= c[i] ? 1 : 0;
            }
            return e;
        }

        /** @return the censored share */
        public double censoring() {
            int k = 0;
            for (int e : event()) {
                k += 1 - e;
            }
            return k / (double) t.length;
        }

        Latent rows(int from, int to) {
            return new Latent(java.util.Arrays.copyOfRange(x, from, to), java.util.Arrays.copyOfRange(t, from, to),
                    java.util.Arrays.copyOfRange(c, from, to));
        }
    }

    /**
     * Weibull accelerated-failure data: {@code T = exp(mu(x)) W}, {@code W} Weibull with shape 1.5;
     * {@code mu = 0.5 + 0.6 x0 - 0.4 x1}, plus {@code 1.2 (|x2| - 0.8)} when nonlinear. Censoring is
     * exponential with rate {@code r exp(0.8 x0)} (covariate-dependent), {@code r} tuned to the
     * target censored share.
     *
     * @param n units
     * @param nonlinear whether the risk has a term a linear Cox model misses
     * @param censoring the target censored share
     * @param seed the seed
     * @return the data
     */
    public static Latent simulate(int n, boolean nonlinear, double censoring, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][4];
        double[] t = new double[n];
        double[] e = new double[n];
        double[] g = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) {
                x[i][j] = r.nextGaussian();
            }
            double mu = 0.5 + 0.6 * x[i][0] - 0.4 * x[i][1] + (nonlinear ? 1.2 * (Math.abs(x[i][2]) - 0.8) : 0);
            t[i] = Math.exp(mu) * Math.pow(-Math.log(1 - r.nextDouble()), 1 / 1.5);
            e[i] = -Math.log(1 - r.nextDouble()) / Math.exp(0.8 * x[i][0]);
        }
        double lo = 1e-4;
        double hi = 1e4;
        for (int it = 0; it < 60; it++) {
            double mid = Math.sqrt(lo * hi);
            int k = 0;
            for (int i = 0; i < n; i++) {
                k += e[i] / mid < t[i] ? 1 : 0;
            }
            if (k / (double) n > censoring) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        double rate = Math.sqrt(lo * hi);
        for (int i = 0; i < n; i++) {
            g[i] = e[i] / rate;
        }
        return new Latent(x, t, g);
    }

    private static io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.ConditionalSurvival forest(
            SurvivalForest f) {
        double[] grid = f.grid();
        return (x, t) -> {
            double[] h = f.cumulativeHazard(x);
            int k = -1;
            while (k + 1 < grid.length && grid[k + 1] <= t) {
                k++;
            }
            return k < 0 ? 1 : Math.exp(-h[k]);
        };
    }

    /**
     * Coverage and tightness of every lower predictive bound.
     *
     * @param n units (split into training, calibration and test thirds)
     * @param nonlinear whether the Cox model is misspecified
     * @param censoring the censored share
     * @param useForest whether the survival model is a random survival forest instead of Cox
     * @param alpha the miscoverage
     * @param seed the seed
     * @return method to {coverage P(T >= L), E[L] / E[T]}
     */
    public static Map<String, double[]> conformal(int n, boolean nonlinear, double censoring, boolean useForest,
            double alpha, long seed) {
        Latent all = simulate(n, nonlinear, censoring, seed);
        int third = n / 3;
        Latent train = all.rows(0, third);
        Latent cal = all.rows(third, 2 * third);
        Latent test = all.rows(2 * third, n);
        double[] obs = train.observed();
        int[] ev = train.event();
        io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.ConditionalSurvival model;
        if (useForest) {
            double[] grid = SurvivalMetrics.grid(obs, 0.01, 0.99, 40);
            model = forest(SurvivalForest.fit(train.x(), obs, ev, grid, 50, 10, 5, seed));
        } else {
            model = io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.cox(
                    new SurvivalModels.Cox(train.x(), obs, ev, 1e-3));
        }
        int[] censEv = new int[ev.length];
        for (int i = 0; i < ev.length; i++) {
            censEv[i] = 1 - ev[i];
        }
        io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.ConditionalSurvival censModel =
                io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.cox(
                        new SurvivalModels.Cox(train.x(), obs, censEv, 1e-3));
        double[] sortedC = train.c().clone();
        java.util.Arrays.sort(sortedC);
        double c0 = sortedC[sortedC.length / 2];
        double horizon = 50;
        io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.Calibration calibration =
                new io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.Calibration(cal.x(),
                        cal.observed(), cal.event(), cal.c());
        Map<String, double[]> out = new LinkedHashMap<>();
        double meanT = 0;
        for (double v : test.t()) {
            meanT += v / test.t().length;
        }
        for (io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.Method m
                : io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.Method.values()) {
            io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival lpb =
                    io.github.zzuegg.brainiac.experimental.survival.ConformalSurvival.fit(m, model, censModel,
                            calibration, alpha, c0, horizon, seed);
            double[] l = lpb.lowerBounds(test.x());
            double cover = 0;
            double mean = 0;
            for (int i = 0; i < l.length; i++) {
                cover += test.t()[i] >= l[i] ? 1 : 0;
                mean += l[i] / l.length;
            }
            out.put(m.toString(), new double[] {cover / l.length, mean / meanT});
        }
        return out;
    }

    /**
     * Mixture-cure data: susceptible with probability {@code sigmoid(0.3 + 1.2 x0)}, Weibull latency
     * (shape 1.5, rate {@code 0.3 exp(0.7 x1)}), uniform censoring on {@code [0, 15]}.
     *
     * @param n units
     * @param seed the seed
     * @return the data and, in {@code c}, the censoring times; {@code t} is infinite for the cured
     */
    public static Latent cureData(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][2];
        double[] t = new double[n];
        double[] c = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextGaussian();
            x[i][1] = r.nextGaussian();
            double pi = 1 / (1 + Math.exp(-(0.3 + 1.2 * x[i][0])));
            t[i] = r.nextDouble() < pi ? Math.pow(-Math.log(1 - r.nextDouble()) / (0.3 * Math.exp(0.7 * x[i][1])),
                    1 / 1.5) : Double.POSITIVE_INFINITY;
            c[i] = 15 * r.nextDouble();
        }
        return new Latent(x, t, c);
    }

    /**
     * Cure fraction and coefficient recovery.
     *
     * @param n units
     * @param seed the seed
     * @return quantity to value
     */
    public static Map<String, Double> cure(int n, long seed) {
        Latent d = cureData(n, seed);
        double[] obs = d.observed();
        int[] ev = d.event();
        double trueCured = 0;
        for (double[] row : d.x()) {
            trueCured += 1 - 1 / (1 + Math.exp(-(0.3 + 1.2 * row[0])));
        }
        trueCured /= n;
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("true cured fraction", trueCured);
        out.put("censored share", d.censoring());
        CureModel w = CureModel.fit(d.x(), obs, ev, CureModel.Latency.WEIBULL, 25);
        out.put("cure model (Weibull) cured fraction", w.curedFraction(d.x()));
        out.put("cure model (Weibull) incidence x0 coefficient (true 1.2)", w.incidenceCoefficients()[0]);
        out.put("cure model (Weibull) latency x1 coefficient (true 0.7)", w.latencyCoefficients()[1]);
        CureModel c = CureModel.fit(d.x(), obs, ev, CureModel.Latency.COX, 15);
        out.put("cure model (Cox) cured fraction", c.curedFraction(d.x()));
        out.put("cure model (Cox) latency x1 coefficient (true 0.7)", c.latencyCoefficients()[1]);
        // the plateau a standard model implies at the end of follow-up, and in the long run
        SurvivalModels.Cox cox = new SurvivalModels.Cox(d.x(), obs, ev, 1e-3);
        double coxEnd = 0;
        double coxLong = 0;
        double cureLong = 0;
        for (double[] row : d.x()) {
            coxEnd += cox.survival(row, 14.9) / n;
            coxLong += Math.exp(Math.log(Math.max(1e-300, cox.survival(row, 14.9))) * Math.pow(30 / 14.9, 1.5)) / n;
            cureLong += w.survival(row, 30) / n;
        }
        out.put("Cox survival at end of follow-up", coxEnd);
        out.put("Cox survival at t=30 (Weibull-shaped extrapolation)", coxLong);
        out.put("cure model survival at t=30", cureLong);
        out.put("Kaplan-Meier plateau", new SurvivalModels.KaplanMeier(obs, ev).survival(14.9));
        out.put("Cox x0 coefficient (mixes incidence and latency)", cox.coefficients()[0]);
        return out;
    }

    /**
     * Fine-Gray data (Fine and Gray 1999's simulation): cause 1 has
     * {@code F_1(t | x) = 1 - (1 - p (1 - e^-t))^exp(b1 . x)} with {@code p = 0.5} and
     * {@code b1 = (0.6, -0.5, 0)}; otherwise cause 2 with an exponential time of rate
     * {@code exp(b2 . x)}, {@code b2 = (-0.6, 0, 0.5)}; uniform censoring on {@code [0, 4]}.
     *
     * @param n units
     * @param seed the seed
     * @return x, observed time, event cause ({@code 0..2}) packed as a {@link SurvivalData.Survival}-like triple
     */
    public static Object[] fineGrayData(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double p = 0.5;
        double[] b1 = {0.6, -0.5, 0};
        double[] b2 = {-0.6, 0, 0.5};
        double[][] x = new double[n][3];
        double[] time = new double[n];
        int[] event = new int[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                x[i][j] = r.nextGaussian();
            }
            double e1 = Math.exp(b1[0] * x[i][0] + b1[1] * x[i][1] + b1[2] * x[i][2]);
            double p1 = 1 - Math.pow(1 - p, e1);
            double t;
            int cause;
            if (r.nextDouble() < p1) {
                double u = r.nextDouble();
                double inner = 1 - Math.pow(1 - u * p1, 1 / e1);
                t = -Math.log(Math.max(1e-12, 1 - inner / p));
                cause = 1;
            } else {
                double e2 = Math.exp(b2[0] * x[i][0] + b2[1] * x[i][1] + b2[2] * x[i][2]);
                t = -Math.log(1 - r.nextDouble()) / e2;
                cause = 2;
            }
            double c = 4 * r.nextDouble();
            time[i] = Math.min(t, c);
            event[i] = t <= c ? cause : 0;
        }
        return new Object[] {x, time, event};
    }

    /**
     * Competing-risks estimators.
     *
     * @param n units
     * @param seed the seed
     * @return quantity to value
     */
    public static Map<String, Double> competing(int n, long seed) {
        Map<String, Double> out = new LinkedHashMap<>();
        // nonparametric curves on the existing generator, scored against the true marginal incidence
        SurvivalData.Survival s = SurvivalData.competingRisks(n, 4, 0.3, seed);
        double[] grid = SurvivalMetrics.grid(s.time(), 0.1, 0.9, 8);
        double[] aj = CompetingRisks.aalenJohansen(s.time(), s.event(), 1, grid);
        double[] km = CompetingRisks.oneMinusKaplanMeier(s.time(), s.event(), 1, grid);
        double ajErr = 0;
        double kmErr = 0;
        double kmBias = 0;
        for (int g = 0; g < grid.length; g++) {
            double truth = 0;
            for (int i = 0; i < s.size(); i++) {
                truth += s.incidence(i, 1, grid[g]) / s.size();
            }
            ajErr += Math.abs(aj[g] - truth) / grid.length;
            kmErr += Math.abs(km[g] - truth) / grid.length;
            kmBias += (km[g] - truth) / grid.length;
        }
        out.put("Aalen-Johansen CIF error", ajErr);
        out.put("1 - KM CIF error", kmErr);
        out.put("1 - KM CIF bias", kmBias);
        // Fine-Gray data: planted subdistribution coefficients
        Object[] fg = fineGrayData(n, seed + 1);
        double[][] x = (double[][]) fg[0];
        double[] time = (double[]) fg[1];
        int[] event = (int[]) fg[2];
        CompetingRisks.FineGray model = CompetingRisks.FineGray.fit(x, time, event, 1, 1e-4);
        double[] b = model.coefficients();
        double[] truth = {0.6, -0.5, 0};
        out.put("Fine-Gray coefficient error", Math.abs(b[0] - truth[0]) + Math.abs(b[1] - truth[1])
                + Math.abs(b[2] - truth[2]));
        int[] cs = new int[event.length];
        for (int i = 0; i < cs.length; i++) {
            cs[i] = event[i] == 1 ? 1 : 0;
        }
        double[] csb = new SurvivalModels.Cox(x, time, cs, 1e-4).coefficients();
        out.put("cause-specific Cox coefficient error", Math.abs(csb[0] - truth[0]) + Math.abs(csb[1] - truth[1])
                + Math.abs(csb[2] - truth[2]));
        out.put("Fine-Gray x2 coefficient (true 0)", b[2]);
        out.put("cause-specific Cox x2 coefficient", csb[2]);
        // individual cumulative incidence at t = 1 against the planted F_1(1 | x)
        DeepHit dh = new DeepHit(3, 32, 2, new double[] {0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4}, seed)
                .fit(x, time, event, 300, 64, 3e-3f, 0.2f, seed);
        double[][][] inc = dh.incidence(x, new double[] {1});
        double fgErr = 0;
        double dhErr = 0;
        for (int i = 0; i < n; i++) {
            double e1 = Math.exp(0.6 * x[i][0] - 0.5 * x[i][1]);
            double f1 = 1 - Math.pow(1 - 0.5 * (1 - Math.exp(-1)), e1);
            fgErr += Math.abs(model.incidence(x[i], 1) - f1) / n;
            dhErr += Math.abs(inc[0][i][0] - f1) / n;
        }
        out.put("Fine-Gray individual CIF(1) error", fgErr);
        out.put("DeepHit individual CIF(1) error", dhErr);
        return out;
    }
}
