package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Simulated quasi-experiments with known effects, for panel, discontinuity and observational
 * estimators:
 * <ul>
 *   <li>{@link #factorPanel}: units by periods with two-way fixed effects plus low-rank
 *       interactive factors (smooth trends), a single treated unit or a block of them adopting at
 *       one time, with treated units' factor loadings shifted (so parallel trends fail) and
 *       optionally their level moved outside the donors' range (no convex pre-period fit);</li>
 *   <li>{@link #staggered}: staggered adoption by cohorts with constant, growing or
 *       cohort-heterogeneous effects, optional anticipation and a covariate that drives both
 *       cohort membership and trends;</li>
 *   <li>{@link #discontinuity}: the Lee (2008) running-variable design of Imbens-Kalyanaraman and
 *       Calonico-Cattaneo-Titiunik (fifth-order curved means, {@code x = 2 Beta(2, 4) - 1}),
 *       sharp or fuzzy, with optional manipulation just below the cutoff;</li>
 *   <li>{@link #observational}: a Kang-Schafer (2007) study with optionally misspecified
 *       observed covariates, limited overlap and a hidden confounder of known strength.</li>
 * </ul>
 */
public final class PanelStudies {

    private PanelStudies() {
    }

    /**
     * A panel.
     *
     * @param y outcomes {@code [units][periods]}
     * @param treated treatment indicators {@code [units][periods]}
     * @param adoption the first treated period per unit, {@code -1} for never treated
     * @param effect the true effect per cell (0 where untreated)
     * @param covariate a unit-level covariate (0 when the design has none)
     */
    public record Panel(double[][] y, boolean[][] treated, int[] adoption, double[][] effect, double[] covariate) {

        /** @return units */
        public int units() {
            return y.length;
        }

        /** @return periods */
        public int periods() {
            return y[0].length;
        }

        /** @return the true average effect over treated cells */
        public double att() {
            double s = 0;
            int c = 0;
            for (int i = 0; i < y.length; i++) {
                for (int t = 0; t < y[0].length; t++) {
                    if (treated[i][t]) {
                        s += effect[i][t];
                        c++;
                    }
                }
            }
            return c == 0 ? 0 : s / c;
        }

        /**
         * The true average effect at an event time over the treated units that reach it.
         *
         * @param e periods since adoption ({@code 0} is the adoption period)
         * @return the mean effect, NaN if no unit reaches it
         */
        public double eventEffect(int e) {
            double s = 0;
            int c = 0;
            for (int i = 0; i < y.length; i++) {
                int t = adoption[i] + e;
                if (adoption[i] >= 0 && t >= 0 && t < y[0].length) {
                    s += effect[i][t];
                    c++;
                }
            }
            return c == 0 ? Double.NaN : s / c;
        }
    }

    private static double[][] factors(int periods, int rank, SplittableRandom r) {
        double[][] f = new double[periods][rank];
        for (int k = 0; k < rank; k++) {
            double freq = 0.5 + r.nextDouble();
            double phase = 2 * Math.PI * r.nextDouble();
            double slope = (r.nextBoolean() ? 1 : -1) * (0.5 + r.nextDouble());
            for (int t = 0; t < periods; t++) {
                double u = t / (double) periods;
                f[t][k] = k == 0 ? slope * 2 * u : Math.sin(2 * Math.PI * freq * u + phase);
            }
        }
        return f;
    }

    /**
     * A factor-model panel with one adoption time.
     *
     * @param units units (the treated ones are the last {@code treatedUnits})
     * @param periods periods
     * @param rank the factor rank, 1 to 3
     * @param treatedUnits treated units
     * @param start the adoption period
     * @param effect the constant effect
     * @param loadingShift added to every treated unit's loadings
     * @param levelShift added to treated units' fixed effects (beyond the donors' range breaks
     *     convex pre-period fit)
     * @param noise the idiosyncratic noise scale
     * @param seed the seed
     * @return the panel
     */
    public static Panel factorPanel(int units, int periods, int rank, int treatedUnits, int start, double effect,
            double loadingShift, double levelShift, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] f = factors(periods, rank, r);
        double[] beta = new double[periods];
        for (int t = 0; t < periods; t++) {
            beta[t] = 0.3 * r.nextGaussian() + 0.05 * t;
        }
        double[][] y = new double[units][periods];
        boolean[][] d = new boolean[units][periods];
        double[][] tau = new double[units][periods];
        int[] adoption = new int[units];
        for (int i = 0; i < units; i++) {
            boolean tr = i >= units - treatedUnits;
            adoption[i] = tr ? start : -1;
            double alpha = r.nextGaussian() + (tr ? levelShift : 0);
            double[] lambda = new double[rank];
            for (int k = 0; k < rank; k++) {
                lambda[k] = r.nextGaussian() + (tr ? loadingShift : 0);
            }
            for (int t = 0; t < periods; t++) {
                double v = alpha + beta[t] + noise * r.nextGaussian();
                for (int k = 0; k < rank; k++) {
                    v += lambda[k] * f[t][k];
                }
                if (tr && t >= start) {
                    d[i][t] = true;
                    tau[i][t] = effect;
                    v += effect;
                }
                y[i][t] = v;
            }
        }
        return new Panel(y, d, adoption, tau, new double[units]);
    }

    /** The effect path in {@link #staggered}. */
    public enum Effects {
        /** The same effect in every treated cell. */
        CONSTANT,
        /** {@code 0.5 (e + 1)} at event time {@code e}. */
        GROWING,
        /** Earlier cohorts have larger and growing effects, later ones small constant ones. */
        HETEROGENEOUS
    }

    /**
     * Staggered adoption.
     *
     * @param units units (a quarter never treated)
     * @param periods periods
     * @param cohorts adoption cohorts, evenly spaced between a quarter and three quarters of the
     *     window
     * @param effects the effect path
     * @param anticipation the effect share already present one period before adoption
     * @param covariateTrend the slope of the trend {@code gamma x_i t}; {@code x} also raises the
     *     chance of early adoption
     * @param seed the seed
     * @return the panel
     */
    public static Panel staggered(int units, int periods, int cohorts, Effects effects, double anticipation,
            double covariateTrend, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] times = new int[cohorts];
        for (int g = 0; g < cohorts; g++) {
            times[g] = periods / 4 + (int) Math.round(g * (periods / 2.0) / Math.max(1, cohorts - 1));
        }
        double[] beta = new double[periods];
        for (int t = 0; t < periods; t++) {
            beta[t] = 0.2 * r.nextGaussian() + 0.1 * t;
        }
        double[][] y = new double[units][periods];
        boolean[][] d = new boolean[units][periods];
        double[][] tau = new double[units][periods];
        int[] adoption = new int[units];
        double[] x = new double[units];
        for (int i = 0; i < units; i++) {
            x[i] = r.nextGaussian();
            if (i < units / 4) {
                adoption[i] = -1;
            } else {
                // higher x -> earlier cohort, through a noisy rank
                double score = -x[i] + r.nextGaussian();
                int g = (int) Math.max(0, Math.min(cohorts - 1, Math.floor((score + 1.5) / 3 * cohorts)));
                adoption[i] = times[g];
            }
            double alpha = r.nextGaussian();
            for (int t = 0; t < periods; t++) {
                double v = alpha + beta[t] + covariateTrend * x[i] * t + 0.5 * r.nextGaussian();
                int a = adoption[i];
                if (a >= 0) {
                    int e = t - a;
                    int cohort = Arrays.binarySearch(times, a);
                    double eff = switch (effects) {
                        case CONSTANT -> 2;
                        case GROWING -> 0.5 * (e + 1);
                        case HETEROGENEOUS -> cohort == 0 ? 1 + 0.6 * (e + 1) : 1.0 / (cohort + 1);
                    };
                    if (e >= 0) {
                        d[i][t] = true;
                        tau[i][t] = eff;
                        v += eff;
                    } else if (e == -1 && anticipation > 0) {
                        double first = switch (effects) {
                            case CONSTANT -> 2;
                            case GROWING -> 0.5;
                            case HETEROGENEOUS -> cohort == 0 ? 1.6 : 1.0 / (cohort + 1);
                        };
                        v += anticipation * first;
                    }
                }
                y[i][t] = v;
            }
        }
        return new Panel(y, d, adoption, tau, x);
    }

    /**
     * A running-variable design.
     *
     * @param x the running variable
     * @param treated the treatment taken
     * @param y the outcome
     * @param cutoff the cutoff (0)
     * @param effect the true local effect of treatment at the cutoff
     */
    public record Discontinuity(double[] x, int[] treated, double[] y, double cutoff, double effect) {
    }

    private static double beta24(SplittableRandom r) {
        // Beta(2, 4) as the 2nd order statistic of 5 uniforms
        double[] u = new double[5];
        for (int k = 0; k < 5; k++) {
            u[k] = r.nextDouble();
        }
        Arrays.sort(u);
        return u[1];
    }

    /**
     * The Lee-design conditional mean below the cutoff (CCT 2014, model 1).
     *
     * @param x the running variable
     * @return {@code mu_-(x)}
     */
    public static double leeBelow(double x) {
        return 0.48 + 1.27 * x + 7.18 * x * x + 20.21 * Math.pow(x, 3) + 21.54 * Math.pow(x, 4)
                + 7.33 * Math.pow(x, 5);
    }

    /**
     * The Lee-design conditional mean above the cutoff.
     *
     * @param x the running variable
     * @return {@code mu_+(x)}
     */
    public static double leeAbove(double x) {
        return 0.52 + 0.84 * x - 3.00 * x * x + 7.99 * Math.pow(x, 3) - 9.01 * Math.pow(x, 4) + 3.56 * Math.pow(x, 5);
    }

    /**
     * Draws a regression-discontinuity data set.
     *
     * @param n units
     * @param fuzzy when true, take-up is 0.8 above and 0.2 below the cutoff; when false, sharp
     * @param manipulation the share of units in {@code (-0.1, 0)} that move just above the cutoff
     * @param seed the seed
     * @return the data, effect 0.04 at the cutoff (the jump of the Lee design)
     */
    public static Discontinuity discontinuity(int n, boolean fuzzy, double manipulation, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = new double[n];
        int[] d = new int[n];
        double[] y = new double[n];
        double effect = 0.04;
        for (int i = 0; i < n; i++) {
            double v = 2 * beta24(r) - 1;
            if (v > -0.1 && v < 0 && r.nextDouble() < manipulation) {
                v = 0.05 * r.nextDouble();
            }
            x[i] = v;
            boolean above = v >= 0;
            d[i] = fuzzy ? (r.nextDouble() < (above ? 0.8 : 0.2) ? 1 : 0) : (above ? 1 : 0);
            // potential outcomes: below-curve plus the effect; above the cutoff the untreated
            // outcome follows mu_+ minus the effect, so the sharp jump is exactly the effect
            double base = above ? leeAbove(v) - effect : leeBelow(v);
            y[i] = base + effect * d[i] + 0.1295 * r.nextGaussian();
        }
        return new Discontinuity(x, d, y, 0, effect);
    }

    /**
     * An observational study.
     *
     * @param x the observed covariates
     * @param t treatment
     * @param y outcome
     * @param hidden the unobserved confounder
     * @param effect the true (constant) effect
     */
    public record Observational(double[][] x, int[] t, double[] y, double[] hidden, double effect) {
    }

    /**
     * A Kang-Schafer study: {@code z ~ N(0, I_4)}, outcome
     * {@code 210 + 27.4 z1 + 13.7 (z2 + z3 + z4) + effect t + gammaY u + e}, propensity
     * {@code logit^-1(overlap (-z1 + 0.5 z2 - 0.25 z3 - 0.1 z4) + gammaT u)}.
     *
     * @param n units
     * @param misspecified when true, only the Kang-Schafer transforms of {@code z} are observed
     * @param overlap the propensity slope; larger means less overlap (1 is the original)
     * @param effect the true effect
     * @param gammaT the hidden confounder's effect on treatment (log-odds)
     * @param gammaY its effect on the outcome
     * @param seed the seed
     * @return the study
     */
    public static Observational observational(int n, boolean misspecified, double overlap, double effect,
            double gammaT, double gammaY, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][4];
        int[] t = new int[n];
        double[] y = new double[n];
        double[] u = new double[n];
        for (int i = 0; i < n; i++) {
            double[] z = {r.nextGaussian(), r.nextGaussian(), r.nextGaussian(), r.nextGaussian()};
            u[i] = r.nextGaussian();
            double logit = overlap * (-z[0] + 0.5 * z[1] - 0.25 * z[2] - 0.1 * z[3]) + gammaT * u[i];
            t[i] = r.nextDouble() < 1 / (1 + Math.exp(-logit)) ? 1 : 0;
            y[i] = 210 + 27.4 * z[0] + 13.7 * (z[1] + z[2] + z[3]) + effect * t[i] + gammaY * u[i] + r.nextGaussian();
            if (misspecified) {
                x[i][0] = Math.exp(z[0] / 2);
                x[i][1] = z[1] / (1 + Math.exp(z[0])) + 10;
                x[i][2] = Math.pow(z[0] * z[2] / 25 + 0.6, 3);
                x[i][3] = Math.pow(z[1] + z[3] + 20, 2);
            } else {
                x[i] = z;
            }
        }
        return new Observational(x, t, y, u, effect);
    }
}
