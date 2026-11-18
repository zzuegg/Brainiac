package io.github.zzuegg.brainiac.experiments.econometrics;

import io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments;
import io.github.zzuegg.brainiac.experimental.data.PanelStudies;
import io.github.zzuegg.brainiac.experimental.panel.StaggeredDid;
import io.github.zzuegg.brainiac.experimental.panel.SyntheticControl;
import io.github.zzuegg.brainiac.experimental.panel.SyntheticDid;
import io.github.zzuegg.brainiac.experimental.panel.TriplyRobustPanel;
import io.github.zzuegg.brainiac.experimental.panel.TwoWayFixedEffects;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Panel estimators against two-way fixed effects (wave 11, items 11.3.6 and 11.3.7).
 *
 * <ul>
 *   <li>{@link #factor}: {@link PanelStudies#factorPanel} panels (30 units, 30 periods, rank 2,
 *       adoption at period 20, effect 1). Synthetic control, synthetic DiD and TROP (tuned by
 *       its leave-one-out criterion on every panel) against the TWFE DiD, matrix completion with
 *       fixed effects (TROP with flat weights, nuclear-norm level tuned the same way) and the
 *       CUPED-adjusted difference of post-period means (pre-period mean as covariate). Scores:
 *       bias and RMSE of the ATT, and the coverage of a placebo-in-space interval (the estimate
 *       plus or minus the largest absolute placebo estimate over control units, with the real
 *       treated units dropped).</li>
 *   <li>{@link #staggered}: {@link PanelStudies#staggered} panels (60 units, 20 periods, three
 *       cohorts). Callaway-Sant'Anna (unconditional, outcome regression, IPW, not-yet-treated)
 *       and Sun-Abraham against the TWFE DiD, the TWFE event study and SDID run per cohort with
 *       never-treated donors. Scores: bias and RMSE of the overall ATT, the coverage of the
 *       unit bootstrap band (CS only), the RMSE of the event-time profile over {@code e = 0..4}
 *       and, for the Goodman-Bacon decomposition, the weight on forbidden comparisons.</li>
 * </ul>
 */
public final class PanelShootout {

    private PanelShootout() {
    }

    /** A single-adoption design. */
    public enum Design {
        /** One treated unit whose factor loadings are shifted (parallel trends fail). */
        SINGLE_SHIFTED,
        /** Five treated units with shifted loadings. */
        BLOCK_SHIFTED,
        /** One treated unit whose level is outside the donors' range (no convex pre-period fit). */
        NO_CONVEX_FIT,
        /** One treated unit drawn like the donors (parallel trends hold). */
        PARALLEL
    }

    @FunctionalInterface
    private interface Estimator {
        double estimate(double[][] y, int[] treatedUnits, int start);
    }

    private static boolean[][] mask(int units, int periods, int[] treatedUnits, int start) {
        boolean[][] d = new boolean[units][periods];
        for (int i : treatedUnits) {
            for (int t = start; t < periods; t++) {
                d[i][t] = true;
            }
        }
        return d;
    }

    /**
     * The CUPED-adjusted difference of post-period means.
     *
     * @param y outcomes
     * @param treatedUnits treated units
     * @param start first treated period
     * @return treated minus control mean of the adjusted post-period means
     */
    static double cuped(double[][] y, int[] treatedUnits, int start) {
        int n = y.length;
        int periods = y[0].length;
        double[] post = new double[n];
        double[] pre = new double[n];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < periods; t++) {
                if (t < start) {
                    pre[i] += y[i][t] / start;
                } else {
                    post[i] += y[i][t] / (periods - start);
                }
            }
        }
        double[] adj = OnlineExperiments.cuped(post, pre, n);
        boolean[] tr = new boolean[n];
        for (int i : treatedUnits) {
            tr[i] = true;
        }
        double st = 0;
        double sc = 0;
        for (int i = 0; i < n; i++) {
            if (tr[i]) {
                st += adj[i] / treatedUnits.length;
            } else {
                sc += adj[i] / (n - treatedUnits.length);
            }
        }
        return st - sc;
    }

    private static Map<String, Estimator> estimators(double[][] y, int[] treatedUnits, int start, long seed) {
        boolean[][] d = mask(y.length, y[0].length, treatedUnits, start);
        double[] inf = {Double.POSITIVE_INFINITY};
        double[] nn = {Double.POSITIVE_INFINITY, 1, 4};
        TriplyRobustPanel mc = TriplyRobustPanel.tune(y, d, new double[] {0}, new double[] {0}, nn, 4, 30, seed);
        TriplyRobustPanel trop = TriplyRobustPanel.tune(y, d, new double[] {0, 1}, new double[] {0, 0.2},
                new double[] {inf[0], 2}, 4, 30, seed);
        Map<String, Estimator> m = new LinkedHashMap<>();
        m.put("TWFE DiD", (yy, tu, s) -> TwoWayFixedEffects.did(yy, mask(yy.length, yy[0].length, tu, s)));
        m.put("CUPED difference", PanelShootout::cuped);
        m.put("matrix completion (FE + nuclear norm)",
                (yy, tu, s) -> mc.fit(yy, mask(yy.length, yy[0].length, tu, s)));
        m.put("synthetic control", (yy, tu, s) -> SyntheticControl.fit(yy, tu, s).effect());
        m.put("synthetic DiD", (yy, tu, s) -> SyntheticDid.fit(yy, tu, s).effect());
        m.put("TROP (LOO-tuned)", (yy, tu, s) -> trop.fit(yy, mask(yy.length, yy[0].length, tu, s)));
        return m;
    }

    /**
     * Single-adoption factor panels.
     *
     * @param design the design
     * @param panels simulated panels
     * @param placebos control units used as placebos per panel
     * @param seed the seed
     * @return method to {@code {bias, RMSE, placebo-interval coverage}}
     */
    public static Map<String, double[]> factor(Design design, int panels, int placebos, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int units = 30;
        int periods = 30;
        int start = 20;
        double effect = 1;
        for (int p = 0; p < panels; p++) {
            PanelStudies.Panel panel = switch (design) {
                case SINGLE_SHIFTED -> PanelStudies.factorPanel(units, periods, 2, 1, start, effect, 1.0, 0, 0.3,
                        seed + p);
                case BLOCK_SHIFTED -> PanelStudies.factorPanel(units, periods, 2, 5, start, effect, 1.0, 0, 0.3,
                        seed + p);
                case NO_CONVEX_FIT -> PanelStudies.factorPanel(units, periods, 2, 1, start, effect, 0.5, 4, 0.3,
                        seed + p);
                case PARALLEL -> PanelStudies.factorPanel(units, periods, 2, 1, start, effect, 0, 0, 0.3, seed + p);
            };
            double[][] y = panel.y();
            int k = design == Design.BLOCK_SHIFTED ? 5 : 1;
            int[] treatedUnits = new int[k];
            for (int i = 0; i < k; i++) {
                treatedUnits[i] = units - k + i;
            }
            double truth = panel.att();
            double[][] pool = Arrays.copyOf(y, units - k);
            SplittableRandom r = new SplittableRandom(seed * 31 + p);
            for (Map.Entry<String, Estimator> e : estimators(y, treatedUnits, start, seed + p).entrySet()) {
                double est = e.getValue().estimate(y, treatedUnits, start);
                double width = 0;
                for (int q = 0; q < placebos; q++) {
                    int[] fake = new int[k];
                    int first = r.nextInt(pool.length - k + 1);
                    for (int i = 0; i < k; i++) {
                        fake[i] = first + i;
                    }
                    width = Math.max(width, Math.abs(e.getValue().estimate(pool, fake, start)));
                }
                double err = est - truth;
                double[] acc = out.computeIfAbsent(e.getKey(), s -> new double[3]);
                acc[0] += err / panels;
                acc[1] += err * err / panels;
                acc[2] += (Math.abs(err) <= width ? 1.0 : 0.0) / panels;
            }
        }
        out.values().forEach(v -> v[1] = Math.sqrt(v[1]));
        return out;
    }

    private static double sdidPerCohort(PanelStudies.Panel panel) {
        int[] adoption = panel.adoption();
        int[] cohorts = Arrays.stream(adoption).filter(a -> a > 0).distinct().sorted().toArray();
        int[] never = java.util.stream.IntStream.range(0, adoption.length).filter(i -> adoption[i] < 0).toArray();
        int periods = panel.periods();
        double num = 0;
        double den = 0;
        for (int g : cohorts) {
            int[] members = java.util.stream.IntStream.range(0, adoption.length).filter(i -> adoption[i] == g)
                    .toArray();
            double[][] sub = new double[members.length + never.length][];
            int[] tr = new int[members.length];
            for (int i = 0; i < members.length; i++) {
                sub[i] = panel.y()[members[i]];
                tr[i] = i;
            }
            for (int j = 0; j < never.length; j++) {
                sub[members.length + j] = panel.y()[never[j]];
            }
            double cells = members.length * (double) (periods - g);
            num += cells * SyntheticDid.fit(sub, tr, g).effect();
            den += cells;
        }
        return num / den;
    }

    /**
     * Staggered-adoption panels.
     *
     * @param effects the effect path
     * @param anticipation the anticipation share
     * @param panels simulated panels
     * @param bootstrap bootstrap replicates for the CS band
     * @param seed the seed
     * @return method to {@code {bias, RMSE, band coverage (NaN if none), event-profile RMSE (NaN if
     *     none), forbidden-comparison weight (NaN except for the decomposition)}}
     */
    public static Map<String, double[]> staggered(PanelStudies.Effects effects, double anticipation, int panels,
            int bootstrap, long seed) {
        int minEvent = -3;
        int maxEvent = 4;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int p = 0; p < panels; p++) {
            PanelStudies.Panel panel = PanelStudies.staggered(60, 20, 3, effects, anticipation, 0.05, seed + p);
            double[][] y = panel.y();
            int[] a = panel.adoption();
            double[] x = panel.covariate();
            double truth = panel.att();
            double[] profile = new double[maxEvent + 1];
            for (int e = 0; e <= maxEvent; e++) {
                profile[e] = panel.eventEffect(e);
            }
            StaggeredDid.Bacon bacon = StaggeredDid.bacon(y, a);
            record(out, "TWFE DiD (Bacon: forbidden weight)", bacon.twfe() - truth, Double.NaN, Double.NaN,
                    bacon.weightOf("late vs early (forbidden)"), panels);
            double[] es = TwoWayFixedEffects.eventStudy(y, a, -minEvent, maxEvent);
            double esOverall = 0;
            for (int e = 0; e <= maxEvent; e++) {
                esOverall += es[e - minEvent] / (maxEvent + 1);
            }
            record(out, "TWFE event study (mean post coefficient)", esOverall - truth, Double.NaN,
                    profileError(es, -minEvent, profile), Double.NaN, panels);
            record(out, "SDID per cohort", sdidPerCohort(panel) - truth, Double.NaN, Double.NaN, Double.NaN, panels);
            for (StaggeredDid.Method m : StaggeredDid.Method.values()) {
                double[] agg = StaggeredDid.aggregate(StaggeredDid.groupTime(y, a, x, m, false), minEvent, maxEvent);
                double cover = Double.NaN;
                if (m == StaggeredDid.Method.UNCONDITIONAL && bootstrap > 0) {
                    double[][] band = StaggeredDid.bootstrap(y, a, x, m, minEvent, maxEvent, bootstrap, seed + p);
                    cover = band[0][0] <= truth && truth <= band[1][0] ? 1 : 0;
                }
                record(out, "Callaway-Sant'Anna (" + m.name().toLowerCase().replace('_', ' ') + ")",
                        agg[0] - truth, cover, profileError(agg, 1 - minEvent, profile), Double.NaN, panels);
            }
            double[] ny = StaggeredDid.aggregate(StaggeredDid.groupTime(y, a, x, StaggeredDid.Method.UNCONDITIONAL,
                    true), minEvent, maxEvent);
            record(out, "Callaway-Sant'Anna (not yet treated)", ny[0] - truth, Double.NaN,
                    profileError(ny, 1 - minEvent, profile), Double.NaN, panels);
            double[] sa = StaggeredDid.sunAbraham(y, a, minEvent, maxEvent);
            record(out, "Sun-Abraham", sa[0] - truth, Double.NaN, profileError(sa, 1 - minEvent, profile),
                    Double.NaN, panels);
        }
        out.values().forEach(v -> v[1] = Math.sqrt(v[1]));
        return out;
    }

    private static double profileError(double[] est, int zeroIndex, double[] truth) {
        double s = 0;
        for (int e = 0; e < truth.length; e++) {
            double d = est[zeroIndex + e] - truth[e];
            s += d * d / truth.length;
        }
        return Math.sqrt(s);
    }

    private static void record(Map<String, double[]> out, String key, double err, double cover, double profile,
            double forbidden, int panels) {
        double[] acc = out.computeIfAbsent(key, s -> new double[5]);
        acc[0] += err / panels;
        acc[1] += err * err / panels;
        acc[2] += cover / panels;
        acc[3] += profile / panels;
        acc[4] += forbidden / panels;
    }
}
