package io.github.zzuegg.brainiac.experimental.panel;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.CausalEstimators;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.TreeSet;

/**
 * Staggered adoption, where units adopt treatment at different times and two-way fixed effects
 * can mislead.
 *
 * <ul>
 *   <li>{@link #bacon}: the Goodman-Bacon decomposition (J. Econometrics 2021,
 *       <a href="https://doi.org/10.1016/j.jeconom.2021.03.014">doi:10.1016/j.jeconom.2021.03.014</a>)
 *       of the TWFE coefficient into all 2x2 DiDs (treated vs never treated, early vs later
 *       treated before the later group adopts, and the "forbidden" later vs already-treated) with
 *       their variance weights; in a balanced panel the weighted sum reproduces TWFE exactly.</li>
 *   <li>{@link #groupTime}: Callaway and Sant'Anna (J. Econometrics 2021,
 *       <a href="https://arxiv.org/abs/1803.09015">arXiv:1803.09015</a>) group-time effects
 *       {@code ATT(g, t)} against never-treated or not-yet-treated units, unconditional, by
 *       outcome regression on a covariate, or by normalised inverse propensity weighting;
 *       {@link #aggregate} averages them overall and by event time, and {@link #bootstrap} gives
 *       unit-cluster bootstrap bands.</li>
 *   <li>{@link #sunAbraham}: Sun and Abraham's interaction-weighted event study (J. Econometrics
 *       2021, <a href="https://arxiv.org/abs/1804.05785">arXiv:1804.05785</a>): a TWFE regression
 *       on cohort-by-event-time dummies (never-treated units as the control cohort, event time
 *       -1 as the reference), averaged with cohort shares.</li>
 * </ul>
 */
public final class StaggeredDid {

    private StaggeredDid() {
    }

    /**
     * One 2x2 comparison.
     *
     * @param kind "treated vs never", "early vs late" or "late vs early (forbidden)"
     * @param treatedTime the treated group's adoption time
     * @param controlTime the control group's adoption time ({@code -1} for never treated)
     * @param estimate the 2x2 DiD
     * @param weight its normalised weight
     */
    public record Comparison(String kind, int treatedTime, int controlTime, double estimate, double weight) {
    }

    /**
     * The decomposition.
     *
     * @param comparisons all 2x2 comparisons
     * @param twfe the TWFE coefficient it decomposes
     */
    public record Bacon(List<Comparison> comparisons, double twfe) {

        /** @return the weighted sum of the 2x2 estimates */
        public double weightedSum() {
            return comparisons.stream().mapToDouble(c -> c.weight() * c.estimate()).sum();
        }

        /**
         * @param kind a comparison kind
         * @return the total weight of that kind
         */
        public double weightOf(String kind) {
            return comparisons.stream().filter(c -> c.kind().equals(kind)).mapToDouble(Comparison::weight).sum();
        }
    }

    private static double mean(double[][] y, List<Integer> units, int from, int to) {
        double s = 0;
        for (int i : units) {
            for (int t = from; t < to; t++) {
                s += y[i][t];
            }
        }
        return s / (units.size() * (double) (to - from));
    }

    private static double did(double[][] y, List<Integer> tr, List<Integer> ct, int from, int split, int to) {
        return (mean(y, tr, split, to) - mean(y, tr, from, split)) - (mean(y, ct, split, to) - mean(y, ct, from, split));
    }

    /**
     * The Goodman-Bacon decomposition.
     *
     * @param y outcomes {@code [units][periods]}
     * @param adoption first treated period per unit, {@code -1} for never (adoption at period 0
     *     is not allowed)
     * @return the comparisons and the TWFE coefficient
     */
    public static Bacon bacon(double[][] y, int[] adoption) {
        int n = y.length;
        int periods = y[0].length;
        TreeSet<Integer> times = new TreeSet<>();
        List<Integer> never = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (adoption[i] < 0) {
                never.add(i);
            } else {
                times.add(adoption[i]);
            }
        }
        Integer[] tk = times.toArray(new Integer[0]);
        List<List<Integer>> groups = new ArrayList<>();
        for (int g : tk) {
            List<Integer> members = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (adoption[i] == g) {
                    members.add(i);
                }
            }
            groups.add(members);
        }
        List<Comparison> out = new ArrayList<>();
        double nu = never.size() / (double) n;
        for (int k = 0; k < tk.length; k++) {
            double nk = groups.get(k).size() / (double) n;
            double dk = (periods - tk[k]) / (double) periods;
            if (!never.isEmpty()) {
                double nkU = nk / (nk + nu);
                double s = (nk + nu) * (nk + nu) * nkU * (1 - nkU) * dk * (1 - dk);
                out.add(new Comparison("treated vs never", tk[k], -1, did(y, groups.get(k), never, 0, tk[k], periods),
                        s));
            }
            for (int l = k + 1; l < tk.length; l++) {
                double nl = groups.get(l).size() / (double) n;
                double dl = (periods - tk[l]) / (double) periods;
                double nkl = nk / (nk + nl);
                double sk = Math.pow((nk + nl) * (1 - dl), 2) * nkl * (1 - nkl) * ((dk - dl) / (1 - dl))
                        * ((1 - dk) / (1 - dl));
                double sl = Math.pow((nk + nl) * dk, 2) * nkl * (1 - nkl) * (dl / dk) * ((dk - dl) / dk);
                out.add(new Comparison("early vs late", tk[k], tk[l],
                        did(y, groups.get(k), groups.get(l), 0, tk[k], tk[l]), sk));
                out.add(new Comparison("late vs early (forbidden)", tk[l], tk[k],
                        did(y, groups.get(l), groups.get(k), tk[k], tk[l], periods), sl));
            }
        }
        double total = out.stream().mapToDouble(Comparison::weight).sum();
        List<Comparison> norm = new ArrayList<>();
        for (Comparison c : out) {
            norm.add(new Comparison(c.kind(), c.treatedTime(), c.controlTime(), c.estimate(), c.weight() / total));
        }
        boolean[][] d = new boolean[n][periods];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < periods; t++) {
                d[i][t] = adoption[i] >= 0 && t >= adoption[i];
            }
        }
        return new Bacon(norm, TwoWayFixedEffects.did(y, d));
    }

    /** How group-time effects adjust for the covariate. */
    public enum Method {
        /** Plain DiD of means. */
        UNCONDITIONAL,
        /** Outcome regression of control changes on the covariate. */
        OUTCOME_REGRESSION,
        /** Normalised inverse propensity weighting of controls. */
        IPW
    }

    /**
     * Group-time effects.
     *
     * @param cohorts the adoption times, ascending
     * @param att {@code att[g][t]}, NaN where undefined
     * @param size units per cohort
     */
    public record GroupTime(int[] cohorts, double[][] att, int[] size) {
    }

    /**
     * Callaway-Sant'Anna group-time effects, base period {@code g - 1} for {@code t >= g} and
     * {@code t - 1} (the "varying" base) before.
     *
     * @param y outcomes
     * @param adoption adoption times ({@code -1} never)
     * @param covariate a unit covariate (used by the adjusted methods)
     * @param method the adjustment
     * @param notYetTreated whether not-yet-treated units join the never-treated controls
     * @return the effects
     */
    public static GroupTime groupTime(double[][] y, int[] adoption, double[] covariate, Method method,
            boolean notYetTreated) {
        int n = y.length;
        int periods = y[0].length;
        int[] cohorts = Arrays.stream(adoption).filter(a -> a > 0).distinct().sorted().toArray();
        double[][] att = new double[cohorts.length][periods];
        int[] size = new int[cohorts.length];
        for (int g = 0; g < cohorts.length; g++) {
            int gt = cohorts[g];
            for (int i = 0; i < n; i++) {
                size[g] += adoption[i] == gt ? 1 : 0;
            }
            for (int t = 0; t < periods; t++) {
                int base = t >= gt ? gt - 1 : t - 1;
                if (base < 0) {
                    att[g][t] = Double.NaN;
                    continue;
                }
                int latest = Math.max(t, base);
                List<Integer> tr = new ArrayList<>();
                List<Integer> ct = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if (adoption[i] == gt) {
                        tr.add(i);
                    } else if (adoption[i] < 0 || (notYetTreated && adoption[i] > latest)) {
                        ct.add(i);
                    }
                }
                att[g][t] = ct.isEmpty() ? Double.NaN : twoByTwo(y, covariate, tr, ct, base, t, method);
            }
        }
        return new GroupTime(cohorts, att, size);
    }

    private static double twoByTwo(double[][] y, double[] x, List<Integer> tr, List<Integer> ct, int base, int t,
            Method method) {
        double[] dtr = new double[tr.size()];
        double[] dct = new double[ct.size()];
        for (int k = 0; k < dtr.length; k++) {
            dtr[k] = y[tr.get(k)][t] - y[tr.get(k)][base];
        }
        for (int k = 0; k < dct.length; k++) {
            dct[k] = y[ct.get(k)][t] - y[ct.get(k)][base];
        }
        double mtr = Arrays.stream(dtr).average().orElse(0);
        switch (method) {
            case UNCONDITIONAL -> {
                return mtr - Arrays.stream(dct).average().orElse(0);
            }
            case OUTCOME_REGRESSION -> {
                double[][] xc = new double[ct.size()][1];
                for (int k = 0; k < xc.length; k++) {
                    xc[k][0] = x[ct.get(k)];
                }
                double[] beta = LeastSquares.fit(xc, dct, true, 1e-9).beta();
                double pred = 0;
                for (int i : tr) {
                    pred += (beta[0] + beta[1] * x[i]) / tr.size();
                }
                return mtr - pred;
            }
            case IPW -> {
                int m = tr.size() + ct.size();
                double[][] xs = new double[m][1];
                int[] g = new int[m];
                for (int k = 0; k < tr.size(); k++) {
                    xs[k][0] = x[tr.get(k)];
                    g[k] = 1;
                }
                for (int k = 0; k < ct.size(); k++) {
                    xs[tr.size() + k][0] = x[ct.get(k)];
                }
                double[] p = CausalEstimators.propensity(xs, g, 1e-6);
                double num = 0;
                double den = 0;
                for (int k = 0; k < ct.size(); k++) {
                    double e = Math.min(p[tr.size() + k], 0.99);
                    double w = e / (1 - e);
                    num += w * dct[k];
                    den += w;
                }
                return mtr - num / den;
            }
            default -> throw new IllegalStateException();
        }
    }

    /**
     * Aggregates group-time effects.
     *
     * @param gt the group-time effects
     * @param minEvent the first event time reported
     * @param maxEvent the last event time reported
     * @return {@code {overall ATT, theta(minEvent), ..., theta(maxEvent)}}; the overall ATT
     *     weights every post cell {@code (g, t >= g)} by its cohort size
     */
    public static double[] aggregate(GroupTime gt, int minEvent, int maxEvent) {
        int periods = gt.att()[0].length;
        double num = 0;
        double den = 0;
        double[] out = new double[maxEvent - minEvent + 2];
        double[] en = new double[out.length];
        double[] ed = new double[out.length];
        for (int g = 0; g < gt.cohorts().length; g++) {
            for (int t = 0; t < periods; t++) {
                double v = gt.att()[g][t];
                if (Double.isNaN(v)) {
                    continue;
                }
                int e = t - gt.cohorts()[g];
                if (e >= 0) {
                    num += gt.size()[g] * v;
                    den += gt.size()[g];
                }
                if (e >= minEvent && e <= maxEvent) {
                    en[e - minEvent + 1] += gt.size()[g] * v;
                    ed[e - minEvent + 1] += gt.size()[g];
                }
            }
        }
        out[0] = num / den;
        for (int k = 1; k < out.length; k++) {
            out[k] = ed[k] == 0 ? Double.NaN : en[k] / ed[k];
        }
        return out;
    }

    /**
     * A unit-cluster bootstrap of {@link #aggregate}.
     *
     * @param y outcomes
     * @param adoption adoption times
     * @param covariate the covariate
     * @param method the adjustment
     * @param minEvent first event time
     * @param maxEvent last event time
     * @param replicates bootstrap draws
     * @param seed the seed
     * @return {@code [2][k]}: the 2.5% and 97.5% quantiles of each aggregate
     */
    public static double[][] bootstrap(double[][] y, int[] adoption, double[] covariate, Method method, int minEvent,
            int maxEvent, int replicates, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = y.length;
        int k = maxEvent - minEvent + 2;
        double[][] draws = new double[k][replicates];
        for (int b = 0; b < replicates; b++) {
            double[][] yb = new double[n][];
            int[] ab = new int[n];
            double[] xb = new double[n];
            for (int i = 0; i < n; i++) {
                int j = r.nextInt(n);
                yb[i] = y[j];
                ab[i] = adoption[j];
                xb[i] = covariate[j];
            }
            double[] agg = aggregate(groupTime(yb, ab, xb, method, false), minEvent, maxEvent);
            for (int q = 0; q < k; q++) {
                draws[q][b] = agg[q];
            }
        }
        double[][] out = new double[2][k];
        for (int q = 0; q < k; q++) {
            double[] s = Arrays.stream(draws[q]).filter(v -> !Double.isNaN(v)).sorted().toArray();
            out[0][q] = s.length == 0 ? Double.NaN : s[(int) Math.floor(0.025 * (s.length - 1))];
            out[1][q] = s.length == 0 ? Double.NaN : s[(int) Math.ceil(0.975 * (s.length - 1))];
        }
        return out;
    }

    /**
     * The Sun-Abraham interaction-weighted event study.
     *
     * @param y outcomes
     * @param adoption adoption times ({@code -1} never; there must be never-treated units)
     * @param minEvent first event time reported
     * @param maxEvent last event time reported
     * @return {@code {overall ATT, theta(minEvent), ..., theta(maxEvent)}}
     */
    public static double[] sunAbraham(double[][] y, int[] adoption, int minEvent, int maxEvent) {
        int n = y.length;
        int periods = y[0].length;
        int[] cohorts = Arrays.stream(adoption).filter(a -> a > 0).distinct().sorted().toArray();
        List<int[]> cells = new ArrayList<>();   // {cohort index, event time}
        for (int g = 0; g < cohorts.length; g++) {
            for (int t = 0; t < periods; t++) {
                int e = t - cohorts[g];
                if (e != -1) {
                    cells.add(new int[] {g, e});
                }
            }
        }
        double[][][] reg = new double[cells.size()][n][periods];
        for (int c = 0; c < cells.size(); c++) {
            int gt = cohorts[cells.get(c)[0]];
            int e = cells.get(c)[1];
            for (int i = 0; i < n; i++) {
                if (adoption[i] == gt && gt + e >= 0 && gt + e < periods) {
                    reg[c][i][gt + e] = 1;
                }
            }
        }
        double[] delta = TwoWayFixedEffects.regress(y, reg);
        int[] size = new int[cohorts.length];
        for (int a : adoption) {
            int g = Arrays.binarySearch(cohorts, a);
            if (a > 0 && g >= 0) {
                size[g]++;
            }
        }
        double[] out = new double[maxEvent - minEvent + 2];
        double[] en = new double[out.length];
        double[] ed = new double[out.length];
        double num = 0;
        double den = 0;
        for (int c = 0; c < cells.size(); c++) {
            int g = cells.get(c)[0];
            int e = cells.get(c)[1];
            if (e >= 0) {
                num += size[g] * delta[c];
                den += size[g];
            }
            if (e >= minEvent && e <= maxEvent) {
                en[e - minEvent + 1] += size[g] * delta[c];
                ed[e - minEvent + 1] += size[g];
            }
        }
        out[0] = num / den;
        for (int k = 1; k < out.length; k++) {
            out[k] = minEvent + k - 1 == -1 ? 0 : ed[k] == 0 ? Double.NaN : en[k] / ed[k];
        }
        return out;
    }
}
