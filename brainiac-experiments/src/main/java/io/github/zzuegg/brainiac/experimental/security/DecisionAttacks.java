package io.github.zzuegg.brainiac.experimental.security;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Decision-based attacks that see only the predicted label ({@link QueryOracle#labels}),
 * untargeted, minimising the {@code ℓ2} distance. Both start from an adversarial point (the
 * first of a pool of other images that the model labels differently), binary-search to the
 * decision boundary and then walk along it toward the clean input:
 *
 * <ul>
 *   <li>{@link Method#BOUNDARY}: the boundary attack proposes random orthogonal steps on the
 *       sphere around the clean input followed by a small contraction toward it, keeps an
 *       adversarial proposal and adapts both step sizes to the acceptance rate (Brendel,
 *       Rauber and Bethge, ICLR 2018, <a href="https://arxiv.org/abs/1712.04248">arXiv:1712.04248</a>);
 *       five proposals are queried per step.</li>
 *   <li>{@link Method#HOP_SKIP_JUMP}: HopSkipJump estimates the boundary normal from the
 *       signs of {@code B_t = B_0 √t} label queries at {@code x_t + δ u_b}, with the
 *       baseline-corrected estimator, takes a geometric step of {@code ‖x_t - x‖/√t} along
 *       it (halved until adversarial) and binary-searches back to the boundary (Chen, Jordan
 *       and Wainwright, IEEE S&amp;P 2020, <a href="https://arxiv.org/abs/1904.02144">arXiv:1904.02144</a>).</li>
 * </ul>
 */
public final class DecisionAttacks {

    /** An attack. */
    public enum Method {
        /** the boundary attack */
        BOUNDARY,
        /** HopSkipJump */
        HOP_SKIP_JUMP
    }

    /**
     * The outcome: per example, the distance reached after each query count.
     *
     * @param adversarial the final adversarial inputs (the start when none was found, then
     *     {@code null})
     * @param trace per example, pairs {@code (queries, distance)} in query order
     */
    public record Result(float[][] adversarial, List<List<double[]>> trace) {

        /**
         * The median {@code ℓ2} distance reached within a query budget.
         *
         * @param queries the budget
         * @return the median over examples (infinite for examples with no adversarial yet)
         */
        public double medianDistance(long queries) {
            double[] d = new double[trace.size()];
            for (int i = 0; i < d.length; i++) {
                d[i] = Double.POSITIVE_INFINITY;
                for (double[] e : trace.get(i)) {
                    if (e[0] <= queries) {
                        d[i] = e[1];
                    }
                }
            }
            Arrays.sort(d);
            int n = d.length;
            return n % 2 == 1 ? d[n / 2] : 0.5 * (d[n / 2 - 1] + d[n / 2]);
        }
    }

    private DecisionAttacks() {
    }

    /**
     * Runs an attack.
     *
     * @param method the attack
     * @param model the label oracle
     * @param x clean inputs, correctly classified
     * @param y true classes
     * @param pool candidate starting points (other images)
     * @param budget the query budget per example
     * @param seed the seed
     * @return the result
     */
    public static Result run(Method method, QueryOracle model, float[][] x, int[] y, float[][] pool, int budget,
            long seed) {
        Objects.requireNonNull(method);
        SplittableRandom r = new SplittableRandom(seed);
        float[][] out = new float[x.length][];
        List<List<double[]>> trace = new ArrayList<>();
        for (int i = 0; i < x.length; i++) {
            List<double[]> t = new ArrayList<>();
            trace.add(t);
            float[] start = start(model, i, x[i], y[i], pool, budget, r);
            if (start == null) {
                continue;
            }
            float[] adv = binarySearch(model, i, x[i], y[i], start, 10);
            t.add(new double[] {model.queries(i), distance(adv, x[i])});
            adv = method == Method.BOUNDARY ? boundary(model, i, x[i], y[i], adv, budget, t, r)
                    : hopSkipJump(model, i, x[i], y[i], adv, budget, t, r);
            out[i] = adv;
        }
        return new Result(out, trace);
    }

    private static float[] start(QueryOracle model, int i, float[] x, int y, float[][] pool, int budget,
            SplittableRandom r) {
        for (int k = 0; k < Math.min(pool.length, 20) && model.queries(i) < budget; k++) {
            float[] c = pool[r.nextInt(pool.length)];
            if (adversarial(model, i, c, y)) {
                return c.clone();
            }
        }
        return null;
    }

    private static float[] boundary(QueryOracle model, int i, float[] x, int y, float[] start, int budget,
            List<double[]> trace, SplittableRandom r) {
        float[] adv = start;
        double spherical = 0.05;
        double source = 0.05;
        int proposals = 5;
        int d = x.length;
        while (model.queries(i) + proposals <= budget) {
            double dist = distance(adv, x);
            float[][] cand = new float[proposals][];
            for (int k = 0; k < proposals; k++) {
                float[] eta = BlackBoxAttacks.gaussian(d, r);
                double en = 0;
                for (float e : eta) {
                    en += e * (double) e;
                }
                float[] c = new float[d];
                for (int j = 0; j < d; j++) {
                    c[j] = (float) (adv[j] + spherical * dist * eta[j] / Math.sqrt(en));
                }
                double cn = distance(c, x);
                double shrink = dist / cn * (1 - source);
                for (int j = 0; j < d; j++) {
                    c[j] = (float) (x[j] + (c[j] - x[j]) * shrink);
                }
                cand[k] = c;
            }
            int[] owner = new int[proposals];
            Arrays.fill(owner, i);
            int[] labels = model.labels(cand, owner);
            int accepted = 0;
            float[] best = null;
            for (int k = 0; k < proposals; k++) {
                if (labels[k] != y) {
                    accepted++;
                    if (best == null || distance(cand[k], x) < distance(best, x)) {
                        best = cand[k];
                    }
                }
            }
            if (best != null) {
                adv = best;
                trace.add(new double[] {model.queries(i), distance(adv, x)});
            }
            double rate = accepted / (double) proposals;
            // The source (contraction) step adapts to the acceptance rate; the orthogonal step only
            // shrinks once the contraction is already negligible, as in Brendel et al.'s two
            // separately tuned step sizes (a fixed ratio makes the rate scale-invariant).
            if (rate > 0.4) {
                source *= 1.5;
            } else if (rate < 0.2) {
                source /= 1.5;
                if (source < 1e-4) {
                    spherical /= 1.5;
                }
            }
            source = Math.min(source, 0.5);
            spherical = Math.min(spherical, 0.5);
        }
        return adv;
    }

    private static float[] hopSkipJump(QueryOracle model, int i, float[] x, int y, float[] start, int budget,
            List<double[]> trace, SplittableRandom r) {
        float[] adv = start;
        int d = x.length;
        for (int t = 1; model.queries(i) < budget; t++) {
            double dist = distance(adv, x);
            double delta = Math.max(dist / d, 1e-4);
            int b = (int) Math.min(100, Math.round(20 * Math.sqrt(t)));
            if (model.queries(i) + b + 12 > budget) {
                break;
            }
            float[][] rows = new float[b][];
            float[][] u = new float[b][];
            for (int k = 0; k < b; k++) {
                u[k] = BlackBoxAttacks.gaussian(d, r);
                double n = Math.sqrt(distance(u[k], new float[d]));
                rows[k] = new float[d];
                for (int j = 0; j < d; j++) {
                    u[k][j] /= (float) n;
                    rows[k][j] = (float) (adv[j] + delta * u[k][j]);
                }
            }
            int[] owner = new int[b];
            Arrays.fill(owner, i);
            int[] labels = model.labels(rows, owner);
            double[] phi = new double[b];
            double mean = 0;
            for (int k = 0; k < b; k++) {
                phi[k] = labels[k] != y ? 1 : -1;
                mean += phi[k] / b;
            }
            boolean constant = Math.abs(mean) == 1;
            double[] v = new double[d];
            for (int k = 0; k < b; k++) {
                double w = constant ? phi[k] : phi[k] - mean;
                for (int j = 0; j < d; j++) {
                    v[j] += w * u[k][j] / b;
                }
            }
            double vn = 0;
            for (double e : v) {
                vn += e * e;
            }
            vn = Math.sqrt(Math.max(vn, 1e-24));
            double xi = dist / Math.sqrt(t);
            float[] next = null;
            while (model.queries(i) < budget) {
                float[] c = new float[d];
                for (int j = 0; j < d; j++) {
                    c[j] = (float) (adv[j] + xi * v[j] / vn);
                }
                if (adversarial(model, i, c, y)) {
                    next = c;
                    break;
                }
                xi /= 2;
                if (xi < 1e-6) {
                    break;
                }
            }
            if (next == null) {
                break;
            }
            float[] cand = binarySearch(model, i, x, y, next, 8);
            if (distance(cand, x) < distance(adv, x)) {
                adv = cand;
            }
            trace.add(new double[] {model.queries(i), distance(adv, x)});
        }
        return adv;
    }

    private static float[] binarySearch(QueryOracle model, int i, float[] x, int y, float[] adv, int steps) {
        double lo = 0;
        double hi = 1;
        float[] best = adv;
        for (int s = 0; s < steps; s++) {
            double mid = 0.5 * (lo + hi);
            float[] c = new float[x.length];
            for (int j = 0; j < x.length; j++) {
                c[j] = (float) (x[j] + mid * (adv[j] - x[j]));
            }
            if (adversarial(model, i, c, y)) {
                hi = mid;
                best = c;
            } else {
                lo = mid;
            }
        }
        return best;
    }

    private static boolean adversarial(QueryOracle model, int i, float[] c, int y) {
        return model.labels(new float[][] {c}, new int[] {i})[0] != y;
    }

    /**
     * The Euclidean distance.
     *
     * @param a one point
     * @param b another
     * @return {@code ‖a - b‖₂}
     */
    public static double distance(float[] a, float[] b) {
        return Math.sqrt(RobustAggregators.distance2(a, b));
    }
}
