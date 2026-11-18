package io.github.zzuegg.brainiac.experimental.explain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Anchors: high-precision rule explanations found by a bandit-guided beam search (Ribeiro, Singh
 * and Guestrin, "Anchors: High-Precision Model-Agnostic Explanations", AAAI 2018).
 *
 * <p>An anchor {@code A} is a conjunction of interval predicates on features that hold for the
 * explained row {@code x}, such that the model's prediction stays {@code f(x)} with probability
 * at least {@code tau} under the perturbation distribution {@code D(z | A)}; among such anchors
 * the one with the largest coverage (the share of data rows satisfying {@code A}) is preferred.
 * Predicates come from quantile bins of the training data: for the bin containing {@code x_j},
 * "{@code x_j <= upper edge}" and "{@code x_j > lower edge}". {@code D(z | A)} draws a training
 * row and replaces every feature that violates a predicate by a training value of that feature
 * that satisfies it.
 *
 * <p>The search grows anchors one predicate at a time, keeping a beam of the best candidates.
 * Choosing the best candidates is a best-arm identification problem whose arms are Bernoulli
 * precisions: {@link Sampler#KL_LUCB} (Kaufmann and Kalyanakrishnan, COLT 2013) samples only the
 * two arms whose KL confidence bounds overlap the most, so model calls go where precision is
 * uncertain; {@link Sampler#UNIFORM} samples every candidate equally until the same confidence
 * condition holds, the baseline. A candidate is accepted when its KL lower bound reaches
 * {@code tau}.
 */
public final class Anchors {

    /** How candidate precisions are sampled. */
    public enum Sampler {
        /** KL-LUCB best-arm identification. */
        KL_LUCB,
        /** Equal samples per candidate. */
        UNIFORM
    }

    /**
     * A predicate {@code lower < x_feature <= upper}.
     *
     * @param feature the feature
     * @param lower the exclusive lower bound (may be negative infinity)
     * @param upper the inclusive upper bound (may be positive infinity)
     */
    public record Predicate(int feature, double lower, double upper) {

        /**
         * @param row a row
         * @return whether the row satisfies the predicate
         */
        public boolean holds(double[] row) {
            return row[feature] > lower && row[feature] <= upper;
        }
    }

    /**
     * An anchor.
     *
     * @param predicates its conjunction
     * @param precision the estimated precision under {@code D(z | A)}
     * @param coverage the share of the training rows that satisfy it
     * @param calls model calls spent on the search
     */
    public record Anchor(List<Predicate> predicates, double precision, double coverage, long calls) {

        /**
         * @param row a row
         * @return whether all predicates hold
         */
        public boolean holds(double[] row) {
            for (Predicate p : predicates) {
                if (!p.holds(row)) {
                    return false;
                }
            }
            return true;
        }

        /** @return the distinct features named */
        public int[] features() {
            return predicates.stream().mapToInt(Predicate::feature).distinct().sorted().toArray();
        }
    }

    private final Function<double[][], int[]> model;
    private final double[][] data;
    private final double[][] sortedColumns;
    private final double[][] edges;
    private final double tau;
    private final double delta;
    private final double epsilon;
    private final int beam;
    private final int batch;
    private final Sampler sampler;

    /**
     * Creates the explainer.
     *
     * @param model the classifier, a batch of rows to predicted classes
     * @param data training rows: the perturbation pool, the bins and the coverage reference
     * @param bins quantile bins per feature, such as 4
     * @param tau the precision threshold, such as 0.95
     * @param delta the confidence parameter, such as 0.1
     * @param beam the beam width, such as 2
     * @param sampler how precisions are estimated
     */
    public Anchors(Function<double[][], int[]> model, double[][] data, int bins, double tau, double delta, int beam,
            Sampler sampler) {
        this.model = model;
        this.data = data;
        this.tau = tau;
        this.delta = delta;
        this.epsilon = 0.1;
        this.beam = beam;
        this.batch = 10;
        this.sampler = sampler;
        int p = data[0].length;
        sortedColumns = new double[p][];
        edges = new double[p][];
        for (int j = 0; j < p; j++) {
            double[] c = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                c[i] = data[i][j];
            }
            Arrays.sort(c);
            sortedColumns[j] = c;
            double[] e = new double[bins - 1];
            for (int b = 1; b < bins; b++) {
                e[b - 1] = c[(int) ((long) b * (c.length - 1) / bins)];
            }
            edges[j] = Arrays.stream(e).distinct().toArray();
        }
    }

    /**
     * The predicates of the bin of {@code x_j}: the upper and the lower edge (where they exist).
     *
     * @param x the row
     * @return the candidate predicates
     */
    public List<Predicate> predicates(double[] x) {
        List<Predicate> out = new ArrayList<>();
        for (int j = 0; j < x.length; j++) {
            double[] e = edges[j];
            int b = 0;
            while (b < e.length && x[j] > e[b]) {
                b++;
            }
            if (b < e.length) {
                out.add(new Predicate(j, Double.NEGATIVE_INFINITY, e[b]));
            }
            if (b > 0) {
                out.add(new Predicate(j, e[b - 1], Double.POSITIVE_INFINITY));
            }
        }
        return out;
    }

    /** One bandit arm: a candidate anchor and its precision counts. */
    private static final class Arm {
        final List<Predicate> predicates;
        long n;
        long hits;
        double coverage;

        Arm(List<Predicate> predicates) {
            this.predicates = predicates;
        }

        double mean() {
            return n == 0 ? 0 : hits / (double) n;
        }
    }

    private long calls;

    /** Draws {@code count} rows from {@code D(z | A)} and counts agreements with the target class. */
    private void sample(Arm arm, int count, int target, SplittableRandom random) {
        double[][] rows = new double[count][];
        for (int k = 0; k < count; k++) {
            double[] z = data[random.nextInt(data.length)].clone();
            for (Predicate p : arm.predicates) {
                if (!p.holds(z)) {
                    z[p.feature()] = draw(p, random);
                }
            }
            rows[k] = z;
        }
        int[] y = model.apply(rows);
        calls += count;
        for (int v : y) {
            if (v == target) {
                arm.hits++;
            }
        }
        arm.n += count;
    }

    private double draw(Predicate p, SplittableRandom random) {
        double[] c = sortedColumns[p.feature()];
        int lo = upperIndex(c, p.lower());
        int hi = upperIndex(c, p.upper());
        if (hi <= lo) {
            return Double.isInfinite(p.upper()) ? p.lower() + 1e-9 : p.upper();
        }
        return c[lo + random.nextInt(hi - lo)];
    }

    /** The number of entries {@code <= v}. */
    private static int upperIndex(double[] sorted, double v) {
        if (v == Double.NEGATIVE_INFINITY) {
            return 0;
        }
        if (v == Double.POSITIVE_INFINITY) {
            return sorted.length;
        }
        int lo = 0;
        int hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    private double coverage(List<Predicate> predicates) {
        int c = 0;
        outer:
        for (double[] row : data) {
            for (Predicate p : predicates) {
                if (!p.holds(row)) {
                    continue outer;
                }
            }
            c++;
        }
        return c / (double) data.length;
    }

    private static double kl(double p, double q) {
        p = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
        q = Math.min(Math.max(q, 1e-12), 1 - 1e-12);
        return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
    }

    /** The KL upper confidence bound: the largest q with {@code n KL(mean, q) <= level}. */
    static double upper(double mean, long n, double level) {
        if (n == 0) {
            return 1;
        }
        double lo = mean;
        double hi = 1;
        for (int it = 0; it < 30; it++) {
            double q = (lo + hi) / 2;
            if (n * kl(mean, q) > level) {
                hi = q;
            } else {
                lo = q;
            }
        }
        return lo;
    }

    /** The KL lower confidence bound. */
    static double lower(double mean, long n, double level) {
        if (n == 0) {
            return 0;
        }
        double lo = 0;
        double hi = mean;
        for (int it = 0; it < 30; it++) {
            double q = (lo + hi) / 2;
            if (n * kl(mean, q) > level) {
                lo = q;
            } else {
                hi = q;
            }
        }
        return hi;
    }

    private double beta(int arms, long t) {
        return Math.log(1.405 * arms * Math.pow(Math.max(t, 1), 1.1) / delta);
    }

    /** Picks the {@code top} arms of highest precision with the configured sampler. */
    private List<Arm> bestArms(List<Arm> arms, int top, int target, SplittableRandom random) {
        int k = arms.size();
        if (k <= top) {
            for (Arm a : arms) {
                sample(a, batch, target, random);
            }
            return arms;
        }
        for (Arm a : arms) {
            sample(a, batch, target, random);
        }
        long t = 1;
        for (int round = 0; round < 2000; round++, t++) {
            List<Arm> sorted = new ArrayList<>(arms);
            sorted.sort(Comparator.comparingDouble(Arm::mean).reversed());
            List<Arm> best = sorted.subList(0, top);
            List<Arm> rest = sorted.subList(top, k);
            double level = beta(k, t);
            Arm lowBest = null;
            double lcb = Double.POSITIVE_INFINITY;
            for (Arm a : best) {
                double v = lower(a.mean(), a.n, level);
                if (v < lcb) {
                    lcb = v;
                    lowBest = a;
                }
            }
            Arm highRest = null;
            double ucb = Double.NEGATIVE_INFINITY;
            for (Arm a : rest) {
                double v = upper(a.mean(), a.n, level);
                if (v > ucb) {
                    ucb = v;
                    highRest = a;
                }
            }
            if (ucb - lcb < epsilon) {
                return new ArrayList<>(best);
            }
            if (sampler == Sampler.KL_LUCB) {
                sample(lowBest, batch, target, random);
                sample(highRest, batch, target, random);
            } else {
                for (Arm a : arms) {
                    sample(a, batch, target, random);
                }
            }
        }
        List<Arm> sorted = new ArrayList<>(arms);
        sorted.sort(Comparator.comparingDouble(Arm::mean).reversed());
        return new ArrayList<>(sorted.subList(0, top));
    }

    /** Samples an arm until its lower bound passes tau or its upper bound falls below it. */
    private boolean verify(Arm arm, int target, SplittableRandom random) {
        for (int round = 0; round < 60; round++) {
            double level = Math.log(1 / delta) + Math.log(1 + round);
            if (arm.n > 0 && lower(arm.mean(), arm.n, level) >= tau) {
                return true;
            }
            if (arm.n > 0 && upper(arm.mean(), arm.n, level) < tau) {
                return false;
            }
            sample(arm, batch, target, random);
        }
        return arm.mean() >= tau;
    }

    /**
     * Finds an anchor for one row.
     *
     * @param x the row
     * @param maxSize the most predicates
     * @param seed the sampling seed
     * @return the anchor (the best found if none reaches {@code tau})
     */
    public Anchor explain(double[] x, int maxSize, long seed) {
        calls = 0;
        SplittableRandom random = new SplittableRandom(seed);
        int target = model.apply(new double[][] {x})[0];
        calls++;
        List<Predicate> all = predicates(x);
        List<Arm> current = List.of(new Arm(List.of()));
        Arm bestValid = null;
        Arm bestSeen = null;
        for (int size = 1; size <= maxSize; size++) {
            List<Arm> candidates = new ArrayList<>();
            for (Arm parent : current) {
                for (Predicate p : all) {
                    boolean used = parent.predicates.stream().anyMatch(q -> q.feature() == p.feature()
                            && q.lower() == p.lower() && q.upper() == p.upper());
                    if (used) {
                        continue;
                    }
                    List<Predicate> next = new ArrayList<>(parent.predicates);
                    next.add(p);
                    if (candidates.stream().anyMatch(c -> c.predicates.containsAll(next)
                            && next.containsAll(c.predicates))) {
                        continue;
                    }
                    Arm a = new Arm(List.copyOf(next));
                    a.coverage = coverage(next);
                    if (bestValid != null && a.coverage <= bestValid.coverage) {
                        continue;
                    }
                    candidates.add(a);
                }
            }
            if (candidates.isEmpty()) {
                break;
            }
            List<Arm> best = bestArms(candidates, Math.min(beam, candidates.size()), target, random);
            for (Arm a : best) {
                if (bestSeen == null || a.mean() > bestSeen.mean()) {
                    bestSeen = a;
                }
                if (verify(a, target, random) && (bestValid == null || a.coverage > bestValid.coverage)) {
                    bestValid = a;
                }
            }
            if (bestValid != null) {
                break;
            }
            current = best;
        }
        Arm chosen = bestValid != null ? bestValid : bestSeen;
        return new Anchor(chosen.predicates, chosen.mean(), chosen.coverage, calls);
    }

    /**
     * The bin predicates of chosen features, the rule that a feature-ranking explanation (LIME or
     * KernelSHAP top-k) turns into.
     *
     * @param x the row
     * @param features the features to anchor
     * @return the rule
     */
    public Anchor ruleOf(double[] x, int[] features) {
        List<Predicate> out = new ArrayList<>();
        for (Predicate p : predicates(x)) {
            for (int f : features) {
                if (p.feature() == f) {
                    out.add(p);
                }
            }
        }
        return new Anchor(List.copyOf(out), Double.NaN, coverage(out), 0);
    }
}
