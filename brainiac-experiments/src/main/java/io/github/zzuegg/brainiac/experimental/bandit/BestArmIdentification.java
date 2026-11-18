package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.experimental.data.BanditSuites;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Pure exploration: find the best arm with as few pulls as possible (fixed confidence) or with
 * the smallest error after a given number of pulls (fixed budget), rather than minimising
 * regret.
 *
 * <ul>
 *   <li>{@link Method#UNIFORM}: round-robin allocation, the naive baseline;</li>
 *   <li>{@link Method#SUCCESSIVE_ELIMINATION} (Even-Dar et al., JMLR 2006): round-robin over the
 *       active arms, dropping every arm whose upper bound falls below the best lower bound;</li>
 *   <li>{@link Method#SUCCESSIVE_HALVING} (Karnin et al., ICML 2013; fixed budget only): log2 K
 *       rounds, each splitting its share of the budget over the survivors and keeping the top
 *       half, the component of core {@code HyperparameterSearch.hyperband};</li>
 *   <li>{@link Method#LUCB} (Kalyanakrishnan et al., ICML 2012): sample the empirical leader and
 *       the arm with the highest upper bound among the rest; stop when the leader's lower bound
 *       clears the challenger's upper bound;</li>
 *   <li>{@link Method#TRACK_AND_STOP} (Garivier &amp; Kaufmann, COLT 2016,
 *       <a href="https://arxiv.org/abs/1602.04589">arXiv:1602.04589</a>): D-tracking of the
 *       optimal proportions {@code w*(mu)} of the lower-bound optimisation (computed by the
 *       paper's nested root finding at the empirical means), with forced exploration;</li>
 *   <li>{@link Method#TOP_TWO_TS} and {@link Method#TOP_TWO_EB} (Russo 2016; Jourdan et al.,
 *       NeurIPS 2022, <a href="https://arxiv.org/abs/2206.05979">arXiv:2206.05979</a>): with
 *       probability 1/2 pull a leader (a Thompson sample's best arm, or the empirical best) and
 *       otherwise the challenger with the smallest transportation cost;</li>
 *   <li>{@link Method#ALMOST_TRACKING} (Komiyama et al., Oct 2025,
 *       <a href="https://arxiv.org/abs/2510.23199">arXiv:2510.23199</a>): anytime batched
 *       tracking of closed-form target proportions for the {@code H1} risk,
 *       {@code w_i = 1 / (D_i Z)} with {@code D_i = gap_i^2 sum_{j != i} (gap_j + gap_i)^-2}
 *       (the best arm taking the smallest {@code D}); each batch goes only to the arms whose
 *       average past allocation is below {@code target / C_suf}, in proportion to their targets.
 *       It needs no budget in advance.</li>
 * </ul>
 *
 * <p>Every fixed-confidence rule other than successive elimination and LUCB stops with the
 * generalised likelihood ratio test {@code Z_t > log((1 + log t) / delta)} of Garivier &amp;
 * Kaufmann.
 */
public final class BestArmIdentification {

    private BestArmIdentification() {
    }

    /** The sampling rule. */
    public enum Method {
        /** round robin */
        UNIFORM,
        /** successive elimination */
        SUCCESSIVE_ELIMINATION,
        /** successive halving (fixed budget only) */
        SUCCESSIVE_HALVING,
        /** lower and upper confidence bounds */
        LUCB,
        /** track-and-stop with D-tracking */
        TRACK_AND_STOP,
        /** top-two with a Thompson-sampling leader */
        TOP_TWO_TS,
        /** top-two with the empirical-best leader */
        TOP_TWO_EB,
        /** anytime almost tracking */
        ALMOST_TRACKING
    }

    /**
     * A fixed-confidence outcome.
     *
     * @param recommendation the arm recommended
     * @param samples pulls taken
     * @param stopped whether the stopping rule fired before the sample cap
     */
    public record Outcome(int recommendation, long samples, boolean stopped) {
    }

    /**
     * The reward model the confidence bounds and divergences assume.
     *
     * @param family Bernoulli or Gaussian
     * @param sigma the Gaussian standard deviation (and the sub-Gaussian scale of the
     *     elimination bounds; 0.5 for Bernoulli)
     */
    public record Model(BanditMath.Family family, double sigma) {

        /** @return Bernoulli rewards */
        public static Model bernoulli() {
            return new Model(BanditMath.Family.BERNOULLI, 0.5);
        }

        /**
         * @param sigma the standard deviation
         * @return Gaussian rewards
         */
        public static Model gaussian(double sigma) {
            return new Model(BanditMath.Family.GAUSSIAN, sigma);
        }

        double d(double p, double q) {
            return BanditMath.kl(family, sigma, p, q);
        }
    }

    private static final class State {
        final BanditSuites.Instance instance;
        final Model model;
        final SplittableRandom random;
        final int k;
        final int[] n;
        final double[] sum;
        long t;
        int round;
        final List<double[]> allocations = new ArrayList<>();
        final List<Integer> queue = new ArrayList<>();
        boolean[] active;
        double[] weights;
        long weightsAt = -1;

        State(BanditSuites.Instance instance, Model model, long seed) {
            this.instance = instance;
            this.model = model;
            this.random = new SplittableRandom(seed);
            this.k = instance.arms();
            this.n = new int[k];
            this.sum = new double[k];
            this.active = new boolean[k];
            java.util.Arrays.fill(active, true);
        }

        void pull(int a) {
            double r = instance.sample(a, (int) Math.min(t, Integer.MAX_VALUE), random);
            n[a]++;
            sum[a] += r;
            t++;
        }

        double mean(int a) {
            return n[a] == 0 ? 0 : sum[a] / n[a];
        }

        int best() {
            int b = 0;
            for (int a = 1; a < k; a++) {
                if (mean(a) > mean(b)) {
                    b = a;
                }
            }
            return b;
        }

        double cost(int leader, int b) {
            double ml = mean(leader);
            double mb = mean(b);
            if (mb >= ml) {
                return 0;
            }
            double m = (n[leader] * ml + n[b] * mb) / (n[leader] + n[b]);
            return n[leader] * model.d(ml, m) + n[b] * model.d(mb, m);
        }

        double glr() {
            int l = best();
            double z = Double.POSITIVE_INFINITY;
            for (int b = 0; b < k; b++) {
                if (b != l) {
                    z = Math.min(z, cost(l, b));
                }
            }
            return z;
        }

        double radius(int a, double level) {
            return model.sigma() * Math.sqrt(2 * level / n[a]);
        }
    }

    /**
     * Runs a sampling rule until it is confident of the best arm.
     *
     * @param method the rule (not {@link Method#SUCCESSIVE_HALVING})
     * @param instance the bandit
     * @param model the reward model
     * @param delta the error probability allowed
     * @param maxSamples the most pulls before giving up
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome fixedConfidence(Method method, BanditSuites.Instance instance, Model model, double delta,
            long maxSamples, long seed) {
        Objects.requireNonNull(method, "method");
        if (method == Method.SUCCESSIVE_HALVING) {
            throw new IllegalArgumentException("successive halving is a fixed-budget rule");
        }
        State s = new State(instance, model, seed);
        for (int a = 0; a < s.k; a++) {
            s.pull(a);
        }
        while (s.t < maxSamples) {
            if (method == Method.SUCCESSIVE_ELIMINATION) {
                if (eliminationRound(s, delta)) {
                    return new Outcome(onlyActive(s), s.t, true);
                }
                continue;
            }
            if (method == Method.LUCB) {
                int h = s.best();
                double level = Math.log(5 * s.k * Math.pow(s.t, 4) / (4 * delta));
                int l = -1;
                double ucb = Double.NEGATIVE_INFINITY;
                for (int a = 0; a < s.k; a++) {
                    if (a != h && s.mean(a) + s.radius(a, level) > ucb) {
                        ucb = s.mean(a) + s.radius(a, level);
                        l = a;
                    }
                }
                if (s.mean(h) - s.radius(h, level) > ucb) {
                    return new Outcome(h, s.t, true);
                }
                s.pull(h);
                s.pull(l);
                continue;
            }
            if (s.glr() > Math.log((1 + Math.log(s.t)) / delta)) {
                return new Outcome(s.best(), s.t, true);
            }
            step(method, s);
        }
        return new Outcome(method == Method.SUCCESSIVE_ELIMINATION ? bestActive(s) : s.best(), s.t, false);
    }

    /**
     * Runs a sampling rule for a fixed number of pulls and recommends the empirical best arm.
     *
     * @param method the rule (successive elimination and LUCB keep sampling past their stopping time)
     * @param instance the bandit
     * @param model the reward model
     * @param budget the pulls
     * @param seed the seed
     * @return the recommended arm
     */
    public static int fixedBudget(Method method, BanditSuites.Instance instance, Model model, long budget, long seed) {
        State s = new State(instance, model, seed);
        if (method == Method.SUCCESSIVE_HALVING) {
            return successiveHalving(s, budget);
        }
        for (int a = 0; a < s.k && s.t < budget; a++) {
            s.pull(a);
        }
        while (s.t < budget) {
            switch (method) {
                case SUCCESSIVE_ELIMINATION -> {
                    eliminationRound(s, 0.05);
                    if (onlyActive(s) >= 0) {
                        java.util.Arrays.fill(s.active, true);
                    }
                }
                case LUCB -> {
                    int h = s.best();
                    double level = Math.log(5 * s.k * Math.pow(s.t, 4) / 0.2);
                    int l = -1;
                    double ucb = Double.NEGATIVE_INFINITY;
                    for (int a = 0; a < s.k; a++) {
                        if (a != h && s.mean(a) + s.radius(a, level) > ucb) {
                            ucb = s.mean(a) + s.radius(a, level);
                            l = a;
                        }
                    }
                    s.pull(h);
                    if (s.t < budget) {
                        s.pull(l);
                    }
                }
                default -> step(method, s);
            }
        }
        return method == Method.SUCCESSIVE_ELIMINATION ? bestActive(s) : s.best();
    }

    private static int onlyActive(State s) {
        int found = -1;
        for (int a = 0; a < s.k; a++) {
            if (s.active[a]) {
                if (found >= 0) {
                    return -1;
                }
                found = a;
            }
        }
        return found;
    }

    private static int bestActive(State s) {
        int b = -1;
        for (int a = 0; a < s.k; a++) {
            if (s.active[a] && (b < 0 || s.mean(a) > s.mean(b))) {
                b = a;
            }
        }
        return b;
    }

    /** One round-robin pass over the active arms, then elimination; true when one arm is left. */
    private static boolean eliminationRound(State s, double delta) {
        for (int a = 0; a < s.k; a++) {
            if (s.active[a]) {
                s.pull(a);
            }
        }
        int m = 0;
        for (int a = 0; a < s.k; a++) {
            if (s.active[a]) {
                m = Math.max(m, s.n[a]);
            }
        }
        double level = Math.log(4 * s.k * (double) m * m / delta);
        double bestLower = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < s.k; a++) {
            if (s.active[a]) {
                bestLower = Math.max(bestLower, s.mean(a) - s.radius(a, level));
            }
        }
        for (int a = 0; a < s.k; a++) {
            if (s.active[a] && s.mean(a) + s.radius(a, level) < bestLower) {
                s.active[a] = false;
            }
        }
        return onlyActive(s) >= 0;
    }

    private static int successiveHalving(State s, long budget) {
        List<Integer> alive = new ArrayList<>();
        for (int a = 0; a < s.k; a++) {
            alive.add(a);
        }
        int rounds = (int) Math.ceil(Math.log(s.k) / Math.log(2));
        for (int r = 0; r < rounds && alive.size() > 1; r++) {
            long per = Math.max(1, budget / ((long) alive.size() * rounds));
            for (int a : alive) {
                for (long i = 0; i < per; i++) {
                    s.pull(a);
                }
            }
            alive.sort((a, b) -> Double.compare(s.mean(b), s.mean(a)));
            alive = new ArrayList<>(alive.subList(0, (alive.size() + 1) / 2));
        }
        return alive.get(0);
    }

    private static void step(Method method, State s) {
        switch (method) {
            case UNIFORM -> s.pull((int) (s.t % s.k));
            case TRACK_AND_STOP -> {
                int min = 0;
                for (int a = 1; a < s.k; a++) {
                    if (s.n[a] < s.n[min]) {
                        min = a;
                    }
                }
                if (s.n[min] < Math.sqrt(s.t) - s.k / 2.0) {
                    s.pull(min);
                    return;
                }
                double[] w = optimalWeights(s);
                int a = 0;
                double gap = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < s.k; i++) {
                    double g = s.t * w[i] - s.n[i];
                    if (g > gap) {
                        gap = g;
                        a = i;
                    }
                }
                s.pull(a);
            }
            case TOP_TWO_TS, TOP_TWO_EB -> {
                int leader = method == Method.TOP_TWO_EB ? s.best() : thompsonLeader(s);
                if (s.random.nextBoolean()) {
                    s.pull(leader);
                    return;
                }
                int challenger = -1;
                double c = Double.POSITIVE_INFINITY;
                for (int b = 0; b < s.k; b++) {
                    if (b != leader) {
                        double v = s.cost(leader, b);
                        if (v < c) {
                            c = v;
                            challenger = b;
                        }
                    }
                }
                s.pull(challenger);
            }
            case ALMOST_TRACKING -> {
                if (s.queue.isEmpty()) {
                    almostTrackingBatch(s);
                }
                s.pull(s.queue.remove(s.queue.size() - 1));
            }
            default -> throw new IllegalArgumentException(method + " has no single-step rule");
        }
    }

    private static int thompsonLeader(State s) {
        double[] draw = new double[s.k];
        for (int a = 0; a < s.k; a++) {
            if (s.model.family() == BanditMath.Family.BERNOULLI) {
                draw[a] = BanditMath.beta(1 + s.sum[a], 1 + s.n[a] - s.sum[a], s.random);
            } else {
                draw[a] = s.mean(a) + s.model.sigma() / Math.sqrt(s.n[a]) * s.random.nextGaussian();
            }
        }
        return BanditMath.argmax(draw, s.random);
    }

    /**
     * The closed-form H1 targets of Almost Tracking at the given means.
     *
     * @param means the empirical means
     * @return the target proportions
     */
    public static double[] almostTrackingTargets(double[] means) {
        int k = means.length;
        int best = 0;
        for (int a = 1; a < k; a++) {
            if (means[a] > means[best]) {
                best = a;
            }
        }
        double[] gap = new double[k];
        for (int a = 0; a < k; a++) {
            gap[a] = means[best] - means[a];
        }
        double[] d = new double[k];
        double minD = Double.POSITIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            if (i == best) {
                continue;
            }
            if (gap[i] <= 0) {
                double[] u = new double[k];
                java.util.Arrays.fill(u, 1.0 / k);
                return u;
            }
            double h = 0;
            for (int j = 0; j < k; j++) {
                if (j != i) {
                    h += 1 / Math.pow(gap[j] + gap[i], 2);
                }
            }
            d[i] = gap[i] * gap[i] * h;
            minD = Math.min(minD, d[i]);
        }
        d[best] = minD;
        double z = 0;
        for (double v : d) {
            z += 1 / v;
        }
        double[] w = new double[k];
        for (int i = 0; i < k; i++) {
            w[i] = 1 / (d[i] * z);
        }
        return w;
    }

    private static void almostTrackingBatch(State s) {
        int batch = Math.max(2 * s.k, 10);
        double cSuf = 0.9;
        double[] means = new double[s.k];
        for (int a = 0; a < s.k; a++) {
            means[a] = s.mean(a);
        }
        double[] target = almostTrackingTargets(means);
        double[] past = new double[s.k];
        for (double[] w : s.allocations) {
            for (int a = 0; a < s.k; a++) {
                past[a] += w[a] / s.allocations.size();
            }
        }
        if (s.allocations.isEmpty()) {
            java.util.Arrays.fill(past, 1.0 / s.k);
        }
        double total = 0;
        boolean[] insufficient = new boolean[s.k];
        for (int a = 0; a < s.k; a++) {
            if (past[a] <= target[a] / cSuf) {
                insufficient[a] = true;
                total += target[a];
            }
        }
        double[] alloc = new double[s.k];
        for (int a = 0; a < s.k; a++) {
            alloc[a] = insufficient[a] ? target[a] / total : 0;
            int pulls = (int) Math.ceil(alloc[a] * batch - 1e-9);
            for (int i = 0; i < pulls; i++) {
                s.queue.add(a);
            }
        }
        java.util.Collections.shuffle(s.queue, new java.util.Random(s.random.nextLong()));
        s.allocations.add(alloc);
    }

    /**
     * The optimal sampling proportions of the fixed-confidence lower bound (Garivier &amp;
     * Kaufmann's Theorem 5), by nested bisection.
     *
     * @param means the arm means
     * @param model the reward model
     * @return the proportions
     */
    public static double[] optimalWeights(double[] means, Model model) {
        int k = means.length;
        int best = 0;
        for (int a = 1; a < k; a++) {
            if (means[a] > means[best]) {
                best = a;
            }
        }
        double mu1 = means[best];
        for (int a = 0; a < k; a++) {
            if (a != best && means[a] >= mu1 - 1e-12) {
                double[] u = new double[k];
                java.util.Arrays.fill(u, 1.0 / k);
                return u;
            }
        }
        double yMax = Double.POSITIVE_INFINITY;
        for (int a = 0; a < k; a++) {
            if (a != best) {
                yMax = Math.min(yMax, model.d(mu1, means[a]));
            }
        }
        double lo = 0;
        double hi = yMax * (1 - 1e-9);
        double[] x = new double[k];
        for (int it = 0; it < 45; it++) {
            double y = (lo + hi) / 2;
            double f = 0;
            for (int a = 0; a < k; a++) {
                if (a == best) {
                    continue;
                }
                x[a] = solveX(mu1, means[a], y, model);
                double m = (mu1 + x[a] * means[a]) / (1 + x[a]);
                f += model.d(mu1, m) / Math.max(model.d(means[a], m), 1e-300);
            }
            if (f > 1) {
                hi = y;
            } else {
                lo = y;
            }
        }
        double y = (lo + hi) / 2;
        double total = 1;
        for (int a = 0; a < k; a++) {
            if (a != best) {
                x[a] = solveX(mu1, means[a], y, model);
                total += x[a];
            }
        }
        double[] w = new double[k];
        for (int a = 0; a < k; a++) {
            w[a] = (a == best ? 1 : x[a]) / total;
        }
        return w;
    }

    /** Solves {@code d(mu1, m) + x d(mua, m) = y} for {@code x}, m the weighted mean. */
    private static double solveX(double mu1, double mua, double y, Model model) {
        double lo = 0;
        double hi = 1;
        while (g(mu1, mua, hi, model) < y && hi < 1e12) {
            hi *= 2;
        }
        for (int it = 0; it < 60; it++) {
            double mid = (lo + hi) / 2;
            if (g(mu1, mua, mid, model) < y) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) / 2;
    }

    private static double g(double mu1, double mua, double x, Model model) {
        double m = (mu1 + x * mua) / (1 + x);
        return model.d(mu1, m) + x * model.d(mua, m);
    }

    private static double[] optimalWeights(State s) {
        if (s.weights != null && s.t - s.weightsAt < Math.max(s.k, s.t / 50)) {
            return s.weights;
        }
        s.weightsAt = s.t;
        double[] means = new double[s.k];
        for (int a = 0; a < s.k; a++) {
            means[a] = s.mean(a);
        }
        s.weights = optimalWeights(means, s.model);
        return s.weights;
    }
}
