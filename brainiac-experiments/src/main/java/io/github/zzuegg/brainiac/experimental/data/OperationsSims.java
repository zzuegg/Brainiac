package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Small generators for operations research and decision-making experiments (wave 12, WP6):
 *
 * <ul>
 *   <li>{@link #newsvendor}: feature-driven demand (seasonality, promotions) with log-normal or
 *       heavy-tailed Pareto noise, with the true conditional quantile for an oracle;</li>
 *   <li>{@link Inventory}: a single-echelon lost-sales inventory problem with lead time, holding
 *       and lost-sales costs and a discrete demand distribution;</li>
 *   <li>{@link #pricing}: price-demand curves (linear, logit, constant elasticity) with unknown
 *       parameters over a price grid, and {@link #valuations} for contextual pricing;</li>
 *   <li>{@link #jobShop}: Taillard-style job-shop instances with a makespan lower bound;</li>
 *   <li>{@link #binItems}: online bin-packing item streams (uniform, few discrete sizes, and
 *       the classic First Fit adversary);</li>
 *   <li>{@link #market}: two-sided matching markets with unknown player preferences;</li>
 *   <li>{@link #restlessArms}: two-state "adherence" restless arms;</li>
 *   <li>{@link #knapsackArms}: bandits with resource consumption;</li>
 *   <li>{@link #deepSeaTreasure} and {@link #resourceGathering}: deterministic two-objective
 *       grid MDPs ({@link VectorMdp}).</li>
 * </ul>
 *
 * <p>Every generator is deterministic for a given seed.
 */
public final class OperationsSims {

    private OperationsSims() {
    }

    // ------------------------------------------------------------------ newsvendor

    /**
     * Feature-driven demand samples {@code d = m(x) * eps}.
     *
     * @param features {@code [n][dims]} features in {@code [0, 1]} (feature 0 a season phase, feature 1 a promotion flag)
     * @param demand the realised demand
     * @param mean the multiplicative base {@code m(x)}
     * @param heavyTailed whether {@code eps} is Pareto (else log-normal)
     * @param spread the log-normal sigma or the Pareto shape
     */
    public record Demand(double[][] features, double[] demand, double[] mean, boolean heavyTailed, double spread) {
        /**
         * @param i a sample
         * @param tau a quantile level
         * @return the true conditional {@code tau}-quantile of sample {@code i}'s demand
         */
        public double quantile(int i, double tau) {
            return mean[i] * noiseQuantile(tau);
        }

        /**
         * @param tau a quantile level
         * @return the quantile of the multiplicative noise
         */
        public double noiseQuantile(double tau) {
            if (heavyTailed) {
                double xm = (spread - 1) / spread;
                return xm * Math.pow(1 - tau, -1 / spread);
            }
            return Math.exp(spread * normalQuantile(tau) - spread * spread / 2);
        }

        /** @return the number of samples */
        public int size() {
            return demand.length;
        }
    }

    /**
     * Newsvendor demand: base {@code 20 (1 + 0.6 sin(2 pi x0)) (1 + 0.8 x1) (1 + x2^2)} with promotion
     * flag {@code x1 in {0, 1}} and further features as noise, times unit-mean noise.
     *
     * @param n samples
     * @param dims features (at least 3)
     * @param heavyTailed Pareto noise (shape 2.5) instead of log-normal (sigma 0.4)
     * @param seed the seed
     * @return the samples
     */
    public static Demand newsvendor(int n, int dims, boolean heavyTailed, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = Math.max(3, dims);
        double[][] x = new double[n][d];
        double[] y = new double[n];
        double[] m = new double[n];
        double spread = heavyTailed ? 2.5 : 0.4;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextDouble();
            }
            x[i][1] = r.nextDouble() < 0.3 ? 1 : 0;
            m[i] = 20 * (1 + 0.6 * Math.sin(2 * Math.PI * x[i][0])) * (1 + 0.8 * x[i][1]) * (1 + x[i][2] * x[i][2]);
            double eps;
            if (heavyTailed) {
                double xm = (spread - 1) / spread;
                eps = xm * Math.pow(1 - r.nextDouble(), -1 / spread);
            } else {
                eps = Math.exp(spread * r.nextGaussian() - spread * spread / 2);
            }
            y[i] = m[i] * eps;
        }
        return new Demand(x, y, m, heavyTailed, spread);
    }

    // ------------------------------------------------------------------ multi-period inventory

    /**
     * A lost-sales inventory problem: each period the order placed {@code leadTime} periods ago
     * arrives, demand is served from stock (unmet demand is lost at {@code penalty} per unit) and
     * left-over stock pays {@code holding} per unit.
     *
     * @param leadTime periods between order and arrival (at least 1)
     * @param holding holding cost per unit and period
     * @param penalty lost-sales cost per unit
     * @param demandPmf the demand distribution over {@code 0..k}
     * @param maxOrder the largest order
     * @param maxStock the stock cap (for dynamic programming)
     */
    public record Inventory(int leadTime, double holding, double penalty, double[] demandPmf, int maxOrder,
            int maxStock) {
        /** @return the mean demand */
        public double meanDemand() {
            double m = 0;
            for (int k = 0; k < demandPmf.length; k++) {
                m += k * demandPmf[k];
            }
            return m;
        }

        /**
         * @param random the source
         * @return one demand draw
         */
        public int sample(SplittableRandom random) {
            double u = random.nextDouble();
            for (int k = 0; k < demandPmf.length; k++) {
                u -= demandPmf[k];
                if (u <= 0) {
                    return k;
                }
            }
            return demandPmf.length - 1;
        }
    }

    /**
     * A lost-sales instance with Poisson-like demand of mean about 2 truncated to {@code 0..5}
     * (the Zipkin 2008 style used to benchmark base-stock against the optimum).
     *
     * @param leadTime the lead time
     * @param penalty the lost-sales cost (holding cost 1)
     * @return the instance
     */
    public static Inventory lostSales(int leadTime, double penalty) {
        double[] pmf = new double[6];
        double lambda = 2;
        double s = 0;
        for (int k = 0; k < pmf.length; k++) {
            pmf[k] = Math.exp(-lambda) * Math.pow(lambda, k) / factorial(k);
            s += pmf[k];
        }
        for (int k = 0; k < pmf.length; k++) {
            pmf[k] /= s;
        }
        return new Inventory(Math.max(1, leadTime), 1, penalty, pmf, 6, 10);
    }

    private static double factorial(int k) {
        double f = 1;
        for (int i = 2; i <= k; i++) {
            f *= i;
        }
        return f;
    }

    // ------------------------------------------------------------------ pricing

    /** Price-demand model families. */
    public enum DemandModel {
        /** purchase probability {@code a - b p}, clipped to {@code [0, 1]} */
        LINEAR,
        /** purchase probability {@code sigmoid(a - b p)} */
        LOGIT,
        /** purchase probability {@code a p^-b}, clipped to {@code [0, 1]} */
        CONSTANT_ELASTICITY
    }

    /**
     * A pricing problem: each period one customer buys at the posted price with an unknown
     * probability {@code d(p)}.
     *
     * @param model the curve family
     * @param a the intercept / scale parameter
     * @param b the slope / elasticity parameter
     * @param prices the allowed prices
     */
    public record Pricing(DemandModel model, double a, double b, double[] prices) {
        /**
         * @param p a price
         * @return the purchase probability at it
         */
        public double demand(double p) {
            double d = switch (model) {
                case LINEAR -> a - b * p;
                case LOGIT -> 1 / (1 + Math.exp(-(a - b * p)));
                case CONSTANT_ELASTICITY -> a * Math.pow(p, -b);
            };
            return Math.max(0, Math.min(1, d));
        }

        /** @return the expected revenue of every price */
        public double[] revenues() {
            double[] r = new double[prices.length];
            for (int k = 0; k < r.length; k++) {
                r[k] = prices[k] * demand(prices[k]);
            }
            return r;
        }

        /** @return the clairvoyant best expected revenue per period */
        public double bestRevenue() {
            return Arrays.stream(revenues()).max().orElse(0);
        }
    }

    /**
     * A random pricing problem with prices evenly spaced in {@code [1, 10]}.
     *
     * @param model the family
     * @param pricePoints the number of prices (5-20)
     * @param seed the seed
     * @return the problem
     */
    public static Pricing pricing(DemandModel model, int pricePoints, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] prices = new double[pricePoints];
        for (int k = 0; k < pricePoints; k++) {
            prices[k] = 1 + 9.0 * k / Math.max(1, pricePoints - 1);
        }
        return switch (model) {
            case LINEAR -> new Pricing(model, 0.9 + 0.1 * r.nextDouble(), 0.06 + 0.06 * r.nextDouble(), prices);
            case LOGIT -> new Pricing(model, 2 + 2 * r.nextDouble(), 0.4 + 0.4 * r.nextDouble(), prices);
            case CONSTANT_ELASTICITY -> new Pricing(model, 1, 0.5 + 0.8 * r.nextDouble(), prices);
        };
    }

    /**
     * Contextual valuations {@code v = theta . x + noise} with logistic noise of scale {@code s};
     * a customer buys when the price is at most the valuation.
     *
     * @param theta the valuation weights
     * @param scale the logistic noise scale
     */
    public record Valuations(double[] theta, double scale) {
        /**
         * @param x a context
         * @param p a price
         * @return the purchase probability
         */
        public double buy(double[] x, double p) {
            double v = 0;
            for (int j = 0; j < x.length; j++) {
                v += theta[j] * x[j];
            }
            return 1 / (1 + Math.exp(-(v - p) / scale));
        }

        /**
         * @param random the source
         * @return a context: a constant 1 then uniform features
         */
        public double[] context(SplittableRandom random) {
            double[] x = new double[theta.length];
            x[0] = 1;
            for (int j = 1; j < x.length; j++) {
                x[j] = random.nextDouble();
            }
            return x;
        }

        /**
         * The clairvoyant price for a context, by golden-section search over {@code [0, 20]}.
         *
         * @param x the context
         * @return the revenue-maximising price
         */
        public double bestPrice(double[] x) {
            double lo = 0;
            double hi = 20;
            for (int it = 0; it < 60; it++) {
                double m1 = lo + (hi - lo) * 0.382;
                double m2 = lo + (hi - lo) * 0.618;
                if (m1 * buy(x, m1) < m2 * buy(x, m2)) {
                    lo = m1;
                } else {
                    hi = m2;
                }
            }
            return (lo + hi) / 2;
        }
    }

    /**
     * @param dims context size (including the constant)
     * @param seed the seed
     * @return random valuations with mean about 5
     */
    public static Valuations valuations(int dims, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] theta = new double[dims];
        theta[0] = 3;
        for (int j = 1; j < dims; j++) {
            theta[j] = 4 * r.nextDouble() / Math.max(1, dims - 1) * 2;
        }
        return new Valuations(theta, 0.6);
    }

    // ------------------------------------------------------------------ job shop

    /**
     * A job-shop instance: job {@code j}'s {@code k}-th operation runs on {@code machine[j][k]}
     * for {@code time[j][k]}.
     *
     * @param machine the machine order of every job
     * @param time the processing times
     */
    public record JobShopInstance(int[][] machine, int[][] time) {
        /** @return the number of jobs */
        public int jobs() {
            return machine.length;
        }

        /** @return the number of machines */
        public int machines() {
            return machine[0].length;
        }

        /** @return the larger of the longest job and the busiest machine */
        public int lowerBound() {
            int[] load = new int[machines()];
            int best = 0;
            for (int j = 0; j < jobs(); j++) {
                int len = 0;
                for (int k = 0; k < machines(); k++) {
                    len += time[j][k];
                    load[machine[j][k]] += time[j][k];
                }
                best = Math.max(best, len);
            }
            for (int l : load) {
                best = Math.max(best, l);
            }
            return best;
        }
    }

    /**
     * A Taillard-style instance: every job visits every machine once in a random order, with
     * times uniform in {@code 1..maxTime}.
     *
     * @param jobs jobs
     * @param machines machines
     * @param maxTime the largest processing time
     * @param seed the seed
     * @return the instance
     */
    public static JobShopInstance jobShop(int jobs, int machines, int maxTime, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] m = new int[jobs][machines];
        int[][] t = new int[jobs][machines];
        for (int j = 0; j < jobs; j++) {
            int[] perm = new int[machines];
            for (int k = 0; k < machines; k++) {
                perm[k] = k;
            }
            for (int k = machines - 1; k > 0; k--) {
                int q = r.nextInt(k + 1);
                int tmp = perm[k];
                perm[k] = perm[q];
                perm[q] = tmp;
            }
            for (int k = 0; k < machines; k++) {
                m[j][k] = perm[k];
                t[j][k] = 1 + r.nextInt(maxTime);
            }
        }
        return new JobShopInstance(m, t);
    }

    // ------------------------------------------------------------------ bin packing

    /** Item-size distributions for online bin packing. */
    public enum ItemDistribution {
        /** sizes uniform in {@code 1..capacity} / 2 range {@code [capacity/10, capacity*0.7]} */
        UNIFORM,
        /** a few discrete sizes with fixed frequencies (the profile-packing regime) */
        DISCRETE,
        /** the First Fit adversary: small, then medium, then large items */
        ADVERSARIAL_FF
    }

    /** The sizes of {@link ItemDistribution#DISCRETE} for capacity 100. */
    public static final int[] DISCRETE_SIZES = {15, 25, 35, 45, 60};

    /** Their frequencies. */
    public static final double[] DISCRETE_FREQUENCIES = {0.3, 0.2, 0.15, 0.2, 0.15};

    /**
     * An item stream.
     *
     * @param distribution the size distribution
     * @param n items
     * @param capacity the bin capacity
     * @param seed the seed
     * @return the sizes in arrival order
     */
    public static int[] binItems(ItemDistribution distribution, int n, int capacity, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] items = new int[n];
        switch (distribution) {
            case UNIFORM -> {
                int lo = Math.max(1, capacity / 10);
                int hi = (int) (capacity * 0.7);
                for (int i = 0; i < n; i++) {
                    items[i] = lo + r.nextInt(hi - lo + 1);
                }
            }
            case DISCRETE -> {
                for (int i = 0; i < n; i++) {
                    items[i] = scaled(sampleIndex(DISCRETE_FREQUENCIES, r), capacity);
                }
            }
            case ADVERSARIAL_FF -> {
                int third = n / 3;
                int small = capacity / 7 + 1;
                int mid = capacity / 3 + 1;
                int big = capacity / 2 + 1;
                for (int i = 0; i < n; i++) {
                    items[i] = i < third ? small : i < 2 * third ? mid : big;
                }
            }
        }
        return items;
    }

    private static int scaled(int k, int capacity) {
        return Math.max(1, DISCRETE_SIZES[k] * capacity / 100);
    }

    /**
     * Draws a discrete stream from given frequencies over {@link #DISCRETE_SIZES}.
     *
     * @param frequencies frequencies over the discrete sizes
     * @param n items
     * @param capacity the capacity
     * @param seed the seed
     * @return the stream
     */
    public static int[] discreteItems(double[] frequencies, int n, int capacity, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] items = new int[n];
        for (int i = 0; i < n; i++) {
            items[i] = scaled(sampleIndex(frequencies, r), capacity);
        }
        return items;
    }

    /**
     * A prediction of the discrete frequencies with total-variation error about {@code error}:
     * mass is moved from a random half of the sizes to the other half.
     *
     * @param error the share of probability mass moved (0-0.3)
     * @param seed the seed
     * @return the predicted frequencies
     */
    public static double[] predictedFrequencies(double error, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] f = DISCRETE_FREQUENCIES.clone();
        int k = f.length;
        double moved = 0;
        boolean[] donor = new boolean[k];
        for (int i = 0; i < k; i++) {
            donor[i] = r.nextBoolean();
        }
        for (int i = 0; i < k; i++) {
            if (donor[i]) {
                double take = Math.min(f[i] * 0.9, error * f[i] * 2.5);
                f[i] -= take;
                moved += take;
            }
        }
        int receivers = 0;
        for (boolean b : donor) {
            receivers += b ? 0 : 1;
        }
        for (int i = 0; i < k; i++) {
            if (!donor[i]) {
                f[i] += receivers == 0 ? moved / k : moved / receivers;
            } else if (receivers == 0) {
                f[i] += moved / k;
            }
        }
        double s = Arrays.stream(f).sum();
        for (int i = 0; i < k; i++) {
            f[i] /= s;
        }
        return f;
    }

    private static int sampleIndex(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        for (int k = 0; k < p.length; k++) {
            u -= p[k];
            if (u <= 0) {
                return k;
            }
        }
        return p.length - 1;
    }

    // ------------------------------------------------------------------ matching markets

    /** How the arms' preferences over players are structured. */
    public enum MarketStructure {
        /** every arm ranks the players identically (a unique stable matching) */
        SERIAL_DICTATORSHIP,
        /** independent random rankings */
        RANDOM
    }

    /**
     * A two-sided market: players learn their arm preferences from noisy rewards; arms rank
     * players by a known order.
     *
     * @param means {@code [players][arms]} expected rewards (Bernoulli)
     * @param armRanks {@code [arms][players]} the rank (0 best) every arm gives every player
     */
    public record Market(double[][] means, int[][] armRanks) {
        /** @return the number of players */
        public int players() {
            return means.length;
        }

        /** @return the number of arms */
        public int arms() {
            return means[0].length;
        }
    }

    /**
     * A market whose player preferences are random permutations with gaps {@code gap} between
     * consecutive arms (means {@code 0.9 - gap * rank}, floored at 0.05).
     *
     * @param players players
     * @param arms arms (at least {@code players})
     * @param gap the preference gap
     * @param structure the arms' preference structure
     * @param seed the seed
     * @return the market
     */
    public static Market market(int players, int arms, double gap, MarketStructure structure, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] means = new double[players][arms];
        for (int i = 0; i < players; i++) {
            int[] perm = permutation(arms, r);
            for (int k = 0; k < arms; k++) {
                means[i][perm[k]] = Math.max(0.05, 0.9 - gap * k);
            }
        }
        int[][] ranks = new int[arms][players];
        int[] shared = permutation(players, r);
        for (int a = 0; a < arms; a++) {
            int[] order = structure == MarketStructure.SERIAL_DICTATORSHIP ? shared : permutation(players, r);
            for (int k = 0; k < players; k++) {
                ranks[a][order[k]] = k;
            }
        }
        return new Market(means, ranks);
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int q = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[q];
            p[q] = t;
        }
        return p;
    }

    // ------------------------------------------------------------------ restless arms

    /**
     * A two-state restless arm (state 1 "adherent" pays 1, state 0 pays 0).
     *
     * @param passive {@code P[s][s']} without intervention
     * @param active {@code P[s][s']} with intervention
     */
    public record RestlessArm(double[][] passive, double[][] active) {
        /**
         * @param action 1 to intervene
         * @return the transition matrix for the action
         */
        public double[][] transition(int action) {
            return action == 1 ? active : passive;
        }
    }

    /**
     * Heterogeneous adherence arms: without intervention an arm drifts to non-adherence,
     * an intervention raises the chance of (staying or becoming) adherent by an arm-specific
     * effect.
     *
     * @param n arms
     * @param seed the seed
     * @return the arms
     */
    public static List<RestlessArm> restlessArms(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<RestlessArm> arms = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double stay1 = 0.5 + 0.4 * r.nextDouble();
            double rise0 = 0.05 + 0.25 * r.nextDouble();
            double effect1 = (1 - stay1) * (0.2 + 0.8 * r.nextDouble());
            double effect0 = (1 - rise0) * (0.1 + 0.7 * r.nextDouble());
            double[][] passive = {{1 - rise0, rise0}, {1 - stay1, stay1}};
            double[][] active = {{1 - rise0 - effect0, rise0 + effect0}, {1 - stay1 - effect1, stay1 + effect1}};
            arms.add(new RestlessArm(passive, active));
        }
        return arms;
    }

    // ------------------------------------------------------------------ knapsack arms

    /**
     * Bandit arms that consume resources: a pull of arm {@code k} pays Bernoulli({@code reward[k]})
     * and uses Bernoulli({@code cost[k][j]}) of resource {@code j}.
     *
     * @param reward the mean rewards
     * @param cost {@code [arms][resources]} the mean consumptions
     */
    public record KnapsackArms(double[] reward, double[][] cost) {
        /** @return the number of arms */
        public int arms() {
            return reward.length;
        }

        /** @return the number of resources */
        public int resources() {
            return cost[0].length;
        }
    }

    /**
     * Random knapsack arms where the high-reward arms are also the most resource-hungry.
     *
     * @param arms arms
     * @param resources resources (1-3)
     * @param seed the seed
     * @return the arms
     */
    public static KnapsackArms knapsackArms(int arms, int resources, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] reward = new double[arms];
        double[][] cost = new double[arms][resources];
        for (int k = 0; k < arms; k++) {
            reward[k] = 0.1 + 0.8 * r.nextDouble();
            for (int j = 0; j < resources; j++) {
                cost[k][j] = Math.min(0.95, Math.max(0.05, reward[k] * (0.5 + 0.7 * r.nextDouble())));
            }
        }
        return new KnapsackArms(reward, cost);
    }

    /**
     * The pricing problem with an inventory as a bandit with knapsacks (shared with 12.6.7):
     * pulling price {@code k} sells with probability {@code d(p_k)}; the reward is the revenue
     * scaled by the highest price and the single resource is one unit of stock per sale.
     *
     * @param pricing the pricing problem
     * @return the arms
     */
    public static KnapsackArms pricingArms(Pricing pricing) {
        double[] p = pricing.prices();
        double max = Arrays.stream(p).max().orElse(1);
        double[] reward = new double[p.length];
        double[][] cost = new double[p.length][1];
        for (int k = 0; k < p.length; k++) {
            double d = pricing.demand(p[k]);
            reward[k] = d * p[k] / max;
            cost[k][0] = d;
        }
        return new KnapsackArms(reward, cost);
    }

    // ------------------------------------------------------------------ multi-objective grids

    /**
     * A deterministic tabular MDP with vector rewards.
     *
     * @param next {@code next[s][a]}
     * @param reward {@code reward[s][a][objective]}
     * @param terminal terminal states
     * @param start the start state
     * @param maxSteps the episode length limit
     * @param rows grid rows (for rendering and features)
     * @param cols grid columns
     */
    public record VectorMdp(int[][] next, double[][][] reward, boolean[] terminal, int start, int maxSteps, int rows,
            int cols) {
        /** @return the number of states */
        public int states() {
            return next.length;
        }

        /** @return the number of actions */
        public int actions() {
            return next[0].length;
        }

        /** @return the number of objectives */
        public int objectives() {
            return reward[0][0].length;
        }
    }

    /** The ten Pareto-optimal returns of {@link #deepSeaTreasure} (treasure, -steps). */
    public static final double[][] DEEP_SEA_TREASURE_FRONT = {{1, -1}, {2, -3}, {3, -5}, {5, -7}, {8, -8}, {16, -9},
            {24, -13}, {50, -14}, {74, -17}, {124, -19}};

    private static final int[] DST_DEPTH = {1, 2, 3, 4, 4, 4, 7, 7, 9, 10};
    private static final double[] DST_VALUE = {1, 2, 3, 5, 8, 16, 24, 50, 74, 124};

    /**
     * Deep Sea Treasure (Vamplew et al., Machine Learning 2011): a submarine starts top-left of an
     * 11 x 10 grid; treasure {@code v_c} lies at depth {@code d_c} in column {@code c} with land
     * below it; every step costs 1 time unit. Objectives: (treasure, -time). Its Pareto front
     * is concave in places, so linear scalarisation finds only part of it.
     *
     * @return the MDP (actions 0 up, 1 right, 2 down, 3 left)
     */
    public static VectorMdp deepSeaTreasure() {
        int rows = 11;
        int cols = 10;
        int n = rows * cols;
        int[][] next = new int[n][4];
        double[][][] reward = new double[n][4][2];
        boolean[] terminal = new boolean[n];
        int[][] moves = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for (int s = 0; s < n; s++) {
            int r = s / cols;
            int c = s % cols;
            terminal[s] = r == DST_DEPTH[c];
            for (int a = 0; a < 4; a++) {
                int nr = r + moves[a][0];
                int nc = c + moves[a][1];
                boolean ok = nr >= 0 && nr < rows && nc >= 0 && nc < cols && nr <= DST_DEPTH[nc];
                int t = ok ? nr * cols + nc : s;
                next[s][a] = t;
                reward[s][a][1] = -1;
                if (t / cols == DST_DEPTH[t % cols]) {
                    reward[s][a][0] = DST_VALUE[t % cols];
                }
            }
        }
        return new VectorMdp(next, reward, terminal, 0, 50, rows, cols);
    }

    /**
     * A two-objective resource-gathering grid (after Barrett &amp; Narayanan, ICML 2008): a 5 x 5
     * grid with home at the bottom middle, gold top-middle behind an enemy, a gem on the right;
     * objectives (resources brought home, -enemy encounters). The state includes what is carried.
     *
     * @return the MDP (actions 0 up, 1 right, 2 down, 3 left)
     */
    public static VectorMdp resourceGathering() {
        int rows = 5;
        int cols = 5;
        int cells = rows * cols;
        int n = cells * 4;
        int home = 4 * cols + 2;
        int gold = 2;
        int gem = cols + 4;
        boolean[] enemy = new boolean[cells];
        enemy[cols + 2] = true;
        enemy[3] = true;
        enemy[1] = true;
        int[][] moves = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        int[][] next = new int[n][4];
        double[][][] reward = new double[n][4][2];
        boolean[] terminal = new boolean[n];
        for (int s = 0; s < n; s++) {
            int cell = s % cells;
            int carry = s / cells;
            int r = cell / cols;
            int c = cell % cols;
            terminal[s] = cell == home && carry != 0;
            for (int a = 0; a < 4; a++) {
                int nr = Math.max(0, Math.min(rows - 1, r + moves[a][0]));
                int nc = Math.max(0, Math.min(cols - 1, c + moves[a][1]));
                int ncell = nr * cols + nc;
                int ncarry = carry | (ncell == gold ? 1 : 0) | (ncell == gem ? 2 : 0);
                next[s][a] = ncarry * cells + ncell;
                if (enemy[ncell]) {
                    reward[s][a][1] = -1;
                }
                if (ncell == home && ncarry != 0) {
                    reward[s][a][0] = Integer.bitCount(ncarry);
                }
            }
        }
        return new VectorMdp(next, reward, terminal, home, 40, rows, cols);
    }

    // ------------------------------------------------------------------ helpers

    /**
     * Acklam's rational approximation of the standard normal quantile.
     *
     * @param p a probability in {@code (0, 1)}
     * @return the quantile
     */
    public static double normalQuantile(double p) {
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
        double q;
        if (p < 0.02425) {
            q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        if (p > 1 - 0.02425) {
            q = Math.sqrt(-2 * Math.log(1 - p));
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
}
