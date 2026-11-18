package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Small bandit benchmark instances for regret, pure-exploration, dueling and ranking studies:
 *
 * <ul>
 *   <li>stochastic arms: {@link #bernoulli Bernoulli}, {@link #gaussian Gaussian} and
 *       {@link #heavyTailed heavy-tailed} (Student-t noise), with generators for small gaps and
 *       means near 0 ({@link #lowMeanBernoulli}) or 0.5;</li>
 *   <li>non-stationary and adversarial reward sequences: {@link #switching} (the best arm
 *       changes every {@code period} pulls) and {@link #adaptiveAdversary} (the arm with the
 *       highest empirical mean so far pays nothing, which exploits deterministic index
 *       policies);</li>
 *   <li>{@link Preferences preference matrices} for dueling bandits: Bradley-Terry utilities
 *       (with a Condorcet winner) and cyclic preferences (none, only a Copeland winner);</li>
 *   <li>{@link ClickItems click models} over items with features for ranking bandits, using
 *       {@link ClickLogs.ClickModel}'s two models and {@link ClickLogs#rankingLog}'s attraction
 *       {@code 0.05 + 0.95 (2^g - 1) / 15} for relevance grades {@code g}.</li>
 * </ul>
 */
public final class BanditSuites {

    private BanditSuites() {
    }

    /**
     * A multi-armed bandit instance. Every round it reveals a full reward vector, of which a
     * policy sees only the entry of the arm it pulled; the full vector is kept so regret
     * against the best fixed arm in hindsight can be computed for adversarial sequences.
     */
    public interface Instance {

        /** @return the number of arms */
        int arms();

        /**
         * @param t the round, from 0
         * @return the expected reward of every arm at round {@code t}
         */
        double[] means(int t);

        /**
         * Draws the rewards of round {@code t}.
         *
         * @param t the round
         * @param pulls how often the policy pulled each arm so far
         * @param sums the reward the policy collected from each arm so far
         * @param random the randomness
         * @return every arm's reward this round
         */
        double[] rewards(int t, int[] pulls, double[] sums, SplittableRandom random);

        /**
         * Draws the reward of one arm only (for pure-exploration rules that never need the
         * full vector); the default draws the full vector with empty histories.
         *
         * @param arm the arm
         * @param t the round
         * @param random the randomness
         * @return the reward
         */
        default double sample(int arm, int t, SplittableRandom random) {
            return rewards(t, new int[arms()], new double[arms()], random)[arm];
        }

        /** @return whether rewards are i.i.d. per arm (so pseudo-regret from the means applies) */
        default boolean stochastic() {
            return true;
        }

        /** @return the arm with the highest mean at round 0 */
        default int bestArm() {
            double[] m = means(0);
            int best = 0;
            for (int i = 1; i < m.length; i++) {
                if (m[i] > m[best]) {
                    best = i;
                }
            }
            return best;
        }
    }

    private record Stationary(String name, double[] mu, Noise noise) implements Instance {
        @Override
        public int arms() {
            return mu.length;
        }

        @Override
        public double[] means(int t) {
            return mu.clone();
        }

        @Override
        public double[] rewards(int t, int[] pulls, double[] sums, SplittableRandom random) {
            double[] r = new double[mu.length];
            for (int i = 0; i < r.length; i++) {
                r[i] = noise.draw(mu[i], random);
            }
            return r;
        }

        @Override
        public double sample(int arm, int t, SplittableRandom random) {
            return noise.draw(mu[arm], random);
        }

        @Override
        public String toString() {
            return name;
        }
    }

    /**
     * An instance whose arms are arbitrary stochastic experiments, such as short training runs
     * that return a validation score.
     *
     * @param name a label
     * @param means each arm's expected reward (for regret; may be estimated)
     * @param arm draws one reward of an arm
     * @return the instance
     */
    public static Instance experiments(String name, double[] means, ArmSampler arm) {
        double[] mu = means.clone();
        return new Instance() {
            @Override
            public int arms() {
                return mu.length;
            }

            @Override
            public double[] means(int t) {
                return mu.clone();
            }

            @Override
            public double[] rewards(int t, int[] pulls, double[] sums, SplittableRandom random) {
                double[] r = new double[mu.length];
                for (int i = 0; i < r.length; i++) {
                    r[i] = arm.draw(i, random);
                }
                return r;
            }

            @Override
            public double sample(int a, int t, SplittableRandom random) {
                return arm.draw(a, random);
            }

            @Override
            public String toString() {
                return name;
            }
        };
    }

    /** Draws one reward of an arm. */
    @FunctionalInterface
    public interface ArmSampler {
        /**
         * @param arm the arm
         * @param random the randomness
         * @return the reward
         */
        double draw(int arm, SplittableRandom random);
    }

    @FunctionalInterface
    private interface Noise {
        double draw(double mean, SplittableRandom random);
    }

    /**
     * @param means the success probability of each arm
     * @return Bernoulli arms
     */
    public static Instance bernoulli(double... means) {
        return new Stationary("bernoulli" + Arrays.toString(means), means.clone(),
                (m, r) -> r.nextDouble() < m ? 1 : 0);
    }

    /**
     * Bernoulli arms with low means: the best arm has mean {@code top}, the others
     * {@code top - gap} or lower (linearly spaced down to {@code top - 2 gap}, clipped at 0.001).
     *
     * @param arms arms, at least 2
     * @param top the best mean
     * @param gap the smallest gap
     * @return the instance
     */
    public static Instance lowMeanBernoulli(int arms, double top, double gap) {
        double[] m = new double[arms];
        m[0] = top;
        for (int i = 1; i < arms; i++) {
            double frac = arms == 2 ? 0 : (i - 1) / (double) (arms - 2);
            m[i] = Math.max(0.001, top - gap - frac * gap);
        }
        return bernoulli(m);
    }

    /**
     * @param sigma the noise standard deviation
     * @param means the arm means
     * @return Gaussian arms
     */
    public static Instance gaussian(double sigma, double... means) {
        return new Stationary("gaussian" + Arrays.toString(means), means.clone(),
                (m, r) -> m + sigma * r.nextGaussian());
    }

    /**
     * Arms with Student-t noise (finite variance for {@code dof > 2}, heavy tails).
     *
     * @param dof the degrees of freedom of the noise, for example 2.5
     * @param scale the noise scale
     * @param means the arm means
     * @return the instance
     */
    public static Instance heavyTailed(double dof, double scale, double... means) {
        return new Stationary("studentT" + Arrays.toString(means), means.clone(), (m, r) -> {
            double z = r.nextGaussian();
            double chi = 0;
            int whole = (int) Math.floor(dof);
            for (int k = 0; k < whole; k++) {
                double g = r.nextGaussian();
                chi += g * g;
            }
            double frac = dof - whole;
            if (frac > 0) {
                double g = r.nextGaussian();
                chi += frac * g * g;
            }
            return m + scale * z / Math.sqrt(chi / dof);
        });
    }

    /**
     * Bernoulli arms whose means rotate every {@code period} rounds, so the best arm changes.
     *
     * @param period rounds between switches
     * @param means the means in the first phase; phase {@code k} shifts them by {@code k} arms
     * @return the instance (not stationary, but still stochastic within a phase)
     */
    public static Instance switching(int period, double... means) {
        double[] base = means.clone();
        return new Instance() {
            @Override
            public int arms() {
                return base.length;
            }

            @Override
            public double[] means(int t) {
                int shift = (t / period) % base.length;
                double[] m = new double[base.length];
                for (int i = 0; i < m.length; i++) {
                    m[i] = base[Math.floorMod(i - shift, base.length)];
                }
                return m;
            }

            @Override
            public double[] rewards(int t, int[] pulls, double[] sums, SplittableRandom random) {
                double[] m = means(t);
                double[] r = new double[m.length];
                for (int i = 0; i < m.length; i++) {
                    r[i] = random.nextDouble() < m[i] ? 1 : 0;
                }
                return r;
            }

            @Override
            public String toString() {
                return "switching(" + period + ")";
            }
        };
    }

    /**
     * An adaptive adversary: every round the arm whose empirical mean (as the policy observed
     * it) is highest pays 0 and every other arm pays 1; unpulled arms count as mean 1. Policies
     * that chase the empirical leader collect almost nothing while the best fixed arm in
     * hindsight collects about {@code (1 - 1/K) T}.
     *
     * @param arms the arms
     * @return the instance
     */
    public static Instance adaptiveAdversary(int arms) {
        return new Instance() {
            @Override
            public int arms() {
                return arms;
            }

            @Override
            public double[] means(int t) {
                double[] m = new double[arms];
                Arrays.fill(m, 1 - 1.0 / arms);
                return m;
            }

            @Override
            public double[] rewards(int t, int[] pulls, double[] sums, SplittableRandom random) {
                int leader = 0;
                double best = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < arms; i++) {
                    double m = pulls[i] == 0 ? 1 : sums[i] / pulls[i];
                    if (m > best + 1e-12) {
                        best = m;
                        leader = i;
                    }
                }
                double[] r = new double[arms];
                Arrays.fill(r, 1);
                r[leader] = 0;
                return r;
            }

            @Override
            public boolean stochastic() {
                return false;
            }

            @Override
            public String toString() {
                return "adaptiveAdversary(" + arms + ")";
            }
        };
    }

    // ------------------------------------------------------------------ dueling

    /**
     * A preference matrix for dueling bandits: {@code p[i][j]} is the probability that arm
     * {@code i} beats arm {@code j}, with {@code p[i][j] + p[j][i] = 1}.
     *
     * @param name a label
     * @param p the matrix
     */
    public record Preferences(String name, double[][] p) {

        /** @return the number of arms */
        public int arms() {
            return p.length;
        }

        /**
         * Plays one duel.
         *
         * @param i the first arm
         * @param j the second arm
         * @param random the randomness
         * @return whether {@code i} won
         */
        public boolean duel(int i, int j, SplittableRandom random) {
            return i == j ? random.nextBoolean() : random.nextDouble() < p[i][j];
        }

        /** @return the Condorcet winner (beats every other arm with probability above 1/2), or -1 */
        public int condorcetWinner() {
            for (int i = 0; i < p.length; i++) {
                boolean all = true;
                for (int j = 0; j < p.length; j++) {
                    if (j != i && p[i][j] <= 0.5) {
                        all = false;
                        break;
                    }
                }
                if (all) {
                    return i;
                }
            }
            return -1;
        }

        /**
         * @param i an arm
         * @return its normalised Copeland score: the share of other arms it beats
         */
        public double copeland(int i) {
            int wins = 0;
            for (int j = 0; j < p.length; j++) {
                if (j != i && p[i][j] > 0.5) {
                    wins++;
                }
            }
            return wins / (double) (p.length - 1);
        }

        /** @return the arms with the highest Copeland score */
        public int[] copelandWinners() {
            double best = -1;
            for (int i = 0; i < p.length; i++) {
                best = Math.max(best, copeland(i));
            }
            final double b = best;
            return java.util.stream.IntStream.range(0, p.length).filter(i -> copeland(i) == b).toArray();
        }

        /**
         * The per-duel regret: with a Condorcet winner {@code c}, the average-regret
         * {@code (p[c][i] + p[c][j] - 1) / 2}; without one, the Copeland regret
         * {@code cpld* - (cpld(i) + cpld(j)) / 2}.
         *
         * @param i the first arm
         * @param j the second arm
         * @return the regret of dueling them
         */
        public double regret(int i, int j) {
            int c = condorcetWinner();
            if (c >= 0) {
                return (p[c][i] + p[c][j] - 1) / 2;
            }
            double best = copeland(copelandWinners()[0]);
            return best - (copeland(i) + copeland(j)) / 2;
        }
    }

    /**
     * Bradley-Terry preferences {@code p[i][j] = sigmoid(u_i - u_j)}; the arm of highest
     * utility is the Condorcet winner.
     *
     * @param utilities the utilities
     * @return the matrix
     */
    public static Preferences bradleyTerry(double... utilities) {
        int k = utilities.length;
        double[][] p = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                p[i][j] = i == j ? 0.5 : 1 / (1 + Math.exp(-(utilities[i] - utilities[j])));
            }
        }
        return new Preferences("bradleyTerry", p);
    }

    /**
     * Cyclic preferences with no Condorcet winner: arms 0, 1, 2 form a rock-paper-scissors
     * cycle (each beats the next with probability {@code 0.5 + strength}), and all three beat
     * every remaining arm with probability {@code 0.5 + strength / 2}; the remaining arms are
     * Bradley-Terry among themselves. The Copeland winners are arms 0, 1 and 2.
     *
     * @param arms arms, at least 4
     * @param strength the cycle's preference margin, at most 0.5
     * @param seed the seed of the remaining arms' utilities
     * @return the matrix
     */
    public static Preferences cyclic(int arms, double strength, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] p = new double[arms][arms];
        double[] u = new double[arms];
        for (int i = 3; i < arms; i++) {
            u[i] = r.nextGaussian();
        }
        for (int i = 0; i < arms; i++) {
            for (int j = 0; j < arms; j++) {
                if (i == j) {
                    p[i][j] = 0.5;
                } else if (i < 3 && j < 3) {
                    p[i][j] = (j == (i + 1) % 3) ? 0.5 + strength : 0.5 - strength;
                } else if (i < 3) {
                    p[i][j] = 0.5 + strength / 2;
                } else if (j < 3) {
                    p[i][j] = 0.5 - strength / 2;
                } else {
                    p[i][j] = 1 / (1 + Math.exp(-(u[i] - u[j])));
                }
            }
        }
        return new Preferences("cyclic", p);
    }

    // ------------------------------------------------------------------ ranking

    /**
     * Items for ranking bandits: features, attraction probabilities and a click model.
     *
     * @param features {@code [items][d]} item features
     * @param attraction the probability that a user examining an item clicks it
     * @param model the click model
     * @param examination the probability of examining each list position (position-based model)
     * @param stopProbability under the cascade model, the chance a user stops after a click
     *     (1 is the cascade model of Kveton et al.; {@link ClickLogs#rankingLog} uses 0.6)
     */
    public record ClickItems(double[][] features, double[] attraction, ClickLogs.ClickModel model,
            double[] examination, double stopProbability) {

        /** @return the number of items */
        public int items() {
            return attraction.length;
        }

        /** @return the list length */
        public int listLength() {
            return examination.length;
        }

        /**
         * Shows a list and simulates the user.
         *
         * @param list item indices, top first
         * @param random the randomness
         * @return a 0/1 click per position
         */
        public int[] clicks(int[] list, SplittableRandom random) {
            int[] c = new int[list.length];
            for (int k = 0; k < list.length; k++) {
                double a = attraction[list[k]];
                if (model == ClickLogs.ClickModel.POSITION_BASED) {
                    c[k] = random.nextDouble() < examination[k] * a ? 1 : 0;
                } else if (random.nextDouble() < a) {
                    c[k] = 1;
                    if (random.nextDouble() < stopProbability) {
                        break;
                    }
                }
            }
            return c;
        }

        /**
         * The expected reward of a list: the probability of at least one click under the
         * cascade model, the expected number of clicks under the position-based model.
         *
         * @param list item indices, top first
         * @return the expected reward
         */
        public double value(int[] list) {
            if (model == ClickLogs.ClickModel.POSITION_BASED) {
                double v = 0;
                for (int k = 0; k < list.length; k++) {
                    v += examination[k] * attraction[list[k]];
                }
                return v;
            }
            double none = 1;
            for (int i : list) {
                none *= 1 - attraction[i];
            }
            return 1 - none;
        }

        /** @return the best list: items by attraction, highest first */
        public int[] bestList() {
            return java.util.stream.IntStream.range(0, attraction.length).boxed()
                    .sorted((a, b) -> Double.compare(attraction[b], attraction[a]))
                    .limit(examination.length).mapToInt(Integer::intValue).toArray();
        }
    }

    /**
     * Items with Gaussian features whose relevance grade (0..4, by the quantiles
     * {@link ClickLogs#rankingLog} uses) comes from a hidden linear score; attraction follows
     * the grade as in {@link ClickLogs#rankingLog}.
     *
     * @param items items
     * @param k the list length
     * @param d feature dimension
     * @param model the click model
     * @param eta the position-bias exponent (examination {@code (1 / rank)^eta})
     * @param stopProbability the cascade stop probability after a click
     * @param seed the seed
     * @return the items
     */
    public static ClickItems clickItems(int items, int k, int d, ClickLogs.ClickModel model, double eta,
            double stopProbability, long seed) {
        Objects.requireNonNull(model, "model");
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = r.nextGaussian();
        }
        double[][] x = new double[items][d];
        double[] score = new double[items];
        for (int i = 0; i < items; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextGaussian();
                score[i] += w[j] * x[i][j];
            }
        }
        double[] sorted = score.clone();
        Arrays.sort(sorted);
        double[] cut = {sorted[(int) (0.5 * items)], sorted[(int) (0.75 * items)], sorted[(int) (0.9 * items)],
            sorted[Math.min(items - 1, (int) (0.97 * items))]};
        double[] attraction = new double[items];
        for (int i = 0; i < items; i++) {
            int g = 0;
            while (g < 4 && score[i] >= cut[g]) {
                g++;
            }
            attraction[i] = 0.05 + 0.95 * (Math.pow(2, g) - 1) / 15;
        }
        double[] exam = new double[k];
        for (int p = 0; p < k; p++) {
            exam[p] = Math.pow(1.0 / (p + 1), eta);
        }
        return new ClickItems(x, attraction, model, exam, stopProbability);
    }
}
