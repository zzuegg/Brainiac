package io.github.zzuegg.brainiac.experimental.ops;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims.Market;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Two-sided matching markets:
 *
 * <ul>
 *   <li>{@link #deferredAcceptance}: Gale-Shapley deferred acceptance (Gale &amp; Shapley, AMM
 *       1962), proposing from either side; the result is stable and optimal for the proposing
 *       side, pessimal for the other (so arm-proposing gives the player-pessimal stable matching);</li>
 *   <li>{@link #stable}: the blocking-pair check;</li>
 *   <li>{@link #run}: bandit learning of the players' preferences from noisy match rewards (an arm
 *       accepts its most preferred proposer, the others get nothing): centralised explore-then-commit
 *       and centralised UCB, where a platform runs player-proposing deferred acceptance on the
 *       players' estimates or upper confidence bounds (Liu, Mania &amp; Jordan, AISTATS 2020,
 *       <a href="https://arxiv.org/abs/1906.05363">arXiv:1906.05363</a>), decentralised
 *       conflict-avoiding CA-UCB (Liu et al., JMLR 2021: with probability {@code lambda} repeat the
 *       last arm, else take the best UCB among the arms that would accept the player given last
 *       round), and the baselines: greedy deferred acceptance on empirical means, random arms, and
 *       independent UCB per player ignoring conflicts.</li>
 * </ul>
 *
 * <p>Regret is per player against the player-pessimal stable matching, the benchmark under which
 * these algorithms have logarithmic regret.
 */
public final class MatchingMarkets {

    private MatchingMarkets() {
    }

    /** The learning scheme. */
    public enum Method {
        /** explore round-robin, then deferred acceptance on the estimates */
        CENTRALIZED_ETC,
        /** deferred acceptance on upper confidence bounds every round */
        CENTRALIZED_UCB,
        /** decentralised conflict-avoiding UCB */
        DECENTRALIZED_CA_UCB,
        /** deferred acceptance on the empirical means, no exploration */
        GREEDY,
        /** every player picks a random arm */
        RANDOM,
        /** every player picks its own UCB arm, ignoring collisions */
        INDEPENDENT_UCB
    }

    /**
     * The outcome of a run.
     *
     * @param regret the stable regret of every player
     * @param unstableShare the share of rounds whose matching was unstable
     */
    public record Run(double[] regret, double unstableShare) {
        /** @return the largest per-player regret */
        public double maxRegret() {
            return Arrays.stream(regret).max().orElse(0);
        }
    }

    /**
     * Deferred acceptance.
     *
     * @param playerScores {@code [players][arms]} the players' preference scores (higher is better)
     * @param armRanks {@code [arms][players]} the arms' ranks (lower is better)
     * @param playersPropose whether players propose (else arms do)
     * @return the arm of every player ({@code -1} for unmatched)
     */
    public static int[] deferredAcceptance(double[][] playerScores, int[][] armRanks, boolean playersPropose) {
        int n = playerScores.length;
        int k = playerScores[0].length;
        int[] armOf = new int[n];
        Arrays.fill(armOf, -1);
        int[] playerOf = new int[k];
        Arrays.fill(playerOf, -1);
        if (playersPropose) {
            int[][] order = new int[n][];
            for (int i = 0; i < n; i++) {
                order[i] = sortedDesc(playerScores[i]);
            }
            int[] nextChoice = new int[n];
            ArrayDeque<Integer> free = new ArrayDeque<>();
            for (int i = 0; i < n; i++) {
                free.add(i);
            }
            while (!free.isEmpty()) {
                int i = free.poll();
                if (nextChoice[i] >= k) {
                    continue;
                }
                int a = order[i][nextChoice[i]++];
                int holder = playerOf[a];
                if (holder < 0) {
                    playerOf[a] = i;
                    armOf[i] = a;
                } else if (armRanks[a][i] < armRanks[a][holder]) {
                    playerOf[a] = i;
                    armOf[i] = a;
                    armOf[holder] = -1;
                    free.add(holder);
                } else {
                    free.add(i);
                }
            }
        } else {
            int[][] order = new int[k][];
            for (int a = 0; a < k; a++) {
                double[] neg = new double[n];
                for (int i = 0; i < n; i++) {
                    neg[i] = -armRanks[a][i];
                }
                order[a] = sortedDesc(neg);
            }
            int[] nextChoice = new int[k];
            ArrayDeque<Integer> free = new ArrayDeque<>();
            for (int a = 0; a < k; a++) {
                free.add(a);
            }
            while (!free.isEmpty()) {
                int a = free.poll();
                if (nextChoice[a] >= n) {
                    continue;
                }
                int i = order[a][nextChoice[a]++];
                int holder = armOf[i];
                if (holder < 0) {
                    armOf[i] = a;
                    playerOf[a] = i;
                } else if (playerScores[i][a] > playerScores[i][holder]) {
                    armOf[i] = a;
                    playerOf[a] = i;
                    playerOf[holder] = -1;
                    free.add(holder);
                } else {
                    free.add(a);
                }
            }
        }
        return armOf;
    }

    private static int[] sortedDesc(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (x, y) -> Double.compare(v[y], v[x]));
        return Arrays.stream(idx).mapToInt(Integer::intValue).toArray();
    }

    /**
     * Whether a matching has no blocking pair under the true preferences.
     *
     * @param armOf the arm of every player ({@code -1} unmatched)
     * @param means the players' true means
     * @param armRanks the arms' ranks
     * @return stability
     */
    public static boolean stable(int[] armOf, double[][] means, int[][] armRanks) {
        int n = means.length;
        int k = means[0].length;
        int[] playerOf = new int[k];
        Arrays.fill(playerOf, -1);
        for (int i = 0; i < n; i++) {
            if (armOf[i] >= 0) {
                playerOf[armOf[i]] = i;
            }
        }
        for (int i = 0; i < n; i++) {
            double mine = armOf[i] < 0 ? Double.NEGATIVE_INFINITY : means[i][armOf[i]];
            for (int a = 0; a < k; a++) {
                if (a == armOf[i] || means[i][a] <= mine) {
                    continue;
                }
                int holder = playerOf[a];
                if (holder < 0 || armRanks[a][i] < armRanks[a][holder]) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Runs a learning scheme on a market.
     *
     * @param market the market
     * @param method the scheme
     * @param horizon rounds
     * @param exploreRounds rounds per arm of the ETC exploration phase
     * @param seed the seed
     * @return the run
     */
    public static Run run(Market market, Method method, int horizon, int exploreRounds, long seed) {
        Objects.requireNonNull(method, "method");
        SplittableRandom r = new SplittableRandom(seed);
        int n = market.players();
        int k = market.arms();
        double[][] means = market.means();
        int[][] ranks = market.armRanks();
        int[] pessimal = deferredAcceptance(means, ranks, false);
        double[] regret = new double[n];
        double[][] sum = new double[n][k];
        int[][] pulls = new int[n][k];
        int[] lastArm = new int[n];
        Arrays.fill(lastArm, -1);
        int[] lastHolder = new int[k];
        Arrays.fill(lastHolder, -1);
        int unstable = 0;
        int[] committed = null;
        for (int t = 1; t <= horizon; t++) {
            int[] proposal = new int[n];
            switch (method) {
                case CENTRALIZED_ETC -> {
                    if (t <= exploreRounds * k) {
                        for (int i = 0; i < n; i++) {
                            proposal[i] = (i + t) % k;
                        }
                    } else {
                        if (committed == null) {
                            committed = deferredAcceptance(estimates(sum, pulls, Double.NaN, t), ranks, true);
                        }
                        proposal = committed.clone();
                    }
                }
                case CENTRALIZED_UCB -> proposal = deferredAcceptance(estimates(sum, pulls, 2, t), ranks, true);
                case GREEDY -> {
                    if (t <= k) {
                        for (int i = 0; i < n; i++) {
                            proposal[i] = (i + t) % k;
                        }
                    } else {
                        proposal = deferredAcceptance(estimates(sum, pulls, Double.NaN, t), ranks, true);
                    }
                }
                case RANDOM -> {
                    for (int i = 0; i < n; i++) {
                        proposal[i] = r.nextInt(k);
                    }
                }
                case INDEPENDENT_UCB -> {
                    double[][] ucb = estimates(sum, pulls, 2, t);
                    for (int i = 0; i < n; i++) {
                        proposal[i] = argmax(ucb[i], r);
                    }
                }
                case DECENTRALIZED_CA_UCB -> {
                    double[][] ucb = estimates(sum, pulls, 2, t);
                    for (int i = 0; i < n; i++) {
                        if (lastArm[i] >= 0 && r.nextDouble() < 0.1) {
                            proposal[i] = lastArm[i];
                            continue;
                        }
                        int best = -1;
                        for (int a = 0; a < k; a++) {
                            int h = lastHolder[a];
                            boolean plausible = h < 0 || h == i || ranks[a][i] < ranks[a][h];
                            if (plausible && (best < 0 || ucb[i][a] > ucb[i][best])) {
                                best = a;
                            }
                        }
                        proposal[i] = best < 0 ? r.nextInt(k) : best;
                    }
                }
            }
            // arms accept their most preferred proposer
            int[] holder = new int[k];
            Arrays.fill(holder, -1);
            for (int i = 0; i < n; i++) {
                int a = proposal[i];
                if (a >= 0 && (holder[a] < 0 || ranks[a][i] < ranks[a][holder[a]])) {
                    holder[a] = i;
                }
            }
            int[] armOf = new int[n];
            Arrays.fill(armOf, -1);
            for (int a = 0; a < k; a++) {
                if (holder[a] >= 0) {
                    armOf[holder[a]] = a;
                }
            }
            for (int i = 0; i < n; i++) {
                double reward = 0;
                if (armOf[i] >= 0) {
                    reward = r.nextDouble() < means[i][armOf[i]] ? 1 : 0;
                    sum[i][armOf[i]] += reward;
                    pulls[i][armOf[i]]++;
                }
                double target = pessimal[i] < 0 ? 0 : means[i][pessimal[i]];
                regret[i] += target - (armOf[i] < 0 ? 0 : means[i][armOf[i]]);
                lastArm[i] = armOf[i];
            }
            lastHolder = holder;
            if (!stable(armOf, means, ranks)) {
                unstable++;
            }
        }
        return new Run(regret, unstable / (double) horizon);
    }

    /** Empirical means (bonus {@code NaN}) or upper confidence bounds with the given constant. */
    private static double[][] estimates(double[][] sum, int[][] pulls, double c, int t) {
        int n = sum.length;
        int k = sum[0].length;
        double[][] e = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < k; a++) {
                if (pulls[i][a] == 0) {
                    e[i][a] = Double.isNaN(c) ? 0.5 : 1e9 - a;
                } else {
                    double m = sum[i][a] / pulls[i][a];
                    e[i][a] = Double.isNaN(c) ? m : m + Math.sqrt(c * Math.log(t) / pulls[i][a]);
                }
            }
        }
        return e;
    }

    private static int argmax(double[] v, SplittableRandom r) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < v.length; a++) {
            if (v[a] > v[best] + 1e-12) {
                best = a;
                ties = 1;
            } else if (Math.abs(v[a] - v[best]) <= 1e-12 && r.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }
}
