package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Policy-Space Response Oracles: grow a population of policies per player; play every pair to
 * fill an empirical payoff matrix (the meta-game); solve the meta-game for a meta-strategy (a
 * mixture over each population); then add, for each player, a best response to the opponent's
 * meta-strategy, found by an {@link Oracle} (exact, or trained by reinforcement learning).
 * With a Nash meta-solver this is the double-oracle algorithm, which converges to a Nash
 * equilibrium and avoids the strategy cycling of naive self-play; a uniform meta-solver is
 * fictitious play over the population (a league), and responding only to the latest policy
 * is naive self-play.
 *
 * <p>The meta-Nash is computed exactly by {@link LinearProgram#solveZeroSum}; the joint
 * mixture is turned into a behavioural policy with {@link Exploitability#mixture}.
 *
 * <p>Lanctot et al. 2017, "A Unified Game-Theoretic Approach to Multiagent Reinforcement
 * Learning", <a href="https://arxiv.org/abs/1711.00832">arXiv:1711.00832</a>; McMahan et al.
 * 2003, "Planning in the Presence of Cost Functions Controlled by an Adversary", ICML.
 */
public final class PolicySpaceResponseOracles {

    /** How the meta-game is solved. */
    public enum MetaSolver {
        /** An exact Nash equilibrium of the meta-game (double oracle). */
        NASH,
        /** Uniform over the population (fictitious play, a league). */
        UNIFORM,
        /** All weight on the newest policy (naive self-play). */
        LATEST
    }

    /** Computes a (possibly approximate) best response. */
    @FunctionalInterface
    public interface Oracle {
        /**
         * @param tree the game
         * @param opponent the behavioural policy table to respond to
         * @param player the responding player
         * @return a policy table whose {@code player} rows respond to {@code opponent}
         */
        double[][] respond(GameTree tree, double[][] opponent, int player);

        /** @return the exact best response of {@link Exploitability#bestResponse} */
        static Oracle exact() {
            return Exploitability::bestResponse;
        }

        /**
         * A reinforcement-learning oracle: a tabular softmax policy trained by policy-gradient
         * steps against the fixed opponent, using the exact counterfactual advantages of the
         * tree (no sampling noise), so it reaches a good but not exact best response.
         *
         * @param iterations gradient steps
         * @param learningRate the step size on the logits
         * @return the oracle
         */
        static Oracle policyGradient(int iterations, double learningRate) {
            return (tree, opponent, player) -> {
                double[][] logits = new double[tree.infoSets()][tree.maxActions()];
                double[][] policy = copy(opponent);
                double[][] q = new double[tree.infoSets()][tree.maxActions()];
                double[] reach = new double[tree.infoSets()];
                for (int it = 0; it < iterations; it++) {
                    softmaxRows(tree, logits, policy, player);
                    tree.counterfactualValues(policy, q, reach);
                    for (int s = 0; s < tree.infoSets(); s++) {
                        if (tree.infoSetPlayer(s) != player || reach[s] <= 0) {
                            continue;
                        }
                        double v = 0;
                        for (int a = 0; a < q[s].length; a++) {
                            v += policy[s][a] * q[s][a];
                        }
                        for (int a = 0; a < q[s].length; a++) {
                            if (tree.legal(s)[a]) {
                                logits[s][a] += learningRate * policy[s][a] * (q[s][a] - v) / reach[s];
                            }
                        }
                    }
                }
                softmaxRows(tree, logits, policy, player);
                return policy;
            };
        }
    }

    private final GameTree tree;
    private final MetaSolver solver;
    private Oracle oracle = Oracle.exact();
    private final List<List<double[][]>> population = List.of(new ArrayList<>(), new ArrayList<>());
    private double[][] meta = {{1}, {1}};
    private double[][] payoffs = new double[0][0];

    private PolicySpaceResponseOracles(GameTree tree, MetaSolver solver) {
        this.tree = tree;
        this.solver = solver;
        population.get(0).add(tree.uniformPolicy());
        population.get(1).add(tree.uniformPolicy());
        refresh();
    }

    /**
     * Starts PSRO with one uniform policy per player.
     *
     * @param tree the game
     * @param solver the meta-solver
     * @return the algorithm
     */
    public static PolicySpaceResponseOracles of(GameTree tree, MetaSolver solver) {
        return new PolicySpaceResponseOracles(Objects.requireNonNull(tree), Objects.requireNonNull(solver));
    }

    /**
     * @param o the best-response oracle (exact by default)
     * @return this
     */
    public PolicySpaceResponseOracles oracle(Oracle o) {
        this.oracle = Objects.requireNonNull(o);
        return this;
    }

    /** Adds one best response per player to the opponent's current meta-strategy. */
    public void iterate() {
        double[][] joint = policy();
        double[][] br0 = oracle.respond(tree, joint, 0);
        double[][] br1 = oracle.respond(tree, joint, 1);
        population.get(0).add(br0);
        population.get(1).add(br1);
        refresh();
    }

    /**
     * Runs iterations.
     *
     * @param count how many
     * @return this
     */
    public PolicySpaceResponseOracles run(int count) {
        for (int i = 0; i < count; i++) {
            iterate();
        }
        return this;
    }

    private void refresh() {
        int n0 = population.get(0).size();
        int n1 = population.get(1).size();
        double[][] m = new double[n0][n1];
        for (int i = 0; i < n0; i++) {
            for (int j = 0; j < n1; j++) {
                m[i][j] = i < payoffs.length && j < payoffs[i].length ? payoffs[i][j]
                        : tree.expectedValue(population.get(0).get(i), population.get(1).get(j));
            }
        }
        payoffs = m;
        meta = switch (solver) {
            case NASH -> {
                LinearProgram.Equilibrium e = LinearProgram.solveZeroSum(m);
                yield new double[][] {e.row(), e.column()};
            }
            case UNIFORM -> new double[][] {fill(n0, 1.0 / n0), fill(n1, 1.0 / n1)};
            case LATEST -> {
                double[] a = new double[n0];
                double[] b = new double[n1];
                a[n0 - 1] = 1;
                b[n1 - 1] = 1;
                yield new double[][] {a, b};
            }
        };
    }

    private static double[] fill(int n, double v) {
        double[] out = new double[n];
        Arrays.fill(out, v);
        return out;
    }

    /** @return the meta-game payoff matrix to player 0 (a copy) */
    public double[][] metaGame() {
        return copy(payoffs);
    }

    /**
     * @param player 0 or 1
     * @return that player's meta-strategy over its population
     */
    public double[] metaStrategy(int player) {
        return meta[player].clone();
    }

    /**
     * @param player 0 or 1
     * @return that player's population size
     */
    public int populationSize(int player) {
        return population.get(player).size();
    }

    /** @return the behavioural policy table of both players' meta-strategy mixtures */
    public double[][] policy() {
        double[][] p0 = Exploitability.mixture(tree, population.get(0), meta[0]);
        double[][] p1 = Exploitability.mixture(tree, population.get(1), meta[1]);
        double[][] out = new double[tree.infoSets()][];
        for (int s = 0; s < out.length; s++) {
            out[s] = tree.infoSetPlayer(s) == 0 ? p0[s] : p1[s];
        }
        return out;
    }

    private static double[][] copy(double[][] a) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i].clone();
        }
        return out;
    }

    static void softmaxRows(GameTree tree, double[][] logits, double[][] policy, int player) {
        for (int s = 0; s < tree.infoSets(); s++) {
            if (player >= 0 && tree.infoSetPlayer(s) != player) {
                continue;
            }
            boolean[] legal = tree.legal(s);
            double max = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < legal.length; a++) {
                if (legal[a]) {
                    max = Math.max(max, logits[s][a]);
                }
            }
            double z = 0;
            for (int a = 0; a < legal.length; a++) {
                policy[s][a] = legal[a] ? Math.exp(logits[s][a] - max) : 0;
                z += policy[s][a];
            }
            for (int a = 0; a < legal.length; a++) {
                policy[s][a] /= z;
            }
        }
    }
}
