package io.github.zzuegg.brainiac.experimental.rl;

import java.util.Objects;

/**
 * Equilibrium learning whose <em>last iterate</em> converges in two-player zero-sum games,
 * so no average-policy network is needed. Plain policy gradient, multiplicative weights and
 * CFR's current strategy cycle around the equilibrium; only their averages converge. Adding a
 * KL "magnet" towards a reference policy makes each step a contraction:
 * <ul>
 *   <li>{@link #mmd} (magnetic mirror descent): per information set,
 *       {@code pi'(a) ∝ [pi(a) rho(a)^(alpha eta) exp(eta q(a))]^(1 / (1 + alpha eta))}, with a
 *       fixed uniform magnet {@code rho}. The last iterate converges to the regularised
 *       (quantal-response) equilibrium, whose exploitability shrinks with {@code alpha}
 *       (Sokota et al. 2023, <a href="https://arxiv.org/abs/2206.05825">arXiv:2206.05825</a>).</li>
 *   <li>{@link #rnad} (regularised Nash dynamics, DeepNash's Stratego algorithm): the same
 *       step, but every {@code period} iterations the magnet moves to the current policy, so
 *       the fixed points converge to the Nash equilibrium itself (Perolat et al. 2022,
 *       <a href="https://arxiv.org/abs/2206.15378">arXiv:2206.15378</a>).</li>
 *   <li>{@link #nashPg} (Nash policy gradient): ordinary softmax policy-gradient steps on the
 *       logits of the KL-regularised objective, with the regulariser's reference refreshed
 *       to the current policy every {@code period} steps (an iteratively refined
 *       regulariser; after <a href="https://arxiv.org/abs/2510.18183">arXiv:2510.18183</a>).</li>
 *   <li>{@link #unregularized}: the same mirror-descent step with {@code alpha = 0}
 *       (multiplicative weights), the cycling baseline.</li>
 * </ul>
 * Both players update simultaneously from exact counterfactual action values
 * ({@link GameTree#counterfactualValues}), normalised by the reach of each information set.
 */
public final class RegularizedEquilibriumLearner {

    /** The update rule. */
    public enum Method {
        /** Magnetic mirror descent with a fixed uniform magnet. */
        MMD,
        /** Mirror descent with a magnet that moves to the current policy periodically. */
        RNAD,
        /** Softmax policy gradient with a periodically refreshed KL reference. */
        NASH_PG,
        /** Mirror descent without regularisation (multiplicative weights). */
        UNREGULARIZED
    }

    private final GameTree tree;
    private final Method method;
    private final double alpha;
    private final double step;
    private final int period;
    private final double[][] policy;
    private final double[][] magnet;
    private final double[][] logits;
    private final double[][] q;
    private final double[] reach;
    private int iterations;

    private RegularizedEquilibriumLearner(GameTree tree, Method method, double alpha, double step, int period) {
        this.tree = Objects.requireNonNull(tree);
        this.method = method;
        this.alpha = alpha;
        this.step = step;
        this.period = period;
        this.policy = tree.uniformPolicy();
        this.magnet = tree.uniformPolicy();
        this.logits = new double[tree.infoSets()][tree.maxActions()];
        this.q = new double[tree.infoSets()][tree.maxActions()];
        this.reach = new double[tree.infoSets()];
    }

    /**
     * @param tree the game
     * @param alpha the magnet strength
     * @param eta the step size
     * @return a magnetic-mirror-descent learner
     */
    public static RegularizedEquilibriumLearner mmd(GameTree tree, double alpha, double eta) {
        return new RegularizedEquilibriumLearner(tree, Method.MMD, alpha, eta, Integer.MAX_VALUE);
    }

    /**
     * @param tree the game
     * @param alpha the magnet strength
     * @param eta the step size
     * @param period iterations between magnet moves
     * @return an R-NaD learner
     */
    public static RegularizedEquilibriumLearner rnad(GameTree tree, double alpha, double eta, int period) {
        return new RegularizedEquilibriumLearner(tree, Method.RNAD, alpha, eta, period);
    }

    /**
     * @param tree the game
     * @param alpha the KL coefficient
     * @param learningRate the logit step size
     * @param period steps between reference refreshes
     * @return a Nash-policy-gradient learner
     */
    public static RegularizedEquilibriumLearner nashPg(GameTree tree, double alpha, double learningRate, int period) {
        return new RegularizedEquilibriumLearner(tree, Method.NASH_PG, alpha, learningRate, period);
    }

    /**
     * @param tree the game
     * @param eta the step size
     * @return a multiplicative-weights learner (no magnet)
     */
    public static RegularizedEquilibriumLearner unregularized(GameTree tree, double eta) {
        return new RegularizedEquilibriumLearner(tree, Method.UNREGULARIZED, 0, eta, Integer.MAX_VALUE);
    }

    /**
     * Starts from a given profile instead of uniform (for example off-equilibrium, to watch
     * the trajectories).
     *
     * @param start the starting policy table
     * @return this
     */
    public RegularizedEquilibriumLearner start(double[][] start) {
        for (int s = 0; s < policy.length; s++) {
            System.arraycopy(start[s], 0, policy[s], 0, policy[s].length);
            for (int a = 0; a < policy[s].length; a++) {
                logits[s][a] = policy[s][a] > 0 ? Math.log(policy[s][a]) : -30;
            }
        }
        return this;
    }

    /** @return the method */
    public Method method() {
        return method;
    }

    /** @return iterations so far */
    public int iterations() {
        return iterations;
    }

    /**
     * Runs iterations.
     *
     * @param count how many
     * @return this
     */
    public RegularizedEquilibriumLearner run(int count) {
        for (int i = 0; i < count; i++) {
            iterate();
        }
        return this;
    }

    /** One simultaneous update of both players. */
    public void iterate() {
        tree.counterfactualValues(policy, q, reach);
        for (int s = 0; s < policy.length; s++) {
            if (reach[s] <= 0) {
                continue;
            }
            boolean[] legal = tree.legal(s);
            double[] pi = policy[s];
            if (method == Method.NASH_PG) {
                double base = 0;
                double[] g = new double[pi.length];
                for (int a = 0; a < pi.length; a++) {
                    if (legal[a]) {
                        g[a] = q[s][a] / reach[s] - alpha * Math.log(Math.max(pi[a], 1e-300) / magnet[s][a]);
                        base += pi[a] * g[a];
                    }
                }
                for (int a = 0; a < pi.length; a++) {
                    if (legal[a]) {
                        logits[s][a] += step * pi[a] * (g[a] - base);
                    }
                }
                continue;
            }
            double scale = 1 / (1 + alpha * step);
            double max = Double.NEGATIVE_INFINITY;
            double[] e = new double[pi.length];
            for (int a = 0; a < pi.length; a++) {
                if (legal[a]) {
                    e[a] = scale * (Math.log(Math.max(pi[a], 1e-300)) + alpha * step * Math.log(magnet[s][a])
                            + step * q[s][a] / reach[s]);
                    max = Math.max(max, e[a]);
                }
            }
            double z = 0;
            for (int a = 0; a < pi.length; a++) {
                pi[a] = legal[a] ? Math.exp(e[a] - max) : 0;
                z += pi[a];
            }
            for (int a = 0; a < pi.length; a++) {
                pi[a] /= z;
            }
        }
        if (method == Method.NASH_PG) {
            PolicySpaceResponseOracles.softmaxRows(tree, logits, policy, -1);
        }
        iterations++;
        if (iterations % period == 0) {
            for (int s = 0; s < policy.length; s++) {
                for (int a = 0; a < policy[s].length; a++) {
                    magnet[s][a] = Math.max(policy[s][a], tree.legal(s)[a] ? 1e-12 : 0);
                }
            }
        }
    }

    /** @return the current (last-iterate) policy table, a copy */
    public double[][] policy() {
        double[][] out = new double[policy.length][];
        for (int s = 0; s < out.length; s++) {
            out[s] = policy[s].clone();
        }
        return out;
    }
}
