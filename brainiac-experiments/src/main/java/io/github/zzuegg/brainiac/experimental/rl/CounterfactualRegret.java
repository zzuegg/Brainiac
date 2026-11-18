package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * Counterfactual regret minimisation for two-player zero-sum imperfect-information games:
 * at every information set, accumulate the regret of not having played each action, weighted
 * by the probability that the opponent and chance reach it (the counterfactual value), and
 * play the next iteration by regret matching (proportional to positive regret). The
 * <em>average</em> strategy converges to a Nash equilibrium, with exploitability
 * {@code O(1/sqrt(T))}.
 * <ul>
 *   <li>{@link #cfr}: vanilla CFR with simultaneous updates and a reach-weighted average
 *       (Zinkevich et al. 2007).</li>
 *   <li>{@link #cfrPlus}: CFR+ clips cumulative regrets at zero (regret-matching+), alternates
 *       the two players' updates and weights the average linearly in the iteration, which in
 *       practice converges far faster (Tammelin 2014,
 *       <a href="https://arxiv.org/abs/1407.5042">arXiv:1407.5042</a>).</li>
 *   <li>{@link #mccfr}: external-sampling Monte Carlo CFR samples one chance outcome and one
 *       opponent action at each of their nodes and explores all of the traverser's actions,
 *       so an iteration touches only a small part of the tree (Lanctot et al. 2009,
 *       <a href="https://papers.nips.cc/paper/3713-monte-carlo-sampling-for-regret-minimization-in-extensive-games">NeurIPS 2009</a>).</li>
 * </ul>
 *
 * <pre>{@code
 * GameTree kuhn = GameTree.compile(new KuhnPoker());
 * CounterfactualRegret solver = CounterfactualRegret.cfrPlus(kuhn);
 * solver.run(1000);
 * double e = Exploitability.of(kuhn, solver.averagePolicy());
 * }</pre>
 */
public final class CounterfactualRegret {

    /** The algorithm. */
    public enum Variant {
        /** Vanilla CFR. */
        CFR,
        /** CFR+. */
        CFR_PLUS,
        /** External-sampling Monte Carlo CFR. */
        MCCFR
    }

    private final GameTree tree;
    private final Variant variant;
    private final double[][] regret;
    private final double[][] strategySum;
    private final double[][] current;
    private final SplittableRandom random;
    private int iterations;
    private long touched;

    private CounterfactualRegret(GameTree tree, Variant variant, long seed) {
        this.tree = tree;
        this.variant = variant;
        int m = tree.infoSets();
        this.regret = new double[m][tree.maxActions()];
        this.strategySum = new double[m][tree.maxActions()];
        this.current = tree.uniformPolicy();
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param tree the game
     * @return a vanilla CFR solver
     */
    public static CounterfactualRegret cfr(GameTree tree) {
        return new CounterfactualRegret(tree, Variant.CFR, 0);
    }

    /**
     * @param tree the game
     * @return a CFR+ solver
     */
    public static CounterfactualRegret cfrPlus(GameTree tree) {
        return new CounterfactualRegret(tree, Variant.CFR_PLUS, 0);
    }

    /**
     * @param tree the game
     * @param seed the sampling seed
     * @return an external-sampling MCCFR solver
     */
    public static CounterfactualRegret mccfr(GameTree tree, long seed) {
        return new CounterfactualRegret(tree, Variant.MCCFR, seed);
    }

    /** @return the algorithm */
    public Variant variant() {
        return variant;
    }

    /** @return iterations run so far */
    public int iterations() {
        return iterations;
    }

    /** @return nodes visited so far, a hardware-independent measure of work */
    public long nodesTouched() {
        return touched;
    }

    /**
     * Runs iterations.
     *
     * @param count how many
     * @return this solver
     */
    public CounterfactualRegret run(int count) {
        for (int i = 0; i < count; i++) {
            iterate();
        }
        return this;
    }

    /** Runs one iteration (for MCCFR, one sampled traversal per player). */
    public void iterate() {
        iterations++;
        switch (variant) {
            case CFR -> {
                refresh();
                cfr(0, 1, 1, 1, -1, 1);
            }
            case CFR_PLUS -> {
                for (int p = 0; p < 2; p++) {
                    refresh();
                    cfr(0, 1, 1, 1, p, iterations);
                }
            }
            case MCCFR -> {
                for (int p = 0; p < 2; p++) {
                    sampled(0, p);
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    private void refresh() {
        for (int s = 0; s < regret.length; s++) {
            match(s, current[s]);
        }
    }

    private void match(int set, double[] out) {
        boolean[] legal = tree.legal(set);
        double total = 0;
        int count = 0;
        for (int a = 0; a < legal.length; a++) {
            if (legal[a]) {
                total += Math.max(regret[set][a], 0);
                count++;
            }
        }
        for (int a = 0; a < legal.length; a++) {
            out[a] = !legal[a] ? 0 : total > 0 ? Math.max(regret[set][a], 0) / total : 1.0 / count;
        }
    }

    /**
     * Full-width CFR pass.
     *
     * @param updater the player whose regrets are updated, or -1 for both
     * @param weight the average-strategy weight of this iteration
     * @return the value of the node to player 0
     */
    private double cfr(int node, double r0, double r1, double rc, int updater, double weight) {
        touched++;
        if (tree.terminal(node)) {
            return tree.payoff(node);
        }
        int[] kids = tree.children(node);
        int p = tree.player(node);
        if (p == ExtensiveFormGame.CHANCE) {
            double v = 0;
            double[] probs = tree.chance(node);
            for (int a = 0; a < kids.length; a++) {
                if (kids[a] >= 0) {
                    v += probs[a] * cfr(kids[a], r0, r1, rc * probs[a], updater, weight);
                }
            }
            return v;
        }
        int s = tree.infoSet(node);
        double[] sigma = current[s];
        double own = p == 0 ? r0 : r1;
        double opp = (p == 0 ? r1 : r0) * rc;
        if (own == 0 && opp == 0) {
            return 0;
        }
        double[] values = new double[kids.length];
        double v = 0;
        for (int a = 0; a < kids.length; a++) {
            if (kids[a] < 0) {
                continue;
            }
            values[a] = cfr(kids[a], p == 0 ? r0 * sigma[a] : r0, p == 1 ? r1 * sigma[a] : r1, rc, updater, weight);
            v += sigma[a] * values[a];
        }
        if (updater < 0 || updater == p) {
            double sign = p == 0 ? 1 : -1;
            for (int a = 0; a < kids.length; a++) {
                if (kids[a] >= 0) {
                    regret[s][a] += opp * sign * (values[a] - v);
                    if (variant == Variant.CFR_PLUS) {
                        regret[s][a] = Math.max(regret[s][a], 0);
                    }
                    strategySum[s][a] += weight * own * sigma[a];
                }
            }
        }
        return v;
    }

    /**
     * External-sampling traversal for one player.
     *
     * @return the sampled value of the node to the traverser
     */
    private double sampled(int node, int traverser) {
        touched++;
        if (tree.terminal(node)) {
            return traverser == 0 ? tree.payoff(node) : -tree.payoff(node);
        }
        int[] kids = tree.children(node);
        int p = tree.player(node);
        if (p == ExtensiveFormGame.CHANCE) {
            return sampled(kids[tree.sampleChance(node, random)], traverser);
        }
        int s = tree.infoSet(node);
        double[] sigma = new double[kids.length];
        match(s, sigma);
        if (p != traverser) {
            for (int a = 0; a < kids.length; a++) {
                strategySum[s][a] += sigma[a];
            }
            return sampled(kids[tree.sample(sigma, s, random)], traverser);
        }
        double[] values = new double[kids.length];
        double v = 0;
        for (int a = 0; a < kids.length; a++) {
            if (kids[a] >= 0) {
                values[a] = sampled(kids[a], traverser);
                v += sigma[a] * values[a];
            }
        }
        for (int a = 0; a < kids.length; a++) {
            if (kids[a] >= 0) {
                regret[s][a] += values[a] - v;
            }
        }
        return v;
    }

    /** @return the average strategy, which converges to a Nash equilibrium */
    public double[][] averagePolicy() {
        double[][] out = tree.uniformPolicy();
        for (int s = 0; s < out.length; s++) {
            double total = 0;
            for (double x : strategySum[s]) {
                total += x;
            }
            if (total > 0) {
                for (int a = 0; a < out[s].length; a++) {
                    out[s][a] = strategySum[s][a] / total;
                }
            }
        }
        return out;
    }

    /** @return the current (last-iterate) regret-matching strategy */
    public double[][] currentPolicy() {
        double[][] out = new double[regret.length][tree.maxActions()];
        for (int s = 0; s < out.length; s++) {
            match(s, out[s]);
        }
        return out;
    }
}
