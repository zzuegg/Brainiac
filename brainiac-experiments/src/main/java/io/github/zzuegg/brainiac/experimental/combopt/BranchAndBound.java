package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Ilp;
import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Branch and bound for binary packing ILPs ({@code max c.x}, {@code A x <= b}, {@code A >= 0},
 * {@code x in {0,1}}) over {@link LinearProgram} relaxations, with pluggable branching rules,
 * the test bed of Gasse et al.'s learned branching.
 *
 * <p>Each node fixes some variables; its relaxation drops integrality of the rest (with
 * {@code x <= 1} rows). A node is pruned when its LP bound cannot beat the incumbent, and the
 * floor of every LP solution (always feasible for a packing ILP) updates the incumbent. The
 * branching variable is chosen among the fractional ones by:
 * <ul>
 *   <li>{@link Rule#MOST_FRACTIONAL}: the value closest to 0.5;</li>
 *   <li>{@link Rule#PSEUDOCOST}: the product of the estimated bound drops of both children,
 *       from per-variable averages of the drops seen so far (unseen variables get the mean);</li>
 *   <li>{@link Rule#STRONG}: solve both child LPs for every candidate and take the best
 *       product of drops, which gives small trees at a high cost per node;</li>
 *   <li>{@link Rule#LEARNED}: a {@link Scorer}, such as {@link BipartiteBranchingGnn}
 *       trained to imitate strong branching.</li>
 * </ul>
 * With {@link #record(boolean)} every node also stores its strong-branching scores, the
 * imitation targets.
 *
 * <p>Paper: Gasse, Chételat, Ferroni, Charlin and Lodi, "Exact Combinatorial Optimization with
 * Graph Convolutional Neural Networks" (NeurIPS 2019),
 * <a href="https://arxiv.org/abs/1906.01629">arXiv:1906.01629</a>.
 */
public final class BranchAndBound {

    /** Branching rules. */
    public enum Rule {
        /** closest to one half */
        MOST_FRACTIONAL,
        /** pseudocost product */
        PSEUDOCOST,
        /** full strong branching */
        STRONG,
        /** a learned scorer */
        LEARNED
    }

    /** Scores branching candidates at a node; higher is better. */
    @FunctionalInterface
    public interface Scorer {
        /**
         * @param node the node with its LP solution
         * @param candidates the fractional variables
         * @return one score per candidate
         */
        double[] score(NodeState node, int[] candidates);
    }

    /**
     * A node's relaxation, the input of a learned rule.
     *
     * @param ilp the problem
     * @param fixed per variable -1 (free), 0 or 1
     * @param x the LP solution over all variables (fixed ones at their values)
     * @param dual the LP duals of the {@code A} rows (0 when the row has no free variable)
     * @param bound the LP bound of the node
     * @param depth the node depth
     */
    public record NodeState(Ilp ilp, int[] fixed, double[] x, double[] dual, double bound, int depth) {
    }

    /**
     * A node visited with strong-branching scores, for imitation learning.
     *
     * @param node the node
     * @param candidates the fractional variables
     * @param scores the strong-branching score of each
     */
    public record Sample(NodeState node, int[] candidates, double[] scores) {
        /** @return the index (into {@code candidates}) of the strong-branching choice */
        public int best() {
            int b = 0;
            for (int i = 1; i < scores.length; i++) {
                if (scores[i] > scores[b]) {
                    b = i;
                }
            }
            return b;
        }
    }

    /**
     * A finished search.
     *
     * @param objective the best {@code c.x} found (optimal unless the node limit was hit)
     * @param x the best solution
     * @param nodes branch-and-bound nodes whose relaxation was solved
     * @param lpSolves all LP solves, strong-branching probes included
     * @param nanos the wall time
     * @param optimal whether the search finished within the node limit
     * @param samples recorded strong-branching samples (empty unless recording)
     */
    public record Result(double objective, int[] x, int nodes, int lpSolves, long nanos, boolean optimal,
            List<Sample> samples) {
    }

    private record Node(int[] fixed, double parentBound, int var, int direction, double frac, int depth) {
    }

    private record Lp(double[] x, double[] dual, double bound, boolean feasible) {
    }

    private static final double EPS = 1e-6;

    private final Rule rule;
    private boolean bestFirst;
    private boolean record;
    private Scorer scorer;
    private int nodeLimit = 100_000;

    private BranchAndBound(Rule rule) {
        this.rule = rule;
    }

    /**
     * A solver with a branching rule (depth-first by default).
     *
     * @param rule the rule
     * @return the solver
     */
    public static BranchAndBound of(Rule rule) {
        return new BranchAndBound(rule);
    }

    /**
     * @param on best-bound node selection instead of depth-first
     * @return this
     */
    public BranchAndBound bestFirst(boolean on) {
        this.bestFirst = on;
        return this;
    }

    /**
     * @param on record strong-branching scores at every node (costs strong branching's LPs)
     * @return this
     */
    public BranchAndBound record(boolean on) {
        this.record = on;
        return this;
    }

    /**
     * @param s the scorer of {@link Rule#LEARNED}
     * @return this
     */
    public BranchAndBound scorer(Scorer s) {
        this.scorer = s;
        return this;
    }

    /**
     * @param limit the most nodes before giving up on proving optimality
     * @return this
     */
    public BranchAndBound nodeLimit(int limit) {
        this.nodeLimit = limit;
        return this;
    }

    /**
     * Solves the ILP.
     *
     * @param ilp a packing ILP
     * @return the result
     */
    public Result solve(Ilp ilp) {
        if (rule == Rule.LEARNED && scorer == null) {
            throw new IllegalStateException("the learned rule needs a scorer");
        }
        long start = System.nanoTime();
        int n = ilp.variables();
        double[] pcSum0 = new double[n];
        double[] pcSum1 = new double[n];
        int[] pcCount0 = new int[n];
        int[] pcCount1 = new int[n];
        int[] lpSolves = {0};
        int nodes = 0;
        double incumbent = 0;
        int[] best = new int[n];
        List<Sample> samples = new ArrayList<>();
        PriorityQueue<Object[]> heap = new PriorityQueue<>((p, q) -> Double.compare((double) q[1], (double) p[1]));
        Deque<Node> stack = new ArrayDeque<>();
        int[] root = new int[n];
        Arrays.fill(root, -1);
        Node first = new Node(root, Double.POSITIVE_INFINITY, -1, 0, 0, 0);
        if (bestFirst) {
            heap.add(new Object[] {first, Double.POSITIVE_INFINITY});
        } else {
            stack.push(first);
        }
        boolean finished = true;
        while (bestFirst ? !heap.isEmpty() : !stack.isEmpty()) {
            Node node = bestFirst ? (Node) heap.poll()[0] : stack.pop();
            if (node.parentBound() <= incumbent + EPS) {
                continue;
            }
            if (nodes >= nodeLimit) {
                finished = false;
                break;
            }
            Lp lp = relax(ilp, node.fixed(), lpSolves);
            nodes++;
            if (node.var() >= 0 && lp.feasible() && Double.isFinite(node.parentBound())) {
                double drop = Math.max(0, node.parentBound() - lp.bound());
                if (node.direction() == 0) {
                    pcSum0[node.var()] += drop / Math.max(EPS, node.frac());
                    pcCount0[node.var()]++;
                } else {
                    pcSum1[node.var()] += drop / Math.max(EPS, 1 - node.frac());
                    pcCount1[node.var()]++;
                }
            }
            if (!lp.feasible() || lp.bound() <= incumbent + EPS) {
                continue;
            }
            int[] floor = new int[n];
            double value = 0;
            List<Integer> frac = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                floor[j] = (int) Math.floor(lp.x()[j] + EPS);
                value += ilp.c()[j] * floor[j];
                if (lp.x()[j] > EPS && lp.x()[j] < 1 - EPS) {
                    frac.add(j);
                }
            }
            if (value > incumbent + EPS) {
                incumbent = value;
                best = floor;
            }
            if (frac.isEmpty() || lp.bound() <= incumbent + EPS) {
                continue;
            }
            int[] cand = frac.stream().mapToInt(Integer::intValue).toArray();
            NodeState state = new NodeState(ilp, node.fixed(), lp.x(), lp.dual(), lp.bound(), node.depth());
            double[] strong = null;
            if (rule == Rule.STRONG || record) {
                strong = new double[cand.length];
                for (int k = 0; k < cand.length; k++) {
                    double b0 = child(ilp, node.fixed(), cand[k], 0, lpSolves);
                    double b1 = child(ilp, node.fixed(), cand[k], 1, lpSolves);
                    strong[k] = Math.max(lp.bound() - b0, 1e-3) * Math.max(lp.bound() - b1, 1e-3);
                }
                if (record) {
                    samples.add(new Sample(state, cand, strong));
                }
            }
            double[] score = switch (rule) {
                case STRONG -> strong;
                case LEARNED -> scorer.score(state, cand);
                case MOST_FRACTIONAL -> {
                    double[] s = new double[cand.length];
                    for (int k = 0; k < cand.length; k++) {
                        s[k] = -Math.abs(lp.x()[cand[k]] - 0.5);
                    }
                    yield s;
                }
                case PSEUDOCOST -> {
                    double mean0 = mean(pcSum0, pcCount0);
                    double mean1 = mean(pcSum1, pcCount1);
                    double[] s = new double[cand.length];
                    for (int k = 0; k < cand.length; k++) {
                        int j = cand[k];
                        double f = lp.x()[j];
                        double p0 = pcCount0[j] > 0 ? pcSum0[j] / pcCount0[j] : mean0;
                        double p1 = pcCount1[j] > 0 ? pcSum1[j] / pcCount1[j] : mean1;
                        s[k] = Math.max(f * p0, 1e-6) * Math.max((1 - f) * p1, 1e-6);
                    }
                    yield s;
                }
            };
            int pick = 0;
            for (int k = 1; k < cand.length; k++) {
                if (score[k] > score[pick]) {
                    pick = k;
                }
            }
            int j = cand[pick];
            double f = lp.x()[j];
            int[] down = node.fixed().clone();
            down[j] = 0;
            int[] up = node.fixed().clone();
            up[j] = 1;
            Node d = new Node(down, lp.bound(), j, 0, f, node.depth() + 1);
            Node u = new Node(up, lp.bound(), j, 1, f, node.depth() + 1);
            if (bestFirst) {
                heap.add(new Object[] {d, lp.bound()});
                heap.add(new Object[] {u, lp.bound()});
            } else if (f >= 0.5) {
                stack.push(d);
                stack.push(u);
            } else {
                stack.push(u);
                stack.push(d);
            }
        }
        return new Result(incumbent, best, nodes, lpSolves[0], System.nanoTime() - start, finished, samples);
    }

    private static double mean(double[] sum, int[] count) {
        double s = 0;
        int c = 0;
        for (int j = 0; j < sum.length; j++) {
            if (count[j] > 0) {
                s += sum[j] / count[j];
                c++;
            }
        }
        return c == 0 ? 1 : s / c;
    }

    private static double child(Ilp ilp, int[] fixed, int j, int value, int[] lpSolves) {
        int[] f = fixed.clone();
        f[j] = value;
        Lp lp = relax(ilp, f, lpSolves);
        return lp.feasible() ? lp.bound() : Double.NEGATIVE_INFINITY;
    }

    /**
     * Solves a node's relaxation.
     *
     * @param ilp the problem
     * @param fixed per variable -1, 0 or 1
     * @param lpSolves a counter, incremented
     * @return the relaxation
     */
    private static Lp relax(Ilp ilp, int[] fixed, int[] lpSolves) {
        int n = ilp.variables();
        int m = ilp.b().length;
        double[] b = ilp.b().clone();
        double base = 0;
        List<Integer> free = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            if (fixed[j] == 1) {
                base += ilp.c()[j];
                for (int i = 0; i < m; i++) {
                    b[i] -= ilp.a()[i][j];
                }
            } else if (fixed[j] < 0) {
                free.add(j);
            }
        }
        for (double v : b) {
            if (v < -EPS) {
                return new Lp(null, null, Double.NEGATIVE_INFINITY, false);
            }
        }
        double[] x = new double[n];
        for (int j = 0; j < n; j++) {
            x[j] = Math.max(0, fixed[j]);
        }
        double[] dual = new double[m];
        if (free.isEmpty()) {
            return new Lp(x, dual, base, true);
        }
        int k = free.size();
        double[] c = new double[k];
        double[][] a = new double[m + k][k];
        double[] bb = new double[m + k];
        for (int q = 0; q < k; q++) {
            c[q] = ilp.c()[free.get(q)];
            a[m + q][q] = 1;
            bb[m + q] = 1;
        }
        for (int i = 0; i < m; i++) {
            bb[i] = Math.max(0, b[i]);
            for (int q = 0; q < k; q++) {
                a[i][q] = ilp.a()[i][free.get(q)];
            }
        }
        lpSolves[0]++;
        LinearProgram.Solution s = LinearProgram.maximize(c, a, bb);
        for (int q = 0; q < k; q++) {
            x[free.get(q)] = Math.max(0, Math.min(1, s.x()[q]));
        }
        System.arraycopy(s.dual(), 0, dual, 0, m);
        return new Lp(x, dual, base + s.objective(), true);
    }
}
