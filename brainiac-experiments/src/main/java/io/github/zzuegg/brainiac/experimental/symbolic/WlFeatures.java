package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Weisfeiler-Leman features of planning states, as in WL-GOOSE (Chen, Trevizan &amp; Thiébaux,
 * "Return to tradition: learning reliable heuristics with classical machine learning", ICAPS 2024,
 * <a href="https://arxiv.org/abs/2403.16508">arXiv:2403.16508</a>).
 *
 * <p>A state and the goal form an instance learning graph (ILG): one node per object and one per
 * fact that holds or is a goal, coloured by its predicate name and status (achieved goal, true
 * non-goal fact, unachieved goal); every fact node is linked to its argument objects by edges
 * labelled with the argument position. A few rounds of WL colour refinement (a node's new colour
 * hashes its colour with the multiset of (edge label, neighbour colour)) are run, and the feature
 * vector counts every colour seen in every round. Colours are named by predicate names, not by
 * indices, so the features of different-sized instances of a domain live in one space; the
 * colour vocabulary grows while fitting and is then frozen (unseen colours are dropped).
 *
 * <p>The same graph optionally carries action nodes (coloured by schema and linked to their
 * argument objects), which gives the action-set features of {@link PartialSpaceSearch}.
 */
public final class WlFeatures {

    private final int iterations;
    private final Map<Long, Integer> vocabulary = new HashMap<>();
    private boolean frozen;

    /**
     * Creates an empty feature map.
     *
     * @param iterations WL rounds (2-4 in the paper)
     */
    public WlFeatures(int iterations) {
        this.iterations = iterations;
    }

    /** @return the number of colours (the feature dimension) */
    public int dimension() {
        return vocabulary.size();
    }

    /** Stops growing the vocabulary. */
    public void freeze() {
        frozen = true;
    }

    /** @return whether the vocabulary is frozen */
    public boolean frozen() {
        return frozen;
    }

    /**
     * An instance learning graph.
     *
     * @param colours the initial colour of every node
     * @param edges per node, pairs {@code (neighbour, label)} flattened
     */
    public record Graph(long[] colours, int[][] edges) {

        /** @return the number of nodes */
        public int nodes() {
            return colours.length;
        }
    }

    private static long hash(String s) {
        long h = 1125899906842597L;
        for (int i = 0; i < s.length(); i++) {
            h = 31 * h + s.charAt(i);
        }
        return mix(h);
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 33)) * 0xff51afd7ed558ccdL;
        z = (z ^ (z >>> 33)) * 0xc4ceb9fe1a85ec53L;
        return z ^ (z >>> 33);
    }

    /**
     * Builds the ILG of a state (and optionally a set of actions).
     *
     * @param task the task
     * @param state the state
     * @param actions actions to add as nodes, or null
     * @return the graph
     */
    public static Graph graph(StripsTask task, BitSet state, List<StripsTask.Action> actions) {
        int objects = task.objects().size();
        BitSet goal = new BitSet();
        for (int g : task.goal()) {
            goal.set(g);
        }
        BitSet facts = (BitSet) state.clone();
        facts.or(goal);
        int nf = facts.cardinality();
        int na = actions == null ? 0 : actions.size();
        int n = objects + nf + na;
        long[] colours = new long[n];
        List<List<Integer>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adj.add(new ArrayList<>());
        }
        long obj = hash("#object");
        for (int i = 0; i < objects; i++) {
            colours[i] = obj;
        }
        int node = objects;
        for (int f = facts.nextSetBit(0); f >= 0; f = facts.nextSetBit(f + 1)) {
            boolean in = state.get(f);
            boolean g = goal.get(f);
            String status = in && g ? "ag" : in ? "t" : "ug";
            colours[node] = hash(task.predicates().get(task.factPredicate(f)) + "/" + status);
            int[] args = task.factArgs(f);
            for (int k = 0; k < args.length; k++) {
                link(adj, node, args[k], k);
            }
            node++;
        }
        if (actions != null) {
            for (StripsTask.Action a : actions) {
                colours[node] = hash("!" + task.schemas().get(a.schema()));
                for (int k = 0; k < a.args().length; k++) {
                    link(adj, node, a.args()[k], 16 + k);
                }
                node++;
            }
        }
        int[][] edges = new int[n][];
        for (int i = 0; i < n; i++) {
            edges[i] = adj.get(i).stream().mapToInt(Integer::intValue).toArray();
        }
        return new Graph(colours, edges);
    }

    private static void link(List<List<Integer>> adj, int a, int b, int label) {
        adj.get(a).add(b);
        adj.get(a).add(label);
        adj.get(b).add(a);
        adj.get(b).add(label);
    }

    /**
     * The colour histogram of a graph over all WL rounds.
     *
     * @param g the graph
     * @param grow whether new colours enter the vocabulary (ignored once frozen)
     * @return the counts, one per vocabulary colour (the dimension at call time)
     */
    public double[] histogram(Graph g, boolean grow) {
        int n = g.nodes();
        long[] c = g.colours().clone();
        List<long[]> rounds = new ArrayList<>();
        rounds.add(c);
        for (int it = 0; it < iterations; it++) {
            long[] next = new long[n];
            for (int v = 0; v < n; v++) {
                int[] e = g.edges()[v];
                long[] msgs = new long[e.length / 2];
                for (int k = 0; k < e.length; k += 2) {
                    msgs[k / 2] = mix(c[e[k]] * 31 + e[k + 1]);
                }
                Arrays.sort(msgs);
                long h = mix(c[v] ^ 0x9e3779b97f4a7c15L);
                for (long m : msgs) {
                    h = mix(h * 1_000_003L + m);
                }
                next[v] = h;
            }
            c = next;
            rounds.add(c);
        }
        Map<Integer, Integer> counts = new HashMap<>();
        for (long[] r : rounds) {
            for (long col : r) {
                Integer idx = vocabulary.get(col);
                if (idx == null) {
                    if (!grow || frozen) {
                        continue;
                    }
                    idx = vocabulary.size();
                    vocabulary.put(col, idx);
                }
                counts.merge(idx, 1, Integer::sum);
            }
        }
        double[] out = new double[vocabulary.size()];
        counts.forEach((k, v) -> out[k] = v);
        return out;
    }

    /**
     * WL features of a state.
     *
     * @param task the task
     * @param state the state
     * @param grow whether new colours enter the vocabulary
     * @return the features (padded to the current dimension)
     */
    public double[] features(StripsTask task, BitSet state, boolean grow) {
        return histogram(graph(task, state, null), grow);
    }

    /**
     * WL features of a state together with a set of actions.
     *
     * @param task the task
     * @param state the state
     * @param actions the action set
     * @param grow whether new colours enter the vocabulary
     * @return the features
     */
    public double[] features(StripsTask task, BitSet state, List<StripsTask.Action> actions, boolean grow) {
        return histogram(graph(task, state, actions), grow);
    }

    /**
     * Pads feature vectors made while the vocabulary grew to the final dimension.
     *
     * @param rows the vectors
     * @return padded copies
     */
    public double[][] pad(List<double[]> rows) {
        double[][] out = new double[rows.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = Arrays.copyOf(rows.get(i), dimension());
        }
        return out;
    }
}
