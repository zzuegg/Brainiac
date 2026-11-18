package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Step-by-step execution traces of classical graph algorithms, for neural algorithmic
 * reasoning (Veličković et al., CLRS, <a href="https://arxiv.org/abs/2205.15659">arXiv:2205.15659</a>).
 * Every trace is a random weighted undirected graph, a source node, and the algorithm's
 * state after each step (the <em>hints</em>); the last state is the output:
 * <ul>
 *   <li>{@link Algorithm#BFS}: reachability from the source, one frontier expansion per
 *       step (node classes 0/1) on a sparse graph that may be disconnected;</li>
 *   <li>{@link Algorithm#BELLMAN_FORD}: shortest-path predecessor pointers after each
 *       synchronous relaxation round (unreached nodes and the source point to
 *       themselves), on a connected graph;</li>
 *   <li>{@link Algorithm#PRIM}: minimum-spanning-tree predecessor pointers after each node
 *       Prim's algorithm adds (nodes outside the tree point to themselves).</li>
 * </ul>
 */
public final class AlgorithmTraces {

    private AlgorithmTraces() {
    }

    /** The traced algorithm. */
    public enum Algorithm {
        /** breadth-first reachability */
        BFS,
        /** Bellman-Ford shortest paths */
        BELLMAN_FORD,
        /** Prim's minimum spanning tree */
        PRIM;

        /** @return whether the per-node output is a pointer to a node (else a 0/1 class) */
        public boolean pointer() {
            return this != BFS;
        }
    }

    /**
     * One execution.
     *
     * @param nodes the node count
     * @param weights {@code nodes x nodes}, row-major, symmetric; 0 = no edge
     * @param source the start node
     * @param hints {@code [steps][nodes]}: the state after each step (class or pointer)
     */
    public record Trace(int nodes, float[] weights, int source, int[][] hints) {

        /** @return the final state, the algorithm's output */
        public int[] output() {
            return hints[hints.length - 1];
        }

        /** @return the number of steps */
        public int steps() {
            return hints.length;
        }
    }

    /**
     * Generates a trace.
     *
     * @param algorithm the algorithm
     * @param nodes the node count
     * @param seed the seed
     * @return the trace
     */
    public static Trace generate(Algorithm algorithm, int nodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] w = new float[nodes * nodes];
        if (algorithm == Algorithm.BFS) {
            double p = 1.2 / nodes;
            for (int i = 0; i < nodes; i++) {
                for (int j = i + 1; j < nodes; j++) {
                    if (r.nextDouble() < p) {
                        link(w, nodes, i, j, 1);
                    }
                }
            }
        } else {
            for (int i = 1; i < nodes; i++) {
                link(w, nodes, i, r.nextInt(i), (float) (0.1 + r.nextDouble()));
            }
            int extra = nodes;
            for (int e = 0; e < extra; e++) {
                int i = r.nextInt(nodes);
                int j = r.nextInt(nodes);
                if (i != j && w[i * nodes + j] == 0) {
                    link(w, nodes, i, j, (float) (0.1 + r.nextDouble()));
                }
            }
        }
        int source = r.nextInt(nodes);
        int[][] hints = switch (algorithm) {
            case BFS -> bfs(w, nodes, source);
            case BELLMAN_FORD -> bellmanFord(w, nodes, source);
            case PRIM -> prim(w, nodes, source);
        };
        return new Trace(nodes, w, source, hints);
    }

    private static void link(float[] w, int n, int i, int j, float weight) {
        w[i * n + j] = weight;
        w[j * n + i] = weight;
    }

    private static int[][] bfs(float[] w, int n, int source) {
        int[] reach = new int[n];
        reach[source] = 1;
        java.util.List<int[]> steps = new java.util.ArrayList<>();
        steps.add(reach.clone());
        while (true) {
            int[] next = reach.clone();
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (reach[j] == 1 && w[i * n + j] != 0) {
                        next[i] = 1;
                    }
                }
            }
            if (Arrays.equals(next, reach)) {
                break;
            }
            reach = next;
            steps.add(reach.clone());
        }
        return steps.toArray(int[][]::new);
    }

    private static int[][] bellmanFord(float[] w, int n, int source) {
        double[] d = new double[n];
        Arrays.fill(d, Double.POSITIVE_INFINITY);
        d[source] = 0;
        int[] pi = new int[n];
        for (int i = 0; i < n; i++) {
            pi[i] = i;
        }
        java.util.List<int[]> steps = new java.util.ArrayList<>();
        steps.add(pi.clone());
        while (true) {
            double[] nd = d.clone();
            int[] np = pi.clone();
            for (int v = 0; v < n; v++) {
                for (int u = 0; u < n; u++) {
                    if (w[v * n + u] != 0 && d[u] + w[v * n + u] < nd[v] - 1e-9) {
                        nd[v] = d[u] + w[v * n + u];
                        np[v] = u;
                    }
                }
            }
            if (Arrays.equals(np, pi) && Arrays.equals(nd, d)) {
                break;
            }
            d = nd;
            pi = np;
            steps.add(pi.clone());
        }
        return steps.toArray(int[][]::new);
    }

    private static int[][] prim(float[] w, int n, int source) {
        boolean[] in = new boolean[n];
        double[] key = new double[n];
        int[] parent = new int[n];
        Arrays.fill(key, Double.POSITIVE_INFINITY);
        key[source] = 0;
        int[] pi = new int[n];
        for (int i = 0; i < n; i++) {
            pi[i] = i;
            parent[i] = i;
        }
        java.util.List<int[]> steps = new java.util.ArrayList<>();
        for (int s = 0; s < n; s++) {
            int best = -1;
            for (int v = 0; v < n; v++) {
                if (!in[v] && (best < 0 || key[v] < key[best])) {
                    best = v;
                }
            }
            if (best < 0 || key[best] == Double.POSITIVE_INFINITY) {
                break;
            }
            in[best] = true;
            pi[best] = parent[best];
            for (int v = 0; v < n; v++) {
                if (!in[v] && w[best * n + v] != 0 && w[best * n + v] < key[v]) {
                    key[v] = w[best * n + v];
                    parent[v] = best;
                }
            }
            steps.add(pi.clone());
        }
        return steps.toArray(int[][]::new);
    }
}
