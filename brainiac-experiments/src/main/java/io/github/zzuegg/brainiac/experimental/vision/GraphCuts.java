package io.github.zzuegg.brainiac.experimental.vision;

import java.util.Arrays;

/**
 * Energy minimisation for Potts MRFs on a 4-connected pixel grid,
 * {@code E(x) = sum_p D_p(x_p) + lambda sum_(p,q) [x_p != x_q]}:
 *
 * <ul>
 *   <li>{@link #binary}: the exact minimum of a two-label energy as a minimum s-t cut (Greig,
 *       Porteous &amp; Seheult, JRSS-B 1989), by Dinic's blocking-flow max-flow (BFS levels
 *       and DFS augmentation) rather than Boykov-Kolmogorov's search-tree reuse;</li>
 *   <li>{@link #alphaExpansion}: one binary cut per label until no expansion lowers the
 *       energy (Boykov, Veksler &amp; Zabih, TPAMI 2001), with pairwise terms reparametrised
 *       as in Kolmogorov &amp; Zabih (TPAMI 2004);</li>
 *   <li>{@link #icm}: iterated conditional modes (Besag, JRSS-B 1986);</li>
 *   <li>{@link #beliefPropagation}: damped min-sum loopy belief propagation.</li>
 * </ul>
 *
 * Unaries are {@code [labels][h * w]}.
 */
public final class GraphCuts {

    private GraphCuts() {
    }

    /** A flow network in compressed adjacency form, built by {@link #addEdge}. */
    public static final class FlowNetwork {

        private final int nodes;
        private int[] to;
        private int[] next;
        private double[] cap;
        private final int[] head;
        private int edges;
        private int[] level;
        private int[] iter;

        /**
         * Creates an empty network.
         *
         * @param nodes the node count
         * @param expectedEdges the expected edge count (grows as needed)
         */
        public FlowNetwork(int nodes, int expectedEdges) {
            this.nodes = nodes;
            this.head = new int[nodes];
            Arrays.fill(head, -1);
            int m = Math.max(4, 2 * expectedEdges);
            to = new int[m];
            next = new int[m];
            cap = new double[m];
        }

        /**
         * Adds {@code u -> v} with capacity {@code c} and {@code v -> u} with {@code reverse}.
         *
         * @param u the tail
         * @param v the head
         * @param c the forward capacity
         * @param reverse the backward capacity
         */
        public void addEdge(int u, int v, double c, double reverse) {
            if (edges + 2 > to.length) {
                to = Arrays.copyOf(to, to.length * 2);
                next = Arrays.copyOf(next, next.length * 2);
                cap = Arrays.copyOf(cap, cap.length * 2);
            }
            to[edges] = v;
            cap[edges] = c;
            next[edges] = head[u];
            head[u] = edges++;
            to[edges] = u;
            cap[edges] = reverse;
            next[edges] = head[v];
            head[v] = edges++;
        }

        private boolean bfs(int s, int t) {
            level = new int[nodes];
            Arrays.fill(level, -1);
            int[] queue = new int[nodes];
            int qh = 0;
            int qt = 0;
            queue[qt++] = s;
            level[s] = 0;
            while (qh < qt) {
                int u = queue[qh++];
                for (int e = head[u]; e >= 0; e = next[e]) {
                    if (cap[e] > 1e-12 && level[to[e]] < 0) {
                        level[to[e]] = level[u] + 1;
                        queue[qt++] = to[e];
                    }
                }
            }
            return level[t] >= 0;
        }

        private double dfs(int s, int t, double limit) {
            int[] stack = new int[nodes];
            int[] via = new int[nodes];
            int depth = 0;
            stack[0] = s;
            double total = 0;
            while (true) {
                int u = stack[depth];
                if (u == t) {
                    double f = limit - total;
                    for (int d = 0; d < depth; d++) {
                        f = Math.min(f, cap[via[d]]);
                    }
                    for (int d = 0; d < depth; d++) {
                        cap[via[d]] -= f;
                        cap[via[d] ^ 1] += f;
                    }
                    total += f;
                    if (total >= limit) {
                        return total;
                    }
                    depth = 0;
                    continue;
                }
                boolean advanced = false;
                for (; iter[u] >= 0; iter[u] = next[iter[u]]) {
                    int e = iter[u];
                    if (cap[e] > 1e-12 && level[to[e]] == level[u] + 1) {
                        via[depth] = e;
                        stack[++depth] = to[e];
                        advanced = true;
                        break;
                    }
                }
                if (!advanced) {
                    if (depth == 0) {
                        return total;
                    }
                    level[u] = -1;
                    depth--;
                    iter[stack[depth]] = next[iter[stack[depth]]];
                }
            }
        }

        /**
         * Computes the maximum flow.
         *
         * @param s the source
         * @param t the sink
         * @return the flow value
         */
        public double maxFlow(int s, int t) {
            double flow = 0;
            while (bfs(s, t)) {
                iter = head.clone();
                flow += dfs(s, t, Double.POSITIVE_INFINITY);
            }
            return flow;
        }

        /**
         * After {@link #maxFlow}, the nodes reachable from the source in the residual graph.
         *
         * @param s the source
         * @return per node whether it is on the source side
         */
        public boolean[] sourceSide(int s) {
            boolean[] seen = new boolean[nodes];
            int[] queue = new int[nodes];
            int qh = 0;
            int qt = 0;
            queue[qt++] = s;
            seen[s] = true;
            while (qh < qt) {
                int u = queue[qh++];
                for (int e = head[u]; e >= 0; e = next[e]) {
                    if (cap[e] > 1e-12 && !seen[to[e]]) {
                        seen[to[e]] = true;
                        queue[qt++] = to[e];
                    }
                }
            }
            return seen;
        }
    }

    /**
     * The Potts energy of a labelling.
     *
     * @param unary {@code [labels][h * w]}
     * @param labels the labelling
     * @param h the height
     * @param w the width
     * @param lambda the smoothness weight
     * @return the energy
     */
    public static double energy(double[][] unary, int[] labels, int h, int w, double lambda) {
        double e = 0;
        for (int p = 0; p < h * w; p++) {
            e += unary[labels[p]][p];
            int x = p % w;
            if (x + 1 < w && labels[p] != labels[p + 1]) {
                e += lambda;
            }
            if (p + w < h * w && labels[p] != labels[p + w]) {
                e += lambda;
            }
        }
        return e;
    }

    /**
     * @param unary {@code [labels][h * w]}
     * @return the per-pixel argmin of the unaries
     */
    public static int[] argmin(double[][] unary) {
        int n = unary[0].length;
        int[] out = new int[n];
        for (int p = 0; p < n; p++) {
            for (int l = 1; l < unary.length; l++) {
                if (unary[l][p] < unary[out[p]][p]) {
                    out[p] = l;
                }
            }
        }
        return out;
    }

    /**
     * The exact minimum of a binary energy by one minimum cut.
     *
     * @param cost0 per pixel the cost of label 0
     * @param cost1 per pixel the cost of label 1
     * @param h the height
     * @param w the width
     * @param lambda the smoothness weight
     * @return the labels
     */
    public static int[] binary(double[] cost0, double[] cost1, int h, int w, double lambda) {
        int n = h * w;
        double[] d1 = new double[n];
        for (int p = 0; p < n; p++) {
            d1[p] = cost1[p] - cost0[p];
        }
        double[][] pair = new double[n][2];
        for (double[] row : pair) {
            row[0] = lambda;
            row[1] = lambda;
        }
        return solve(d1, pair, h, w);
    }

    /**
     * Solves {@code sum_p d1_p x_p + sum_(p,q) c_pq [x_p = 0, x_q = 1] + c_qp [x_p = 1, x_q = 0]}
     * with pairwise weights per right and down neighbour, symmetric here.
     */
    private static int[] solve(double[] d1, double[][] pair, int h, int w) {
        int n = h * w;
        int s = n;
        int t = n + 1;
        FlowNetwork g = new FlowNetwork(n + 2, 3 * n);
        for (int p = 0; p < n; p++) {
            if (d1[p] > 0) {
                g.addEdge(s, p, d1[p], 0);
            } else if (d1[p] < 0) {
                g.addEdge(p, t, -d1[p], 0);
            }
            int x = p % w;
            if (x + 1 < w && pair[p][0] > 0) {
                g.addEdge(p, p + 1, pair[p][0], pair[p][0]);
            }
            if (p + w < n && pair[p][1] > 0) {
                g.addEdge(p, p + w, pair[p][1], pair[p][1]);
            }
        }
        g.maxFlow(s, t);
        boolean[] src = g.sourceSide(s);
        int[] out = new int[n];
        for (int p = 0; p < n; p++) {
            out[p] = src[p] ? 0 : 1;
        }
        return out;
    }

    /**
     * Alpha-expansion for a Potts energy.
     *
     * @param unary {@code [labels][h * w]}
     * @param h the height
     * @param w the width
     * @param lambda the smoothness weight
     * @param cycles the most cycles over all labels
     * @return the labels
     */
    public static int[] alphaExpansion(double[][] unary, int h, int w, double lambda, int cycles) {
        int n = h * w;
        int[] x = argmin(unary);
        double best = energy(unary, x, h, w, lambda);
        for (int c = 0; c < cycles; c++) {
            boolean improved = false;
            for (int alpha = 0; alpha < unary.length; alpha++) {
                double[] d1 = new double[n];
                int s = n;
                int t = n + 1;
                FlowNetwork g = new FlowNetwork(n + 2, 3 * n);
                for (int p = 0; p < n; p++) {
                    d1[p] += unary[alpha][p] - unary[x[p]][p];
                }
                for (int p = 0; p < n; p++) {
                    int[] nb = {p % w + 1 < w ? p + 1 : -1, p + w < n ? p + w : -1};
                    for (int q : nb) {
                        if (q < 0) {
                            continue;
                        }
                        double a = x[p] != x[q] ? lambda : 0;
                        double b = x[p] != alpha ? lambda : 0;
                        double cc = x[q] != alpha ? lambda : 0;
                        d1[p] += cc - a;
                        d1[q] += 0 - cc;
                        double edge = b + cc - a;
                        if (edge > 0) {
                            g.addEdge(p, q, edge, 0);
                        }
                    }
                }
                for (int p = 0; p < n; p++) {
                    if (d1[p] > 0) {
                        g.addEdge(s, p, d1[p], 0);
                    } else if (d1[p] < 0) {
                        g.addEdge(p, t, -d1[p], 0);
                    }
                }
                g.maxFlow(s, t);
                boolean[] src = g.sourceSide(s);
                int[] cand = x.clone();
                for (int p = 0; p < n; p++) {
                    if (!src[p]) {
                        cand[p] = alpha;
                    }
                }
                double e = energy(unary, cand, h, w, lambda);
                if (e < best - 1e-9) {
                    best = e;
                    x = cand;
                    improved = true;
                }
            }
            if (!improved) {
                break;
            }
        }
        return x;
    }

    /**
     * Iterated conditional modes from the unary argmin.
     *
     * @param unary {@code [labels][h * w]}
     * @param h the height
     * @param w the width
     * @param lambda the smoothness weight
     * @param sweeps the sweeps
     * @return the labels
     */
    public static int[] icm(double[][] unary, int h, int w, double lambda, int sweeps) {
        int n = h * w;
        int[] x = argmin(unary);
        for (int s = 0; s < sweeps; s++) {
            boolean changed = false;
            for (int p = 0; p < n; p++) {
                int best = x[p];
                double bestCost = Double.POSITIVE_INFINITY;
                for (int l = 0; l < unary.length; l++) {
                    double c = unary[l][p];
                    int px = p % w;
                    if (px > 0 && x[p - 1] != l) {
                        c += lambda;
                    }
                    if (px + 1 < w && x[p + 1] != l) {
                        c += lambda;
                    }
                    if (p - w >= 0 && x[p - w] != l) {
                        c += lambda;
                    }
                    if (p + w < n && x[p + w] != l) {
                        c += lambda;
                    }
                    if (c < bestCost) {
                        bestCost = c;
                        best = l;
                    }
                }
                changed |= best != x[p];
                x[p] = best;
            }
            if (!changed) {
                break;
            }
        }
        return x;
    }

    /**
     * Damped min-sum loopy belief propagation.
     *
     * @param unary {@code [labels][h * w]}
     * @param h the height
     * @param w the width
     * @param lambda the smoothness weight
     * @param iterations the message-passing rounds
     * @param damping the share of the old message kept, such as 0.5
     * @return the labels
     */
    public static int[] beliefPropagation(double[][] unary, int h, int w, double lambda, int iterations,
            double damping) {
        int n = h * w;
        int k = unary.length;
        // msg[d][p][l]: message into p from its neighbour in direction d (0 left, 1 right, 2 up, 3 down)
        double[][][] msg = new double[4][n][k];
        int[] dx = {-1, 1, 0, 0};
        int[] dy = {0, 0, -1, 1};
        int[] opposite = {1, 0, 3, 2};
        for (int it = 0; it < iterations; it++) {
            double[][][] next = new double[4][n][k];
            for (int p = 0; p < n; p++) {
                int px = p % w;
                int py = p / w;
                for (int d = 0; d < 4; d++) {
                    int qx = px + dx[d];
                    int qy = py + dy[d];
                    if (qx < 0 || qy < 0 || qx >= w || qy >= h) {
                        continue;
                    }
                    int q = qy * w + qx;
                    // message from p to q: q sees it as coming from direction opposite[d]
                    double[] h0 = new double[k];
                    double min = Double.POSITIVE_INFINITY;
                    for (int l = 0; l < k; l++) {
                        h0[l] = unary[l][p];
                        for (int e = 0; e < 4; e++) {
                            if (e != d) {
                                h0[l] += msg[e][p][l];
                            }
                        }
                        min = Math.min(min, h0[l]);
                    }
                    double[] m = next[opposite[d]][q];
                    double norm = Double.POSITIVE_INFINITY;
                    for (int l = 0; l < k; l++) {
                        m[l] = Math.min(h0[l], min + lambda);
                        norm = Math.min(norm, m[l]);
                    }
                    for (int l = 0; l < k; l++) {
                        m[l] = damping * msg[opposite[d]][q][l] + (1 - damping) * (m[l] - norm);
                    }
                }
            }
            msg = next;
        }
        int[] x = new int[n];
        for (int p = 0; p < n; p++) {
            double best = Double.POSITIVE_INFINITY;
            for (int l = 0; l < k; l++) {
                double b = unary[l][p];
                for (int d = 0; d < 4; d++) {
                    b += msg[d][p][l];
                }
                if (b < best) {
                    best = b;
                    x[p] = l;
                }
            }
        }
        return x;
    }
}
