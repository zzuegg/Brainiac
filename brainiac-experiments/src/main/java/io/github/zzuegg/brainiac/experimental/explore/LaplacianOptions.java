package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Options from the graph Laplacian of a grid world's state-transition graph:
 *
 * <ul>
 *   <li>{@link #eigenvectors}: the exact eigenvectors of the combinatorial Laplacian
 *       {@code L = D - W} (proto-value functions), by core {@link Linalg#eigh};</li>
 *   <li>{@link #eigenoptions}: eigenoptions (Machado, Bellemare &amp; Bowling, ICML 2017,
 *       <a href="https://arxiv.org/abs/1703.00956">arXiv:1703.00956</a>): for eigenvector
 *       {@code e} (both signs) the option maximising the intrinsic reward {@code e(s') - e(s)}
 *       until no step increases it, so it runs to an end of the graph's slowest direction
 *       (through the bottlenecks);</li>
 *   <li>{@link #coveringOptions}: covering options (Jinnai et al., ICML 2019): connect the two
 *       extremes of the Fiedler vector by a point-to-point option (both directions), add that
 *       edge to the graph and repeat, which provably lowers the expected cover time;</li>
 *   <li>{@link #learnedLaplacian}: the Laplacian representation learned from sampled transitions
 *       by the spectral graph-drawing objective of Wu, Tucker &amp; Nachum (ICLR 2019,
 *       <a href="https://arxiv.org/abs/1810.04586">arXiv:1810.04586</a>):
 *       {@code E_(s,s')||f(s) - f(s')||^2 + beta E_(u,v)[(f(u).f(v))^2 - ||f(u)||^2 - ||f(v)||^2 + d]},
 *       compared with the true eigenvectors by subspace cosines.</li>
 * </ul>
 *
 * <p>{@link #coverTime} and {@link #hittingTime} measure exploration by a uniformly random agent
 * that may also start any available option (each option step counts as a primitive step).
 */
public final class LaplacianOptions {

    private LaplacianOptions() {
    }

    /**
     * A grid world as a graph: free cells, 4-neighbourhood moves (blocked moves stay put).
     *
     * @param rows grid rows
     * @param cols grid columns
     * @param free which cells are free
     * @param index the node index of every free cell ({@code -1} for walls)
     * @param cells the cell of every node
     */
    public record GridGraph(int rows, int cols, boolean[] free, int[] index, int[] cells) {
        /** @return the number of nodes */
        public int nodes() {
            return cells.length;
        }

        /**
         * @param node a node
         * @param action 0 up, 1 right, 2 down, 3 left
         * @return the node reached (the same node when blocked)
         */
        public int move(int node, int action) {
            int c = cells[node];
            int r = c / cols + (action == 0 ? -1 : action == 2 ? 1 : 0);
            int k = c % cols + (action == 1 ? 1 : action == 3 ? -1 : 0);
            if (r < 0 || r >= rows || k < 0 || k >= cols || !free[r * cols + k]) {
                return node;
            }
            return index[r * cols + k];
        }

        /** @return the symmetric adjacency matrix of distinct neighbours */
        public double[][] adjacency() {
            int n = nodes();
            double[][] w = new double[n][n];
            for (int v = 0; v < n; v++) {
                for (int a = 0; a < 4; a++) {
                    int u = move(v, a);
                    if (u != v) {
                        w[v][u] = 1;
                    }
                }
            }
            return w;
        }

        private static GridGraph of(int rows, int cols, boolean[] free) {
            int[] index = new int[rows * cols];
            List<Integer> cells = new ArrayList<>();
            for (int c = 0; c < rows * cols; c++) {
                index[c] = free[c] ? cells.size() : -1;
                if (free[c]) {
                    cells.add(c);
                }
            }
            return new GridGraph(rows, cols, free, index, cells.stream().mapToInt(Integer::intValue).toArray());
        }

        /**
         * Rooms separated by walls with one door per wall segment.
         *
         * @param roomsPerSide 2 for four rooms, 3 for nine rooms
         * @param roomSize free cells per room side
         * @return the graph
         */
        public static GridGraph rooms(int roomsPerSide, int roomSize) {
            int side = roomsPerSide * roomSize + roomsPerSide - 1;
            boolean[] free = new boolean[side * side];
            Arrays.fill(free, true);
            for (int k = 1; k < roomsPerSide; k++) {
                int wall = k * (roomSize + 1) - 1;
                for (int i = 0; i < side; i++) {
                    free[wall * side + i] = false;
                    free[i * side + wall] = false;
                }
            }
            for (int k = 1; k < roomsPerSide; k++) {
                int wall = k * (roomSize + 1) - 1;
                for (int room = 0; room < roomsPerSide; room++) {
                    int door = room * (roomSize + 1) + roomSize / 2;
                    free[wall * side + door] = true;
                    free[door * side + wall] = true;
                }
            }
            return of(side, side, free);
        }

        /**
         * An open grid with random wall cells (kept connected by construction: walls only where
         * they do not disconnect the free cells).
         *
         * @param side the side
         * @param wallShare the share of cells to try to wall off
         * @param seed the seed
         * @return the graph
         */
        public static GridGraph walled(int side, double wallShare, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            boolean[] free = new boolean[side * side];
            Arrays.fill(free, true);
            int target = (int) (wallShare * side * side);
            for (int tries = 0, placed = 0; placed < target && tries < 20 * target; tries++) {
                int c = r.nextInt(side * side);
                if (!free[c] || c == 0) {
                    continue;
                }
                free[c] = false;
                if (connected(side, free)) {
                    placed++;
                } else {
                    free[c] = true;
                }
            }
            return of(side, side, free);
        }

        private static boolean connected(int side, boolean[] free) {
            int total = 0;
            int startCell = -1;
            for (int c = 0; c < free.length; c++) {
                if (free[c]) {
                    total++;
                    startCell = startCell < 0 ? c : startCell;
                }
            }
            boolean[] seen = new boolean[free.length];
            ArrayDeque<Integer> q = new ArrayDeque<>();
            q.add(startCell);
            seen[startCell] = true;
            int count = 0;
            while (!q.isEmpty()) {
                int c = q.poll();
                count++;
                int r = c / side;
                int k = c % side;
                int[][] nb = {{r - 1, k}, {r + 1, k}, {r, k - 1}, {r, k + 1}};
                for (int[] p : nb) {
                    if (p[0] >= 0 && p[0] < side && p[1] >= 0 && p[1] < side) {
                        int d = p[0] * side + p[1];
                        if (free[d] && !seen[d]) {
                            seen[d] = true;
                            q.add(d);
                        }
                    }
                }
            }
            return count == total;
        }
    }

    /**
     * A deterministic option: where it may start, and the next node it moves to from every node
     * ({@code -1} where it terminates).
     *
     * @param initiation the nodes where it can start
     * @param policy the next node per node, {@code -1} to stop
     */
    public record Option(boolean[] initiation, int[] policy) {
    }

    /**
     * @param w a symmetric adjacency matrix
     * @return {eigenvalues ascending, eigenvectors as columns {@code [n][n]}}
     */
    public static double[][][] eigenvectors(double[][] w) {
        int n = w.length;
        double[] l = new double[n * n];
        for (int i = 0; i < n; i++) {
            double deg = 0;
            for (int j = 0; j < n; j++) {
                deg += w[i][j];
                l[i * n + j] = -w[i][j];
            }
            l[i * n + i] += deg;
        }
        Tensor[] eig = Linalg.eigh(Tensor.of(Shape.of(n, n), l, DType.FLOAT64, Device.cpu()));
        double[] values = eig[0].toDoubleArray();
        double[] vec = eig[1].toDoubleArray();
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                v[i][k] = vec[i * n + k];
            }
        }
        return new double[][][] {{values}, v};
    }

    /**
     * Eigenoptions for the eigenvectors {@code 1..count} (skipping the constant one), both signs.
     *
     * @param graph the graph
     * @param count eigenvectors to use
     * @param gamma the discount of the option's value iteration
     * @return {@code 2 count} options
     */
    public static List<Option> eigenoptions(GridGraph graph, int count, double gamma) {
        double[][] vec = eigenvectors(graph.adjacency())[1];
        int n = graph.nodes();
        List<Option> out = new ArrayList<>();
        for (int k = 1; k <= count && k < n; k++) {
            for (int sign = 1; sign >= -1; sign -= 2) {
                double[] e = new double[n];
                for (int v = 0; v < n; v++) {
                    e[v] = sign * vec[v][k];
                }
                out.add(eigenoption(graph, e, gamma));
            }
        }
        return out;
    }

    private static Option eigenoption(GridGraph graph, double[] e, double gamma) {
        int n = graph.nodes();
        double[] v = new double[n];
        for (int it = 0; it < 500; it++) {
            double change = 0;
            for (int s = 0; s < n; s++) {
                double best = 0;
                for (int a = 0; a < 4; a++) {
                    int t = graph.move(s, a);
                    if (t != s) {
                        best = Math.max(best, e[t] - e[s] + gamma * v[t]);
                    }
                }
                change = Math.max(change, Math.abs(best - v[s]));
                v[s] = best;
            }
            if (change < 1e-10) {
                break;
            }
        }
        int[] policy = new int[n];
        boolean[] init = new boolean[n];
        for (int s = 0; s < n; s++) {
            double best = 1e-9;
            int arg = -1;
            for (int a = 0; a < 4; a++) {
                int t = graph.move(s, a);
                if (t != s) {
                    double qv = e[t] - e[s] + gamma * v[t];
                    if (qv > best) {
                        best = qv;
                        arg = t;
                    }
                }
            }
            policy[s] = arg;
            init[s] = arg >= 0;
        }
        return new Option(init, policy);
    }

    /**
     * Covering options: {@code pairs} rounds of connecting the Fiedler extremes of the graph
     * augmented by the previous rounds' edges, each as two point options along shortest paths.
     *
     * @param graph the graph
     * @param pairs rounds (each adds two options)
     * @return the options
     */
    public static List<Option> coveringOptions(GridGraph graph, int pairs) {
        double[][] w = graph.adjacency();
        int n = graph.nodes();
        List<Option> out = new ArrayList<>();
        for (int k = 0; k < pairs; k++) {
            double[][] vec = eigenvectors(w)[1];
            int lo = 0;
            int hi = 0;
            for (int v = 1; v < n; v++) {
                if (vec[v][1] < vec[lo][1]) {
                    lo = v;
                }
                if (vec[v][1] > vec[hi][1]) {
                    hi = v;
                }
            }
            out.add(pointOption(graph, lo, hi));
            out.add(pointOption(graph, hi, lo));
            w[lo][hi] = 1;
            w[hi][lo] = 1;
        }
        return out;
    }

    /**
     * @param graph the graph
     * @param from the only initiation node
     * @param to where it terminates
     * @return an option following a shortest path
     */
    public static Option pointOption(GridGraph graph, int from, int to) {
        int n = graph.nodes();
        int[] dist = new int[n];
        Arrays.fill(dist, -1);
        ArrayDeque<Integer> q = new ArrayDeque<>();
        dist[to] = 0;
        q.add(to);
        while (!q.isEmpty()) {
            int v = q.poll();
            for (int a = 0; a < 4; a++) {
                int u = graph.move(v, a);
                if (dist[u] < 0) {
                    dist[u] = dist[v] + 1;
                    q.add(u);
                }
            }
        }
        int[] policy = new int[n];
        for (int v = 0; v < n; v++) {
            policy[v] = -1;
            if (v == to) {
                continue;
            }
            for (int a = 0; a < 4; a++) {
                int u = graph.move(v, a);
                if (dist[u] >= 0 && dist[u] == dist[v] - 1) {
                    policy[v] = u;
                    break;
                }
            }
        }
        boolean[] init = new boolean[n];
        init[from] = true;
        return new Option(init, policy);
    }

    /**
     * Mean steps until a random agent has visited every node.
     *
     * @param graph the graph
     * @param options options the agent may also pick (uniformly among primitives and available options)
     * @param start the start node
     * @param runs runs to average
     * @param maxSteps the step cap per run
     * @param seed the seed
     * @return the mean cover time (capped runs count at the cap)
     */
    public static double coverTime(GridGraph graph, List<Option> options, int start, int runs, int maxSteps,
            long seed) {
        return walk(graph, options, start, -1, runs, maxSteps, seed);
    }

    /**
     * Mean steps until a random agent first reaches a goal node.
     *
     * @param graph the graph
     * @param options options the agent may pick
     * @param start the start node
     * @param goal the goal node
     * @param runs runs to average
     * @param maxSteps the step cap per run
     * @param seed the seed
     * @return the mean hitting time
     */
    public static double hittingTime(GridGraph graph, List<Option> options, int start, int goal, int runs,
            int maxSteps, long seed) {
        return walk(graph, options, start, goal, runs, maxSteps, seed);
    }

    private static double walk(GridGraph graph, List<Option> options, int start, int goal, int runs, int maxSteps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = graph.nodes();
        double total = 0;
        List<Option> avail = new ArrayList<>();
        for (int run = 0; run < runs; run++) {
            boolean[] seen = new boolean[n];
            int left = n - 1;
            int s = start;
            seen[s] = true;
            int steps = 0;
            while (steps < maxSteps && (goal < 0 ? left > 0 : s != goal)) {
                avail.clear();
                for (Option o : options) {
                    if (o.initiation()[s]) {
                        avail.add(o);
                    }
                }
                int pick = r.nextInt(4 + avail.size());
                if (pick < 4) {
                    s = graph.move(s, pick);
                    steps++;
                    if (!seen[s]) {
                        seen[s] = true;
                        left--;
                    }
                } else {
                    Option o = avail.get(pick - 4);
                    for (int k = 0; k < 2 * n && o.policy()[s] >= 0 && steps < maxSteps; k++) {
                        s = o.policy()[s];
                        steps++;
                        if (!seen[s]) {
                            seen[s] = true;
                            left--;
                        }
                        if (s == goal) {
                            break;
                        }
                    }
                }
            }
            total += steps;
        }
        return total / runs;
    }

    /**
     * Learns a {@code d}-dimensional Laplacian representation (a table, one row per node) from
     * random-walk transitions with the graph-drawing objective, by SGD.
     *
     * @param graph the graph
     * @param d the dimension
     * @param updates SGD steps
     * @param beta the orthonormality weight
     * @param lr the step size
     * @param seed the seed
     * @return the representation {@code [nodes][d]}
     */
    public static double[][] learnedLaplacian(GridGraph graph, int d, int updates, double beta, double lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = graph.nodes();
        double[][] f = new double[n][d];
        for (double[] row : f) {
            for (int j = 0; j < d; j++) {
                row[j] = r.nextGaussian() * 0.1;
            }
        }
        int s = r.nextInt(n);
        double[] gu = new double[d];
        double[] gv = new double[d];
        for (int it = 0; it < updates; it++) {
            int t = graph.move(s, r.nextInt(4));
            // attractive term on the sampled transition
            for (int j = 0; j < d; j++) {
                double diff = f[s][j] - f[t][j];
                f[s][j] -= lr * 2 * diff;
                f[t][j] += lr * 2 * diff;
            }
            // repulsive (orthonormality) term on two independent nodes
            int u = r.nextInt(n);
            int v = r.nextInt(n);
            double dot = 0;
            for (int j = 0; j < d; j++) {
                dot += f[u][j] * f[v][j];
            }
            for (int j = 0; j < d; j++) {
                gu[j] = beta * (2 * dot * f[v][j] - 2 * f[u][j]);
                gv[j] = beta * (2 * dot * f[u][j] - 2 * f[v][j]);
            }
            // clip the single-sample repulsive step, whose quartic term otherwise diverges
            double norm = 0;
            for (int j = 0; j < d; j++) {
                norm += gu[j] * gu[j] + gv[j] * gv[j];
            }
            double scale = norm > 25 ? 5 / Math.sqrt(norm) : 1;
            for (int j = 0; j < d; j++) {
                f[u][j] -= lr * scale * gu[j];
                f[v][j] -= lr * scale * gv[j];
            }
            s = t;
        }
        return f;
    }

    /**
     * How well a learned representation spans the first {@code d} Laplacian eigenvectors: the
     * learned columns are orthonormalised and each true eigenvector's projection norm onto
     * their span is its cosine with the subspace.
     *
     * @param graph the graph
     * @param learned the representation {@code [nodes][d]}
     * @return one cosine per eigenvector {@code 0..d-1}
     */
    public static double[] subspaceCosines(GridGraph graph, double[][] learned) {
        int n = learned.length;
        int d = learned[0].length;
        double[][] basis = new double[d][];
        int rank = 0;
        for (int j = 0; j < d; j++) {
            double[] col = new double[n];
            for (int v = 0; v < n; v++) {
                col[v] = learned[v][j];
            }
            for (int k = 0; k < rank; k++) {
                double dot = dot(col, basis[k]);
                for (int v = 0; v < n; v++) {
                    col[v] -= dot * basis[k][v];
                }
            }
            double norm = Math.sqrt(dot(col, col));
            if (norm > 1e-9) {
                for (int v = 0; v < n; v++) {
                    col[v] /= norm;
                }
                basis[rank++] = col;
            }
        }
        double[][] vec = eigenvectors(graph.adjacency())[1];
        double[] cos = new double[d];
        for (int k = 0; k < d; k++) {
            double[] e = new double[n];
            for (int v = 0; v < n; v++) {
                e[v] = vec[v][k];
            }
            double norm = Math.sqrt(dot(e, e));
            double proj = 0;
            for (int b = 0; b < rank; b++) {
                double p = dot(e, basis[b]) / norm;
                proj += p * p;
            }
            cos[k] = Math.sqrt(proj);
        }
        return cos;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}
