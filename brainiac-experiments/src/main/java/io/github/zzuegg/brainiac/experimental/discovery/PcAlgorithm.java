package io.github.zzuegg.brainiac.experimental.discovery;

import java.util.ArrayList;
import java.util.List;

/**
 * The PC algorithm (Spirtes, Glymour and Scheines, <i>Causation, Prediction, and Search</i>, MIT
 * Press 2000), in its order-independent "stable" form (Colombo and Maathuis, JMLR 2014,
 * <a href="https://arxiv.org/abs/1211.3295">arXiv:1211.3295</a>).
 *
 * <ol>
 *   <li>Skeleton: start from the complete graph; for growing conditioning-set sizes
 *       {@code l = 0, 1, ...}, remove {@code i - j} when {@code X_i} and {@code X_j} test
 *       independent given some {@code l}-subset of the neighbours of {@code i} (taken from the
 *       adjacency frozen at the start of the level, which makes the result independent of the
 *       variable order), and record the separating set.</li>
 *   <li>V-structures: for every unshielded triple {@code i - k - j} with {@code k} not in
 *       {@code sep(i, j)}, orient {@code i -> k <- j}.</li>
 *   <li>Meek's rules R1 to R3 until nothing changes, giving the CPDAG (the Markov equivalence
 *       class).</li>
 * </ol>
 *
 * The CPDAG is returned as a 0/1 matrix in the convention of
 * {@code GreedyEquivalenceSearch.cpdag}: {@code g[i][j] = 1, g[j][i] = 0} is {@code i -> j}, both
 * 1 is an undirected edge.
 */
public final class PcAlgorithm {

    private final double alpha;
    private final int maxConditioning;

    /**
     * Creates the learner.
     *
     * @param alpha the significance level of the CI tests, for example {@code 0.01}
     * @param maxConditioning the largest conditioning set tried
     */
    public PcAlgorithm(double alpha, int maxConditioning) {
        this.alpha = alpha;
        this.maxConditioning = maxConditioning;
    }

    /**
     * The skeleton phase.
     *
     * @param adjacent {@code adjacent[i][j]} after the phase
     * @param sepsets the separating set of every removed pair ({@code null} if adjacent)
     * @param tests the number of CI tests run
     */
    public record Skeleton(boolean[][] adjacent, int[][][] sepsets, int tests) {

        /** @return the undirected skeleton as a 0/1 matrix */
        public double[][] matrix() {
            int d = adjacent.length;
            double[][] g = new double[d][d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    g[i][j] = adjacent[i][j] ? 1 : 0;
                }
            }
            return g;
        }
    }

    /**
     * The result.
     *
     * @param cpdag the CPDAG
     * @param skeleton the skeleton phase
     */
    public record Result(double[][] cpdag, Skeleton skeleton) {
    }

    /**
     * Runs PC.
     *
     * @param test the CI test
     * @param d the number of variables
     * @return the CPDAG and the skeleton
     */
    public Result fit(CiTest test, int d) {
        Skeleton sk = skeleton(test, d, alpha, maxConditioning);
        int[][] g = new int[d][d];   // g[i][j] = 1: an edge mark from i towards j exists
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                g[i][j] = sk.adjacent()[i][j] ? 1 : 0;
            }
        }
        orientColliders(g, sk);
        meek(g);
        double[][] out = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = g[i][j];
            }
        }
        return new Result(out, sk);
    }

    /**
     * The stable skeleton search shared with {@link Fci}.
     *
     * @param test the CI test
     * @param d variables
     * @param alpha the level
     * @param maxConditioning the largest conditioning set
     * @return the skeleton
     */
    public static Skeleton skeleton(CiTest test, int d, double alpha, int maxConditioning) {
        boolean[][] adj = new boolean[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                adj[i][j] = i != j;
            }
        }
        int[][][] sep = new int[d][d][];
        int tests = 0;
        for (int l = 0; l <= maxConditioning; l++) {
            boolean[][] frozen = new boolean[d][];
            for (int i = 0; i < d; i++) {
                frozen[i] = adj[i].clone();
            }
            boolean any = false;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    if (i == j || !adj[i][j]) {
                        continue;
                    }
                    List<Integer> nb = new ArrayList<>();
                    for (int k = 0; k < d; k++) {
                        if (k != j && frozen[i][k]) {
                            nb.add(k);
                        }
                    }
                    if (nb.size() < l) {
                        continue;
                    }
                    any = true;
                    for (int[] s : subsets(nb, l)) {
                        tests++;
                        if (test.pValue(i, j, s) > alpha) {
                            adj[i][j] = false;
                            adj[j][i] = false;
                            sep[i][j] = s;
                            sep[j][i] = s;
                            break;
                        }
                    }
                }
            }
            if (!any) {
                break;
            }
        }
        return new Skeleton(adj, sep, tests);
    }

    /**
     * All subsets of a given size.
     *
     * @param items the items
     * @param size the subset size
     * @return the subsets
     */
    public static List<int[]> subsets(List<Integer> items, int size) {
        List<int[]> out = new ArrayList<>();
        int[] idx = new int[size];
        for (int i = 0; i < size; i++) {
            idx[i] = i;
        }
        int n = items.size();
        if (size > n) {
            return out;
        }
        while (true) {
            int[] s = new int[size];
            for (int i = 0; i < size; i++) {
                s[i] = items.get(idx[i]);
            }
            out.add(s);
            int p = size - 1;
            while (p >= 0 && idx[p] == n - size + p) {
                p--;
            }
            if (p < 0) {
                return out;
            }
            idx[p]++;
            for (int q = p + 1; q < size; q++) {
                idx[q] = idx[q - 1] + 1;
            }
        }
    }

    static boolean contains(int[] s, int k) {
        if (s == null) {
            return false;
        }
        for (int v : s) {
            if (v == k) {
                return true;
            }
        }
        return false;
    }

    private static void orientColliders(int[][] g, Skeleton sk) {
        int d = g.length;
        boolean[][] adj = sk.adjacent();
        for (int k = 0; k < d; k++) {
            for (int i = 0; i < d; i++) {
                for (int j = i + 1; j < d; j++) {
                    if (adj[i][k] && adj[j][k] && !adj[i][j] && !contains(sk.sepsets()[i][j], k)) {
                        // only orient when it does not reverse an earlier collider
                        if (g[i][k] == 1 && g[j][k] == 1) {
                            g[k][i] = 0;
                            g[k][j] = 0;
                        }
                    }
                }
            }
        }
    }

    private static boolean directed(int[][] g, int a, int b) {
        return g[a][b] == 1 && g[b][a] == 0;
    }

    private static boolean undirected(int[][] g, int a, int b) {
        return g[a][b] == 1 && g[b][a] == 1;
    }

    private static boolean adjacent(int[][] g, int a, int b) {
        return g[a][b] == 1 || g[b][a] == 1;
    }

    /**
     * Applies Meek's rules R1 to R3 in place to a partially directed graph.
     *
     * @param g {@code g[i][j] = 1} marks, as in the CPDAG convention
     */
    public static void meek(int[][] g) {
        int d = g.length;
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int b = 0; b < d; b++) {
                for (int c = 0; c < d; c++) {
                    if (b == c || !undirected(g, b, c)) {
                        continue;
                    }
                    boolean orient = false;
                    for (int a = 0; a < d && !orient; a++) {
                        // R1: a -> b - c, a and c not adjacent
                        if (a != c && directed(g, a, b) && !adjacent(g, a, c)) {
                            orient = true;
                        }
                    }
                    for (int a = 0; a < d && !orient; a++) {
                        // R2: b -> a -> c with b - c
                        if (directed(g, b, a) && directed(g, a, c)) {
                            orient = true;
                        }
                    }
                    // R3: b - x, b - y, x -> c, y -> c, x and y not adjacent
                    for (int x = 0; x < d && !orient; x++) {
                        for (int y = x + 1; y < d && !orient; y++) {
                            if (x != c && y != c && undirected(g, b, x) && undirected(g, b, y) && directed(g, x, c)
                                    && directed(g, y, c) && !adjacent(g, x, y)) {
                                orient = true;
                            }
                        }
                    }
                    if (orient) {
                        g[c][b] = 0;
                        changed = true;
                    }
                }
            }
        }
    }
}
