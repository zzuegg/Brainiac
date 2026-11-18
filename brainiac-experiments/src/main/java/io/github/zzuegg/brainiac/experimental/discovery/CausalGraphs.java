package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.GreedyEquivalenceSearch;
import io.github.zzuegg.brainiac.experimental.ml.Notears;

import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Scores and oracles for causal-graph learners: SHD against the true CPDAG, orientation
 * precision and recall, the order divergence of a topological order, a d-separation oracle (so a
 * true PAG can be computed by running {@link Fci} on it), and ancestral mark scores for PAGs.
 * Graphs are adjacency matrices with non-zero {@code g[i][j]} meaning {@code i -> j}; CPDAGs use
 * the convention of {@link GreedyEquivalenceSearch#cpdag}.
 */
public final class CausalGraphs {

    private CausalGraphs() {
    }

    /**
     * SHD between the true DAG's CPDAG and an estimated CPDAG (or DAG).
     *
     * @param dag the true DAG
     * @param estimate the estimate
     * @return the number of pairs whose edge type differs
     */
    public static int cpdagShd(double[][] dag, double[][] estimate) {
        return Notears.shd(GreedyEquivalenceSearch.cpdag(dag), estimate);
    }

    /**
     * Orientation precision and recall against the true CPDAG: of the estimated directed edges,
     * the share that the CPDAG directs the same way; of the CPDAG's directed edges, the share the
     * estimate directs the same way.
     *
     * @param dag the true DAG
     * @param estimate an estimated CPDAG or DAG
     * @return {@code {precision, recall}} (precision 1 when nothing is directed)
     */
    public static double[] orientation(double[][] dag, double[][] estimate) {
        double[][] t = GreedyEquivalenceSearch.cpdag(dag);
        int d = t.length;
        int est = 0;
        int truth = 0;
        int hit = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                boolean e = estimate[i][j] != 0 && estimate[j][i] == 0;
                boolean r = t[i][j] != 0 && t[j][i] == 0;
                est += e ? 1 : 0;
                truth += r ? 1 : 0;
                hit += e && r ? 1 : 0;
            }
        }
        return new double[] {est == 0 ? 1 : hit / (double) est, truth == 0 ? 1 : hit / (double) truth};
    }

    /**
     * The order divergence: the number of true edges pointing against an estimated order.
     *
     * @param order a topological order (first is the most upstream)
     * @param dag the true DAG
     * @return the count
     */
    public static int orderDivergence(int[] order, double[][] dag) {
        int d = order.length;
        int[] pos = new int[d];
        for (int k = 0; k < d; k++) {
            pos[order[k]] = k;
        }
        int bad = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (dag[i][j] != 0 && pos[i] > pos[j]) {
                    bad++;
                }
            }
        }
        return bad;
    }

    /**
     * The share of ordered pairs {@code (i, j)} with {@code i} an ancestor of {@code j} that an
     * order places correctly.
     *
     * @param order the order
     * @param dag the true DAG
     * @return the share, 1 when there are no ancestral pairs
     */
    public static double orderedPairs(int[] order, double[][] dag) {
        boolean[][] anc = ancestors(dag);
        int d = order.length;
        int[] pos = new int[d];
        for (int k = 0; k < d; k++) {
            pos[order[k]] = k;
        }
        int all = 0;
        int ok = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i != j && anc[i][j]) {
                    all++;
                    ok += pos[i] < pos[j] ? 1 : 0;
                }
            }
        }
        return all == 0 ? 1 : ok / (double) all;
    }

    /**
     * The ancestor relation of a DAG.
     *
     * @param dag the DAG
     * @return {@code anc[i][j]}: a directed path {@code i -> ... -> j} exists
     */
    public static boolean[][] ancestors(double[][] dag) {
        int d = dag.length;
        boolean[][] a = new boolean[d][d];
        for (int i = 0; i < d; i++) {
            Deque<Integer> stack = new ArrayDeque<>();
            stack.push(i);
            while (!stack.isEmpty()) {
                int u = stack.pop();
                for (int v = 0; v < d; v++) {
                    if (dag[u][v] != 0 && !a[i][v]) {
                        a[i][v] = true;
                        stack.push(v);
                    }
                }
            }
        }
        return a;
    }

    /**
     * Whether {@code i} and {@code j} are d-separated by {@code s} in a DAG (moralised ancestral
     * graph criterion of Lauritzen et al. 1990).
     *
     * @param dag the DAG
     * @param i one node
     * @param j the other
     * @param s the conditioning set
     * @return true when d-separated
     */
    public static boolean dSeparated(double[][] dag, int i, int j, int[] s) {
        int d = dag.length;
        boolean[] keep = new boolean[d];
        Deque<Integer> stack = new ArrayDeque<>();
        stack.push(i);
        stack.push(j);
        for (int v : s) {
            stack.push(v);
        }
        while (!stack.isEmpty()) {
            int u = stack.pop();
            if (keep[u]) {
                continue;
            }
            keep[u] = true;
            for (int p = 0; p < d; p++) {
                if (dag[p][u] != 0 && !keep[p]) {
                    stack.push(p);
                }
            }
        }
        boolean[][] moral = new boolean[d][d];
        for (int c = 0; c < d; c++) {
            if (!keep[c]) {
                continue;
            }
            for (int p = 0; p < d; p++) {
                if (keep[p] && dag[p][c] != 0) {
                    moral[p][c] = true;
                    moral[c][p] = true;
                    for (int q = 0; q < d; q++) {
                        if (q != p && keep[q] && dag[q][c] != 0) {
                            moral[p][q] = true;
                        }
                    }
                }
            }
        }
        boolean[] blocked = new boolean[d];
        for (int v : s) {
            blocked[v] = true;
        }
        boolean[] seen = new boolean[d];
        stack.push(i);
        seen[i] = true;
        while (!stack.isEmpty()) {
            int u = stack.pop();
            if (u == j) {
                return false;
            }
            for (int v = 0; v < d; v++) {
                if (moral[u][v] && keep[v] && !blocked[v] && !seen[v]) {
                    seen[v] = true;
                    stack.push(v);
                }
            }
        }
        return true;
    }

    /**
     * A perfect CI test from the true DAG over a subset of observed nodes (the rest are hidden).
     *
     * @param dag the full DAG
     * @param observed the observed nodes; test indices refer to positions in this array
     * @return p-value 1 for d-separation, 0 otherwise
     */
    public static CiTest oracle(double[][] dag, int[] observed) {
        return (i, j, s) -> {
            int[] mapped = new int[s.length];
            for (int k = 0; k < s.length; k++) {
                mapped[k] = observed[s[k]];
            }
            return dSeparated(dag, observed[i], observed[j], mapped) ? 1 : 0;
        };
    }

    /**
     * Converts a CPDAG to PAG marks (directed {@code i -> j} becomes tail at {@code i}, arrow at
     * {@code j}; undirected edges become circles on both ends).
     *
     * @param cpdag the CPDAG
     * @return marks in the {@link Fci} convention
     */
    public static int[][] marks(double[][] cpdag) {
        int d = cpdag.length;
        int[][] m = new int[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                boolean ij = cpdag[i][j] != 0;
                boolean ji = cpdag[j][i] != 0;
                if (ij && ji) {
                    m[i][j] = Fci.CIRCLE;
                } else if (ij) {
                    m[i][j] = Fci.ARROW;
                } else if (ji) {
                    m[i][j] = Fci.TAIL;
                }
            }
        }
        return m;
    }

    /**
     * Ancestral mark scores of PAG-style marks against the true DAG restricted to the observed
     * nodes: an arrowhead at {@code j} on {@code i *-> j} is correct when {@code j} is not an
     * ancestor of {@code i}; a tail at {@code i} on {@code i -> j} is correct when {@code i} is an
     * ancestor of {@code j}.
     *
     * @param marks estimated marks over the observed nodes
     * @param dag the full DAG
     * @param observed the observed nodes
     * @return {@code {arrowhead precision, arrowhead recall, tail precision, arrowheads, tails}}
     */
    public static double[] ancestralScores(int[][] marks, double[][] dag, int[] observed) {
        boolean[][] anc = ancestors(dag);
        int d = observed.length;
        int arrows = 0;
        int arrowsOk = 0;
        int trueArrows = 0;
        int tails = 0;
        int tailsOk = 0;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (i == j || marks[i][j] == Fci.NONE) {
                    continue;
                }
                boolean jAncI = anc[observed[j]][observed[i]];
                trueArrows += jAncI ? 0 : 1;
                if (marks[i][j] == Fci.ARROW) {
                    arrows++;
                    arrowsOk += jAncI ? 0 : 1;
                }
                if (marks[j][i] == Fci.TAIL) {
                    tails++;
                    tailsOk += anc[observed[i]][observed[j]] ? 1 : 0;
                }
            }
        }
        return new double[] {arrows == 0 ? 1 : arrowsOk / (double) arrows,
            trueArrows == 0 ? 1 : arrowsOk / (double) trueArrows, tails == 0 ? 1 : tailsOk / (double) tails, arrows,
            tails};
    }

    /**
     * The number of observed pairs whose PAG marks differ between two PAGs (an SHD for PAGs).
     *
     * @param a one PAG
     * @param b the other
     * @return differing pairs
     */
    public static int pagDistance(int[][] a, int[][] b) {
        int d = a.length;
        int dist = 0;
        for (int i = 0; i < d; i++) {
            for (int j = i + 1; j < d; j++) {
                if (a[i][j] != b[i][j] || a[j][i] != b[j][i]) {
                    dist++;
                }
            }
        }
        return dist;
    }
}
