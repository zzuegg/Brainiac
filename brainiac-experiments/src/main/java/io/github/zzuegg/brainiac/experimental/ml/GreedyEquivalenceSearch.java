package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Score-based causal structure learning in the style of Greedy Equivalence Search: a
 * forward phase that greedily adds the single edge improving the BIC score most, until no
 * addition helps, then a backward phase that greedily deletes edges while that helps, then
 * a turning phase that reverses edges while that helps. The linear-Gaussian BIC
 * {@code -n/2 log(residual variance) - (log n)/2 (parents + 1)} is decomposable and score
 * equivalent, so each step moves between Markov equivalence classes and two DAGs of the
 * same class score alike.
 *
 * <p>Simplification: the search walks over DAG representatives with single-edge
 * operators instead of Chickering's Insert/Delete operators on completed partially
 * directed graphs, so it keeps the two-phase schedule and the score but not GES's
 * large-sample optimality guarantee. {@link #cpdag} converts the result to its
 * equivalence class (compelled edges kept directed) for orientation-free comparisons.
 *
 * <p>Paper: Chickering, "Optimal Structure Identification With Greedy Search" (JMLR 2002),
 * <a href="https://www.jmlr.org/papers/v3/chickering02b.html">JMLR 3</a>.
 */
public final class GreedyEquivalenceSearch {

    private final double penalty;
    private final int maxParents;

    /**
     * Creates the search.
     *
     * @param penalty multiplies the BIC complexity term ({@code 1} is plain BIC)
     * @param maxParents the most parents a node may get
     */
    public GreedyEquivalenceSearch(double penalty, int maxParents) {
        this.penalty = penalty;
        this.maxParents = maxParents;
    }

    /**
     * Learns a DAG.
     *
     * @param x samples {@code [n][d]}
     * @return {@code W[i][j]}: the least-squares weight of edge {@code i -> j}, 0 where there is none
     */
    public double[][] fit(double[][] x) {
        int n = x.length;
        double[][] cov = Dagma.covariance(x);
        int d = cov.length;
        boolean[][] g = new boolean[d][d];
        double[] local = new double[d];
        for (int j = 0; j < d; j++) {
            local[j] = localScore(cov, g, j, n);
        }
        // forward: add edges
        while (true) {
            double best = 1e-9;
            int bi = -1;
            int bj = -1;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    if (i == j || g[i][j] || g[j][i] || parents(g, j) >= maxParents || reaches(g, j, i)) {
                        continue;
                    }
                    g[i][j] = true;
                    double gain = localScore(cov, g, j, n) - local[j];
                    g[i][j] = false;
                    if (gain > best) {
                        best = gain;
                        bi = i;
                        bj = j;
                    }
                }
            }
            if (bi < 0) {
                break;
            }
            g[bi][bj] = true;
            local[bj] = localScore(cov, g, bj, n);
        }
        // then the best single addition, deletion or reversal, until none helps
        for (int round = 0; round < 4 * d * d; round++) {
            double best = 1e-9;
            int bi = -1;
            int bj = -1;
            boolean reverse = false;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    if (!g[i][j]) {
                        continue;
                    }
                    g[i][j] = false;
                    double del = localScore(cov, g, j, n) - local[j];
                    double rev = Double.NEGATIVE_INFINITY;
                    if (parents(g, i) < maxParents && !reaches(g, i, j)) {
                        g[j][i] = true;
                        rev = del + localScore(cov, g, i, n) - local[i];
                        g[j][i] = false;
                    }
                    g[i][j] = true;
                    if (del > best) {
                        best = del;
                        bi = i;
                        bj = j;
                        reverse = false;
                    }
                    if (rev > best) {
                        best = rev;
                        bi = i;
                        bj = j;
                        reverse = true;
                    }
                }
            }
            int ai = -1;
            int aj = -1;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    if (i == j || g[i][j] || g[j][i] || parents(g, j) >= maxParents || reaches(g, j, i)) {
                        continue;
                    }
                    g[i][j] = true;
                    double gain = localScore(cov, g, j, n) - local[j];
                    g[i][j] = false;
                    if (gain > best) {
                        best = gain;
                        ai = i;
                        aj = j;
                    }
                }
            }
            if (ai >= 0) {
                g[ai][aj] = true;
                local[aj] = localScore(cov, g, aj, n);
                continue;
            }
            if (bi < 0) {
                break;
            }
            g[bi][bj] = false;
            if (reverse) {
                g[bj][bi] = true;
                local[bi] = localScore(cov, g, bi, n);
            }
            local[bj] = localScore(cov, g, bj, n);
        }
        return weights(cov, g);
    }

    private static int parents(boolean[][] g, int j) {
        int c = 0;
        for (boolean[] row : g) {
            c += row[j] ? 1 : 0;
        }
        return c;
    }

    /** True if a directed path leads from {@code from} to {@code to}. */
    private static boolean reaches(boolean[][] g, int from, int to) {
        int d = g.length;
        boolean[] seen = new boolean[d];
        int[] stack = new int[d];
        int top = 0;
        stack[top++] = from;
        seen[from] = true;
        while (top > 0) {
            int u = stack[--top];
            if (u == to) {
                return true;
            }
            for (int v = 0; v < d; v++) {
                if (g[u][v] && !seen[v]) {
                    seen[v] = true;
                    stack[top++] = v;
                }
            }
        }
        return false;
    }

    private static int[] parentList(boolean[][] g, int j) {
        int[] p = new int[parents(g, j)];
        int k = 0;
        for (int i = 0; i < g.length; i++) {
            if (g[i][j]) {
                p[k++] = i;
            }
        }
        return p;
    }

    /** Regression of {@code j} on its parents: {coefficients..., residual variance}. */
    private static double[] regress(double[][] cov, int[] pa, int j) {
        int k = pa.length;
        if (k == 0) {
            return new double[] {cov[j][j]};
        }
        double[] a = new double[k * k];
        double[] b = new double[k];
        for (int r = 0; r < k; r++) {
            for (int c = 0; c < k; c++) {
                a[r * k + c] = cov[pa[r]][pa[c]] + (r == c ? 1e-9 : 0);
            }
            b[r] = cov[pa[r]][j];
        }
        double[] beta = Linalg.solve(Tensor.of(Shape.of(k, k), a, DType.FLOAT64, Device.cpu()),
                Tensor.of(Shape.of(k, 1), b, DType.FLOAT64, Device.cpu())).toDoubleArray();
        double res = cov[j][j];
        for (int r = 0; r < k; r++) {
            res -= beta[r] * b[r];
        }
        double[] out = java.util.Arrays.copyOf(beta, k + 1);
        out[k] = Math.max(res, 1e-12);
        return out;
    }

    private double localScore(double[][] cov, boolean[][] g, int j, int n) {
        int[] pa = parentList(g, j);
        double[] r = regress(cov, pa, j);
        return -0.5 * n * Math.log(r[pa.length]) - penalty * 0.5 * Math.log(n) * (pa.length + 1);
    }

    private static double[][] weights(double[][] cov, boolean[][] g) {
        int d = g.length;
        double[][] w = new double[d][d];
        for (int j = 0; j < d; j++) {
            int[] pa = parentList(g, j);
            double[] r = regress(cov, pa, j);
            for (int k = 0; k < pa.length; k++) {
                w[pa[k]][j] = r[k] == 0 ? 1e-12 : r[k];
            }
        }
        return w;
    }

    /**
     * The completed partially directed graph of a DAG's equivalence class: edges in a
     * v-structure, and those Meek's first three rules then force, stay directed
     * ({@code out[i][j] = 1}, {@code out[j][i] = 0}); the rest become undirected (both 1).
     *
     * @param dag an adjacency, non-zero meaning {@code i -> j}
     * @return the CPDAG as a 0/1 matrix
     */
    public static double[][] cpdag(double[][] dag) {
        int d = dag.length;
        int[][] e = new int[d][d];   // 1 directed i->j, 2 undirected
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                if (dag[i][j] != 0) {
                    e[i][j] = 2;
                    e[j][i] = 2;
                }
            }
        }
        for (int j = 0; j < d; j++) {
            for (int a = 0; a < d; a++) {
                for (int b = a + 1; b < d; b++) {
                    if (dag[a][j] != 0 && dag[b][j] != 0 && dag[a][b] == 0 && dag[b][a] == 0) {
                        e[a][j] = 1;
                        e[j][a] = 0;
                        e[b][j] = 1;
                        e[j][b] = 0;
                    }
                }
            }
        }
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    if (e[a][b] != 2) {
                        continue;
                    }
                    boolean orient = false;
                    for (int c = 0; c < d && !orient; c++) {
                        // R1: c -> a - b, c and b non-adjacent  =>  a -> b
                        orient = e[c][a] == 1 && e[a][c] == 0 && e[c][b] == 0 && e[b][c] == 0 && c != b;
                        // R2: a -> c -> b and a - b  =>  a -> b
                        orient |= e[a][c] == 1 && e[c][a] == 0 && e[c][b] == 1 && e[b][c] == 0;
                    }
                    for (int c = 0; c < d && !orient; c++) {
                        for (int k = c + 1; k < d && !orient; k++) {
                            // R3: a - c -> b, a - k -> b, c and k non-adjacent  =>  a -> b
                            orient = e[a][c] == 2 && e[c][b] == 1 && e[b][c] == 0 && e[a][k] == 2 && e[k][b] == 1
                                    && e[b][k] == 0 && e[c][k] == 0 && e[k][c] == 0;
                        }
                    }
                    if (orient) {
                        e[a][b] = 1;
                        e[b][a] = 0;
                        changed = true;
                    }
                }
            }
        }
        double[][] out = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = e[i][j] != 0 ? 1 : 0;
            }
        }
        return out;
    }
}
