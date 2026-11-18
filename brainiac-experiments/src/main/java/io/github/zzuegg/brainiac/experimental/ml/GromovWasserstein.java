package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Entropic Gromov-Wasserstein alignment of two metric spaces (Memoli 2011; Peyre, Cuturi and
 * Solomon 2016, <a href="https://proceedings.mlr.press/v48/peyre16.html">ICML</a>).
 *
 * <p>The spaces are given only by their intra-domain distance matrices {@code C1 [n][n]} and
 * {@code C2 [m][m]}, so they may live in different domains (two graphs, two embedding spaces
 * related by an unknown rotation). The square-loss objective
 * {@code sum |C1(i,k) - C2(j,l)|^2 T_ij T_kl} is minimised by entropic projected gradient:
 * each outer step linearises it at the current coupling, giving the cost
 * {@code (C1^2 a) 1^T + 1 (C2^2 b)^T - 2 C1 T C2^T}, and solves an entropic OT problem
 * ({@link EntropicTransport#sinkhorn}) for the next coupling. The regularisation is annealed from
 * a large value down to {@code epsilon}, and several perturbed starts can be tried, since the
 * problem is non-convex.
 */
public final class GromovWasserstein {

    private GromovWasserstein() {
    }

    /**
     * A Gromov-Wasserstein coupling.
     *
     * @param coupling the coupling {@code [n][m]}
     * @param value the square-loss GW objective at it
     */
    public record Result(double[][] coupling, double value) {

        /**
         * The hard one-to-one matching that maximises the coupling mass (Hungarian), for
         * equal sizes.
         *
         * @return {@code p}: point {@code i} of the first space matches {@code p[i]} of the second
         */
        public int[] matching() {
            double[][] neg = new double[coupling.length][];
            for (int i = 0; i < neg.length; i++) {
                neg[i] = new double[coupling[i].length];
                for (int j = 0; j < neg[i].length; j++) {
                    neg[i][j] = -coupling[i][j];
                }
            }
            return MinibatchOptimalTransport.assignCost(neg);
        }
    }

    /**
     * Euclidean distance matrix of a point cloud, scaled so its largest entry is 1.
     *
     * @param x the points {@code [n][d]}
     * @return {@code [n][n]}
     */
    public static double[][] distances(double[][] x) {
        double[][] d = MlMatrices.squaredDistances(x);
        for (double[] row : d) {
            for (int j = 0; j < row.length; j++) {
                row[j] = Math.sqrt(Math.max(0, row[j]));
            }
        }
        return normalise(d);
    }

    /**
     * Shortest-path (hop) distances of an unweighted graph, scaled so the largest finite entry
     * is 1; disconnected pairs get 1.5.
     *
     * @param adjacency the {@code n x n} adjacency, row-major, non-zero for an edge
     * @param n the node count
     * @return {@code [n][n]}
     */
    public static double[][] shortestPaths(float[] adjacency, int n) {
        double[][] d = new double[n][n];
        for (int s = 0; s < n; s++) {
            Arrays.fill(d[s], -1);
            d[s][s] = 0;
            ArrayDeque<Integer> queue = new ArrayDeque<>();
            queue.add(s);
            while (!queue.isEmpty()) {
                int u = queue.poll();
                for (int v = 0; v < n; v++) {
                    if (adjacency[u * n + v] != 0 && d[s][v] < 0) {
                        d[s][v] = d[s][u] + 1;
                        queue.add(v);
                    }
                }
            }
        }
        double max = 0;
        for (double[] row : d) {
            for (double v : row) {
                max = Math.max(max, v);
            }
        }
        for (double[] row : d) {
            for (int j = 0; j < n; j++) {
                row[j] = row[j] < 0 ? 1.5 : row[j] / max;
            }
        }
        return d;
    }

    private static double[][] normalise(double[][] d) {
        double max = 0;
        for (double[] row : d) {
            for (double v : row) {
                max = Math.max(max, v);
            }
        }
        for (double[] row : d) {
            for (int j = 0; j < row.length; j++) {
                row[j] /= max;
            }
        }
        return d;
    }

    /**
     * The linearised cost {@code (C1^2 a) 1^T + 1 (C2^2 b)^T - 2 C1 T C2^T}.
     *
     * @param c1 the first distance matrix
     * @param c2 the second distance matrix
     * @param a the first weights
     * @param b the second weights
     * @param t the coupling
     * @return {@code [n][m]}
     */
    public static double[][] linearisedCost(double[][] c1, double[][] c2, double[] a, double[] b, double[][] t) {
        int n = a.length;
        int m = b.length;
        double[] ra = new double[n];
        double[] rb = new double[m];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                ra[i] += c1[i][k] * c1[i][k] * a[k];
            }
        }
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < m; l++) {
                rb[j] += c2[j][l] * c2[j][l] * b[l];
            }
        }
        // C1 T: [n][m]
        double[][] c1t = new double[n][m];
        for (int i = 0; i < n; i++) {
            double[] out = c1t[i];
            for (int k = 0; k < n; k++) {
                double v = c1[i][k];
                if (v != 0) {
                    double[] row = t[k];
                    for (int l = 0; l < m; l++) {
                        out[l] += v * row[l];
                    }
                }
            }
        }
        double[][] cost = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double s = 0;
                double[] left = c1t[i];
                double[] right = c2[j];
                for (int l = 0; l < m; l++) {
                    s += left[l] * right[l];
                }
                cost[i][j] = ra[i] + rb[j] - 2 * s;
            }
        }
        return cost;
    }

    private static double objective(double[][] cost, double[][] t) {
        double v = 0;
        for (int i = 0; i < t.length; i++) {
            for (int j = 0; j < t[i].length; j++) {
                v += cost[i][j] * t[i][j];
            }
        }
        return v;
    }

    /**
     * Entropic GW from a given starting coupling.
     *
     * @param c1 the first distance matrix {@code [n][n]}
     * @param c2 the second distance matrix {@code [m][m]}
     * @param a the first weights (sum 1)
     * @param b the second weights (sum 1)
     * @param start the initial coupling, for example {@code a b^T}
     * @param epsilon the final entropic regularisation, for example 0.002 on unit-scale distances
     * @param outer the outer (linearisation) steps; the regularisation decays geometrically from
     *     {@code 20 epsilon} to {@code epsilon} over the first half of them
     * @param sinkhornIterations Sinkhorn iterations per outer step
     * @return the coupling and its objective
     */
    public static Result solve(double[][] c1, double[][] c2, double[] a, double[] b, double[][] start,
            double epsilon, int outer, int sinkhornIterations) {
        double[][] t = start;
        int anneal = Math.max(1, outer / 2);
        for (int it = 0; it < outer; it++) {
            double eps = epsilon * Math.pow(20, Math.max(0, 1 - (double) it / anneal));
            double[][] cost = linearisedCost(c1, c2, a, b, t);
            t = EntropicTransport.sinkhorn(a, b, cost, eps, sinkhornIterations, 1e-7).coupling();
        }
        return new Result(t, objective(linearisedCost(c1, c2, a, b, t), t));
    }

    /**
     * Entropic GW with uniform weights, the best of several starts: the product coupling and
     * randomly perturbed copies of it.
     *
     * @param c1 the first distance matrix
     * @param c2 the second distance matrix
     * @param epsilon the final regularisation
     * @param outer the outer steps per start
     * @param restarts the number of starts, at least 1
     * @param seed the perturbation seed
     * @return the result with the lowest objective
     */
    public static Result align(double[][] c1, double[][] c2, double epsilon, int outer, int restarts, long seed) {
        double[] a = EntropicTransport.uniform(c1.length);
        double[] b = EntropicTransport.uniform(c2.length);
        SplittableRandom r = new SplittableRandom(seed);
        Result best = null;
        for (int s = 0; s < restarts; s++) {
            double[][] start = new double[a.length][b.length];
            double total = 0;
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < b.length; j++) {
                    start[i][j] = a[i] * b[j] * (s == 0 ? 1 : Math.exp(r.nextGaussian()));
                    total += start[i][j];
                }
            }
            for (double[] row : start) {
                for (int j = 0; j < row.length; j++) {
                    row[j] /= total;
                }
            }
            Result res = solve(c1, c2, a, b, start, epsilon, outer, 200);
            if (best == null || res.value() < best.value()) {
                best = res;
            }
        }
        return best;
    }
}
