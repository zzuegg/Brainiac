package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.combopt.BranchAndBound;
import io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Small combinatorial-optimisation instances with exact or best-known optima, the shared
 * test bed of the neural combinatorial-optimisation blocks:
 *
 * <ul>
 *   <li>Euclidean TSP on the unit square, uniform or clustered (Gaussian blobs), with the
 *       Held–Karp optimum for {@code n <= 13} and the best of multi-start 2-opt + Or-opt
 *       otherwise;</li>
 *   <li>CVRP with a depot and integer demands 1-9 (capacity 30 for 20 customers, as in
 *       Kool et al.), with a best-known value from nearest-neighbour and sweep starts
 *       improved route by route;</li>
 *   <li>Erdős–Rényi and random regular graphs for MaxCut and maximum independent set,
 *       with exact references for small graphs (Gray-code enumeration of cuts up to 22
 *       nodes; bitmask branch and bound for MIS up to 64 nodes);</li>
 *   <li>0/1 knapsack (strongly correlated, exact by dynamic programming) and set cover
 *       (exact by branch and bound over {@code LinearProgram} relaxations), both written as
 *       packing ILPs {@code max c.x, A x <= b, A >= 0, x in {0,1}}; set cover is stored in
 *       complemented variables {@code y = 1 - x}, so its cover cost is
 *       {@code offset - c.y}.</li>
 * </ul>
 */
public final class CombinatorialInstances {

    private CombinatorialInstances() {
    }

    /**
     * A Euclidean TSP instance.
     *
     * @param coords {@code [n][2]} in the unit square
     * @param optimum the optimal (or best-known) tour length
     * @param exact whether {@code optimum} is proven optimal
     */
    public record Tsp(double[][] coords, double optimum, boolean exact) {
        /** @return the node count */
        public int size() {
            return coords.length;
        }

        /** @return the distance matrix */
        public double[][] distances() {
            return ClassicHeuristics.distances(coords);
        }
    }

    /**
     * A CVRP instance; node 0 is the depot.
     *
     * @param coords {@code [n + 1][2]}, depot first
     * @param demand demands, {@code demand[0] = 0}
     * @param capacity the vehicle capacity
     * @param bestKnown the best-known solution length
     */
    public record Cvrp(double[][] coords, double[] demand, double capacity, double bestKnown) {
        /** @return customers plus the depot */
        public int size() {
            return coords.length;
        }
    }

    /**
     * A reference value of a graph problem.
     *
     * @param value the best value known
     * @param exact whether it is proven optimal
     */
    public record Reference(double value, boolean exact) {
    }

    /**
     * A packing ILP: maximise {@code c.x} subject to {@code A x <= b}, {@code A >= 0},
     * {@code x} binary.
     *
     * @param name "knapsack" or "setcover"
     * @param c the objective
     * @param a the constraints {@code [m][n]}
     * @param b the bounds
     * @param offset for set cover, the total cost (cover cost is {@code offset - c.x});
     *     0 otherwise
     * @param optimum the optimal objective {@code c.x}
     */
    public record Ilp(String name, double[] c, double[][] a, double[] b, double offset, double optimum) {
        /** @return the variable count */
        public int variables() {
            return c.length;
        }
    }

    /**
     * A TSP instance.
     *
     * @param n nodes
     * @param clustered draw from 3 Gaussian clusters instead of uniformly
     * @param seed the seed
     * @return the instance with its optimum
     */
    public static Tsp tsp(int n, boolean clustered, long seed) {
        double[][] c = points(n, clustered, new SplittableRandom(seed));
        return withOptimum(c);
    }

    /**
     * A TSP instance from given points, with its optimum.
     *
     * @param coords the points
     * @return the instance
     */
    public static Tsp withOptimum(double[][] coords) {
        double[][] d = ClassicHeuristics.distances(coords);
        if (coords.length <= 13) {
            return new Tsp(coords, ClassicHeuristics.tourLength(d, heldKarp(d)), true);
        }
        return new Tsp(coords, ClassicHeuristics.multiStart(d, 4, 1).length(), false);
    }

    /**
     * Points without an optimum (cheap, for training batches).
     *
     * @param n nodes
     * @param clustered clustered or uniform
     * @param random the source
     * @return {@code [n][2]}
     */
    public static double[][] points(int n, boolean clustered, SplittableRandom random) {
        double[][] c = new double[n][2];
        double[][] centres = new double[3][2];
        for (double[] ce : centres) {
            ce[0] = 0.2 + 0.6 * random.nextDouble();
            ce[1] = 0.2 + 0.6 * random.nextDouble();
        }
        for (int i = 0; i < n; i++) {
            if (clustered) {
                double[] ce = centres[random.nextInt(3)];
                c[i][0] = clamp01(ce[0] + 0.08 * gaussian(random));
                c[i][1] = clamp01(ce[1] + 0.08 * gaussian(random));
            } else {
                c[i][0] = random.nextDouble();
                c[i][1] = random.nextDouble();
            }
        }
        return c;
    }

    /**
     * Several TSP instances with optima.
     *
     * @param count how many
     * @param n nodes
     * @param clustered clustered or uniform
     * @param seed the seed
     * @return the instances
     */
    public static List<Tsp> tspSet(int count, int n, boolean clustered, long seed) {
        List<Tsp> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(tsp(n, clustered, seed * 1_000_003L + i));
        }
        return out;
    }

    /**
     * Rotates (about the centre of the unit square) and optionally reflects an instance,
     * which leaves its optimum unchanged; points are rescaled into the unit square if needed.
     *
     * @param t the instance
     * @param angle the rotation in radians
     * @param reflect reflect across the vertical axis first
     * @return the transformed instance
     */
    public static Tsp rotated(Tsp t, double angle, boolean reflect) {
        int n = t.size();
        double[][] c = new double[n][2];
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            double x = (reflect ? 1 - t.coords()[i][0] : t.coords()[i][0]) - 0.5;
            double y = t.coords()[i][1] - 0.5;
            c[i][0] = cos * x - sin * y;
            c[i][1] = sin * x + cos * y;
            lo = Math.min(lo, Math.min(c[i][0], c[i][1]));
            hi = Math.max(hi, Math.max(c[i][0], c[i][1]));
        }
        double scale = Math.max(1, hi - lo);
        for (double[] p : c) {
            p[0] = (p[0] - lo) / scale;
            p[1] = (p[1] - lo) / scale;
        }
        return new Tsp(c, t.optimum() / scale, t.exact());
    }

    /**
     * The Held–Karp dynamic program: the optimal tour in {@code O(2^n n^2)}.
     *
     * @param d distances, {@code n <= 16}
     * @return an optimal tour starting at node 0
     */
    public static int[] heldKarp(double[][] d) {
        int n = d.length;
        if (n <= 3) {
            int[] t = new int[n];
            for (int i = 0; i < n; i++) {
                t[i] = i;
            }
            return t;
        }
        if (n > 16) {
            throw new IllegalArgumentException("Held-Karp is limited to 16 nodes, got " + n);
        }
        int m = n - 1;
        int full = 1 << m;
        double[] cost = new double[full * m];
        int[] parent = new int[full * m];
        Arrays.fill(cost, Double.POSITIVE_INFINITY);
        for (int j = 0; j < m; j++) {
            cost[(1 << j) * m + j] = d[0][j + 1];
            parent[(1 << j) * m + j] = -1;
        }
        for (int s = 1; s < full; s++) {
            for (int j = 0; j < m; j++) {
                double cj = cost[s * m + j];
                if ((s & (1 << j)) == 0 || cj == Double.POSITIVE_INFINITY) {
                    continue;
                }
                for (int k = 0; k < m; k++) {
                    if ((s & (1 << k)) != 0) {
                        continue;
                    }
                    int t = s | (1 << k);
                    double c = cj + d[j + 1][k + 1];
                    if (c < cost[t * m + k]) {
                        cost[t * m + k] = c;
                        parent[t * m + k] = j;
                    }
                }
            }
        }
        int last = 0;
        double best = Double.POSITIVE_INFINITY;
        for (int j = 0; j < m; j++) {
            double c = cost[(full - 1) * m + j] + d[j + 1][0];
            if (c < best) {
                best = c;
                last = j;
            }
        }
        int[] tour = new int[n];
        int s = full - 1;
        for (int k = n - 1; k >= 1; k--) {
            tour[k] = last + 1;
            int p = parent[s * m + last];
            s &= ~(1 << last);
            last = p;
        }
        return tour;
    }

    /**
     * A CVRP instance with uniform depot and customers, demands 1-9 and capacity 30 for 20
     * customers (scaled as in Kool et al.: 20 for 10, 30 for 20, 40 for 50).
     *
     * @param customers the customer count
     * @param seed the seed
     * @return the instance with a best-known value
     */
    public static Cvrp cvrp(int customers, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] c = new double[customers + 1][2];
        double[] demand = new double[customers + 1];
        for (int i = 0; i <= customers; i++) {
            c[i][0] = r.nextDouble();
            c[i][1] = r.nextDouble();
            demand[i] = i == 0 ? 0 : 1 + r.nextInt(9);
        }
        double cap = customers <= 10 ? 20 : customers <= 20 ? 30 : 40;
        double[][] d = ClassicHeuristics.distances(c);
        double best = ClassicHeuristics.routeLength(d,
                ClassicHeuristics.improveRoutes(d, ClassicHeuristics.nearestNeighbourCvrp(d, demand, cap)));
        for (int k = 0; k < 8; k++) {
            int[] route = ClassicHeuristics.improveRoutes(d, ClassicHeuristics.sweepCvrp(c, demand, cap, 2 * Math.PI * k / 8));
            best = Math.min(best, ClassicHeuristics.routeLength(d, route));
        }
        return new Cvrp(c, demand, cap, best);
    }

    /**
     * An Erdős–Rényi graph.
     *
     * @param n nodes
     * @param p the edge probability
     * @param seed the seed
     * @return the 0/1 adjacency
     */
    public static double[][] erdosRenyi(int n, double p, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (r.nextDouble() < p) {
                    a[i][j] = a[j][i] = 1;
                }
            }
        }
        return a;
    }

    /**
     * A random {@code d}-regular graph by the pairing model with restarts.
     *
     * @param n nodes ({@code n d} even)
     * @param degree the degree
     * @param seed the seed
     * @return the 0/1 adjacency
     */
    public static double[][] randomRegular(int n, int degree, long seed) {
        if ((n * degree) % 2 != 0) {
            throw new IllegalArgumentException("n * degree must be even");
        }
        SplittableRandom r = new SplittableRandom(seed);
        for (int attempt = 0; attempt < 1000; attempt++) {
            double[][] a = new double[n][n];
            List<Integer> stubs = new ArrayList<>();
            for (int v = 0; v < n; v++) {
                for (int k = 0; k < degree; k++) {
                    stubs.add(v);
                }
            }
            boolean ok = true;
            while (!stubs.isEmpty() && ok) {
                ok = false;
                for (int tries = 0; tries < 50; tries++) {
                    int i = r.nextInt(stubs.size());
                    int j = r.nextInt(stubs.size());
                    int u = stubs.get(i);
                    int v = stubs.get(j);
                    if (i != j && u != v && a[u][v] == 0) {
                        a[u][v] = a[v][u] = 1;
                        stubs.remove(Math.max(i, j));
                        stubs.remove(Math.min(i, j));
                        ok = true;
                        break;
                    }
                }
            }
            if (ok) {
                return a;
            }
        }
        throw new IllegalStateException("could not build a regular graph");
    }

    /**
     * The MaxCut reference: exact by Gray-code enumeration for {@code n <= 22}, otherwise
     * the best of 20 greedy-plus-local-search restarts from random sides.
     *
     * @param w weights
     * @param seed the seed for restarts
     * @return the reference cut
     */
    public static Reference maxCut(double[][] w, long seed) {
        int n = w.length;
        if (n <= 22) {
            int[] side = new int[n];
            double cut = 0;
            double best = 0;
            long total = 1L << (n - 1);
            for (long g = 1; g < total; g++) {
                int v = Long.numberOfTrailingZeros(g);
                double delta = 0;
                for (int u = 0; u < n; u++) {
                    if (u != v && w[v][u] != 0) {
                        delta += side[u] == side[v] ? w[v][u] : -w[v][u];
                    }
                }
                side[v] = 1 - side[v];
                cut += delta;
                best = Math.max(best, cut);
            }
            return new Reference(best, true);
        }
        SplittableRandom r = new SplittableRandom(seed);
        double best = ClassicHeuristics.cut(w, ClassicHeuristics.greedyMaxCut(w));
        for (int k = 0; k < 20; k++) {
            int[] s = new int[n];
            for (int i = 0; i < n; i++) {
                s[i] = r.nextInt(2);
            }
            best = Math.max(best, ClassicHeuristics.cut(w, ClassicHeuristics.localSearchCut(w, s)));
        }
        return new Reference(best, false);
    }

    /**
     * The maximum independent set size: exact by bitmask branch and bound for
     * {@code n <= 64}, otherwise greedy.
     *
     * @param adjacency the graph
     * @return the reference size
     */
    public static Reference maxIndependentSet(double[][] adjacency) {
        int n = adjacency.length;
        if (n > 64) {
            return new Reference(Arrays.stream(ClassicHeuristics.greedyMis(adjacency)).sum(), false);
        }
        long[] nb = new long[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (adjacency[i][j] != 0 && i != j) {
                    nb[i] |= 1L << j;
                }
            }
        }
        long all = n == 64 ? -1L : (1L << n) - 1;
        int[] best = {Arrays.stream(ClassicHeuristics.greedyMis(adjacency)).sum()};
        mis(nb, all, 0, best);
        return new Reference(best[0], true);
    }

    private static void mis(long[] nb, long alive, int size, int[] best) {
        if (alive == 0) {
            best[0] = Math.max(best[0], size);
            return;
        }
        if (size + Long.bitCount(alive) <= best[0]) {
            return;
        }
        int minV = -1;
        int minD = Integer.MAX_VALUE;
        int maxV = -1;
        int maxD = -1;
        for (long m = alive; m != 0; m &= m - 1) {
            int v = Long.numberOfTrailingZeros(m);
            int deg = Long.bitCount(nb[v] & alive);
            if (deg < minD) {
                minD = deg;
                minV = v;
            }
            if (deg > maxD) {
                maxD = deg;
                maxV = v;
            }
        }
        if (minD <= 1) {
            mis(nb, alive & ~nb[minV] & ~(1L << minV), size + 1, best);
            return;
        }
        mis(nb, alive & ~nb[maxV] & ~(1L << maxV), size + 1, best);
        mis(nb, alive & ~(1L << maxV), size, best);
    }

    /**
     * A strongly correlated 0/1 knapsack (weights 1-30, values weight + 10, capacity half
     * the total weight), solved exactly by dynamic programming over integer weights.
     *
     * @param n items
     * @param seed the seed
     * @return the ILP with its optimum
     */
    public static Ilp knapsack(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] c = new double[n];
        double[][] a = new double[1][n];
        int total = 0;
        int[] w = new int[n];
        for (int i = 0; i < n; i++) {
            w[i] = 1 + r.nextInt(30);
            a[0][i] = w[i];
            c[i] = w[i] + 10;
            total += w[i];
        }
        int cap = total / 2;
        double[] dp = new double[cap + 1];
        for (int i = 0; i < n; i++) {
            for (int k = cap; k >= w[i]; k--) {
                dp[k] = Math.max(dp[k], dp[k - w[i]] + c[i]);
            }
        }
        return new Ilp("knapsack", c, a, new double[] {cap}, 0, dp[cap]);
    }

    /**
     * A random set-cover instance: {@code 2n} elements, each set holding each element with
     * probability 0.15 (every element in at least two sets), integer costs 1-10. Stored as a
     * packing ILP over {@code y = 1 - x}: maximise {@code c.y} with
     * {@code sum_{s contains e} y_s <= deg(e) - 1}; the optimum comes from best-bound
     * branch and bound.
     *
     * @param n sets (variables)
     * @param seed the seed
     * @return the ILP with its optimum
     */
    public static Ilp setCover(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int m = 2 * n;
        double[][] a = new double[m][n];
        double[] b = new double[m];
        double[] c = new double[n];
        double total = 0;
        for (int j = 0; j < n; j++) {
            c[j] = 1 + r.nextInt(10);
            total += c[j];
        }
        for (int e = 0; e < m; e++) {
            int deg = 0;
            for (int j = 0; j < n; j++) {
                if (r.nextDouble() < 0.15) {
                    a[e][j] = 1;
                    deg++;
                }
            }
            while (deg < 2) {
                int j = r.nextInt(n);
                if (a[e][j] == 0) {
                    a[e][j] = 1;
                    deg++;
                }
            }
            b[e] = deg - 1;
        }
        Ilp raw = new Ilp("setcover", c, a, b, total, Double.NaN);
        double opt = BranchAndBound.of(BranchAndBound.Rule.PSEUDOCOST).bestFirst(true).solve(raw).objective();
        return new Ilp("setcover", c, a, b, total, opt);
    }

    private static double gaussian(SplittableRandom r) {
        double u = Math.max(1e-12, r.nextDouble());
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble());
    }

    private static double clamp01(double x) {
        return Math.max(0, Math.min(1, x));
    }
}
