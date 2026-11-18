package io.github.zzuegg.brainiac.experimental.combopt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The classical reference heuristics for the combinatorial-optimisation blocks, written once
 * so every learned solver is compared with the same code: tour construction (nearest
 * neighbour, farthest insertion), tour improvement (first- and best-improvement 2-opt, Or-opt
 * and their combination, simulated annealing over 2-opt moves), a nearest-neighbour CVRP
 * construction, and greedy rules for maximum independent set and MaxCut.
 *
 * <p>Tours are permutations of {@code 0..n-1}, closed (the last node returns to the first).
 * CVRP routes are node sequences over customers {@code 1..n-1} with the depot {@code 0}
 * marking returns; the route starts and ends at the depot implicitly.
 *
 * <p>References: Rosenkrantz, Stearns and Lewis, "An analysis of several heuristics for the
 * traveling salesman problem" (SIAM J. Computing 1977); Croes, "A method for solving
 * traveling-salesman problems" (Operations Research 1958); Or, "Traveling salesman-type
 * combinatorial problems" (PhD thesis, 1976).
 */
public final class ClassicHeuristics {

    private ClassicHeuristics() {
    }

    /**
     * The result of a local search.
     *
     * @param tour the improved tour
     * @param moves the improving moves applied
     * @param length its length
     */
    public record Improvement(int[] tour, int moves, double length) {
    }

    /**
     * Euclidean distances.
     *
     * @param coords {@code [n][2]} points
     * @return {@code [n][n]}
     */
    public static double[][] distances(double[][] coords) {
        int n = coords.length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double dx = coords[i][0] - coords[j][0];
                double dy = coords[i][1] - coords[j][1];
                d[i][j] = d[j][i] = Math.sqrt(dx * dx + dy * dy);
            }
        }
        return d;
    }

    /**
     * The length of a closed tour.
     *
     * @param d distances
     * @param tour a permutation
     * @return its length
     */
    public static double tourLength(double[][] d, int[] tour) {
        double s = 0;
        for (int i = 0; i < tour.length; i++) {
            s += d[tour[i]][tour[(i + 1) % tour.length]];
        }
        return s;
    }

    /**
     * The length of a CVRP solution: depot, the sequence (zeros are returns), depot.
     *
     * @param d distances, depot {@code 0}
     * @param route the sequence
     * @return its length
     */
    public static double routeLength(double[][] d, int[] route) {
        double s = 0;
        int prev = 0;
        for (int v : route) {
            s += d[prev][v];
            prev = v;
        }
        return s + d[prev][0];
    }

    /**
     * Whether a CVRP sequence visits every customer exactly once within capacity.
     *
     * @param demand demands, {@code demand[0] = 0}
     * @param capacity the vehicle capacity
     * @param route the sequence
     * @return validity
     */
    public static boolean validRoute(double[] demand, double capacity, int[] route) {
        boolean[] seen = new boolean[demand.length];
        double load = 0;
        for (int v : route) {
            if (v == 0) {
                load = 0;
                continue;
            }
            if (seen[v]) {
                return false;
            }
            seen[v] = true;
            load += demand[v];
            if (load > capacity + 1e-9) {
                return false;
            }
        }
        for (int v = 1; v < demand.length; v++) {
            if (!seen[v]) {
                return false;
            }
        }
        return true;
    }

    /**
     * A uniformly random tour.
     *
     * @param n nodes
     * @param random the source
     * @return the tour
     */
    public static int[] randomTour(int n, SplittableRandom random) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int x = t[i];
            t[i] = t[j];
            t[j] = x;
        }
        return t;
    }

    /**
     * Nearest neighbour: always go to the closest unvisited node.
     *
     * @param d distances
     * @param start the first node
     * @return the tour
     */
    public static int[] nearestNeighbour(double[][] d, int start) {
        int n = d.length;
        int[] t = new int[n];
        boolean[] used = new boolean[n];
        t[0] = start;
        used[start] = true;
        for (int k = 1; k < n; k++) {
            int cur = t[k - 1];
            int best = -1;
            for (int j = 0; j < n; j++) {
                if (!used[j] && (best < 0 || d[cur][j] < d[cur][best])) {
                    best = j;
                }
            }
            t[k] = best;
            used[best] = true;
        }
        return t;
    }

    /**
     * Farthest insertion: repeatedly take the node farthest from the tour and insert it where
     * it lengthens the tour least.
     *
     * @param d distances
     * @return the tour
     */
    public static int[] farthestInsertion(double[][] d) {
        int n = d.length;
        if (n < 3) {
            return nearestNeighbour(d, 0);
        }
        List<Integer> tour = new ArrayList<>();
        boolean[] in = new boolean[n];
        int a = 0;
        int b = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (d[i][j] > d[a][b]) {
                    a = i;
                    b = j;
                }
            }
        }
        tour.add(a);
        tour.add(b);
        in[a] = in[b] = true;
        double[] near = new double[n];
        for (int v = 0; v < n; v++) {
            near[v] = Math.min(d[v][a], d[v][b]);
        }
        for (int k = 2; k < n; k++) {
            int far = -1;
            for (int v = 0; v < n; v++) {
                if (!in[v] && (far < 0 || near[v] > near[far])) {
                    far = v;
                }
            }
            int pos = 0;
            double inc = Double.POSITIVE_INFINITY;
            for (int i = 0; i < tour.size(); i++) {
                int u = tour.get(i);
                int w = tour.get((i + 1) % tour.size());
                double c = d[u][far] + d[far][w] - d[u][w];
                if (c < inc) {
                    inc = c;
                    pos = i + 1;
                }
            }
            tour.add(pos, far);
            in[far] = true;
            for (int v = 0; v < n; v++) {
                near[v] = Math.min(near[v], d[v][far]);
            }
        }
        return tour.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * The gain of the 2-opt move that reverses {@code tour[i+1..j]}, i.e. replaces edges
     * {@code (t_i, t_i+1)} and {@code (t_j, t_j+1)} with {@code (t_i, t_j)} and
     * {@code (t_i+1, t_j+1)}.
     *
     * @param d distances
     * @param tour the tour
     * @param i the first cut, {@code 0 <= i < j < n}
     * @param j the second cut
     * @return the length saved (positive is an improvement)
     */
    public static double twoOptGain(double[][] d, int[] tour, int i, int j) {
        int n = tour.length;
        int a = tour[i];
        int b = tour[(i + 1) % n];
        int c = tour[j];
        int e = tour[(j + 1) % n];
        return d[a][b] + d[c][e] - d[a][c] - d[b][e];
    }

    /**
     * Applies a 2-opt move in place: reverses {@code tour[i+1..j]}.
     *
     * @param tour the tour
     * @param i the first cut
     * @param j the second cut
     */
    public static void applyTwoOpt(int[] tour, int i, int j) {
        for (int lo = i + 1, hi = j; lo < hi; lo++, hi--) {
            int x = tour[lo];
            tour[lo] = tour[hi];
            tour[hi] = x;
        }
    }

    /**
     * 2-opt local search.
     *
     * @param d distances
     * @param start the starting tour (not changed)
     * @param firstImprovement take the first improving move found instead of the best one
     * @param maxMoves the most moves applied
     * @return the local optimum (or the tour after {@code maxMoves})
     */
    public static Improvement twoOpt(double[][] d, int[] start, boolean firstImprovement, int maxMoves) {
        int[] t = start.clone();
        int n = t.length;
        int moves = 0;
        while (moves < maxMoves) {
            int bi = -1;
            int bj = -1;
            double best = 1e-10;
            search:
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 2; j < n; j++) {
                    if (i == 0 && j == n - 1) {
                        continue;
                    }
                    double g = twoOptGain(d, t, i, j);
                    if (g > best) {
                        best = g;
                        bi = i;
                        bj = j;
                        if (firstImprovement) {
                            break search;
                        }
                    }
                }
            }
            if (bi < 0) {
                break;
            }
            applyTwoOpt(t, bi, bj);
            moves++;
        }
        return new Improvement(t, moves, tourLength(d, t));
    }

    /**
     * Or-opt: move a segment of 1-3 consecutive nodes (either direction) to its best place,
     * until no move improves.
     *
     * @param d distances
     * @param start the starting tour
     * @param maxMoves the most moves applied
     * @return the improved tour
     */
    public static Improvement orOpt(double[][] d, int[] start, int maxMoves) {
        List<Integer> t = new ArrayList<>();
        for (int v : start) {
            t.add(v);
        }
        int n = t.size();
        int moves = 0;
        boolean improved = true;
        while (improved && moves < maxMoves) {
            improved = false;
            for (int len = 1; len <= 3 && !improved; len++) {
                for (int i = 0; i < n && !improved; i++) {
                    if (len >= n - 2) {
                        break;
                    }
                    int prev = t.get((i - 1 + n) % n);
                    int first = t.get(i);
                    int last = t.get((i + len - 1) % n);
                    int next = t.get((i + len) % n);
                    double removeGain = d[prev][first] + d[last][next] - d[prev][next];
                    List<Integer> seg = new ArrayList<>();
                    for (int k = 0; k < len; k++) {
                        seg.add(t.get((i + k) % n));
                    }
                    List<Integer> rest = new ArrayList<>();
                    for (int k = 0; k < n - len; k++) {
                        rest.add(t.get((i + len + k) % n));
                    }
                    double bestDelta = -1e-10;
                    int bestPos = -1;
                    boolean bestRev = false;
                    for (int p = 0; p < rest.size(); p++) {
                        int u = rest.get(p);
                        int w = rest.get((p + 1) % rest.size());
                        double base = d[u][w];
                        double fwd = removeGain - (d[u][first] + d[last][w] - base);
                        double rev = removeGain - (d[u][last] + d[first][w] - base);
                        if (fwd > -bestDelta && fwd > 1e-10 && p != rest.size() - 1) {
                            bestDelta = -fwd;
                            bestPos = p;
                            bestRev = false;
                        }
                        if (rev > -bestDelta && rev > 1e-10) {
                            bestDelta = -rev;
                            bestPos = p;
                            bestRev = true;
                        }
                    }
                    if (bestPos >= 0) {
                        if (bestRev) {
                            java.util.Collections.reverse(seg);
                        }
                        rest.addAll(bestPos + 1, seg);
                        t = rest;
                        moves++;
                        improved = true;
                    }
                }
            }
        }
        int[] out = t.stream().mapToInt(Integer::intValue).toArray();
        return new Improvement(out, moves, tourLength(d, out));
    }

    /**
     * Alternates best-improvement 2-opt and Or-opt until neither improves.
     *
     * @param d distances
     * @param start the starting tour
     * @return the local optimum
     */
    public static Improvement twoOptOrOpt(double[][] d, int[] start) {
        int[] t = start.clone();
        int moves = 0;
        for (int round = 0; round < 50; round++) {
            Improvement a = twoOpt(d, t, false, 10_000);
            Improvement b = orOpt(d, a.tour(), 10_000);
            moves += a.moves() + b.moves();
            t = b.tour();
            if (b.moves() == 0) {
                break;
            }
        }
        return new Improvement(t, moves, tourLength(d, t));
    }

    /**
     * The best of several 2-opt + Or-opt local searches from nearest-neighbour starts and
     * random starts: a best-known reference for instances too large for an exact solver.
     *
     * @param d distances
     * @param restarts random restarts on top of {@code min(n, 8)} nearest-neighbour starts
     * @param seed the seed
     * @return the best tour found
     */
    public static Improvement multiStart(double[][] d, int restarts, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Improvement best = null;
        int n = d.length;
        for (int s = 0; s < Math.min(n, 8) + restarts; s++) {
            int[] start = s < Math.min(n, 8) ? nearestNeighbour(d, s * n / Math.min(n, 8)) : randomTour(n, r);
            Improvement i = twoOptOrOpt(d, start);
            if (best == null || i.length() < best.length()) {
                best = i;
            }
        }
        return best;
    }

    /**
     * Simulated annealing over random 2-opt moves with a geometric temperature schedule,
     * keeping the best tour seen.
     *
     * @param d distances
     * @param start the starting tour
     * @param moves proposals evaluated
     * @param t0 the starting temperature (in length units)
     * @param t1 the final temperature
     * @param seed the seed
     * @return the best tour
     */
    public static Improvement annealTwoOpt(double[][] d, int[] start, int moves, double t0, double t1, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] t = start.clone();
        int n = t.length;
        double len = tourLength(d, t);
        int[] best = t.clone();
        double bestLen = len;
        int accepted = 0;
        for (int m = 0; m < moves; m++) {
            double temp = t0 * Math.pow(t1 / t0, m / (double) Math.max(1, moves - 1));
            int i = r.nextInt(n - 1);
            int j = i + 2 + r.nextInt(Math.max(1, n - i - 2));
            if (j >= n || (i == 0 && j == n - 1)) {
                continue;
            }
            double g = twoOptGain(d, t, i, j);
            if (g > 0 || r.nextDouble() < Math.exp(g / temp)) {
                applyTwoOpt(t, i, j);
                len -= g;
                accepted++;
                if (len < bestLen - 1e-12) {
                    bestLen = len;
                    best = t.clone();
                }
            }
        }
        return new Improvement(best, accepted, tourLength(d, best));
    }

    /**
     * Nearest-neighbour CVRP: from the current position go to the closest customer that still
     * fits, returning to the depot when none fits.
     *
     * @param d distances, depot {@code 0}
     * @param demand demands
     * @param capacity the capacity
     * @return the sequence (zeros are depot returns)
     */
    public static int[] nearestNeighbourCvrp(double[][] d, double[] demand, double capacity) {
        int n = d.length;
        boolean[] done = new boolean[n];
        List<Integer> route = new ArrayList<>();
        int cur = 0;
        double load = 0;
        int left = n - 1;
        while (left > 0) {
            int best = -1;
            for (int j = 1; j < n; j++) {
                if (!done[j] && load + demand[j] <= capacity + 1e-9 && (best < 0 || d[cur][j] < d[cur][best])) {
                    best = j;
                }
            }
            if (best < 0) {
                route.add(0);
                cur = 0;
                load = 0;
                continue;
            }
            route.add(best);
            done[best] = true;
            load += demand[best];
            cur = best;
            left--;
        }
        return route.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * The sweep heuristic of Gillett and Miller: order customers by polar angle around the
     * depot from a starting angle and cut the order into routes whenever the next customer
     * would exceed the capacity.
     *
     * @param coords points, depot first
     * @param demand demands
     * @param capacity the capacity
     * @param startAngle the angle the sweep starts at
     * @return the sequence (zeros are depot returns)
     */
    public static int[] sweepCvrp(double[][] coords, double[] demand, double capacity, double startAngle) {
        int n = coords.length;
        Integer[] order = new Integer[n - 1];
        double[] angle = new double[n];
        for (int i = 1; i < n; i++) {
            order[i - 1] = i;
            double a = Math.atan2(coords[i][1] - coords[0][1], coords[i][0] - coords[0][0]) - startAngle;
            angle[i] = ((a % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
        }
        Arrays.sort(order, (x, y) -> Double.compare(angle[x], angle[y]));
        List<Integer> route = new ArrayList<>();
        double load = 0;
        for (int v : order) {
            if (load + demand[v] > capacity + 1e-9) {
                route.add(0);
                load = 0;
            }
            route.add(v);
            load += demand[v];
        }
        return route.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Improves each route of a CVRP solution separately with 2-opt (the depot included in
     * each route's cycle).
     *
     * @param d distances, depot {@code 0}
     * @param route the sequence
     * @return the improved sequence, same routes in the same order
     */
    public static int[] improveRoutes(double[][] d, int[] route) {
        List<Integer> out = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        for (int k = 0; k <= route.length; k++) {
            if (k == route.length || route[k] == 0) {
                if (!cur.isEmpty()) {
                    int[] cycle = new int[cur.size() + 1];
                    for (int i = 0; i < cur.size(); i++) {
                        cycle[i + 1] = cur.get(i);
                    }
                    int[] better = twoOpt(subDistances(d, cycle), identity(cycle.length), false, 1000).tour();
                    int z = 0;
                    while (better[z] != 0) {
                        z++;
                    }
                    if (!out.isEmpty()) {
                        out.add(0);
                    }
                    for (int i = 1; i < better.length; i++) {
                        out.add(cycle[better[(z + i) % better.length]]);
                    }
                    cur.clear();
                }
            } else {
                cur.add(route[k]);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static int[] identity(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = i;
        }
        return t;
    }

    private static double[][] subDistances(double[][] d, int[] nodes) {
        double[][] s = new double[nodes.length][nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            for (int j = 0; j < nodes.length; j++) {
                s[i][j] = d[nodes[i]][nodes[j]];
            }
        }
        return s;
    }

    /**
     * Greedy maximum independent set: repeatedly take a node of minimum remaining degree
     * and delete its neighbours.
     *
     * @param adjacency 0/1 symmetric
     * @return the chosen nodes as a 0/1 mask
     */
    public static int[] greedyMis(double[][] adjacency) {
        int n = adjacency.length;
        boolean[] alive = new boolean[n];
        Arrays.fill(alive, true);
        int[] in = new int[n];
        while (true) {
            int pick = -1;
            int pickDeg = Integer.MAX_VALUE;
            for (int v = 0; v < n; v++) {
                if (!alive[v]) {
                    continue;
                }
                int deg = 0;
                for (int u = 0; u < n; u++) {
                    if (alive[u] && adjacency[v][u] != 0) {
                        deg++;
                    }
                }
                if (deg < pickDeg) {
                    pickDeg = deg;
                    pick = v;
                }
            }
            if (pick < 0) {
                return in;
            }
            in[pick] = 1;
            alive[pick] = false;
            for (int u = 0; u < n; u++) {
                if (adjacency[pick][u] != 0) {
                    alive[u] = false;
                }
            }
        }
    }

    /**
     * Whether a 0/1 mask is an independent set.
     *
     * @param adjacency the graph
     * @param set the mask
     * @return independence
     */
    public static boolean independent(double[][] adjacency, int[] set) {
        for (int i = 0; i < set.length; i++) {
            for (int j = i + 1; j < set.length; j++) {
                if (set[i] == 1 && set[j] == 1 && adjacency[i][j] != 0) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * The cut value of a 0/1 partition.
     *
     * @param w weights
     * @param side the side of each node, 0 or 1
     * @return total weight across
     */
    public static double cut(double[][] w, int[] side) {
        double c = 0;
        for (int i = 0; i < w.length; i++) {
            for (int j = i + 1; j < w.length; j++) {
                c += side[i] != side[j] ? w[i][j] : 0;
            }
        }
        return c;
    }

    /**
     * Greedy MaxCut followed by one-flip local search: place nodes one by one on the side
     * that cuts more of the edges to placed nodes, then flip any node whose flip increases
     * the cut until none does.
     *
     * @param w weights
     * @return the sides, 0 or 1
     */
    public static int[] greedyMaxCut(double[][] w) {
        int n = w.length;
        int[] side = new int[n];
        boolean[] placed = new boolean[n];
        for (int v = 0; v < n; v++) {
            double to0 = 0;
            double to1 = 0;
            for (int u = 0; u < n; u++) {
                if (placed[u]) {
                    if (side[u] == 0) {
                        to0 += w[v][u];
                    } else {
                        to1 += w[v][u];
                    }
                }
            }
            side[v] = to0 >= to1 ? 1 : 0;
            placed[v] = true;
        }
        return localSearchCut(w, side);
    }

    /**
     * One-flip local search for MaxCut.
     *
     * @param w weights
     * @param start the sides
     * @return a local optimum
     */
    public static int[] localSearchCut(double[][] w, int[] start) {
        int n = w.length;
        int[] side = start.clone();
        boolean improved = true;
        while (improved) {
            improved = false;
            for (int v = 0; v < n; v++) {
                double same = 0;
                double other = 0;
                for (int u = 0; u < n; u++) {
                    if (u != v) {
                        if (side[u] == side[v]) {
                            same += w[v][u];
                        } else {
                            other += w[v][u];
                        }
                    }
                }
                if (same > other + 1e-12) {
                    side[v] = 1 - side[v];
                    improved = true;
                }
            }
        }
        return side;
    }
}
