package io.github.zzuegg.brainiac.experimental.ops;

import io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Routing metaheuristics on the wave 11 instances ({@link CombinatorialInstances}), with anytime
 * traces (best length after each iteration):
 *
 * <ul>
 *   <li>{@link #antColonySystem}: ant colony system (Dorigo &amp; Gambardella, IEEE TEC 1997): ants
 *       build tours from pheromone times inverse distance with the pseudo-random proportional rule,
 *       local pheromone evaporation on used edges and a global update on the best tour;</li>
 *   <li>{@link #tabuSearch}: tabu search (Glover, 1989) over the full 2-opt neighbourhood: the best
 *       non-tabu move is taken even if it worsens the tour, recently changed cities are tabu for
 *       {@code tenure} iterations, with aspiration for new bests;</li>
 *   <li>{@link #geneticOx}: a genetic algorithm with order crossover (Davis, 1985), tournament
 *       selection, a random 2-opt mutation and elitism;</li>
 *   <li>{@link #lns}: large-neighbourhood search for CVRP: destroy a share of the customers
 *       ({@link Destroy#RANDOM}, {@link Destroy#WORST} by removal gain, {@link Destroy#RELATED} by
 *       proximity to a seed, or {@link Destroy#LEARNED}) and repair by cheapest feasible insertion,
 *       accepting improvements and small deteriorations (record-to-record);</li>
 *   <li>{@link LearnedDestroy}: a learned destroy operator in the spirit of neural LNS (Hottung &amp;
 *       Tierney, ECAI 2020, <a href="https://arxiv.org/abs/1911.09539">arXiv:1911.09539</a>, which
 *       learns the repair): a softmax policy over customers on (removal gain, depot distance, seed
 *       distance, demand) trained by REINFORCE on the improvement it enables.</li>
 * </ul>
 */
public final class RoutingMetaheuristics {

    private RoutingMetaheuristics() {
    }

    /**
     * An anytime run.
     *
     * @param trace the best length after every iteration
     * @param best the best solution (a tour, or a CVRP sequence with depot returns as 0)
     * @param length its length
     * @param millis wall-clock time
     */
    public record Trace(double[] trace, int[] best, double length, long millis) {
    }

    // ------------------------------------------------------------------ TSP

    /**
     * Ant colony system.
     *
     * @param d distances
     * @param ants ants per iteration
     * @param iterations iterations
     * @param seed the seed
     * @return the trace
     */
    public static Trace antColonySystem(double[][] d, int ants, int iterations, long seed) {
        long t0 = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.length;
        double beta = 2;
        double q0 = 0.9;
        double rho = 0.1;
        double xi = 0.1;
        int[] nn = ClassicHeuristics.nearestNeighbour(d, 0);
        double lnn = ClassicHeuristics.tourLength(d, nn);
        double tau0 = 1 / (n * lnn);
        double[][] tau = new double[n][n];
        for (double[] row : tau) {
            Arrays.fill(row, tau0);
        }
        int[] best = nn.clone();
        double bestLen = lnn;
        double[] trace = new double[iterations];
        for (int it = 0; it < iterations; it++) {
            for (int a = 0; a < ants; a++) {
                int[] tour = new int[n];
                boolean[] seen = new boolean[n];
                tour[0] = r.nextInt(n);
                seen[tour[0]] = true;
                for (int k = 1; k < n; k++) {
                    int cur = tour[k - 1];
                    int pick = -1;
                    if (r.nextDouble() < q0) {
                        double bv = -1;
                        for (int j = 0; j < n; j++) {
                            if (!seen[j]) {
                                double v = tau[cur][j] * Math.pow(1 / Math.max(1e-12, d[cur][j]), beta);
                                if (v > bv) {
                                    bv = v;
                                    pick = j;
                                }
                            }
                        }
                    } else {
                        double total = 0;
                        double[] w = new double[n];
                        for (int j = 0; j < n; j++) {
                            if (!seen[j]) {
                                w[j] = tau[cur][j] * Math.pow(1 / Math.max(1e-12, d[cur][j]), beta);
                                total += w[j];
                            }
                        }
                        double u = r.nextDouble() * total;
                        for (int j = 0; j < n; j++) {
                            if (!seen[j]) {
                                u -= w[j];
                                pick = j;
                                if (u <= 0) {
                                    break;
                                }
                            }
                        }
                    }
                    tour[k] = pick;
                    seen[pick] = true;
                    tau[cur][pick] = (1 - xi) * tau[cur][pick] + xi * tau0;
                    tau[pick][cur] = tau[cur][pick];
                }
                double len = ClassicHeuristics.tourLength(d, tour);
                if (len < bestLen) {
                    bestLen = len;
                    best = tour;
                }
            }
            for (int k = 0; k < n; k++) {
                int i = best[k];
                int j = best[(k + 1) % n];
                tau[i][j] = (1 - rho) * tau[i][j] + rho / bestLen;
                tau[j][i] = tau[i][j];
            }
            trace[it] = bestLen;
        }
        return new Trace(trace, best, bestLen, (System.nanoTime() - t0) / 1_000_000);
    }

    /**
     * Tabu search over 2-opt moves: the edges a move removes may not be added back for
     * {@code tenure} iterations unless that gives a new best tour (aspiration).
     *
     * @param d distances
     * @param start the initial tour
     * @param iterations iterations
     * @param tenure iterations a removed edge stays tabu
     * @return the trace
     */
    public static Trace tabuSearch(double[][] d, int[] start, int iterations, int tenure) {
        long t0 = System.nanoTime();
        int n = d.length;
        int[] tour = start.clone();
        double len = ClassicHeuristics.tourLength(d, tour);
        int[] best = tour.clone();
        double bestLen = len;
        int[][] tabuUntil = new int[n][n];
        double[] trace = new double[iterations];
        for (int it = 0; it < iterations; it++) {
            double bestGain = Double.NEGATIVE_INFINITY;
            int bi = -1;
            int bj = -1;
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 2; j < n; j++) {
                    if (i == 0 && j == n - 1) {
                        continue;
                    }
                    double gain = ClassicHeuristics.twoOptGain(d, tour, i, j);
                    int a = tour[i];
                    int b = tour[i + 1];
                    int c = tour[j];
                    int e = tour[(j + 1) % n];
                    boolean tabu = tabuUntil[a][c] > it || tabuUntil[b][e] > it;
                    if (tabu && len - gain >= bestLen - 1e-12) {
                        continue;
                    }
                    if (gain > bestGain) {
                        bestGain = gain;
                        bi = i;
                        bj = j;
                    }
                }
            }
            if (bi < 0) {
                trace[it] = bestLen;
                continue;
            }
            int a = tour[bi];
            int b = tour[bi + 1];
            int c = tour[bj];
            int e = tour[(bj + 1) % n];
            tabuUntil[a][b] = it + tenure;
            tabuUntil[b][a] = it + tenure;
            tabuUntil[c][e] = it + tenure;
            tabuUntil[e][c] = it + tenure;
            ClassicHeuristics.applyTwoOpt(tour, bi, bj);
            len -= bestGain;
            if (len < bestLen - 1e-12) {
                bestLen = len;
                best = tour.clone();
            }
            trace[it] = bestLen;
        }
        return new Trace(trace, best, bestLen, (System.nanoTime() - t0) / 1_000_000);
    }

    /**
     * A genetic algorithm with order crossover.
     *
     * @param d distances
     * @param population individuals
     * @param generations generations
     * @param seed the seed
     * @return the trace
     */
    public static Trace geneticOx(double[][] d, int population, int generations, long seed) {
        long t0 = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.length;
        int[][] pop = new int[population][];
        double[] fit = new double[population];
        for (int p = 0; p < population; p++) {
            pop[p] = ClassicHeuristics.randomTour(n, r);
            fit[p] = ClassicHeuristics.tourLength(d, pop[p]);
        }
        double[] trace = new double[generations];
        int bestIdx = argmin(fit);
        int[] best = pop[bestIdx].clone();
        double bestLen = fit[bestIdx];
        for (int g = 0; g < generations; g++) {
            int[][] next = new int[population][];
            double[] nf = new double[population];
            next[0] = best.clone();
            nf[0] = bestLen;
            for (int p = 1; p < population; p++) {
                int[] a = pop[tournament(fit, r)];
                int[] b = pop[tournament(fit, r)];
                int[] child = orderCrossover(a, b, r);
                if (r.nextDouble() < 0.3) {
                    int i = r.nextInt(n - 2);
                    int j = i + 2 + r.nextInt(n - i - 2);
                    if (!(i == 0 && j == n - 1)) {
                        ClassicHeuristics.applyTwoOpt(child, i, j);
                    }
                }
                next[p] = child;
                nf[p] = ClassicHeuristics.tourLength(d, child);
                if (nf[p] < bestLen) {
                    bestLen = nf[p];
                    best = child.clone();
                }
            }
            pop = next;
            fit = nf;
            trace[g] = bestLen;
        }
        return new Trace(trace, best, bestLen, (System.nanoTime() - t0) / 1_000_000);
    }

    private static int tournament(double[] fit, SplittableRandom r) {
        int a = r.nextInt(fit.length);
        int b = r.nextInt(fit.length);
        return fit[a] <= fit[b] ? a : b;
    }

    /**
     * Order crossover: a slice of {@code a}, the rest in {@code b}'s order.
     *
     * @param a the first parent
     * @param b the second parent
     * @param r the source
     * @return the child
     */
    public static int[] orderCrossover(int[] a, int[] b, SplittableRandom r) {
        int n = a.length;
        int i = r.nextInt(n);
        int j = r.nextInt(n);
        int lo = Math.min(i, j);
        int hi = Math.max(i, j);
        int[] child = new int[n];
        Arrays.fill(child, -1);
        boolean[] used = new boolean[n];
        for (int k = lo; k <= hi; k++) {
            child[k] = a[k];
            used[a[k]] = true;
        }
        int pos = (hi + 1) % n;
        for (int k = 0; k < n; k++) {
            int city = b[(hi + 1 + k) % n];
            if (!used[city]) {
                child[pos] = city;
                used[city] = true;
                pos = (pos + 1) % n;
            }
        }
        return child;
    }

    private static int argmin(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] < v[b]) {
                b = i;
            }
        }
        return b;
    }

    // ------------------------------------------------------------------ CVRP LNS

    /** Destroy operators of the LNS. */
    public enum Destroy {
        /** uniformly random customers */
        RANDOM,
        /** the customers whose removal saves the most */
        WORST,
        /** a random seed customer and its nearest neighbours */
        RELATED,
        /** the learned operator */
        LEARNED
    }

    private static List<List<Integer>> routes(int[] seq) {
        List<List<Integer>> out = new ArrayList<>();
        List<Integer> cur = new ArrayList<>();
        for (int v : seq) {
            if (v == 0) {
                if (!cur.isEmpty()) {
                    out.add(cur);
                }
                cur = new ArrayList<>();
            } else {
                cur.add(v);
            }
        }
        if (!cur.isEmpty()) {
            out.add(cur);
        }
        return out;
    }

    private static int[] sequence(List<List<Integer>> routes) {
        List<Integer> s = new ArrayList<>();
        for (List<Integer> route : routes) {
            if (route.isEmpty()) {
                continue;
            }
            if (!s.isEmpty()) {
                s.add(0);
            }
            s.addAll(route);
        }
        return s.stream().mapToInt(Integer::intValue).toArray();
    }

    private static double cost(double[][] d, List<List<Integer>> routes) {
        double c = 0;
        for (List<Integer> route : routes) {
            int prev = 0;
            for (int v : route) {
                c += d[prev][v];
                prev = v;
            }
            c += d[prev][0];
        }
        return c;
    }

    private static double removalGain(double[][] d, List<List<Integer>> routes, int customer) {
        for (List<Integer> route : routes) {
            int k = route.indexOf(customer);
            if (k >= 0) {
                int prev = k == 0 ? 0 : route.get(k - 1);
                int next = k == route.size() - 1 ? 0 : route.get(k + 1);
                return d[prev][customer] + d[customer][next] - d[prev][next];
            }
        }
        return 0;
    }

    private static void repair(CombinatorialInstances.Cvrp inst, double[][] d, List<List<Integer>> routes,
            List<Integer> removed, SplittableRandom r) {
        double[] demand = inst.demand();
        for (int i = removed.size() - 1; i > 0; i--) {
            int q = r.nextInt(i + 1);
            int t = removed.get(i);
            removed.set(i, removed.get(q));
            removed.set(q, t);
        }
        for (int c : removed) {
            double best = Double.POSITIVE_INFINITY;
            int br = -1;
            int bp = -1;
            for (int ri = 0; ri < routes.size(); ri++) {
                List<Integer> route = routes.get(ri);
                double load = 0;
                for (int v : route) {
                    load += demand[v];
                }
                if (load + demand[c] > inst.capacity()) {
                    continue;
                }
                for (int p = 0; p <= route.size(); p++) {
                    int prev = p == 0 ? 0 : route.get(p - 1);
                    int next = p == route.size() ? 0 : route.get(p);
                    double delta = d[prev][c] + d[c][next] - d[prev][next];
                    if (delta < best) {
                        best = delta;
                        br = ri;
                        bp = p;
                    }
                }
            }
            if (br < 0) {
                List<Integer> fresh = new ArrayList<>();
                fresh.add(c);
                routes.add(fresh);
            } else {
                routes.get(br).add(bp, c);
            }
        }
        routes.removeIf(List::isEmpty);
    }

    /**
     * Large-neighbourhood search for CVRP.
     *
     * @param inst the instance
     * @param destroy the destroy operator
     * @param learned the learned operator ({@link Destroy#LEARNED} only)
     * @param iterations iterations
     * @param share the share of customers destroyed per iteration
     * @param seed the seed
     * @return the trace
     */
    public static Trace lns(CombinatorialInstances.Cvrp inst, Destroy destroy, LearnedDestroy learned, int iterations,
            double share, long seed) {
        long t0 = System.nanoTime();
        SplittableRandom r = new SplittableRandom(seed);
        double[][] d = ClassicHeuristics.distances(inst.coords());
        List<List<Integer>> current = routes(ClassicHeuristics.nearestNeighbourCvrp(d, inst.demand(), inst.capacity()));
        double curCost = cost(d, current);
        List<List<Integer>> best = copy(current);
        double bestCost = curCost;
        int n = inst.size() - 1;
        int k = Math.max(2, (int) Math.round(share * n));
        double[] trace = new double[iterations];
        for (int it = 0; it < iterations; it++) {
            List<List<Integer>> cand = copy(current);
            List<Integer> removed = select(inst, d, cand, destroy, learned, k, r, null);
            for (List<Integer> route : cand) {
                route.removeAll(removed);
            }
            repair(inst, d, cand, removed, r);
            double c = cost(d, cand);
            if (c < curCost - 1e-12 || c < bestCost * 1.01) {
                current = cand;
                curCost = c;
            }
            if (c < bestCost - 1e-12) {
                bestCost = c;
                best = copy(cand);
            }
            trace[it] = bestCost;
        }
        return new Trace(trace, sequence(best), bestCost, (System.nanoTime() - t0) / 1_000_000);
    }

    private static List<List<Integer>> copy(List<List<Integer>> routes) {
        List<List<Integer>> c = new ArrayList<>();
        for (List<Integer> route : routes) {
            c.add(new ArrayList<>(route));
        }
        return c;
    }

    private static List<Integer> select(CombinatorialInstances.Cvrp inst, double[][] d, List<List<Integer>> routes,
            Destroy destroy, LearnedDestroy learned, int k, SplittableRandom r, List<double[]> gradOut) {
        int n = inst.size() - 1;
        List<Integer> customers = new ArrayList<>();
        for (int c = 1; c <= n; c++) {
            customers.add(c);
        }
        List<Integer> out = new ArrayList<>();
        switch (destroy) {
            case RANDOM -> {
                for (int i = 0; i < k; i++) {
                    out.add(customers.remove(r.nextInt(customers.size())));
                }
            }
            case WORST -> {
                double[] gain = new double[n + 1];
                for (int c = 1; c <= n; c++) {
                    gain[c] = removalGain(d, routes, c) * (0.8 + 0.4 * r.nextDouble());
                }
                customers.sort((a, b) -> Double.compare(gain[b], gain[a]));
                out.addAll(customers.subList(0, k));
            }
            case RELATED -> {
                int seedC = 1 + r.nextInt(n);
                customers.sort((a, b) -> Double.compare(d[seedC][a], d[seedC][b]));
                out.addAll(customers.subList(0, k));
            }
            case LEARNED -> out.addAll(learned.sample(inst, d, routes, k, r, gradOut));
        }
        return out;
    }

    /** A learned destroy operator: a softmax over customer features, trained by REINFORCE. */
    public static final class LearnedDestroy {
        private static final int F = 5;
        private final double[] w = new double[F];
        private double baseline;

        /** Creates an untrained operator (uniform, like random destroy). */
        public LearnedDestroy() {
        }

        private static double[] features(CombinatorialInstances.Cvrp inst, double[][] d, List<List<Integer>> routes,
                int c, int seedC) {
            double scale = 0;
            for (int j = 1; j <= inst.size() - 1; j++) {
                scale = Math.max(scale, d[0][j]);
            }
            return new double[] {removalGain(d, routes, c) / scale, d[0][c] / scale, d[seedC][c] / scale,
                inst.demand()[c] / inst.capacity(), c == seedC ? 1 : 0};
        }

        List<Integer> sample(CombinatorialInstances.Cvrp inst, double[][] d, List<List<Integer>> routes, int k,
                SplittableRandom r, List<double[]> gradOut) {
            int n = inst.size() - 1;
            int seedC = 1 + r.nextInt(n);
            double[][] f = new double[n + 1][];
            for (int c = 1; c <= n; c++) {
                f[c] = features(inst, d, routes, c, seedC);
            }
            boolean[] taken = new boolean[n + 1];
            List<Integer> out = new ArrayList<>();
            double[] grad = new double[F];
            for (int i = 0; i < k; i++) {
                double[] s = new double[n + 1];
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 1; c <= n; c++) {
                    if (!taken[c]) {
                        s[c] = dot(w, f[c]);
                        max = Math.max(max, s[c]);
                    }
                }
                double sum = 0;
                for (int c = 1; c <= n; c++) {
                    if (!taken[c]) {
                        s[c] = Math.exp(s[c] - max);
                        sum += s[c];
                    }
                }
                double u = r.nextDouble() * sum;
                int pick = -1;
                for (int c = 1; c <= n; c++) {
                    if (!taken[c]) {
                        u -= s[c];
                        pick = c;
                        if (u <= 0) {
                            break;
                        }
                    }
                }
                for (int c = 1; c <= n; c++) {
                    if (!taken[c]) {
                        double p = s[c] / sum;
                        for (int j = 0; j < F; j++) {
                            grad[j] += ((c == pick ? 1 : 0) - p) * f[c][j];
                        }
                    }
                }
                taken[pick] = true;
                out.add(pick);
            }
            if (gradOut != null) {
                gradOut.add(grad);
            }
            return out;
        }

        private static double dot(double[] a, double[] b) {
            double s = 0;
            for (int i = 0; i < a.length; i++) {
                s += a[i] * b[i];
            }
            return s;
        }

        /**
         * Trains by REINFORCE: from LNS states on the training instances, a destroy-and-repair step
         * is rewarded by its relative cost improvement (against a running baseline).
         *
         * @param instances training instances
         * @param iterations LNS iterations per instance
         * @param share the destroyed share
         * @param lr the step size
         * @param seed the seed
         * @return this operator
         */
        public LearnedDestroy train(List<CombinatorialInstances.Cvrp> instances, int iterations, double share,
                double lr, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            for (CombinatorialInstances.Cvrp inst : instances) {
                double[][] d = ClassicHeuristics.distances(inst.coords());
                List<List<Integer>> current = routes(ClassicHeuristics.nearestNeighbourCvrp(d, inst.demand(),
                        inst.capacity()));
                double curCost = cost(d, current);
                int k = Math.max(2, (int) Math.round(share * (inst.size() - 1)));
                for (int it = 0; it < iterations; it++) {
                    List<List<Integer>> cand = copy(current);
                    List<double[]> grads = new ArrayList<>();
                    List<Integer> removed = sample(inst, d, cand, k, r, grads);
                    for (List<Integer> route : cand) {
                        route.removeAll(removed);
                    }
                    repair(inst, d, cand, removed, r);
                    double c = cost(d, cand);
                    double reward = (curCost - c) / curCost;
                    double adv = reward - baseline;
                    baseline += 0.05 * (reward - baseline);
                    double[] g = grads.getFirst();
                    for (int j = 0; j < F; j++) {
                        w[j] += lr * adv * g[j];
                    }
                    if (c < curCost * 1.01) {
                        current = cand;
                        curCost = c;
                    }
                }
            }
            return this;
        }

        /** @return the learned feature weights (removal gain, depot distance, seed distance, demand, seed) */
        public double[] weights() {
            return w.clone();
        }
    }
}
