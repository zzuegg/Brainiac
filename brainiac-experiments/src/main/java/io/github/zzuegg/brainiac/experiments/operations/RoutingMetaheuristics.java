package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wave 12 item 12.6.10: routing metaheuristics and learned destroy operators.
 *
 * <ul>
 *   <li>{@link #tsp}: the optimality gap and wall-clock time of ant colony system, tabu search and a
 *       genetic algorithm with order crossover against the wave 11 nearest-neighbour and 2-opt
 *       heuristics, on TSP instances with exact (Held-Karp) or best-known optima;</li>
 *   <li>{@link #cvrp}: large-neighbourhood search with random, worst, related and learned
 *       (REINFORCE-trained) destroy operators at an equal iteration budget, against nearest
 *       neighbour with route improvement.</li>
 * </ul>
 *
 * <p>The wave 11 attention model and POMO rows are not re-run here (their training does not fit the
 * test budget); their gaps are in the wave 11 {@code RoutingShootout} results.
 */
public final class RoutingMetaheuristics {

    private RoutingMetaheuristics() {
    }

    /**
     * TSP.
     *
     * @param n cities
     * @param instances instances
     * @param iterations iterations of every metaheuristic
     * @param seed the seed
     * @return method to {@code {mean gap to the reference, mean milliseconds}}
     */
    public static Map<String, double[]> tsp(int n, int instances, int iterations, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int i = 0; i < instances; i++) {
            CombinatorialInstances.Tsp t = CombinatorialInstances.tsp(n, false, seed + i);
            double[][] d = t.distances();
            double ref = t.optimum();
            long t0 = System.nanoTime();
            int[] nn = ClassicHeuristics.nearestNeighbour(d, 0);
            addTimed(out, "nearest neighbour", ClassicHeuristics.tourLength(d, nn), ref, t0, instances);
            t0 = System.nanoTime();
            ClassicHeuristics.Improvement two = ClassicHeuristics.twoOpt(d, nn, false, Integer.MAX_VALUE);
            addTimed(out, "nearest neighbour + 2-opt", two.length(), ref, t0, instances);
            var acs = io.github.zzuegg.brainiac.experimental.ops.RoutingMetaheuristics.antColonySystem(d, 10,
                    iterations, seed + i);
            add(out, "ant colony system", acs.length(), ref, acs.millis(), instances);
            var tabu = io.github.zzuegg.brainiac.experimental.ops.RoutingMetaheuristics.tabuSearch(d, nn,
                    iterations, Math.max(5, n / 4));
            add(out, "tabu search", tabu.length(), ref, tabu.millis(), instances);
            var ga = io.github.zzuegg.brainiac.experimental.ops.RoutingMetaheuristics.geneticOx(d, 40, iterations,
                    seed + i);
            add(out, "genetic (order crossover)", ga.length(), ref, ga.millis(), instances);
        }
        return out;
    }

    private static void addTimed(Map<String, double[]> out, String key, double length, double ref, long t0,
            int count) {
        add(out, key, length, ref, (System.nanoTime() - t0) / 1_000_000L, count);
    }

    private static void add(Map<String, double[]> out, String key, double length, double ref, long millis,
            int count) {
        double[] row = out.computeIfAbsent(key, k -> new double[2]);
        row[0] += (length - ref) / ref / count;
        row[1] += millis / (double) count;
    }

    /**
     * CVRP.
     *
     * @param customers customers
     * @param instances test instances
     * @param trainInstances training instances of the learned destroy operator
     * @param iterations LNS iterations
     * @param seed the seed
     * @return method to the mean cost relative to nearest neighbour with route improvement ({@code < 1} is
     *     better)
     */
    public static Map<String, Double> cvrp(int customers, int instances, int trainInstances, int iterations,
            long seed) {
        List<CombinatorialInstances.Cvrp> train = new ArrayList<>();
        for (int i = 0; i < trainInstances; i++) {
            train.add(CombinatorialInstances.cvrp(customers, 50_000 + i));
        }
        var learned = new io.github.zzuegg.brainiac.experimental.ops.RoutingMetaheuristics.LearnedDestroy()
                .train(train, iterations, 0.2, 0.05, seed);
        Map<String, Double> out = new LinkedHashMap<>();
        for (int i = 0; i < instances; i++) {
            CombinatorialInstances.Cvrp inst = CombinatorialInstances.cvrp(customers, seed + i);
            double[][] d = ClassicHeuristics.distances(inst.coords());
            double base = ClassicHeuristics.routeLength(d, ClassicHeuristics.improveRoutes(d,
                    ClassicHeuristics.nearestNeighbourCvrp(d, inst.demand(), inst.capacity())));
            for (var destroy : io.github.zzuegg.brainiac.experimental.ops.RoutingMetaheuristics.Destroy.values()) {
                var trace = io.github.zzuegg.brainiac.experimental.ops.RoutingMetaheuristics.lns(inst, destroy,
                        learned, iterations, 0.2, seed + i);
                out.merge("LNS " + destroy.name(), trace.length() / base / instances, Double::sum);
            }
        }
        return out;
    }
}
