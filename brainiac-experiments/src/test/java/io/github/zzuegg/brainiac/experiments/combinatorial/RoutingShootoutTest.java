package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Tsp;
import io.github.zzuegg.brainiac.experiments.combinatorial.RoutingShootout.Row;

import java.util.List;
import java.util.StringJoiner;

import org.junit.jupiter.api.Test;

/** Item 11.2.1: supervised and actor-critic pointer networks against the classical heuristics. */
class RoutingShootoutTest {

    static final List<Tsp> TSP10 = CombinatorialInstances.tspSet(20, 10, false, 101);
    static final List<Tsp> TSP20 = CombinatorialInstances.tspSet(10, 20, false, 202);

    static String line(List<Row> rows) {
        StringJoiner j = new StringJoiner("; ");
        rows.forEach(r -> j.add(r.toString()));
        return j.toString();
    }

    static Row find(List<Row> rows, String method, String label) {
        return rows.stream().filter(r -> r.method().equals(method) && r.label().startsWith(label)).findFirst()
                .orElseThrow();
    }

    @Test
    void pointerNetworksAgainstHeuristics() {
        List<Row> heur = RoutingShootout.heuristics(TSP10, 1);
        List<Row> rows = RoutingShootout.pointerNetworks(10, 100, 32, List.of(TSP10, TSP20), 1);
        System.out.println("TSP-10 heuristics, gap to Held-Karp: " + line(heur));
        System.out.println("Pointer networks (100 steps x 32 TSP-10): " + line(rows));
        double random = find(heur, "random", "").gap();
        double sup = find(rows, "pointer supervised", "TSP-10").gap();
        double ac = find(rows, "pointer actor-critic", "TSP-10").gap();
        System.out.printf("11.2.1 TSP-10 gap: supervised %.1f%%, actor-critic %.1f%%, NN %.1f%%, 2-opt %.1f%%, random %.1f%%%n",
                sup, ac, find(heur, "nearest neighbour", "").gap(), find(heur, "2-opt", "NN").gap(), random);
        assertTrue(find(heur, "2-opt+Or-opt", "").gap() < find(heur, "nearest neighbour", "").gap(),
                "local search improves nearest neighbour");
        assertTrue(Math.min(sup, ac) < random, "a pointer network beats random tours");
    }
}
