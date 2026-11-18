package io.github.zzuegg.brainiac.experiments.operations;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RoutingMetaheuristicsTest {

    @Test
    void tsp() {
        Map<String, double[]> r = RoutingMetaheuristics.tsp(20, 3, 300, 1);
        System.out.printf("12.6.10 TSP-20 gap/ms: %s%n", InventoryAndPricingTest.table(r));
        double nn = r.get("nearest neighbour")[0];
        assertTrue(r.get("tabu search")[0] < nn && r.get("ant colony system")[0] < nn,
                "tabu search and ACS improve on nearest neighbour");
        assertTrue(r.get("tabu search")[0] < 0.05);
    }

    @Test
    void cvrpLns() {
        Map<String, Double> r = RoutingMetaheuristics.cvrp(20, 3, 3, 150, 1);
        System.out.printf("12.6.10 CVRP-20 LNS cost / (NN + route improvement): %s%n", InventoryAndPricingTest.row(r));
        assertTrue(r.get("LNS RANDOM") < 1.0, "LNS improves the constructive start");
        assertTrue(r.get("LNS LEARNED") < 1.0);
    }
}
