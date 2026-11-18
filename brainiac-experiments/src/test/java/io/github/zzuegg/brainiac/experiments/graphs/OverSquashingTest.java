package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/** Wave 7 items 7.1.7 and 7.1.17. */
class OverSquashingTest {

    @Test
    void attentionAndWalksReachTheRoot() {
        Map<String, Double> r = GraphStructureShootout.overSquashing(
                List.of("GIN", "GIN + virtual node", "GIN + FoSR", "GPS", "NeuralWalker"), new int[] {2, 3}, 128, 100, 1);
        System.out.printf("7.1.7/7.1.17 neighbours-match depth 2 / 3: GPS %.3f / %.3f, NeuralWalker %.3f / %.3f,"
                + " virtual node %.3f / %.3f, FoSR %.3f / %.3f; GIN %.3f / %.3f (chance 0.25 / 0.125)%n",
                r.get("GPS @ 2"), r.get("GPS @ 3"), r.get("NeuralWalker @ 2"), r.get("NeuralWalker @ 3"),
                r.get("GIN + virtual node @ 2"), r.get("GIN + virtual node @ 3"), r.get("GIN + FoSR @ 2"),
                r.get("GIN + FoSR @ 3"), r.get("GIN @ 2"), r.get("GIN @ 3"));
        assertTrue(r.get("GPS @ 3") > r.get("GIN @ 3") + 0.2, r.toString());
        assertTrue(r.get("GPS @ 2") > 0.8, r.toString());
    }
}
