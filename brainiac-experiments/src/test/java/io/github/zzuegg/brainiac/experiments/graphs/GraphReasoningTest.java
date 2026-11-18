package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.AlgorithmTraces;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/** Wave 7 item 7.1.14. */
class GraphReasoningTest {

    @Test
    void maxAggregationExtrapolatesBfsToLargerGraphs() {
        Map<String, Double> r = GraphReasoning.algorithmic(AlgorithmTraces.Algorithm.BFS,
                List.of("max + hints", "max", "sum + hints", "GIN", "GAT", "GPS"), 64, 150, 1);
        System.out.printf("7.1.14 BFS at 64 nodes (trained 8-16): max+hints %.3f, max %.3f, sum+hints %.3f; GIN %.3f,"
                + " GAT %.3f, GPS %.3f%n", r.get("max + hints"), r.get("max"), r.get("sum + hints"), r.get("GIN"),
                r.get("GAT"), r.get("GPS"));
        assertTrue(r.get("max + hints") > r.get("GIN") + 0.2, r.toString());
        assertTrue(r.get("max + hints") >= r.get("sum + hints"), r.toString());
    }

    @Test
    void bellmanFordAndPrimPointers() {
        Map<String, Double> bf = GraphReasoning.algorithmic(AlgorithmTraces.Algorithm.BELLMAN_FORD,
                List.of("max + hints", "max", "sum + hints", "GIN"), 64, 150, 1);
        Map<String, Double> prim = GraphReasoning.algorithmic(AlgorithmTraces.Algorithm.PRIM,
                List.of("max + hints", "max", "sum + hints"), 32, 100, 1);
        System.out.printf("7.1.14 Bellman-Ford pointers at 64 nodes: max+hints %.3f, max %.3f, sum+hints %.3f, GIN %.3f."
                + " Prim at 32: max+hints %.3f, max %.3f, sum+hints %.3f%n", bf.get("max + hints"), bf.get("max"),
                bf.get("sum + hints"), bf.get("GIN"), prim.get("max + hints"), prim.get("max"), prim.get("sum + hints"));
        assertTrue(bf.get("max + hints") > bf.get("sum + hints") + 0.1, bf.toString());
        assertTrue(bf.get("max + hints") > bf.get("GIN") + 0.2, bf.toString());
    }
}
