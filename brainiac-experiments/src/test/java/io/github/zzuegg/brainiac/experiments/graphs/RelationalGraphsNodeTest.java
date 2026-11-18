package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RelationalGraphsNodeTest {

    @Test
    void relationTypesCarryTheGeneration() {
        Map<String, Double> r = RelationalGraphs.nodeClassification(8, 150, 1);
        System.out.printf("KG node classification (generation) acc: R-GCN %.3f, CompGCN %.3f, untyped GCN %.3f%n",
                r.get("RGCN_BASIS"), r.get("COMPGCN_MULT"), r.get("GCN (untyped)"));
        assertTrue(r.get("RGCN_BASIS") > r.get("GCN (untyped)"), r.toString());
        assertTrue(r.get("RGCN_BASIS") > 0.8, r.toString());
    }
}
