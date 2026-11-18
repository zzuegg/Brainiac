package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RelationalGraphsHierarchyTest {

    @Test
    void twoHyperbolicDimensionsBeatEuclideanOnATree() {
        Map<String, double[]> r = RelationalGraphs.hierarchy(6, new int[] {2, 10}, 40, 1);
        String line = RelationalGraphsLinkTest.line("Hierarchy mean-rank/MAP/norm-depth-r/ms", r);
        System.out.println(line);
        assertTrue(r.get("poincare 2d")[0] < r.get("euclidean 2d")[0], line);
        assertTrue(r.get("lorentz 2d")[1] > r.get("euclidean 2d")[1], line);
        assertTrue(r.get("poincare 2d")[2] > 0.5, line);
    }
}
