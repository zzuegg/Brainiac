package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TransportGeometryTest {

    @Test
    void gromovWassersteinAndBarycenters() {
        Map<String, double[]> g = TransportGeometry.graphMatching(new double[] {0, 0.05, 0.15}, 3);
        System.out.printf("Graph matching acc {noise 0, 0.05, 0.15}: %s%n", ProbabilisticProgramsTest.line(g));
        Map<String, double[]> a = TransportGeometry.embeddingAlignment(new int[] {3, 8, 20}, 4);
        System.out.printf("Embedding alignment P@1: %s%n", ProbabilisticProgramsTest.line(a));
        Map<String, double[]> b = TransportGeometry.barycenter(2, 12, 5);
        System.out.printf("Glyph barycenter {mean W2 cost, entropy}: %s%n", ProbabilisticProgramsTest.line(b));
        assertTrue(g.get("Gromov-Wasserstein")[0] > 0.8, "GW matches the clean graph");
        assertTrue(a.get("GW (no anchors)")[0] > a.get("Procrustes k=3")[0], "GW beats few-anchor Procrustes");
        assertTrue(b.get("Wasserstein barycenter")[1] < b.get("Euclidean mean")[1], "the barycenter stays sharp");
    }
}
