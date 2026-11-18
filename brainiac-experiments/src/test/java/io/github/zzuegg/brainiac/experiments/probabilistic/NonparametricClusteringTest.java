package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class NonparametricClusteringTest {

    @Test
    void dirichletProcessInfersK() {
        Map<String, double[]> r = NonparametricClustering.unknownK(new int[] {2, 4, 6, 8, 10, 12}, 400, 1);
        System.out.printf("Unknown K {ARI, |K err|, held-out LL}: %s%n", ProbabilisticProgramsTest.line(r));
        Map<String, double[]> s = NonparametricClustering.stream(5, 120, 2);
        System.out.printf("Stream {ARI, |K err|}: %s%n", ProbabilisticProgramsTest.line(s));
        assertTrue(r.get("DP Gibbs")[0] > 0.9, "Gibbs recovers the clusters");
        assertTrue(r.get("DP Gibbs")[1] <= r.get("k-means + elbow")[1], "K at least as good as the elbow rule");
        assertTrue(s.get("DP Gibbs (warm start)")[0] > s.get("k-means, K from chunk 1")[0], "adapts to new clusters");
    }
}
