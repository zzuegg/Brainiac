package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SinkhornLossTest {

    @Test
    void sinkhornDivergenceTrainsAGenerator() {
        Map<String, double[]> t = TransportFlows.timing(new int[] {256, 1024, 2048}, 1);
        System.out.printf("Sinkhorn timing {log ms/it, scaling ms/it, Hungarian ms}: %s%n",
                ProbabilisticProgramsTest.line(t));
        Map<String, double[]> r = TransportFlows.generatorLoss(120, 3);
        System.out.printf("Generator losses {final sliced W}: %s%n", ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("Sinkhorn divergence")[0] < 0.3, "the divergence fits the rings");
    }
}
