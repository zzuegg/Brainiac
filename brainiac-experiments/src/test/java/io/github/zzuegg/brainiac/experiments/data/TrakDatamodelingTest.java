package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TrakDatamodelingTest {

    @Test
    void trakPredictsSubsetRetraining() {
        long t = System.nanoTime();
        Map<String, Double> lds = TrainingAttributionExperiment.datamodeling(false, 3, 24, 16, 1);
        lds.forEach((k, v) -> System.out.printf("LDS linear digits %-28s %.3f%n", k, v));
        System.out.printf("(%d ms)%n", (System.nanoTime() - t) / 1_000_000);
        assertTrue(lds.get("TRAK (3 models)") > 0.05, "TRAK correlates with subset retraining");
    }
}
