package io.github.zzuegg.brainiac.experiments.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class IrregularSeriesClassificationTest {

    @Test
    void subsampledSeriesAreClassifiedAboveChance() {
        Map<String, double[]> r = IrregularSeriesShootout.classification(0.6,
                new IrregularSeriesShootout.Budget(240, 90, 64, 1, 120, 24, 24), 1);
        String line = PointProcessShootoutTest.line("irregular CBF 60% missing accuracy, ms", r);
        System.out.println(line);
        assertTrue(r.get("GRU-D")[0] > 0.45, line);
        assertTrue(r.get("mTAN")[0] > 0.45, line);
    }
}
