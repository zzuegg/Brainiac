package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class HydrologyTest {

    @Test
    void calibratedAndHybridModelsTrackRunoffUnderWarming() {
        Map<String, double[]> r = Hydrology.run(7, 3, 2.0, 150, 60, 60, 1);
        System.out.printf("13.3.11 hydrology {NSE cold, NSE warm, KGE cold, KGE warm, |d mean err|, |d Q95 err|,"
                + " s}: %s%n", Lines.of(r));
        assertTrue(r.get("HBV CMA-ES")[0] > 0.6, Lines.of(r));
        assertTrue(r.get("delta-HBV static (from CMA-ES)")[0] > 0.5, Lines.of(r));
        assertTrue(r.get("LSTM")[0] > 0.0, Lines.of(r));
        assertTrue(r.get("delta-HBV static (from CMA-ES)")[5] < r.get("LSTM")[5], Lines.of(r));
    }
}
