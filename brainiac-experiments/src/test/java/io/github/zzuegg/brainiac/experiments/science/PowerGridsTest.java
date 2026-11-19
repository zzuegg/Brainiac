package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PowerGridsTest {

    @Test
    void completionAndCorrectionCutViolationsOfLearnedDispatch() {
        Map<String, double[]> r = PowerGrids.run(new int[] {9, 14, 30}, 300, 100, 300, 1);
        System.out.printf("13.3.12 DC-OPF: %s%n", Lines.of(r));
        String k = " {opt gap %, violation rate, mean violation MW, us per scenario}";
        for (int nb : new int[] {9, 14, 30}) {
            assertTrue(r.get(nb + "-bus DC3" + k)[1] <= r.get(nb + "-bus MLP_DIRECT" + k)[1], Lines.of(r));
            double[] se = r.get(nb + "-bus state estimate {RMSE raw injections, RMSE WLS} MW");
            assertTrue(se[1] < se[0], Lines.of(r));
        }
    }
}
