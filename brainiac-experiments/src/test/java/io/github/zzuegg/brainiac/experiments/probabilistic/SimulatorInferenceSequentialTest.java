package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SimulatorInferenceSequentialTest {

    @Test
    void sequentialRowsOnLotkaVolterra() {
        Map<String, double[]> lv = SimulatorInference.lotkaVolterraSequential(2, 500, 3);
        System.out.printf("SBI Lotka-Volterra sequential 2x500 {RMSE of log-rates, posterior std}: %s%n",
                ProbabilisticProgramsTest.line(lv));
        assertTrue(lv.get("TSNPE")[0] < 0.5, "TSNPE finds the rates");
    }
}
