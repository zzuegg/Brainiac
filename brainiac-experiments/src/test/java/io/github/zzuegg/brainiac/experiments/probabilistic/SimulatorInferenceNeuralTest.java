package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SimulatorInferenceNeuralTest {

    @Test
    void waveThirteenRowsOnGaussianLinear() {
        Map<String, double[]> g = SimulatorInference.gaussianLinearNeural(1000, 1);
        System.out.printf("SBI Gaussian-linear wave-13 rows C2ST: %s%n", ProbabilisticProgramsTest.line(g));
        assertTrue(g.get("NLE n=1000")[0] < 0.75, "NLE matches the analytic posterior");
    }
}
