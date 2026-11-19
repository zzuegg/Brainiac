package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SimulatorInferenceMoonsTest {

    @Test
    void waveThirteenRowsOnTwoMoons() {
        Map<String, double[]> m = SimulatorInference.twoMoonsNeural(1000, 2);
        System.out.printf("SBI two moons wave-13 rows C2ST: %s%n", ProbabilisticProgramsTest.line(m));
        assertTrue(m.get("NLE n=1000")[0] < 0.9, "NLE finds the crescent");
    }
}
