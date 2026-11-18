package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SimulatorInferenceTest {

    @Test
    void gaussianLinearAndTwoMoons() {
        Map<String, double[]> g = SimulatorInference.gaussianLinear(new int[] {1000, 4000}, 1);
        System.out.printf("SBI Gaussian-linear C2ST: %s%n", ProbabilisticProgramsTest.line(g));
        Map<String, double[]> m = SimulatorInference.twoMoons(new int[] {4000}, 2);
        System.out.printf("SBI two moons C2ST: %s%n", ProbabilisticProgramsTest.line(m));
        assertTrue(g.get("NUTS (exact likelihood)")[0] < 0.6, "NUTS matches the analytic posterior");
        assertTrue(g.get("NPE n=4000")[0] < 0.6, "NPE close to the analytic posterior");
        assertTrue(m.get("NPE n=4000")[0] < 0.9, "NPE two moons");
    }

    @Test
    void lotkaVolterra() {
        Map<String, double[]> r = SimulatorInference.lotkaVolterra(3000, 3);
        System.out.printf("SBI Lotka-Volterra {RMSE of log-rates, posterior std}: %s%n",
                ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("NPE")[0] < 0.3, "NPE finds the rates");
    }
}
