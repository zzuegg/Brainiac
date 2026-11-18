package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class GaussianProcessKernelsTest {

    @Test
    void deepGpsOnNonStationaryData() {
        for (String task : new String[] {"step", "heteroscedastic"}) {
            Map<String, double[]> r = GaussianProcessKernels.deepGp(task, 1);
            System.out.printf("Deep GP %s {NLL, CRPS}: %s%n", task, ProbabilisticProgramsTest.line(r));
            r.forEach((k, v) -> assertTrue(Double.isFinite(v[0]), k));
            if (task.equals("step")) {
                assertTrue(r.get("deep GP L=2")[1] < r.get("SVGP (RBF, M=20)")[1], "the deep GP models the step");
            }
        }
    }

    @Test
    void spectralMixtureExtrapolatesTheSeason() {
        Map<String, double[]> r = GaussianProcessKernels.spectralForecast(2);
        System.out.printf("Seasonal forecast {MSE over 1-4 periods}: %s%n", ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("GP spectral mixture (Q=3)")[3] < r.get("GP RBF")[3], "SM beats RBF at 4 periods");
    }
}
