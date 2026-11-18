package io.github.zzuegg.brainiac.experiments.estimation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class InferenceAndTestingTest {

    private static String line(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append(' ');
            for (double x : v) {
                sb.append(String.format("%.3g ", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void expectationPropagationTracksHmcCloserThanLaplace() {
        Map<String, double[]> r = InferenceAndTesting.gpClassification(60, 1000, 1);
        System.out.printf("GP classification, NLL / ECE / marginal KL / log Z / sweeps: %s%n", line(r));
        for (String d : new String[] {"moons", "noisy xor", "imbalanced blobs"}) {
            assertTrue(r.get(d + " / EP sequential")[2] < r.get(d + " / Laplace probit")[2], line(r));
            double truth = r.get(d + " / truth (HMC, AIS evidence)")[3];
            assertTrue(Math.abs(r.get(d + " / EP sequential")[3] - truth)
                    < Math.abs(r.get(d + " / Laplace probit")[3] - truth), line(r));
            assertTrue(Math.abs(r.get(d + " / EP parallel (damping 0.5)")[3] - r.get(d + " / EP sequential")[3]) < 0.05,
                    line(r));
        }
    }

    @Test
    void nestedSamplingRecoversKnownEvidence() {
        Map<String, double[]> r = InferenceAndTesting.evidence(300, 2);
        System.out.printf("evidence, log Z / reported error / |error| / k-evals: %s%n", line(r));
        for (String p : new String[] {"gaussian mixture d=5", "gaussian shells d=2"}) {
            assertTrue(r.get(p + " / NS SLICE")[2] < 0.5, line(r));
            assertTrue(r.get(p + " / NS MULTI_ELLIPSOID")[2] < 0.5, line(r));
        }
    }
}
