package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class PosteriorDiagnosticsTest {

    @Test
    void plantedFailuresAreDetected() {
        Map<String, double[]> m = PosteriorDiagnostics.detection(150, 6, 1);
        System.out.printf("Diagnostics (13.5.6) rejection rate at 5%% {SBC, coverage, TARP, L-C2ST}: %s%n",
                NeuralLikelihoodTest.line(m));
        double[] lc = PosteriorDiagnostics.localAgainstOracle(300, 10, 2);
        System.out.printf("L-C2ST vs oracle C2ST on 10 observations: Spearman %.3f, oracle C2ST %.3f, statistic %.4f%n",
                lc[0], lc[1], lc[2]);
        assertTrue(m.get("linear underdispersed")[0] > 0.5, "SBC catches underdispersion");
        assertTrue(m.get("linear exact")[0] < 0.5, "SBC rarely rejects the exact posterior");
    }
}
