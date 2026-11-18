package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class NeuralRatioTest {

    @Test
    void ratioVariantsAgainstNpe() {
        Map<String, double[]> m = LikelihoodFree.ratios(1000, 80, 4);
        System.out.printf("NRE, Gaussian linear, n=1000 (13.5.2) {C2ST, coverage@0.9, coverage@0.5}: %s%n",
                NeuralLikelihoodTest.line(m));
        assertTrue(m.get("NRE_A")[0] < 0.8, "NRE-A learns the posterior");
        assertTrue(m.get("BNRE")[1] > 0.7, "BNRE covers the truth");
    }
}
