package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class NeuralLikelihoodTest {

    static String line(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k);
            for (double x : v) {
                sb.append(String.format(" %.3g", x));
            }
            sb.append("; ");
        });
        return sb.toString();
    }

    @Test
    void nleAgainstNpeAndAbcOnSlcp() {
        Map<String, double[]> m = LikelihoodFree.neuralLikelihood(2000, 1);
        System.out.printf("NLE on SLCP, n=2000 (13.5.1) C2ST: %s%n", line(m));
        assertTrue(m.get("NLE-MDN")[0] < 0.95, "NLE finds the SLCP posterior region");
    }
}
