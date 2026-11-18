package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class MisspecificationTest {

    @Test
    void robustMethodsRestoreCoverage() {
        Map<String, double[]> m = Misspecification.table(2000, 50, 40, 1);
        System.out.printf("Misspecification (13.5.7) {coverage@0.9, log p(true), sd}: %s%n",
                NeuralLikelihoodTest.line(m));
        assertTrue(m.get("MMD test {stat, p}")[1] < 0.05, "MMD flags the misspecification");
        assertTrue(m.get("RoPE tau=0.05")[0] > m.get("NPE (simulations only)")[0], "RoPE improves coverage");
    }
}
