package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SequentialPosteriorTest {

    @Test
    void sequentialMethodsAgainstAmortised() {
        Map<String, double[]> m = LikelihoodFree.sequential(2, 600, 5);
        System.out.printf("Sequential SLCP, 2x600 sims (13.5.4) {C2ST, leakage}: %s%n", NeuralLikelihoodTest.line(m));
        assertTrue(m.get("TSNPE")[0] < 0.97, "TSNPE runs and finds the posterior region");
        assertTrue(m.get("TSNPE")[1] < 0.2, "TSNPE keeps its mass inside the prior");
    }
}
