package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class FlowPosteriorTest {

    @Test
    void flowMatchingAgainstMdn() {
        Map<String, double[]> m = LikelihoodFree.flows(2000, 600, 3);
        System.out.printf("FMPE vs NPE, n=2000 (13.5.3) C2ST: %s%n", NeuralLikelihoodTest.line(m));
        assertTrue(m.get("FMPE Gaussian mixture")[0] < 0.85, "FMPE fits the mixture posterior");
    }
}
