package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DiscreteGeneratorsTest {

    @Test
    void discreteFlowMatchingAndBfnWriteWords() {
        Map<String, Double> m = DiscreteGenerators.words(TextDiffusionVsAutoregressiveTest.WORDS, 200,
                new int[] {3, 12}, 1);
        StringBuilder line = new StringBuilder("valid-word share (6-letter words) by sampling steps:");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f;", k, v)));
        System.out.println(line);
        assertTrue(m.get("DFM mask 12") > 0.4 && m.get("DFM uniform 12") > 0.3, "DFM writes words");
        assertTrue(m.get("BFN 3") > m.get("masked diffusion 3"), "BFN beats masked diffusion at 3 steps");
    }
}
