package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BayesianFlowImagesTest {

    @Test
    void bayesianFlowNetworksOnBinaryAndRealImages() {
        Map<String, GenerativeScores.ImageScore> m = DiscreteGenerators.images(600, new int[] {10, 50}, 1);
        StringBuilder line = new StringBuilder("ToyImages (distance per pixel / real share / distinct):");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f/%.2f/%d;", k, v.distance(), v.realShare(),
                v.distinct())));
        System.out.println(line);
        assertTrue(m.get("BFN discrete 50").realShare() > 0.3, "discrete BFN draws real shapes");
        assertTrue(m.get("BFN continuous 50").distance() < 0.5, "continuous BFN approaches the shapes");
    }
}
