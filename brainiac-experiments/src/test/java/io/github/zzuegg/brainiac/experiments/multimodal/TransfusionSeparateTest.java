package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TransfusionSeparateTest {

    @Test
    void captionsAndGenerates() {
        FusionShootout.GenerationScores s = FusionShootout.generation(FusionShootout.Arrangement.TRANSFUSION_SEPARATE, 300, 3, 1);
        System.out.printf("Generation %-46s caption %.3f generation %.3f distance %.3f params %d (%d ms)%n",
                s.method(), s.captionAccuracy(), s.generationAccuracy(), s.distance(), s.parameters(), s.millis());
        assertTrue(s.captionAccuracy() > 0.2, "the Transfusion captioner beats chance (0.1)");
    }
}
