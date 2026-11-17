package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PerceiverImagesTest {

    @Test
    void perceiverClassifiesRawPixels() {
        double[] small = PerceiverExperiments.perceiverImages(4, 300, 1);
        double[] large = PerceiverExperiments.perceiverImages(16, 300, 1);
        double[] vit = PerceiverExperiments.vitImages(300, 1);
        System.out.printf("Raw-pixel shapes accuracy (MACs), 300 steps: Perceiver 4 latents %.3f (%.0f),"
                + " 16 latents %.3f (%.0f), ViT %.3f (%.0f)%n", small[0], small[1], large[0], large[1], vit[0], vit[1]);
        assertTrue(Math.max(small[0], large[0]) > 0.3, "the Perceiver learns shapes from raw pixel elements (chance 0.17)");
        assertTrue(vit[0] > large[0], "at this budget the patch ViT, with its built-in locality, learns faster");
        assertTrue(small[1] < large[1], "fewer latents cost fewer multiply-adds");
    }
}
