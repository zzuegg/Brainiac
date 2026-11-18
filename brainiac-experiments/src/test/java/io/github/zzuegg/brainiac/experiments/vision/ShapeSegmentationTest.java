package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ShapeSegmentationTest {

    @Test
    void skipConnectionsSharpenSegmentation() {
        double[] unet = ShapeSegmentation.run(true, 200, 1);
        double[] plain = ShapeSegmentation.run(false, 200, 1);
        System.out.printf("shape segmentation mIoU/pixel accuracy: U-Net %.3f/%.3f, no skips %.3f/%.3f%n", unet[0],
                unet[1], plain[0], plain[1]);
        assertTrue(unet[0] > plain[0] + 0.05, "skip connections improve IoU");
        assertTrue(unet[1] > 0.9, "the U-Net labels most pixels right");
    }
}
