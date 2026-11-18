package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GroupConvLowDataTest {

    @Test
    void groupConvolutionBeatsAugmentationOnFewImages() {
        GeometricVision.RotationScore aug = GeometricVision.rotation(GeometricVision.RotationModel.CNN_AUGMENTED, 64,
                120, 1);
        GeometricVision.RotationScore gcnn = GeometricVision.rotation(GeometricVision.RotationModel.GCNN, 64, 120, 1);
        GeometricVision.RotationScore cnn = GeometricVision.rotation(GeometricVision.RotationModel.CNN, 64, 120, 1);
        System.out.printf("64 training images, rotated accuracy: G-CNN %.3f, CNN + augmentation %.3f, CNN %.3f%n",
                gcnn.rotated(), aug.rotated(), cnn.rotated());
        assertTrue(gcnn.rotated() > aug.rotated(), "built-in invariance beats learned invariance on little data");
    }
}
