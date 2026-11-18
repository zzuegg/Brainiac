package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GroupConvRotationTest {

    @Test
    void groupConvolutionIsRotationInvariant() {
        GeometricVision.RotationScore cnn = GeometricVision.rotation(GeometricVision.RotationModel.CNN, 0, 120, 1);
        GeometricVision.RotationScore aug = GeometricVision.rotation(GeometricVision.RotationModel.CNN_AUGMENTED, 0,
                120, 1);
        GeometricVision.RotationScore gcnn = GeometricVision.rotation(GeometricVision.RotationModel.GCNN, 0, 120, 1);
        System.out.printf("trained upright, accuracy upright/rotated (params): G-CNN %.3f/%.3f (%d), CNN %.3f/%.3f"
                + " (%d), CNN + rotation augmentation %.3f/%.3f%n", gcnn.upright(), gcnn.rotated(), gcnn.parameters(),
                cnn.upright(), cnn.rotated(), cnn.parameters(), aug.upright(), aug.rotated());
        assertTrue(Math.abs(gcnn.upright() - gcnn.rotated()) < 0.02, "the G-CNN is invariant to quarter turns");
        assertTrue(gcnn.rotated() > cnn.rotated() + 0.15, "the plain CNN does not generalise to rotations");
    }
}
