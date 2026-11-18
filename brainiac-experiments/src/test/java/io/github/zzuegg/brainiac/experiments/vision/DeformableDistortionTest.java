package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DeformableDistortionTest {

    @Test
    void deformableConvolutionIsAStrongRival() {
        GeometricVision.DistortionScore cnn = GeometricVision.distortion(GeometricVision.DistortionModel.CNN, 120, 1);
        GeometricVision.DistortionScore dcn = GeometricVision.distortion(GeometricVision.DistortionModel.DEFORMABLE_CNN,
                120, 1);
        System.out.printf("distorted glyphs after 120 steps, accuracy (params): deformable CNN %.3f (%d), CNN %.3f"
                + " (%d)%n", dcn.accuracy(), dcn.parameters(), cnn.accuracy(), cnn.parameters());
        assertTrue(dcn.accuracy() > cnn.accuracy() + 0.05, "deformable sampling absorbs the distortion");
    }
}
