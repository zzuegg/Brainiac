package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GcnnDistortionTest {

    @Test
    void groupConvolutionOnDistortedGlyphs() {
        GeometricVision.DistortionScore cnn = GeometricVision.distortion(GeometricVision.DistortionModel.CNN, 150, 1);
        GeometricVision.DistortionScore gcnn = GeometricVision.distortion(GeometricVision.DistortionModel.GCNN, 150, 1);
        System.out.printf("distorted glyphs after 150 steps, accuracy (params): G-CNN %.3f (%d), CNN %.3f (%d)%n",
                gcnn.accuracy(), gcnn.parameters(), cnn.accuracy(), cnn.parameters());
        assertTrue(gcnn.accuracy() > 1.5 / GeometricVision.GLYPHS, "the G-CNN learns the glyphs");
    }
}
