package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class StnDistortionTest {

    @Test
    void spatialTransformerHelpsTheCnn() {
        GeometricVision.DistortionScore cnn = GeometricVision.distortion(GeometricVision.DistortionModel.CNN, 300, 1);
        GeometricVision.DistortionScore stn = GeometricVision.distortion(GeometricVision.DistortionModel.STN_CNN, 300,
                1);
        System.out.printf("distorted glyphs, accuracy (params): STN + CNN %.3f (%d, transform spread %.4f), CNN %.3f"
                + " (%d)%n", stn.accuracy(), stn.parameters(), stn.transformSpread(), cnn.accuracy(), cnn.parameters());
        assertTrue(stn.accuracy() > cnn.accuracy(), "the spatial transformer canonicalises the glyphs");
    }
}
