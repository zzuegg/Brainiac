package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PointCloudTest {

    @Test
    void pointNetBeatsSortedCoordinates() {
        GeometricVision.PointScore plain = GeometricVision.points(GeometricVision.PointModel.POINTNET, 150, 1);
        GeometricVision.PointScore tnet = GeometricVision.points(GeometricVision.PointModel.POINTNET_TNET, 150, 1);
        GeometricVision.PointScore sets = GeometricVision.points(GeometricVision.PointModel.DEEP_SETS, 150, 1);
        GeometricVision.PointScore sorted = GeometricVision.points(GeometricVision.PointModel.SORTED_MLP, 150, 1);
        System.out.printf("point clouds, accuracy upright/rotated: PointNet %.3f/%.3f, PointNet+T-Net %.3f/%.3f,"
                + " Deep Sets %.3f/%.3f, sorted-coordinate MLP %.3f/%.3f%n", plain.accuracy(), plain.rotatedAccuracy(),
                tnet.accuracy(), tnet.rotatedAccuracy(), sets.accuracy(), sets.rotatedAccuracy(), sorted.accuracy(),
                sorted.rotatedAccuracy());
        assertTrue(plain.accuracy() > 0.85, "PointNet classifies upright clouds");
        assertTrue(plain.accuracy() > sorted.accuracy() + 0.1, "symmetric pooling beats sorting the points");
    }
}
