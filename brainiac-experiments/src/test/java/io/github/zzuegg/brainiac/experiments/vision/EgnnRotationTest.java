package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EgnnRotationTest {

    @Test
    void onlyEgnnIgnoresRotation() {
        GeometricVision.PointScore egnn = GeometricVision.points(GeometricVision.PointModel.EGNN, 100, 1);
        GeometricVision.PointScore pointNet = GeometricVision.points(GeometricVision.PointModel.POINTNET, 150, 1);
        System.out.printf("point clouds, accuracy upright/rotated: EGNN %.3f/%.3f, PointNet %.3f/%.3f%n",
                egnn.accuracy(), egnn.rotatedAccuracy(), pointNet.accuracy(), pointNet.rotatedAccuracy());
        assertTrue(Math.abs(egnn.accuracy() - egnn.rotatedAccuracy()) < 0.1, "EGNN is rotation invariant");
        assertTrue(pointNet.accuracy() - pointNet.rotatedAccuracy() > 0.4, "PointNet is not");
    }
}
