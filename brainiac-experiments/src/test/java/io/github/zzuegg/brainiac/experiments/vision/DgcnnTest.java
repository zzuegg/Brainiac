package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DgcnnTest {

    @Test
    void edgeConvClassifiesClouds() {
        GeometricVision.PointScore dgcnn = GeometricVision.points(GeometricVision.PointModel.DGCNN, 150, 1);
        GeometricVision.PointScore pointNet = GeometricVision.points(GeometricVision.PointModel.POINTNET, 150, 1);
        System.out.printf("point clouds, accuracy upright/rotated (params, s): DGCNN %.3f/%.3f (%d, %.1f s), PointNet"
                + " %.3f/%.3f (%d, %.1f s)%n", dgcnn.accuracy(), dgcnn.rotatedAccuracy(), dgcnn.parameters(),
                dgcnn.seconds(), pointNet.accuracy(), pointNet.rotatedAccuracy(), pointNet.parameters(),
                pointNet.seconds());
        assertTrue(dgcnn.accuracy() > 0.85, "DGCNN classifies upright clouds");
    }
}
