package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.SignedDistanceField;
import org.junit.jupiter.api.Test;

class SdfSirenTest {

    @Test
    void sirenAndReluFitOutlines() {
        SceneFields.SdfScore siren = SceneFields.sdf(SignedDistanceField.Kind.SIREN, 100, 1);
        SceneFields.SdfScore relu = SceneFields.sdf(SignedDistanceField.Kind.RELU, 100, 1);
        System.out.printf("SDF auto-decoder, 6 outlines, Chamfer / zero-set IoU / eikonal residual: SIREN"
                + " %.3f / %.3f / %.3f (%.1f s), ReLU %.3f / %.3f / %.3f (%.1f s)%n", siren.chamfer(), siren.iou(),
                siren.eikonal(), siren.seconds(), relu.chamfer(), relu.iou(), relu.eikonal(), relu.seconds());
        assertTrue(relu.iou() > 0.8, "the ReLU field recovers the shapes");
        assertTrue(siren.iou() > 0.3 && siren.chamfer() < 0.3, "the SIREN field has a zero set near the outlines");
    }
}
