package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.SignedDistanceField;
import org.junit.jupiter.api.Test;

class SdfFourierTest {

    @Test
    void fourierFeaturesBreakTheDistanceField() {
        SceneFields.SdfScore fourier = SceneFields.sdf(SignedDistanceField.Kind.FOURIER, 100, 1);
        SceneFields.SdfScore relu = SceneFields.sdf(SignedDistanceField.Kind.RELU, 100, 1);
        System.out.printf("SDF auto-decoder, Chamfer / zero-set IoU / eikonal residual: Fourier features"
                + " %.3f / %.3f / %.3f, ReLU %.3f / %.3f / %.3f%n", fourier.chamfer(), fourier.iou(),
                fourier.eikonal(), relu.chamfer(), relu.iou(), relu.eikonal());
        assertTrue(relu.iou() > fourier.iou(), "at this budget the plain ReLU field is the cleaner SDF");
    }
}
