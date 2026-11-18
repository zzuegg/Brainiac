package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class FlowUNetTest {

    @Test
    void flowNetStyleUNetOnLargeMotions() {
        double[] unet = DensePrediction.regressedFlow(DensePrediction.FlowNet.UNET, true, 80, 1);
        System.out.printf("FlowNet-style U-Net EPE all/moving, 6-10 px: %.3f/%.3f%n", unet[0], unet[1]);
        assertTrue(Double.isFinite(unet[1]), "the U-Net regressor trains");
    }
}
