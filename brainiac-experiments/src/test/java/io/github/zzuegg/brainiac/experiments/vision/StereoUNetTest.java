package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class StereoUNetTest {

    @Test
    void unetRegressesDisparityLessAccurately() {
        double[] unet = DensePrediction.stereo(DensePrediction.StereoMethod.UNET, 150, 1);
        System.out.printf("stereo disparity error/share off by >1 px: U-Net on the stacked pair %.3f/%.3f%n", unet[0],
                unet[1]);
        assertTrue(unet[0] < 2, "the U-Net learns something about disparity");
    }
}
