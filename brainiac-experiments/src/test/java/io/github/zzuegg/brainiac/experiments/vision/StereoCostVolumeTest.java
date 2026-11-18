package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class StereoCostVolumeTest {

    @Test
    void costVolumeBeatsBlockMatching() {
        double[] cv = DensePrediction.stereo(DensePrediction.StereoMethod.COST_VOLUME, 150, 1);
        double[] bm = DensePrediction.stereo(DensePrediction.StereoMethod.BLOCK_MATCHING, 0, 1);
        System.out.printf("stereo disparity error/share off by >1 px: cost volume + soft-argmin %.3f/%.3f,"
                + " block matching %.3f/%.3f%n", cv[0], cv[1], bm[0], bm[1]);
        assertTrue(cv[0] < bm[0], "the learned cost volume is more accurate than block matching");
    }
}
