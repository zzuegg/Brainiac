package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SimVpPredictionTest {

    @Test
    void simVpBeatsCopyingTheLastFrame() {
        VideoTracking.PredictionScore copy = VideoTracking.predict(VideoTracking.Predictor.COPY_LAST, 0, 1);
        VideoTracking.PredictionScore simvp = VideoTracking.predict(VideoTracking.Predictor.SIM_VP, 150, 1);
        System.out.printf("video prediction MSE/SSIM at horizons 1 and 4: copy last %.4f/%.3f, %.4f/%.3f;"
                + " SimVP %.4f/%.3f, %.4f/%.3f (%.1f ms per step)%n", copy.mse()[0], copy.ssim()[0], copy.mse()[3],
                copy.ssim()[3], simvp.mse()[0], simvp.ssim()[0], simvp.mse()[3], simvp.ssim()[3],
                simvp.seconds() * 1000 / 150);
        assertTrue(simvp.mse()[3] < copy.mse()[3], "SimVP extrapolates the motion");
    }
}
