package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class UNetPredictionTest {

    @Test
    void unetFramePredictorRollsOut() {
        VideoTracking.PredictionScore copy = VideoTracking.predict(VideoTracking.Predictor.COPY_LAST, 0, 1);
        VideoTracking.PredictionScore unet = VideoTracking.predict(VideoTracking.Predictor.UNET, 120, 1);
        System.out.printf("video prediction MSE/SSIM at horizons 1 and 4: U-Net rollout %.4f/%.3f, %.4f/%.3f"
                + " (%.1f ms per step; copy last %.4f at horizon 4)%n", unet.mse()[0], unet.ssim()[0], unet.mse()[3],
                unet.ssim()[3], unet.seconds() * 1000 / 120, copy.mse()[3]);
        assertTrue(unet.mse()[0] < copy.mse()[0], "the U-Net predicts the next frame");
    }
}
