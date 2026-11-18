package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ConvLstmPredictionTest {

    @Test
    void convLstmBeatsCopyingTheLastFrame() {
        VideoTracking.PredictionScore copy = VideoTracking.predict(VideoTracking.Predictor.COPY_LAST, 0, 1);
        VideoTracking.PredictionScore lstm = VideoTracking.predict(VideoTracking.Predictor.CONV_LSTM, 70, 1);
        System.out.printf("video prediction MSE/SSIM at horizons 1 and 4: ConvLSTM %.4f/%.3f, %.4f/%.3f"
                + " (%.1f ms per step; copy last %.4f at horizon 4)%n", lstm.mse()[0], lstm.ssim()[0], lstm.mse()[3],
                lstm.ssim()[3], lstm.seconds() * 1000 / 70, copy.mse()[3]);
        assertTrue(lstm.mse()[3] < copy.mse()[3], "ConvLSTM extrapolates the motion");
    }
}
