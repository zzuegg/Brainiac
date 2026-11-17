package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TrainingTricksExperimentTest {

    @Test
    void decayingSchedulesEndLowerThanAConstantRate() {
        Map<String, Float> mse = TrainingTricksExperiment.schedules(280, 1);
        System.out.println(mse);
        float constant = mse.get("constant");
        for (String s : new String[] {"cosine", "wsd", "1cycle"}) {
            assertTrue(mse.get(s) < constant, s + " beats a constant rate: " + mse);
        }
    }

    @Test
    void zLossKeepsTheNormalizerSmall() {
        float[] plain = TrainingTricksExperiment.zLoss(0f, 2);
        float[] z = TrainingTricksExperiment.zLoss(1e-2f, 2);
        System.out.println("z-loss: plain " + plain[0] + " |logZ| " + plain[1] + ", z " + z[0] + " |logZ| " + z[1]);
        assertTrue(z[1] < 0.7f * plain[1], "z-loss shrinks |log Z|");
        assertTrue(z[0] > plain[0] - 0.05f, "without hurting accuracy");
    }

    @Test
    void rDropAndGradientNoiseTrain() {
        float plain = TrainingTricksExperiment.rDrop(0f, 3);
        float rdrop = TrainingTricksExperiment.rDrop(1f, 3);
        System.out.println("R-Drop: plain " + plain + ", r-drop " + rdrop);
        assertTrue(plain > 0.3f && rdrop > 0.3f, plain + " / " + rdrop);
        float noiseless = TrainingTricksExperiment.gradientNoise(0f, 4);
        float noisy = TrainingTricksExperiment.gradientNoise(0.01f, 4);
        System.out.println("gradient noise: without " + noiseless + ", with " + noisy);
        assertTrue(Float.isFinite(noisy) && noisy < 0.9f, "the deep net trains with noise: " + noisy);
    }

    @Test
    void lossScalingRescuesUnderflowingFloat16Gradients() {
        float[] plain = TrainingTricksExperiment.lossScaling(false, 5);
        float[] scaled = TrainingTricksExperiment.lossScaling(true, 5);
        System.out.println("fp16: unscaled " + plain[0] + " -> " + plain[1] + ", scaled " + scaled[0] + " -> "
                + scaled[1]);
        assertTrue(plain[1] > 0.9f * plain[0], "without scaling the gradients underflow");
        assertTrue(scaled[1] < 0.05f * scaled[0], "with scaling it trains");
    }
}
