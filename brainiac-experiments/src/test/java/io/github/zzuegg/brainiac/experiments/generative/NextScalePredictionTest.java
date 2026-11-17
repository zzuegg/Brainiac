package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.diffusion.DiffusionLossWeighting;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NextScalePredictionTest {

    @Test
    void coarseToFineInThreeCalls() {
        NextScalePrediction var = NextScalePrediction.of(256, 1);
        var.train(700, 64);
        double[] v = NextScalePrediction.score(var.sample(128, 0.6f, 5));
        MaskGit mg = MaskGit.of(256, 1);
        mg.trainMaskGit(700, 64);
        double[] mg3 = MaskGit.score(mg.sampleMaskGit(128, 3, 4.5f, 5));
        double ddpm3 = DiffusionTargets.run(Diffusion.Prediction.VELOCITY, DiffusionLossWeighting.uniform(), 700, 3, 1);
        System.out.printf("next-scale prediction (VAR), 3 calls: distance %.3f, real share %.2f, distinct %.0f, classes "
                + "%.0f (tokenizer error %.4f); MaskGIT 3 steps: %.3f/%.2f/%.0f; v-prediction DDPM 3 DDIM steps: "
                + "distance %.3f%n", v[0], v[1], v[2], v[3], var.reconstructionError(), mg3[0], mg3[1], mg3[2], ddpm3);
        assertTrue(var.reconstructionError() < 0.04, "the residual tokenizer reconstructs the images");
        assertTrue(v[1] > 0.05 && v[3] >= 3, "VAR makes real shapes: " + v[1]);
        assertTrue(v[0] < 1.15 * Math.min(mg3[0], ddpm3), "at three calls VAR is on par with MaskGIT and DDPM");
    }
}
