package io.github.zzuegg.brainiac.experiments.multimodal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.multimodal.FusionShootout.LatentScores;

import java.util.List;

import org.junit.jupiter.api.Test;

/**
 * Transfusion on a VAE latent with U-Net patch layers, at the multiply-adds of the wave 9
 * pixel model's 200 steps (wave 9: caption 0.133, generation 0.067 for the pixel model).
 */
class LatentTransfusionTest {

    @Test
    void latentUNetTransfusion() {
        // at 300 pixel-model steps (239 matched steps, 300 VAE steps): caption 0.333, generation 0.133
        List<LatentScores> rows = FusionShootout.transfusionFollowUp(200, 200, 5f, 3, false, 1);
        System.out.printf("latent Transfusion: %s%n", rows);
        LatentScores u = rows.get(0);
        assertTrue(u.captionAccuracy() >= 0 && u.generationAccuracy() >= 0 && u.trainingMacs() > 0, u.toString());
    }
}
