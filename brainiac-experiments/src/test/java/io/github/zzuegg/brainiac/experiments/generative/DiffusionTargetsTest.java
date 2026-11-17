package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.diffusion.DiffusionLossWeighting;
import io.github.zzuegg.brainiac.nn.diffusion.Diffusion;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiffusionTargetsTest {

    @Test
    void minSnrWeightsMatchThePaper() {
        DiffusionLossWeighting w = DiffusionLossWeighting.minSnr(5);
        double ab = 0.9;   // SNR 9, clamped to 5
        assertEquals(5.0 / 9, w.weight(Diffusion.Prediction.NOISE, ab), 1e-9);
        assertEquals(5.0, w.weight(Diffusion.Prediction.SAMPLE, ab), 1e-9);
        assertEquals(5.0 / 10, w.weight(Diffusion.Prediction.VELOCITY, ab), 1e-9);
        assertEquals(1.0, w.weight(Diffusion.Prediction.NOISE, 0.5), 1e-9);   // SNR 1 below the clamp
        assertEquals(1.0, DiffusionLossWeighting.uniform().weight(Diffusion.Prediction.VELOCITY, 0.99), 0);
    }

    @Test
    void targetsAndWeightingsAtAFixedBudget() {
        int it = 500;
        int steps = 10;
        double eps = DiffusionTargets.run(Diffusion.Prediction.NOISE, DiffusionLossWeighting.uniform(), it, steps, 1);
        double epsSnr = DiffusionTargets.run(Diffusion.Prediction.NOISE, DiffusionLossWeighting.minSnr(5), it, steps, 1);
        double v = DiffusionTargets.run(Diffusion.Prediction.VELOCITY, DiffusionLossWeighting.uniform(), it, steps, 1);
        double vSnr = DiffusionTargets.run(Diffusion.Prediction.VELOCITY, DiffusionLossWeighting.minSnr(5), it, steps, 1);
        System.out.printf("diffusion targets, 10 DDIM steps, distance to nearest real image per pixel: "
                + "eps %.4f, eps+min-SNR %.4f, v %.4f, v+min-SNR %.4f%n", eps, epsSnr, v, vSnr);
        assertTrue(epsSnr < eps, "min-SNR helps epsilon prediction: " + epsSnr + " vs " + eps);
        assertTrue(v < 0.8 * eps && vSnr < 0.8 * eps, "velocity prediction beats epsilon prediction");
    }
}
