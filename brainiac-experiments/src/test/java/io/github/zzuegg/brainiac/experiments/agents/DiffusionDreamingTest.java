package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DiffusionDreamingTest {

    @Test
    void aDiffusionWorldModelRollsOutAndTrainsADreamer() {
        DiffusionDreaming.Fidelity[] f = DiffusionDreaming.fidelity(3000, 800, 1);
        double[] returns = DiffusionDreaming.dreamer(3000, 1);
        String line = String.format("Cart-pole world models, 1-step/10-step RMSE: MLP %.4f/%.4f, diffusion %.4f/%.4f;"
                + " Dreamer return: MLP %.1f, diffusion %.1f", f[0].oneStep(), f[0].tenStep(), f[1].oneStep(),
                f[1].tenStep(), returns[0], returns[1]);
        System.out.println(line);
        assertTrue(f[1].tenStep() < 0.5, line);
        assertTrue(returns[1] > 40, "a random policy gets about 22: " + line);
    }
}
