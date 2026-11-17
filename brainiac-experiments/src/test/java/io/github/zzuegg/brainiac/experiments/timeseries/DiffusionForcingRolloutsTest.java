package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class DiffusionForcingRolloutsTest {

    @Test
    void diffusionForcingStaysOnTheAttractor() {
        Map<String, DiffusionForcingRollouts.Stats> r = DiffusionForcingRollouts.run(3000, 200, 16, 1);
        StringBuilder line = new StringBuilder("Lorenz 200-step rollouts, 10-step RMSE/attractor distance/diverged:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.2f;", k, v.shortError(), v.attractorDistance(),
                v.diverged())));
        System.out.println(line);
        DiffusionForcingRollouts.Stats noisy = r.get("diffusion forcing, noisy past");
        DiffusionForcingRollouts.Stats full = r.get("full-sequence diffusion");
        assertTrue(noisy.diverged() == 0, line.toString());
        assertTrue(noisy.attractorDistance() < 0.2, line.toString());
        assertTrue(noisy.shortError() < full.shortError(), "conditioning beats inpainting: " + line);
    }
}
