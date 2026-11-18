package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ValueBasedRecipesPixelPendulumTest {

    @Test
    void drqOnThePixelPendulumRuns() {
        LearningCurveHarness.Report r = ValueBasedRecipes.pixelPendulum(200, 1);
        System.out.printf("W10 4.9 pixel pendulum 200 steps: SAC %.0f, DrQ-v2-style SAC %.0f%n",
                r.curve("SAC (pixels)").finalIqm(), r.curve("DrQ-v2-style SAC (shifts, 3-step)").finalIqm());
        assertTrue(Double.isFinite(r.curve("DrQ-v2-style SAC (shifts, 3-step)").finalIqm()));
    }
}
