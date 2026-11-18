package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ValueBasedRecipesPixelTest {

    @Test
    void sampleEfficiencyRecipesTrainFromPixels() {
        LearningCurveHarness.Report r = ValueBasedRecipes.sampleEfficient(ValueBasedRecipes.Task.PIXEL_GRID, 800, 1, 2);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.2f, ", k, c.finalIqm())));
        System.out.printf("W10 4.8/4.9 pixel grid 800 steps return: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()), c.name());
        }
    }

    @Test
    void drqOnThePixelPendulumRuns() {
        LearningCurveHarness.Report r = ValueBasedRecipes.pixelPendulum(400, 1);
        System.out.printf("W10 4.9 pixel pendulum 400 steps: SAC %.0f, DrQ-v2-style SAC %.0f%n",
                r.curve("SAC (pixels)").finalIqm(), r.curve("DrQ-v2-style SAC (shifts, 3-step)").finalIqm());
        assertTrue(Double.isFinite(r.curve("DrQ-v2-style SAC (shifts, 3-step)").finalIqm()));
    }
}
