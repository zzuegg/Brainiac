package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ValueBasedRecipesPixelTest {

    @Test
    void sampleEfficiencyRecipesTrainFromPixels() {
        LearningCurveHarness.Report r = ValueBasedRecipes.sampleEfficient(ValueBasedRecipes.Task.PIXEL_GRID, 400, 1, 2);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.2f, ", k, c.finalIqm())));
        System.out.printf("W10 4.8/4.9 pixel grid 400 steps return: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()), c.name());
        }
    }
}
