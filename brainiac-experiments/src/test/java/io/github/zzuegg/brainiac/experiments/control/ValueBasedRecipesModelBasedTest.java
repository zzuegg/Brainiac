package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ValueBasedRecipesModelBasedTest {

    @Test
    void efficientZeroAndItsAblationsTrain() {
        LearningCurveHarness.Report r = ValueBasedRecipes.modelBased(ValueBasedRecipes.Task.GRID, 300, 1, 3);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.2f, ", k, c.finalIqm())));
        System.out.printf("W10 4.12 grid 300 steps return: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()), c.name());
        }
    }
}
