package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ValueBasedRecipesMrQTest {

    @Test
    void mrqUsesItsContinuousConfigurationOnDiscreteActions() {
        LearningCurveHarness.Report r = ValueBasedRecipes.mrqDiscrete(ValueBasedRecipes.Task.CARTPOLE, 600, 1);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.0f, ", k, c.finalIqm())));
        System.out.printf("W10 4.6 cart-pole 600 steps return: %s%n", b);
        assertTrue(r.curve("MR.Q").finalIqm() > 0, b.toString());
    }
}
