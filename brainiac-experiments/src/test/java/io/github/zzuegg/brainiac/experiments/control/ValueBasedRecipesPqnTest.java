package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ValueBasedRecipesPqnTest {

    @Test
    void pqnLearnsCartPoleWithoutReplayOrTargetNetwork() {
        LearningCurveHarness.Report r = ValueBasedRecipes.pqn(ValueBasedRecipes.Task.CARTPOLE, 2_000, 1, 4);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.0f (%.2fs, %.2f upd/step), ", k, c.finalIqm(),
                c.meanSeconds(), c.updatesPerStep())));
        System.out.printf("W10 4.5 cart-pole 2000 steps return: %s%n", b);
        assertTrue(r.curve("PQN (LayerNorm, lambda 0.65)").finalIqm() > 20, b.toString());
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()), c.name());
        }
    }
}
