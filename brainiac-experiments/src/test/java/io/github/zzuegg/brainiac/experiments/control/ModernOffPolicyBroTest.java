package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ModernOffPolicyBroTest {

    @Test
    void broTrainsAtTwoCriticWidthsWithItsAblations() {
        LearningCurveHarness.Report r = ModernOffPolicy.broScaling(ModernOffPolicy.Task.PENDULUM, 200, 1,
                new int[] {16, 32}, 1);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.0f, ", k, c.finalIqm())));
        System.out.printf("W10 4.3 pendulum 200 steps return by critic width: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()), c.name());
        }
    }
}
