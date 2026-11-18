package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ModernOffPolicyFujimotoTest {

    @Test
    void td7AndItsAblationsTrain() {
        LearningCurveHarness.Report r = ModernOffPolicy.td7Ablation(ModernOffPolicy.Task.PENDULUM,
                new long[] {400, 600}, 1, 16);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.0f, ", k, c.finalIqm())));
        System.out.printf("W10 4.7 pendulum 600 steps return: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()) && c.updatesPerStep() > 0, c.name());
        }
    }

    @Test
    void mrqTrainsWithAndWithoutItsModelLosses() {
        LearningCurveHarness.Report r = ModernOffPolicy.mrq(ModernOffPolicy.Task.PENDULUM, 500, 1, 16);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.0f, ", k, c.finalIqm())));
        System.out.printf("W10 4.6 pendulum 500 steps return: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()), c.name());
        }
    }
}
