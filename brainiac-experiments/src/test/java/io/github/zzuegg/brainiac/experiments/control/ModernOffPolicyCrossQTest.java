package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ModernOffPolicyCrossQTest {

    @Test
    void crossQRunsAtOneUpdatePerStepWithoutTargetNetworks() {
        LearningCurveHarness.Report r = ModernOffPolicy.crossQ(ModernOffPolicy.Task.PENDULUM, 600, 1, 16, 4);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.0f (%.1f upd/step, %.1fs), ", k, c.finalIqm(),
                c.updatesPerStep(), c.meanSeconds())));
        System.out.printf("W10 4.2 pendulum 600 steps return: %s%n", b);
        for (LearningCurveHarness.Curve c : r.curves().values()) {
            assertTrue(Double.isFinite(c.finalIqm()) && c.finalIqm() > -2_000, c.name() + " " + c.finalIqm());
        }
        double crossQ = r.curve("CrossQ (UTD 1)").updatesPerStep();
        double redq = r.curve("REDQ (UTD 4)").updatesPerStep();
        assertEquals(4, redq / crossQ, 0.2);
    }
}
