package io.github.zzuegg.brainiac.experiments.exotic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class WorkspaceBroadcastTest {
    @Test
    void flaggedSpecialistWinsTheWorkspaceAndIsBroadcast() {
        WorkspaceBroadcast w = WorkspaceBroadcast.of(16, 1e-2f, 1);
        double[] before = w.evaluate(200, 9);
        w.train(400);
        double[] after = w.evaluate(200, 9);
        System.out.printf("workspace: mse %.3f -> %.3f, flagged writer %.2f -> %.2f%n", before[0], after[0], before[1],
                after[1]);
        assertTrue(after[0] < 0.15, "mse " + after[0]);   // predicting zero would give about 1
        assertTrue(after[1] > 0.95, "flagged writer " + after[1]);
    }
}
