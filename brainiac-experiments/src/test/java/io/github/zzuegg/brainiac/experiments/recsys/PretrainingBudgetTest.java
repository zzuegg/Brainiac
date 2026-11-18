package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PretrainingBudgetTest {

    @Test
    void pretrainingBudgetSweepRuns() {
        Map<String, double[]> r = TabularShootout.pretrainingBudget(12, new int[] {1, 4}, 4, 1);
        StringBuilder b = new StringBuilder("ICL prior-acc/trees-acc/pretrain-ms:");
        r.forEach((k, v) -> b.append(String.format(" %s %.3f/%.3f/%.0f |", k, v[0], v[1], v[2])));
        String line = b.toString();
        System.out.printf("%s%n", line);
        // Genuine negative at test-sized budgets (at most 48 pre-training steps): both in-context
        // learners stay far below per-task GBT and random forests, and TabICL-lite is near chance.
        // Sanity checks only; the budget claim needs the 16x and 64x rows outside the test.
        assertTrue(r.get("GBT")[0] > 0.7, line);
        assertTrue(r.get("PFN x4")[0] > r.get("PFN x1")[0] - 0.05, line);
        for (double[] v : r.values()) {
            assertTrue(v[0] > 0.35 && v[0] <= 1 && v[1] > 0.35 && v[1] <= 1, line);
        }
    }
}
