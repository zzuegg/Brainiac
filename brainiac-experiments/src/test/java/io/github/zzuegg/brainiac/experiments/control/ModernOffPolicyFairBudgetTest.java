package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import org.junit.jupiter.api.Test;

class ModernOffPolicyFairBudgetTest {

    @Test
    void everyRecipeRunsOnOneHarnessWithRankStatistics() {
        LearningCurveHarness.Report r = ModernOffPolicy.fairBudget(ModernOffPolicy.Task.MOUNTAIN_CAR,
                new long[] {100, 200}, 1, 16);
        double tau = r.rankStability(0, 1);
        double poi = ModernOffPolicy.probabilityOfImprovement(r, "SAC", "random", 1);
        StringBuilder b = new StringBuilder();
        r.curves().forEach((k, c) -> b.append(String.format("%s %.1f, ", k, c.finalIqm())));
        System.out.printf("W10 4.14 mountain car 200 steps (1 seed): %s Kendall tau 100 vs 200 %.2f, "
                + "P(SAC > random) %.2f%n", b, tau, poi);
        assertTrue(tau >= -1 && tau <= 1);
        assertTrue(poi >= 0 && poi <= 1);
        assertTrue(r.curves().size() == 10);
    }
}
