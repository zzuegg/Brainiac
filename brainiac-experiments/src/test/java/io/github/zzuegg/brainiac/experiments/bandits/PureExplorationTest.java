package io.github.zzuegg.brainiac.experiments.bandits;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.bandit.BestArmIdentification;
import io.github.zzuegg.brainiac.experimental.data.BanditSuites;

import java.util.Map;

import org.junit.jupiter.api.Test;

class PureExplorationTest {

    private static String fmt(Map<String, double[]> t) {
        StringBuilder b = new StringBuilder();
        t.forEach((k, v) -> b.append(k).append(' ').append(java.util.Arrays.toString(
                java.util.Arrays.stream(v).map(x -> Math.round(x * 1000) / 1000.0).toArray())).append("; "));
        return b.toString();
    }

    @Test
    void trackingRulesNeedFewerSamplesThanEliminationAndUniform() {
        BanditSuites.Instance g = BanditSuites.gaussian(1, 1, 0.6, 0.5, 0.4, 0.3);
        Map<String, double[]> fc = PureExploration.fixedConfidence(g, BestArmIdentification.Model.gaussian(1), 0.05,
                40_000, 12);
        System.out.printf("11.6.3 fixed confidence delta=0.05 {samples, error, stopped}: %s%n", fmt(fc));
        double se = fc.get("SUCCESSIVE_ELIMINATION")[0];
        assertTrue(fc.get("TRACK_AND_STOP")[0] < se && fc.get("TOP_TWO_EB")[0] < se, fmt(fc));
        assertTrue(fc.get("TRACK_AND_STOP")[1] <= 0.1, fmt(fc));
    }

    @Test
    void anytimeAlmostTrackingAgainstBudgetedRules() {
        BanditSuites.Instance g = BanditSuites.gaussian(1, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3);
        Map<String, double[]> fb = PureExploration.fixedBudget(g, BestArmIdentification.Model.gaussian(1),
                new long[] {200, 800}, 60);
        System.out.printf("11.6.3 fixed budget error at {200, 800}: %s%n", fmt(fb));
        assertTrue(fb.get("ALMOST_TRACKING")[1] <= fb.get("UNIFORM")[1] + 0.05, fmt(fb));
        PureExploration.HyperRow h = PureExploration.hyperparameters(9, 6);
        System.out.printf("11.6.3 lr selection at %d runs: Almost Tracking regret %.4f, Hyperband %.4f (best acc %.3f)%n",
                h.pulls(), h.almostTrackingRegret(), h.hyperbandRegret(), h.bestAccuracy());
        assertTrue(h.almostTrackingRegret() < 0.05, String.valueOf(h));
    }

    @Test
    void doubleThompsonCopesWithCyclicPreferences() {
        Map<String, double[]> bt = PureExploration.dueling(BanditSuites.bradleyTerry(1.0, 0.5, 0.2, 0, -0.3), 3_000, 6);
        Map<String, double[]> cy = PureExploration.dueling(BanditSuites.cyclic(5, 0.3, 1), 3_000, 6);
        System.out.printf("11.6.4 dueling {regret, winner share}: BT: %s| cyclic: %s%n", fmt(bt), fmt(cy));
        assertTrue(bt.get("DOUBLE_TS")[0] < bt.get("UNIFORM")[0] && bt.get("RUCB")[0] < bt.get("UNIFORM")[0], fmt(bt));
        assertTrue(cy.get("DOUBLE_TS")[0] < cy.get("UNIFORM")[0], fmt(cy));
    }
}
