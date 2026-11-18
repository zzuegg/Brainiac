package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.LocalRules.RecourseMethod;
import io.github.zzuegg.brainiac.experiments.explainability.LocalRules.RecourseScores;
import io.github.zzuegg.brainiac.experiments.explainability.LocalRules.RuleMethod;
import io.github.zzuegg.brainiac.experiments.explainability.LocalRules.RuleScores;

import java.util.Map;

import org.junit.jupiter.api.Test;

class LocalRulesTest {

    @Test
    void anchorsAreMorePreciseThanFeatureRankingRules() {
        for (boolean trees : new boolean[] {true, false}) {
            Map<RuleMethod, RuleScores> m = LocalRules.anchors(trees, 12, 1);
            m.forEach((k, s) -> System.out.printf("Anchors %s %-17s held-out precision %.3f coverage %.3f planted-only %.2f"
                    + " calls %.0f%n", trees ? "GBT" : "MLP", k, s.precision(), s.coverage(), s.planted(), s.calls()));
            assertTrue(m.get(RuleMethod.ANCHOR_KL_LUCB).precision() > 0.85, "anchors are precise");
        }
    }

    @Test
    void integerProgramRecourseIsActionable() {
        for (boolean linear : new boolean[] {true, false}) {
            Map<RecourseMethod, RecourseScores> m = LocalRules.recourse(linear, 15, 1);
            m.forEach((k, s) -> System.out.printf("Recourse %s %-15s valid %.2f actionable %.2f cost %.2f density %.2f"
                    + " diversity %.2f features %.2f valid-after-update %.2f%n", linear ? "LR " : "MLP", k, s.validity(),
                    s.actionable(), s.cost(), s.density(), s.diversity(), s.featuresCovered(), s.validityAfterUpdate()));
            if (linear) {
                assertEquals(1.0, m.get(RecourseMethod.LINEAR_IP).actionable(), 1e-9);
                assertTrue(m.get(RecourseMethod.LINEAR_IP).validity() > 0.9);
            }
        }
    }
}
