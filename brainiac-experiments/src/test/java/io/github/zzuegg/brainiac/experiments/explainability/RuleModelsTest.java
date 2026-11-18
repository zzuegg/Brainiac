package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableModels.RuleRow;

import java.util.List;

import org.junit.jupiter.api.Test;

class RuleModelsTest {

    @Test
    void corelsRecoversThePlantedRuleList() {
        for (double noise : new double[] {0, 0.05, 0.10}) {
            List<RuleRow> rows = InterpretableModels.rules(noise, 1000, 1);
            for (RuleRow r : rows) {
                System.out.printf("Rules noise %.2f %-12s accuracy %.3f truth agreement %.3f size %3d time %7.1f ms%s%n",
                        r.noise(), r.method(), r.accuracy(), r.truth(), r.size(), r.millis(),
                        r.certified() ? " certified" : "");
            }
            RuleRow corels = rows.get(0);
            RuleRow l1 = rows.get(rows.size() - 1);
            assertTrue(corels.truth() >= 0.97, "CORELS recovers the planted list");
            assertTrue(corels.size() <= 4, "the list stays short");
            assertTrue(noise > 0.05 || corels.certified(), "CORELS certifies optimality at low noise");
            assertTrue(corels.truth() > l1.truth(), "the rule list beats the linear baseline");
        }
    }
}
