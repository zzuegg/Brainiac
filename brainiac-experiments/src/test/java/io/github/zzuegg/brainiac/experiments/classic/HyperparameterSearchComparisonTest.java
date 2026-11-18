package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class HyperparameterSearchComparisonTest {

    private static String line(Map<String, Double> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %.4f, ", k, v)));
        return sb.toString();
    }

    @Test
    void gpBayesOptBeatsRandomSearchAndTpeAtEqualTrials() {
        Map<String, Double> r = HyperparameterSearchComparison.fixedBudget(30, 4);
        System.out.printf("best score after 30 trials (floor 0.01): %s%n", line(r));
        assertTrue(r.get("GP-BO (EI)") < r.get("random"), r.toString());
        assertTrue(r.get("GP-BO (EI)") < r.get("tpe") + 0.02, r.toString());
        assertTrue(r.get("GP-BO (EI)") < 0.1, r.toString());
    }

    @Test
    void bohbBeatsHyperbandAtEqualTotalBudget() {
        Map<String, Double> r = HyperparameterSearchComparison.multiFidelity(27, 3, 3, 4);
        System.out.printf("best full-budget score at equal total budget: %s%n", line(r));
        assertTrue(r.get("BOHB") < r.get("hyperband"), r.toString());
        assertTrue(r.get("BOHB") < r.get("random (same budget)"), r.toString());
    }
}
