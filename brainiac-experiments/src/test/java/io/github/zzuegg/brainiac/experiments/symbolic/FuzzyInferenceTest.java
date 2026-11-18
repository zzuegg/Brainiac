package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

class FuzzyInferenceTest {

    private static FuzzyInference.Row find(List<FuzzyInference.Row> rows, String task, String method) {
        return rows.stream().filter(r -> r.task().startsWith(task) && r.method().startsWith(method)).findFirst()
            .orElseThrow();
    }

    @Test
    void neuroFuzzy() {
        List<FuzzyInference.Row> rows = new ArrayList<>(FuzzyInference.mackeyGlass(10, 1));
        rows.addAll(FuzzyInference.sinc(10, 1));
        rows.addAll(FuzzyInference.cartPole(1000, 10, 1));
        rows.addAll(FuzzyInference.ruleGrowth(1));
        System.out.printf("13.2.15 neuro-fuzzy (test RMSE or steps): %s%n",
            rows.stream().map(FuzzyInference.Row::toString).collect(Collectors.joining("; ")));
        double anfis = find(rows, "mackey-glass", "ANFIS hybrid").rmse();
        double mlpShort = find(rows, "mackey-glass", "MLP").rmse();
        assertTrue(anfis < mlpShort, "ANFIS after 10 hybrid epochs beats the MLP after 100 epochs");
        assertTrue(anfis < 0.05, "ANFIS predicts Mackey-Glass");
        assertTrue(find(rows, "mackey-glass d=8", "subtractive").rules() < find(rows, "mackey-glass d=8", "grid 2")
            .rules(), "clustering keeps the rule base small");
    }
}
