package io.github.zzuegg.brainiac.experiments.selfsupervised;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.CurvePoint;
import io.github.zzuegg.brainiac.experiments.selfsupervised.SelfSupervisedBenchmark.Method;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 7.6.18: I-JEPA-lite against the full recipe, as learning curves at matched budgets. */
class JepaFollowUpTest {

    @Test
    void fullRecipeAgainstLiteAtMatchedBudgets() {
        int[] checkpoints = {100, 200};
        List<CurvePoint> lite = SelfSupervisedBenchmark.learningCurve(Method.IJEPA, checkpoints, 1);
        List<CurvePoint> full = SelfSupervisedBenchmark.learningCurve(Method.IJEPA_FULL, checkpoints, 1);
        StringBuilder line = new StringBuilder("I-JEPA probe acc (effective rank) at steps");
        for (int i = 0; i < lite.size(); i++) {
            line.append(String.format(" | %d: lite %.3f (%.1f) full %.3f (%.1f)", lite.get(i).steps(),
                    lite.get(i).probeAccuracy(), lite.get(i).effectiveRank(), full.get(i).probeAccuracy(),
                    full.get(i).effectiveRank()));
        }
        System.out.println(line + " (random encoder 0.341, MAE 0.59 and SimCLR 0.56 at 100-150 steps)");
        CurvePoint last = full.get(full.size() - 1);
        assertTrue(last.effectiveRank() > 2, "the full recipe does not collapse: " + last);
        assertTrue(last.probeAccuracy() > 1.0f / 6 + 0.1f, "the full recipe's probe beats chance: " + last);
    }
}
