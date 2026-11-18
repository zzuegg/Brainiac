package io.github.zzuegg.brainiac.experiments.code;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.14 to 9.2.16: execution traces, self-repair and execution-based reranking. */
class CodeExecutionTest {

    @Test
    void executionPrediction() {
        Map<String, CodeExecution.PredictionRow> r = CodeExecution.executionPrediction(300, 20, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s in %.3f longer %.3f (%.0f tok);", k, row.inDistribution(),
                row.longer(), row.tokensPerProgram())));
        System.out.printf("9.2.14 output accuracy, lengths 2-4 vs 5-8 (direct prediction is the baseline):%s%n", sb);
        assertTrue(r.size() >= 3);
    }

    @Test
    void selfRepairAtEqualBudget() {
        CodeExecution.RepairResult r = CodeExecution.repair(200, 200, 40, 2);
        StringBuilder sb = new StringBuilder();
        r.passAt().forEach((k, v) -> sb.append(' ').append(k).append(' ').append(Arrays.toString(v)).append(';'));
        System.out.printf("9.2.15 solve rate at budgets 1/2/4/8/16 (resampling is the baseline, first sample %.3f):%s%n",
                r.firstSample(), sb);
        assertFalse(r.passAt().isEmpty());
    }

    @Test
    void executionReranking() {
        Map<String, Double> r = CodeExecution.rerank(2, new int[] {8, 32}, 200, 40, 3);
        System.out.printf("9.2.16 hidden-test top-1 accuracy (likelihood is the baseline): %s%n", r);
        assertFalse(r.isEmpty());
    }
}
