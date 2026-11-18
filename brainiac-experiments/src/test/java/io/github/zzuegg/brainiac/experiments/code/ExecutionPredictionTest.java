package io.github.zzuegg.brainiac.experiments.code;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.14: predicting program outputs directly, through states, or through CWM-style traces. */
class ExecutionPredictionTest {

    @Test
    void executionPrediction() {
        Map<String, CodeExecution.PredictionRow> r = CodeExecution.executionPrediction(300, 30, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> sb.append(String.format(" %s 1-3 ops %.3f, 4-6 ops %.3f (%.1f target tok);", k,
                row.inDistribution(), row.longer(), row.tokensPerProgram())));
        System.out.printf("9.2.14 exact-output accuracy (direct prediction is the baseline):%s%n", sb);
        assertEquals(3, r.size());
    }
}
