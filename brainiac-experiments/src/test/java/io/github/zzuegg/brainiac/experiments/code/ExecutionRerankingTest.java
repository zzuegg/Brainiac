package io.github.zzuegg.brainiac.experiments.code;

import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.16: which sampled program to submit: likelihood, filtering, AlphaCode, CodeT, LEVER. */
class ExecutionRerankingTest {

    @Test
    void executionReranking() {
        Map<String, Double> r = CodeExecution.rerank(2, new int[] {8, 32}, 200, 80, 3);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format(" %s %.3f;", k, v)));
        System.out.printf("9.2.16 hidden-test top-1 accuracy from 2 given examples (likelihood is the baseline):%s%n",
                sb);
        assertFalse(r.isEmpty());
    }
}
