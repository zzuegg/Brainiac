package io.github.zzuegg.brainiac.experiments.code;

import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.15: self-repair with execution feedback against resampling at equal budget. */
class SelfRepairTest {

    @Test
    void selfRepairAtEqualBudget() {
        CodeExecution.RepairResult r = CodeExecution.repair(200, 150, 50, 2);
        StringBuilder sb = new StringBuilder();
        r.passAt().forEach((k, v) -> {
            sb.append(' ').append(k);
            Arrays.stream(v).forEach(x -> sb.append(String.format(" %.3f", x)));
            sb.append(';');
        });
        System.out.printf("9.2.15 solve rate at 1/2/4/8/16 model calls (resampling is the baseline):%s%n", sb);
        assertFalse(r.passAt().isEmpty());
    }
}
