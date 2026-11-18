package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.13: RLVR on one well-chosen example against the full training set. */
class OneShotRlvrTest {

    @Test
    void oneExampleAgainstFullSet() {
        Map<String, ArithmeticRl.OneShotRow> r = ArithmeticRl.oneShot(ArithmeticTask.addition(9), 30, 2, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, row) -> {
            sb.append(' ').append(k).append(" (").append(row.trainProblems()).append(") [");
            for (double a : row.testAccuracy()) {
                sb.append(String.format(" %.3f", a));
            }
            sb.append(String.format(" ] train solved@%d;", row.solvedAt()));
        });
        System.out.printf("9.2.13 test accuracy by checkpoint (full train set is the baseline):%s%n", sb);
        assertTrue(r.size() >= 6);
    }
}
