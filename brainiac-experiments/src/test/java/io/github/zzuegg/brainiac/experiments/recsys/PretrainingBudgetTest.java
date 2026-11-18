package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PretrainingBudgetTest {

    @Test
    void columnRowIclLearnsInContext() {
        Map<String, double[]> r = TabularShootout.pretrainingBudget(20, new int[] {1, 4}, 4, 1);
        StringBuilder b = new StringBuilder("ICL prior-acc/trees-acc/pretrain-ms:");
        r.forEach((k, v) -> b.append(String.format(" %s %.3f/%.3f/%.0f |", k, v[0], v[1], v[2])));
        String line = b.toString();
        System.out.printf("%s%n", line);
        assertTrue(r.get("TabICL-lite x4")[0] > 0.6, line);
        assertTrue(r.get("PFN x4")[0] > 0.6, line);
    }
}
