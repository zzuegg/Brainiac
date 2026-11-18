package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BudgetPacingTest {

    @Test
    void pacingBeatsEarlyExhaustion() {
        Map<String, double[]> r = BudgetPacing.run(20_000, 0.2, 1);
        StringBuilder b = new StringBuilder("pacing value-vs-hindsight/spend/depletion (budget 20%):");
        r.forEach((k, v) -> b.append(String.format(" %s %.3f/%.3f/%.3f |", k, v[0], v[1], v[2])));
        String line = b.toString();
        System.out.printf("%s%n", line);
        double truthful = r.get("truthful until exhausted")[0];
        assertTrue(r.get("dual bid shading")[0] > truthful, line);
        assertTrue(r.get("PID throttle")[0] > r.get("fixed throttle")[0] * 0.9, line);
        assertTrue(r.get("truthful until exhausted")[2] < 0.9, line);
        for (double[] v : r.values()) {
            assertTrue(v[0] <= 1 + 1e-9, line);
        }
    }

    @Test
    void greedyHindsightMatchesSimplex() {
        double[] c = BudgetPacing.hindsightCheck(150, 0.3, 2);
        System.out.printf("hindsight greedy %.5f simplex %.5f%n", c[0], c[1]);
        assertEquals(c[0], c[1], 1e-6 * Math.max(1, c[0]));
    }
}
