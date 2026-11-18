package io.github.zzuegg.brainiac.experiments.classic;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MonotonicNetworksTest {

    @Test
    void constrainedModelsHaveNoViolations() {
        for (int[] setting : new int[][] {{100, 3}, {500, 3}, {100, 1}}) {
            Map<String, ClassicMlShootout.MonotoneRow> rows = ClassicMlShootout.monotonic(setting[0], setting[1], 300,
                    1);
            StringBuilder b = new StringBuilder("11.4.17 monotone n=" + setting[0] + " k=" + setting[1]
                    + " RMSE/extrapolation RMSE/violations:");
            rows.forEach((m, r) -> b.append(String.format(" %s %.3f/%.3f/%.3f;", m, r.rmse(), r.extrapolationRmse(),
                    r.violations())));
            System.out.println(b);
            for (String m : new String[] {"MIN_MAX", "LATTICE", "CONSTRAINED_MLP"}) {
                assertEquals(0, rows.get(m).violations(), m + " is monotone by construction");
                assertTrue(rows.get(m).rmse() < 1.0, m + " fits the target");
            }
        }
    }
}
