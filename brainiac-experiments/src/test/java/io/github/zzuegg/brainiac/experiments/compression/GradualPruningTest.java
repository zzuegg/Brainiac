package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.compress.GradualPruningSchedule;

import java.util.List;

import org.junit.jupiter.api.Test;

class GradualPruningTest {

    @Test
    void cubicScheduleRisesFastThenFlattens() {
        assertEquals(0, GradualPruningSchedule.cubic(0, 0.9, 10, 110, 5), 1e-12);
        assertEquals(0.9 * (1 - 0.125), GradualPruningSchedule.cubic(0, 0.9, 10, 110, 60), 1e-12);
        assertEquals(0.9, GradualPruningSchedule.cubic(0, 0.9, 10, 110, 500), 1e-12);
    }

    @Test
    void gradualMagnitudePruningAgainstOneShotLotteryAndRigl() {
        List<PruningShootout.Row> rows = PruningShootout.schedules(new double[] {0.9, 0.97}, 100, 1);
        StringBuilder line = new StringBuilder("8.5.1 GMP vs one-shot/LT/RigL, digits MLP acc:");
        for (PruningShootout.Row r : rows) {
            line.append(String.format(" %s@%.2f %.3f;", r.method(), r.sparsity(), r.accuracy()));
        }
        System.out.println(line);
        double dense = rows.get(0).accuracy();
        assertTrue(dense > 0.6, "dense reference trains: " + dense);
        for (int i = 1; i < rows.size(); i += 4) {
            PruningShootout.Row gmp = rows.get(i + 1);
            assertTrue(gmp.sparsity() > 0.85, "GMP reaches its sparsity: " + gmp);
            assertTrue(gmp.accuracy() > 0.15, "GMP network still classifies: " + gmp);
        }
    }
}
