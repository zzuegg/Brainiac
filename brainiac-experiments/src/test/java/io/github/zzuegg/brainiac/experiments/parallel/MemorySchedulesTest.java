package io.github.zzuegg.brainiac.experiments.parallel;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.parallel.CheckpointPlanner;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class MemorySchedulesTest {

    @Test
    void selectiveRecomputationAndTheOptimumBeatWholeBlocks() {
        Map<String, MemorySchedules.Point> r = MemorySchedules.pareto(8, 64, 1);
        r.forEach((k, v) -> System.out.printf("9.6.5 depth 8 seq 64 %-52s peak %8d B (saves %.2f), extra compute %.3f%n",
                k, v.peakBytes(), v.memorySaving(), v.extraCompute()));
        MemorySchedules.Point block = r.get("Recompute every block");
        MemorySchedules.Point selective = r.get("selective [attention]");
        MemorySchedules.Point optSqrt = r.get("optimal at the peak of sqrt(n) layer segments");
        MemorySchedules.Point sqrt = r.get("sqrt(n) layer segments");
        assertTrue(selective.memorySaving() > 0.3, selective.toString());
        assertTrue(selective.extraCompute() < block.extraCompute(), selective + " vs " + block);
        assertTrue(optSqrt.extraCompute() <= sqrt.extraCompute() + 1e-12);
        for (double[] row : MemorySchedules.dominance(8, 64, 1)) {
            assertTrue(row[3] <= row[2] + 1e-12, "optimal must not lose to every-" + row[0]);
        }
        System.out.printf("9.6.5 Revolve: 32 steps with 4 slots need %d forward steps (store-all: 31)%n",
                CheckpointPlanner.revolveForwards(32, 4));
        assertEquals(31, CheckpointPlanner.revolveForwards(32, 32));
    }

    @Test
    void executedPlansMatchGradientsThroughCoreRecompute() {
        List<MemorySchedules.Executed> rows = MemorySchedules.execute(8, 64, 2);
        for (MemorySchedules.Executed e : rows) {
            System.out.printf("9.6.5 executed %-24s %.2f ms/step, overhead %+.2f, grad err %.1e%n", e.method(),
                    e.millis(), e.overhead(), e.gradientError());
            assertTrue(e.gradientError() < 1e-4, e.toString());
        }
    }
}
