package io.github.zzuegg.brainiac.experiments.parallel;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.parallel.ParallelCostModel;
import io.github.zzuegg.brainiac.experiments.parallel.ParallelismShootout.PipelineRow;

import java.util.List;

import org.junit.jupiter.api.Test;

class PipelineMemoryFollowUpTest {

    private static PipelineRow find(List<PipelineRow> rows, String prefix) {
        return rows.stream().filter(r -> r.schedule().startsWith(prefix)).findFirst().orElseThrow();
    }

    @Test
    void memoryAwareSchedulesTradeBubbleForMemory() {
        ParallelCostModel cost = new ParallelCostModel(1.0, 1.0, 1.0, 0.02, 10e9, 1024, true);
        List<PipelineRow> rows = ParallelismShootout.pipelineMemoryFollowUp(cost, new int[] {4}, new int[] {8},
                new double[] {0.5, 1.0}, 20_000, true, 1);
        for (PipelineRow r : rows) {
            System.out.printf("%-34s p=%d m=%d bubble %.3f measured %.3f peak %.2f err %.2e%n", r.schedule(),
                    r.devices(), r.microBatches(), r.simulatedBubble(), r.measuredBubble(), r.peakActivations(),
                    r.gradientError());
            if (!Double.isNaN(r.gradientError())) {
                assertTrue(r.gradientError() < 1e-4, r.toString());
            }
        }
        PipelineRow f1b1 = find(rows, "1F1B");
        PipelineRow h2 = find(rows, "ZB-H2");
        PipelineRow greedyH2 = find(rows, "memory-aware greedy @ZB-H2");
        PipelineRow vZb = find(rows, "V-ZB");
        PipelineRow vMin = find(rows, "V-Min");
        assertTrue(greedyH2.peakActivations() <= 2 * 4 - 1 + 1e-9, greedyH2.toString());
        assertTrue(greedyH2.simulatedBubble() <= h2.simulatedBubble() + 0.05, greedyH2 + " vs " + h2);
        assertTrue(vZb.simulatedBubble() < f1b1.simulatedBubble(), vZb + " vs " + f1b1);
        assertTrue(vMin.peakActivations() < f1b1.peakActivations(), vMin + " vs " + f1b1);
    }
}
