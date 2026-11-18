package io.github.zzuegg.brainiac.experiments.parallel;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.parallel.ParallelCostModel;

import java.util.List;
import org.junit.jupiter.api.Test;

class ParallelismShootoutTest {

    private static ParallelismShootout.PipelineRow find(List<ParallelismShootout.PipelineRow> rows, String name,
            int p, int m) {
        return rows.stream().filter(r -> r.schedule().startsWith(name) && r.devices() == p && r.microBatches() == m)
                .findFirst().orElseThrow();
    }

    @Test
    void pipelineSchedulesMatchOneDeviceAndZeroBubbleFillsTheBubble() {
        List<ParallelismShootout.PipelineRow> real = ParallelismShootout.pipelines(4, 8, 16, 32, 1);
        for (ParallelismShootout.PipelineRow r : real) {
            System.out.printf("9.6.1/9.6.2 p=%d m=%d %-22s sim bubble %.3f (theory %.3f), measured %.3f, peak act %.1f,"
                    + " msgs %d, grad err %.1e, step %.3f of 1 device%n", r.devices(), r.microBatches(), r.schedule(),
                    r.simulatedBubble(), r.theoryBubble(), r.measuredBubble(), r.peakActivations(), r.messages(),
                    r.gradientError(), r.stepTime());
            if (!Double.isNaN(r.gradientError())) {
                assertTrue(r.gradientError() < 1e-4, r.toString());
            }
        }
        ParallelCostModel cost = ParallelismShootout.calibrate(4, 16, 32, 1);
        List<ParallelismShootout.PipelineRow> grid = ParallelismShootout.simulatedGrid(cost, new int[] {4, 8},
                new int[] {8, 32});
        for (ParallelismShootout.PipelineRow r : grid) {
            System.out.printf("9.6.1/9.6.2 grid p=%d m=%2d %-22s bubble %.3f (theory %.3f) peak act %.1f msgs %d%n",
                    r.devices(), r.microBatches(), r.schedule(), r.simulatedBubble(), r.theoryBubble(),
                    r.peakActivations(), r.messages());
        }
        ParallelismShootout.PipelineRow gpipe = find(grid, "GPipe", 4, 8);
        ParallelismShootout.PipelineRow f1b1 = find(grid, "1F1B", 4, 8);
        ParallelismShootout.PipelineRow inter = find(grid, "interleaved", 4, 8);
        ParallelismShootout.PipelineRow h1 = find(grid, "ZB-H1", 4, 8);
        ParallelismShootout.PipelineRow h2 = find(grid, "ZB-H2", 4, 8);
        assertTrue(Math.abs(gpipe.simulatedBubble() - gpipe.theoryBubble()) < 0.08, gpipe.toString());
        assertTrue(f1b1.peakActivations() < gpipe.peakActivations(), "1F1B holds fewer activations");
        assertTrue(inter.simulatedBubble() < f1b1.simulatedBubble(), "interleaving shrinks the bubble");
        assertTrue(h1.simulatedBubble() < f1b1.simulatedBubble(), "ZB-H1 below 1F1B");
        assertTrue(h2.simulatedBubble() <= h1.simulatedBubble() + 0.02, "ZB-H2 not above ZB-H1");
    }

    @Test
    void tensorAndSequenceParallelismMatchOneDevice() {
        List<ParallelismShootout.TensorRow> rows = ParallelismShootout.tensorParallel(new int[] {1, 2, 4, 8}, 3);
        for (ParallelismShootout.TensorRow r : rows) {
            System.out.printf("9.6.3 %-16s t=%d %7d B/rank, %d msgs/rank, %6d params/rank, %6d act floats/rank,"
                    + " grad err %.1e%n", r.mode(), r.ranks(), r.bytesPerRank(), r.messages(),
                    r.parametersPerRank(), r.activationFloatsPerRank(), r.gradientError());
            assertTrue(r.gradientError() < 1e-4, r.toString());
        }
        ParallelismShootout.TensorRow tp4 = rows.stream().filter(r -> r.mode().equals("tensor") && r.ranks() == 4)
                .findFirst().orElseThrow();
        ParallelismShootout.TensorRow sp4 = rows.stream()
                .filter(r -> r.mode().equals("tensor+sequence") && r.ranks() == 4).findFirst().orElseThrow();
        assertTrue(sp4.activationFloatsPerRank() < tp4.activationFloatsPerRank());
        assertTrue(Math.abs(sp4.bytesPerRank() - tp4.bytesPerRank()) <= 0.3 * tp4.bytesPerRank(), "same volume");
    }

    @Test
    void zeroShardsMemoryAndMatchesDataParallel() {
        List<ParallelismShootout.ZeroRow> rows = ParallelismShootout.zero(4, 6, 5);
        for (ParallelismShootout.ZeroRow r : rows) {
            System.out.printf("9.6.4 N=%d %-34s params %7d B, grads %7d B, states %7d B (total %7d), comm %7d B/rank/step,"
                    + " loss diff vs DP %.1e, loss %.4f%n", r.ranks(), r.method(), r.parameterBytes(), r.gradientBytes(),
                    r.stateBytes(), r.total(), r.bytesPerRankPerStep(), r.maxLossDifference(), r.finalLoss());
        }
        ParallelismShootout.ZeroRow dp = rows.get(0);
        ParallelismShootout.ZeroRow z1 = rows.get(2);
        ParallelismShootout.ZeroRow z3 = rows.get(4);
        assertTrue(z1.maxLossDifference() < 1e-4 && z3.maxLossDifference() < 1e-4, rows.toString());
        assertTrue(z3.total() < z1.total() && z1.total() < dp.total());
        assertTrue(z3.bytesPerRankPerStep() > 1.3 * z1.bytesPerRankPerStep());
    }

    @Test
    void expertParallelCapacityTradesDropsAgainstPadding() {
        List<ParallelismShootout.MoeRow> rows = ParallelismShootout.experts(new float[] {1.0f, 1.5f, 2.0f}, 120, 7);
        for (ParallelismShootout.MoeRow r : rows) {
            System.out.printf("9.6.6 %-36s cf %.2f dropped %.3f, a2a %6d B (padded %6d), slowest rank %.2fx, val loss %.3f%n",
                    r.method(), r.capacityFactor(), r.dropped(), r.allToAllBytes(), r.paddedBytes(), r.slowestRank(),
                    r.validationLoss());
        }
        ParallelismShootout.MoeRow aux1 = rows.get(0);
        ParallelismShootout.MoeRow aux2 = rows.get(4);
        assertTrue(aux1.dropped() > aux2.dropped(), "more capacity drops fewer");
        assertTrue(aux2.paddedBytes() > aux1.paddedBytes(), "more capacity pads more");
    }
}
