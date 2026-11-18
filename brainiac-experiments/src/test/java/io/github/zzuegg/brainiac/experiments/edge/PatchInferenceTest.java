package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class PatchInferenceTest {

    @Test
    void patchInferenceCutsPeakMemoryExactly() {
        List<EdgeDeployment.PatchRow> rows = EdgeDeployment.patches(64, new int[] {2, 4}, 2);
        for (EdgeDeployment.PatchRow r : rows) {
            System.out.printf("9.6.14 %-22s grid %d peak %7d B, %9d MACs (+%.1f%%), max diff %.2e%n", r.network(),
                    r.grid(), r.peakBytes(), r.macs(), 100 * r.extraMacShare(), r.maxDifference());
        }
        for (EdgeDeployment.OrderRow r : EdgeDeployment.reordering(2)) {
            System.out.printf("9.6.14 %-46s given order %6d B, best %6d B (%s)%n", r.graph(), r.givenPeak(),
                    r.bestPeak(), r.bestOrder());
            assertTrue(r.bestPeak() <= r.givenPeak());
        }
        for (EdgeDeployment.BudgetRow r : EdgeDeployment.budget(16, 130, 2)) {
            System.out.printf("9.6.14 %-48s side %2d peak %6d B, %8d MACs, accuracy %.3f%n", r.method(), r.side(),
                    r.peakBytes(), r.macs(), r.accuracy());
        }
        EdgeDeployment.PatchRow layer = rows.get(0);
        EdgeDeployment.PatchRow patch = rows.get(2);
        assertTrue(patch.peakBytes() * 2 < layer.peakBytes(), patch + " vs " + layer);
        assertTrue(patch.maxDifference() < 1e-3, "patch execution matches layer by layer");
    }
}
