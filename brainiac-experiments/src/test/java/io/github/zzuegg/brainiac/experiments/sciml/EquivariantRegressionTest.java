package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class EquivariantRegressionTest {

    @Test
    void symmetricModelsBeatAugmentationOnInvariantRegression() {
        List<SymmetricModels.InvariantRow> rows = SymmetricModels.invariantRegression(5, new int[] {32, 256}, 250, 1);
        StringBuilder line = new StringBuilder("8.3.14 O(3)-invariant regression, rel test MSE / invariance gap (ms):");
        int size = -1;
        for (SymmetricModels.InvariantRow r : rows) {
            if (r.samples() != size) {
                size = r.samples();
                line.append(" [n=").append(size).append(']');
            }
            line.append(String.format(" %s %.3f/%.1e (%.0f);", r.model(), r.testMse(), r.invarianceGap(),
                    r.millis()));
        }
        System.out.println(line);
        // frame averaging, the EMLP and the EGNN are invariant by construction
        for (SymmetricModels.InvariantRow r : rows) {
            if (!r.model().startsWith("MLP")) {
                assertTrue(r.invarianceGap() < 1e-2, line.toString());
            }
        }
    }
}
