package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PruningAtInitTest {

    @Test
    void pruningAtInitialisationAgainstRandom() {
        List<PruningShootout.Row> rows = PruningShootout.atInit(new double[] {0.9, 0.99}, 80, 1);
        System.out.println(PruningCriteriaTest.join("8.5.4 pruning at init, acc (last-layer survival):", rows, true));
        for (PruningShootout.Row r : rows) {
            assertTrue(r.accuracy() >= 0.0 && r.sparsity() > 0.85, "pruned to sparsity: " + r);
        }
        assertTrue(rows.get(2).accuracy() > 0.2, "SynFlow at 90% trains: " + rows.get(2));
    }
}
