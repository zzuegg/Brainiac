package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SparseFormatsLmTest {

    @Test
    void blockAndNmAgainstUnstructured() {
        List<PruningShootout.Row> rows = PruningShootout.sparseFormats(110, 15, 1);
        StringBuilder line = new StringBuilder("8.5.18 LM 50% sparse (val loss after recovery, one-shot):");
        for (PruningShootout.Row r : rows) {
            line.append(String.format(" %s %.3f (%.3f) @%.2f;", r.method(), r.accuracy(), r.extra(), r.sparsity()));
        }
        System.out.println(line);
        for (PruningShootout.Row r : rows) {
            assertTrue(r.accuracy() < 4, "recovers: " + r);
        }
    }
}
