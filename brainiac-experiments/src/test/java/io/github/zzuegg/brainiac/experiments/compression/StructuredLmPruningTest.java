package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class StructuredLmPruningTest {

    @Test
    void structuredPruningOfAToyLanguageModel() {
        List<PruningShootout.Row> rows = PruningShootout.languageModel(new double[] {0.25, 0.5}, 130, 20, 1);
        StringBuilder line = new StringBuilder("8.5.5 structured LM pruning, val loss recovered (one-shot) @removed:");
        for (PruningShootout.Row r : rows) {
            line.append(String.format(" %s@%.2f %.3f (%.3f);", r.method(), r.sparsity(), r.accuracy(), r.extra()));
        }
        System.out.println(line);
        double dense = rows.get(0).accuracy();
        assertTrue(dense < Math.log(32), "the toy LM learned something: " + dense);
        for (PruningShootout.Row r : rows.subList(1, rows.size())) {
            assertTrue(r.accuracy() <= r.extra() + 0.05, "recovery does not hurt: " + r);
            assertTrue(r.accuracy() < Math.log(32), "pruned model beats uniform: " + r);
        }
    }
}
