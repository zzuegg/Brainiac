package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SparseSelectionShootoutTest {

    @Test
    void sparseSelectionKeepsLanguageModelLoss() {
        List<SparseSelectionShootout.Row> rows = SparseSelectionShootout.languageModel(1, 16, 200, 1);
        System.out.print(SparseSelectionShootout.table("val loss", rows));
        double dense = rows.get(0).metrics()[0];
        SparseSelectionShootout.Row dsa = rows.get(1);
        SparseSelectionShootout.Row moba = rows.get(2);
        System.out.printf("toy LM val loss: DSA %.4f (%.1f keys/query), MoBA %.4f (%.1f keys/query), NSA %.4f vs dense "
                + "%.4f (12.5 keys/query on average)%n", dsa.metrics()[0], dsa.attended(), moba.metrics()[0],
                moba.attended(), rows.get(3).metrics()[0], dense);
        assertTrue(dsa.attended() <= 8.0 + 1e-9);
        assertTrue(moba.attended() <= 8.0 + 1e-9);
        assertTrue(dsa.metrics()[0] < dense + 0.4);
        assertTrue(moba.metrics()[0] < dense + 0.5);
    }

    @Test
    void indexerFindsNeedlesBeyondTrainingLength() {
        List<SparseSelectionShootout.Row> rows = SparseSelectionShootout.longRecall(1, 16, 250, 1,
                List.of("dense", "DSA", "MoBA"));
        System.out.print(SparseSelectionShootout.table("recall @ 8/16/32 pairs", rows));
        SparseSelectionShootout.Row dense = rows.get(0);
        SparseSelectionShootout.Row dsa = rows.get(1);
        SparseSelectionShootout.Row moba = rows.get(2);
        System.out.printf("recall at 8/32 pairs (65 tokens): DSA %.3f/%.3f with %.1f keys/query, MoBA %.3f/%.3f with "
                + "%.1f keys/query vs dense %.3f/%.3f (chance 0.125)%n", dsa.metrics()[0], dsa.metrics()[2],
                dsa.attended(), moba.metrics()[0], moba.metrics()[2], moba.attended(), dense.metrics()[0],
                dense.metrics()[2]);
        assertTrue(dsa.metrics()[0] > 0.3);
        assertTrue(dsa.attended() <= 8.0 + 1e-9);
        assertTrue(moba.attended() <= 8.0 + 1e-9);
    }
}
