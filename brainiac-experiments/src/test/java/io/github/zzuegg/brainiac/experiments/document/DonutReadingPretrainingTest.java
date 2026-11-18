package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.document.DocumentParsing.DonutRow;

import java.util.List;

import org.junit.jupiter.api.Test;

class DonutReadingPretrainingTest {

    @Test
    void readingPretrainingThenParsing() {
        // a smoke run: 150 + 150 steps take over three minutes (F1 0.104 against 0.031 without
        // reading pretraining at 300 total steps, reading CER 0.662); curves are a main() Run
        List<DonutRow> rows = DocumentParsing.donutFollowUp(10, 40, new int[] {5}, 8, 0.02, false, 1);
        System.out.printf("Donut reading pretraining smoke run (10 steps on 40 pages, 5 parsing): %s%n", rows);
        assertTrue(rows.size() == 2, rows.toString());
        assertTrue(Double.isFinite(rows.get(0).readingCer()), "reading CER is measured: " + rows.get(0));
        for (DonutRow r : rows) {
            assertTrue(r.f1() >= 0 && r.f1() <= 1, r.toString());
        }
    }
}
