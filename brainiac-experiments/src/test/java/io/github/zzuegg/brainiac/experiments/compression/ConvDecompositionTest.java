package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ConvDecompositionTest {

    @Test
    void cpAndTuckerAgainstSvd() {
        List<CompactRow> conv = CompactModels.convDecompositions(new int[] {8, 4}, 40, 20, 1);
        TensorDecompositionTest.print("8.5.12 CNN conv2 32x16x3x3 (acc @weights, ms):", conv);
        for (CompactRow row : conv) {
            if (row.method().endsWith("+ft")) {
                assertTrue(row.metric() > 0.5, "fine-tuned conv decomposition works: " + row);
            }
        }
    }
}
