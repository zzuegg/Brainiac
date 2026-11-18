package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class CompactEmbeddingTest {

    @Test
    void compactEmbeddingsKeepMostOfTheAuc() {
        Map<String, double[]> r = ClickModels.memorySweep(new double[] {0.1}, 100, 1);
        String line = InteractionOrderTest.line("compact embeddings bytes-ratio/AUC/tail-AUC:", r);
        System.out.printf("%s%n", line);
        double full = r.get("full")[1];
        assertTrue(r.get("QUOTIENT_REMAINDER @0.1")[0] < 0.2, line);
        assertTrue(r.get("QUOTIENT_REMAINDER @0.1")[1] > full - 0.05, line);
        assertTrue(r.get("HASHED @0.1")[1] > full - 0.05, line);
    }
}
