package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ModelSurgeryTest {

    @Test
    void droppingTheLeastImportantBlockHurtsLeast() {
        HookedDecoder m = TinyLm.deeplySupervised();
        int[][] calib = TinyLm.counting(new SplittableRandom(3), 16, 12);
        double[] bi = BlockImportance.scores(m, calib);
        int low = BlockImportance.leastImportant(bi, 1)[0];
        int high = 0;
        for (int i = 1; i < bi.length; i++) {
            high = bi[i] > bi[high] ? i : high;
        }
        try {
            m.skip(low);
            float lossLow = TinyLm.meanLoss(m, 12, 5);
            m.skip(high);
            float lossHigh = TinyLm.meanLoss(m, 12, 5);
            assertTrue(lossLow < lossHigh, "drop " + low + ": " + lossLow + ", drop " + high + ": " + lossHigh);
            assertArrayEquals(new int[] {low}, BlockImportance.prune(m, calib, 1));
        } finally {
            m.skip();
        }
    }

    @Test
    void trimmedVocabularyKeepsTheScoresOfKeptTokens() {
        HookedDecoder m = TinyLm.trained();
        int[][] evenOnly = {{0, 2, 4, 6, 8, 10}, {4, 6, 8, 10, 0, 2}};
        int[] keep = VocabularyTrimming.used(evenOnly);
        assertEquals(6, keep.length);
        VocabularyTrimming.Trimmed t = VocabularyTrimming.trim(m, keep);
        assertEquals(6, t.model().vocabSize());
        int[] seq = {2, 4, 6};
        float[] full = m.lastLogits(seq);
        float[] small = t.model().lastLogits(t.encode(seq));
        for (int i = 0; i < keep.length; i++) {
            assertEquals(full[keep[i]], small[i], 1e-4f);
        }
        assertArrayEquals(new int[] {8}, t.decode(new int[] {io.github.zzuegg.brainiac.experimental.llm.LogitMath.argmax(small)}));
        long before = m.parameters().stream().mapToLong(v -> v.shape().size()).sum();
        long after = t.model().parameters().stream().mapToLong(v -> v.shape().size()).sum();
        assertTrue(after < before);
    }
}
