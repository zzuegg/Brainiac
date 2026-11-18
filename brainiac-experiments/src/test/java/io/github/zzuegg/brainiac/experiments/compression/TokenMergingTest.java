package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.compress.TokenMerging;

import java.util.List;

import org.junit.jupiter.api.Test;

class TokenMergingTest {

    @Test
    void bipartiteMatchingMergesTheMostSimilarPair() {
        // tokens: cls, a1=(1,0), b1=(0,1), a2=(0,1), b2=(1,0.1)
        float[] keys = {5, 5, 1, 0, 0, 1, 0, 1, 1, 0.1f};
        int[][] groups = TokenMerging.bipartiteMatch(keys, 0, 5, 2, 1);
        assertEquals(4, groups.length);
        assertArrayEquals(new int[] {0}, groups[0]);
        int merged = 0;
        for (int[] g : groups) {
            merged += g.length == 2 ? 1 : 0;
        }
        assertEquals(1, merged);
    }

    @Test
    void tomeAgainstPruning() {
        List<CompactRow> rows = CompactModels.tokenMerging(new int[] {2, 5}, 120, 20, 1);
        StringBuilder line = new StringBuilder("8.5.13 ViT 17 tokens (acc @MACs, ms):");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f @%.0f %.1fms;", r.method(), r.metric(), r.cost(), r.millis()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).metric() > 0.6, "dense ViT learns: " + rows.get(0));
        assertTrue(rows.get(1).metric() > 0.6, "tome r=2 works without training: " + rows.get(1));
    }
}
