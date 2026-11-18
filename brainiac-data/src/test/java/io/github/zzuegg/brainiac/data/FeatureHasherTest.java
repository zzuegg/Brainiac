package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class FeatureHasherTest {

    private static int hash(String s, int seed) {
        return FeatureHasher.murmur3(s.getBytes(StandardCharsets.UTF_8), seed);
    }

    @Test
    void murmur3MatchesReferenceVectors() {
        assertEquals(0, hash("", 0));
        assertEquals(0x514E28B7, hash("", 1));
        assertEquals(0x248bfa47, hash("hello", 0));
        assertEquals(0x2e4ff723, hash("The quick brown fox jumps over the lazy dog", 0));
    }

    @Test
    void hashesIntoFixedColumns() {
        FeatureHasher h = new FeatureHasher(16, true);
        float[] row = h.transform(List.of("cat", "dog", "cat"));
        assertEquals(16, row.length);
        assertEquals(2 * h.sign("cat"), row[h.column("cat")] - (h.column("dog") == h.column("cat") ? h.sign("dog") : 0));
        float[] weighted = h.transform(Map.of("price", 2.5, "cat", 1));
        float[] expected = new float[16];
        expected[h.column("price")] += 2.5f * h.sign("price");
        expected[h.column("cat")] += h.sign("cat");
        assertArrayEquals(expected, weighted);
        FeatureHasher plain = new FeatureHasher(16, false);
        for (float v : plain.transform(List.of("a", "b", "c", "d", "e"))) {
            assertEquals(true, v >= 0);
        }
    }
}
