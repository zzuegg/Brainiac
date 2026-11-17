package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SeedPrefixTest {

    @Test
    void prefixesWithReservedSeedTokens() {
        SeedPrefix p = new SeedPrefix(12, 8, 4);
        assertEquals(20, p.vocabularySize());
        int[] out = p.apply(new int[] {11, 3}, new SplittableRandom(1));
        assertEquals(6, out.length);
        assertArrayEquals(new int[] {11, 3}, Arrays.copyOfRange(out, 4, 6));
        for (int i = 0; i < 4; i++) {
            assertTrue(out[i] >= 12 && out[i] < 20);
        }
    }
}
