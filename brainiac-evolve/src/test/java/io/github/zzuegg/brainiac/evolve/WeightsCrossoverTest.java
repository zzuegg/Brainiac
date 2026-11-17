package io.github.zzuegg.brainiac.evolve;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class WeightsCrossoverTest {

    private static Weights of(Map<Long, Float> connections) {
        return new Weights(connections, Map.of(), Map.of(), null, Map.of(), Map.of());
    }

    @Test
    void sharedValuesMixAndOthersAreKept() {
        Weights a = of(Map.of(1L, 1f, 2L, 2f, 3L, 3f));
        Weights b = of(Map.of(2L, 20f, 3L, 30f, 4L, 40f));
        Map<Long, Float> blended = a.blend(b, 0.5f).connectionValues();
        assertEquals(Map.of(1L, 1f, 2L, 11f, 3L, 16.5f, 4L, 40f), blended);
        int fromA = 0;
        for (int seed = 0; seed < 200; seed++) {
            Map<Long, Float> c = a.crossover(b, new SplittableRandom(seed)).connectionValues();
            assertEquals(1f, c.get(1L));
            assertEquals(40f, c.get(4L));
            assertTrue(c.get(2L) == 2f || c.get(2L) == 20f);
            fromA += c.get(2L) == 2f ? 1 : 0;
        }
        assertTrue(fromA > 70 && fromA < 130, "uniform: " + fromA);
    }
}
