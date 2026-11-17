package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CompressiveKvMemoryTest {

    private static Tensor row(float... x) {
        return Tensor.of(Shape.of(1, 1, x.length), x);
    }

    @Test
    void deltaRuleOverwritesARepeatedKeyWhileLinearAveragesIt() {
        Tensor key = row(1, 1, 0, 0);
        Tensor other = row(0, 0, 1, 1);
        for (CompressiveKvMemory.FeatureMap map : CompressiveKvMemory.FeatureMap.values()) {
            CompressiveKvMemory linear = CompressiveKvMemory.create(1, 4, 2, 1, CompressiveKvMemory.Mode.LINEAR)
                    .featureMap(map);
            CompressiveKvMemory delta = CompressiveKvMemory.create(1, 4, 2, 1, CompressiveKvMemory.Mode.DELTA)
                    .featureMap(map);
            for (CompressiveKvMemory m : new CompressiveKvMemory[] {linear, delta}) {
                m.append(key, row(1, 0));
                m.append(key, row(0, 1));   // evicts the first write
                m.append(other, row(5, 5)); // evicts the second write
            }
            float[] l = linear.readMemory(key).toArray();
            float[] d = delta.readMemory(key).toArray();
            assertArrayEquals(new float[] {0.5f, 0.5f}, l, 1e-4f, map.toString());
            assertArrayEquals(new float[] {0, 1}, d, 1e-4f, map.toString());
        }
    }

    @Test
    void windowStaysExactAndMemoryStaysConstant() {
        CompressiveKvMemory m = CompressiveKvMemory.create(2, 3, 2, 4, CompressiveKvMemory.Mode.LINEAR);
        long before = m.memoryFloats();
        m.append(Tensor.ones(Shape.of(2, 10, 3)), Tensor.ones(Shape.of(2, 10, 2)));
        assertEquals(4, m.windowSize());
        assertEquals(6, m.evicted());
        assertEquals(before, m.memoryFloats());
        float[] out = m.attend(Tensor.ones(Shape.of(2, 1, 3)), 1).toArray();
        for (float v : out) {
            assertEquals(1, v, 1e-5f);
        }
        CompressiveKvMemory pool = CompressiveKvMemory.create(1, 3, 2, 2, CompressiveKvMemory.Mode.POOL).pooling(2, 2);
        pool.append(Tensor.ones(Shape.of(1, 12, 3)), Tensor.ones(Shape.of(1, 12, 2)));
        assertTrue(pool.memoryFloats() <= 2 * (3 + 2) + 1 * (3 + 2));
    }
}
