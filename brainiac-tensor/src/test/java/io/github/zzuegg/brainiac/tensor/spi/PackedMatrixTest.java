package io.github.zzuegg.brainiac.tensor.spi;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PackedMatrixTest {

    private static PackedMatrix random(int rows, int cols, int bits, int group, boolean offsets, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int limit = (1 << (bits - 1)) - 1;
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) r.nextInt(-limit, limit + 1);
        }
        float[] scales = new float[rows / group * cols];
        float[] off = offsets ? new float[scales.length] : null;
        for (int i = 0; i < scales.length; i++) {
            scales[i] = (float) r.nextDouble(0.01, 1);
            if (off != null) {
                off[i] = (float) r.nextDouble(-1, 1);
            }
        }
        return PackedMatrix.of(levels, scales, off, rows, cols, bits, group);
    }

    @Test
    void joinedColumnsHoldEachPartsValues() {
        for (int bits : new int[] {4, 8}) {
            for (boolean offsets : new boolean[] {false, true}) {
                PackedMatrix a = random(8, 3, bits, 4, offsets, 1);
                PackedMatrix b = random(8, 5, bits, 4, offsets, 2);
                PackedMatrix c = random(8, 2, bits, 4, offsets, 3);
                PackedMatrix j = PackedMatrix.concatColumns(a, b, c);
                assertEquals(10, j.cols());
                float[] va = new float[24];
                float[] vb = new float[40];
                float[] vc = new float[16];
                float[] vj = new float[80];
                a.dequantize(va);
                b.dequantize(vb);
                c.dequantize(vc);
                j.dequantize(vj);
                float[] expected = new float[80];
                for (int r = 0; r < 8; r++) {
                    System.arraycopy(va, r * 3, expected, r * 10, 3);
                    System.arraycopy(vb, r * 5, expected, r * 10 + 3, 5);
                    System.arraycopy(vc, r * 2, expected, r * 10 + 8, 2);
                }
                assertArrayEquals(expected, vj, 0);
            }
        }
        assertThrows(IllegalArgumentException.class,
                () -> PackedMatrix.concatColumns(random(8, 3, 4, 4, false, 1), random(8, 3, 8, 4, false, 1)));
    }
}
