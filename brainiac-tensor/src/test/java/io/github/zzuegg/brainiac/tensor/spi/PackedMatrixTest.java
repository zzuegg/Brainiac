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

    /** A k-quant-like matrix: codes per group of 16 times a value per 64 rows, offsets shifted by 8. */
    private static PackedMatrix factored(int cols, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int rows = 128;
        int group = 16;
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) r.nextInt(-8, 8);
        }
        byte[] scaleCodes = new byte[rows / group * cols];
        byte[] offsetCodes = new byte[scaleCodes.length];
        for (int i = 0; i < scaleCodes.length; i++) {
            scaleCodes[i] = (byte) r.nextInt(-32, 32);
            offsetCodes[i] = (byte) r.nextInt(0, 64);
        }
        float[] superScales = new float[rows / 64 * cols];
        float[] superOffsets = new float[superScales.length];
        for (int i = 0; i < superScales.length; i++) {
            superScales[i] = (float) r.nextDouble(1e-3, 1e-2);
            superOffsets[i] = (float) -r.nextDouble(1e-3, 1e-2);
        }
        return PackedMatrix.of(levels, new PackedMatrix.Factors(scaleCodes, superScales, offsetCodes, superOffsets, 64, 8),
                rows, cols, 4, group);
    }

    @Test
    void factoredScalesAreCodesTimesSuperValues() {
        PackedMatrix p = factored(5, 7);
        PackedMatrix.Factors f = p.factors();
        for (int g = 0; g < 8; g++) {
            for (int c = 0; c < 5; c++) {
                float s = f.scaleCodes()[g * 5 + c] * f.superScales()[g / 4 * 5 + c];
                assertEquals(s, p.scales()[g * 5 + c]);
                float o = f.offsetCodes()[g * 5 + c] * f.superOffsets()[g / 4 * 5 + c];
                assertEquals((float) (o + (double) s * 8), p.offsets()[g * 5 + c]);
            }
        }
        assertEquals(128 * 5 / 2 + 2 * 8 * 5 + 2 * 4 * 2 * 5, p.bytes());
    }

    @Test
    void joinedFactoredMatricesStayFactored() {
        PackedMatrix a = factored(3, 1);
        PackedMatrix b = factored(4, 2);
        PackedMatrix j = PackedMatrix.concatColumns(a, b);
        assertEquals(64, j.factors().superGroup());
        float[] va = new float[128 * 3];
        float[] vb = new float[128 * 4];
        float[] vj = new float[128 * 7];
        a.dequantize(va);
        b.dequantize(vb);
        j.dequantize(vj);
        for (int r = 0; r < 128; r++) {
            for (int c = 0; c < 7; c++) {
                assertEquals(c < 3 ? va[r * 3 + c] : vb[r * 4 + c - 3], vj[r * 7 + c]);
            }
        }
    }
}
