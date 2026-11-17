package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;

import java.util.SplittableRandom;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class PackedMatmulTest {

    static PackedMatrix random(int rows, int cols, int bits, int group, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int limit = (1 << (bits - 1)) - 1;
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) r.nextInt(-limit, limit + 1);
        }
        float[] scales = new float[rows / group * cols];
        for (int i = 0; i < scales.length; i++) {
            scales[i] = (float) r.nextDouble(0.001, 0.05);
        }
        return PackedMatrix.of(levels, scales, rows, cols, bits, group);
    }

    @ParameterizedTest
    @CsvSource({
        "8, 64, 1, 64", "8, 64, 3, 16", "4, 64, 1, 64", "4, 64, 5, 32", "4, 96, 2, 3",
        "3, 48, 4, 48", "2, 32, 7, 8", "8, 50, 1, 25", "4, 40, 20, 40", "8, 64, 17, 64",
    })
    void matchesTheProductWithTheUnpackedMatrix(int bits, int rows, int m, int group) {
        for (int cols : new int[] {7, 64, 200, 333}) {
            PackedMatrix w = random(rows, cols, bits, group, bits * 31L + cols);
            assertEquals(bits <= 4 && group % 2 == 0, w.nibbles());
            Tensor x = Tensor.uniform(Shape.of(m, rows), -1, 1, new SplittableRandom(cols));
            Tensor packed = Tensor.packed(w, Device.cpu());
            assertTrue(packed.isPacked());
            float[] dense = new float[rows * cols];
            w.dequantize(dense);
            float[] expected = x.matmul(Tensor.of(Shape.of(rows, cols), dense)).toArray();
            assertArrayEquals(expected, x.matmul(packed).toArray(), 1e-4f, bits + " bits, " + cols + " columns");
            assertArrayEquals(dense, packed.toArray());
            assertArrayEquals(dense, packed.unpacked().toArray());

            float[] scalar = new float[m * cols];
            float[] simd = new float[m * cols];
            for (int j0 = 0; j0 < cols; j0 += PackedKernels.CHUNK) {
                int j1 = Math.min(cols, j0 + PackedKernels.CHUNK);
                PackedKernels.matmul(x.toArray(), w, scalar, m, j0, j1);
                if (VectorPackedKernels.usable()) {
                    VectorPackedKernels.matmul(x.toArray(), w, simd, m, j0, j1);
                }
            }
            assertArrayEquals(expected, scalar, 1e-4f);
            if (VectorPackedKernels.usable()) {
                assertArrayEquals(expected, simd, 1e-4f);
            }
        }
    }

    static PackedMatrix withOffsets(PackedMatrix w, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] offsets = new float[w.scales().length];
        for (int i = 0; i < offsets.length; i++) {
            offsets[i] = (float) r.nextDouble(-0.3, 0.3);
        }
        byte[] levels = new byte[w.rows() * w.cols()];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) w.level(i / w.cols(), i % w.cols());
        }
        return PackedMatrix.of(levels, w.scales(), offsets, w.rows(), w.cols(), w.bits(), w.group());
    }

    @ParameterizedTest
    @CsvSource({"8, 64, 1, 32", "4, 64, 3, 32", "5, 48, 2, 16", "2, 32, 9, 16", "6, 32, 4, 16"})
    void offsetsAddToEveryValueOfTheirGroup(int bits, int rows, int m, int group) {
        for (int cols : new int[] {7, 200}) {
            PackedMatrix plain = random(rows, cols, bits, group, cols);
            PackedMatrix w = withOffsets(plain, bits);
            float[] dense = new float[rows * cols];
            w.dequantize(dense);
            float[] levels = new float[rows * cols];
            plain.dequantize(levels);
            for (int i = 0; i < dense.length; i++) {
                assertEquals(levels[i] + w.offsets()[i / cols / group * cols + i % cols], dense[i], 1e-6f);
            }
            Tensor x = Tensor.uniform(Shape.of(m, rows), -1, 1, new SplittableRandom(cols));
            float[] expected = x.matmul(Tensor.of(Shape.of(rows, cols), dense)).toArray();
            Tensor packed = Tensor.packed(w, Device.cpu());
            assertArrayEquals(expected, x.matmul(packed).toArray(), 1e-4f, bits + " bits, " + cols + " columns");
            assertArrayEquals(dense, packed.unpacked().toArray());
            assertEquals(2 * plain.bytes() - plain.data().length, w.bytes());
        }
    }

    @ParameterizedTest
    @CsvSource({"8, 1", "4, 2"})
    void otherOperationsSeeTheValues(int bits, int m) {
        PackedMatrix w = random(8, 6, bits, 8, 5);
        Tensor packed = Tensor.packed(w, Device.cpu());
        float[] dense = new float[48];
        w.dequantize(dense);
        Tensor plain = Tensor.of(Shape.of(8, 6), dense);
        assertArrayEquals(plain.transpose().toArray(), packed.transpose().toArray());
        assertArrayEquals(plain.mul(2).add(1).toArray(), packed.mul(2).add(1).toArray());
        assertArrayEquals(plain.reshape(6, 8).toArray(), packed.reshape(6, 8).toArray());
        assertEquals(w.bytes(), packed.bytes());
        assertEquals(4L * 48, plain.bytes());
        Tensor x = Tensor.uniform(Shape.of(2, m, 8), -1, 1, new SplittableRandom(1));
        assertArrayEquals(x.matmul(plain).toArray(), x.matmul(packed).toArray(), 1e-5f);
    }
}
