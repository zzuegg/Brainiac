package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD version of {@link PackedKernels}: levels are loaded as bytes, one per float lane,
 * widened to floats and accumulated with fused multiply-add in registers over a whole
 * group of rows, then scaled into the output once per group. A single left row works on
 * four vectors of columns at a time (two for half-byte levels); more rows go four at a time on two vectors. Only
 * loaded when the {@code jdk.incubator.vector} module is present, and used where floats have
 * at least 8 lanes (so the bytes fill a 64-bit vector).
 */
final class VectorPackedKernels {

    private static final VectorSpecies<Float> FLOATS = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = FLOATS.length();
    private static final VectorSpecies<Byte> BYTES = LANES >= 8
            ? VectorSpecies.of(byte.class, VectorShape.forBitSize(LANES * 8)) : null;

    private VectorPackedKernels() {
    }

    /** @return whether this CPU's vectors are wide enough */
    static boolean usable() {
        return BYTES != null;
    }

    /** As {@link PackedKernels#matmul}. */
    static void matmul(float[] x, PackedMatrix w, float[] z, int m, int j0, int j1) {
        int n = w.cols();
        for (int i = 0; i < m; i++) {
            java.util.Arrays.fill(z, i * n + j0, i * n + j1, 0f);
        }
        int j = j0;
        if (m == 1 && !w.nibbles()) {
            for (; j + 4 * LANES <= j1; j += 4 * LANES) {
                oneRow(x, w, z, j);
            }
        } else if (m == 1) {
            for (; j + 2 * LANES <= j1; j += 2 * LANES) {
                oneRowNibbles(x, w, z, j);
            }
        }
        for (; j + 2 * LANES <= j1; j += 2 * LANES) {
            for (int i = 0; i < m; i += 4) {
                fourRows(x, w, z, i, Math.min(4, m - i), j);
            }
        }
        if (j < j1) {
            PackedKernels.matmul(x, w, z, m, j, j1);
        }
    }

    private static FloatVector widen(ByteVector b) {
        return (FloatVector) b.convertShape(VectorOperators.B2F, FLOATS, 0);
    }

    private static FloatVector low(ByteVector b) {
        return widen(b.lanewise(VectorOperators.LSHL, 4).lanewise(VectorOperators.ASHR, 4));
    }

    private static FloatVector high(ByteVector b) {
        return widen(b.lanewise(VectorOperators.ASHR, 4));
    }

    private static FloatVector scaled(float[] scales, int at, FloatVector acc, float[] z, int zAt) {
        return acc.fma(FloatVector.fromArray(FLOATS, scales, at), FloatVector.fromArray(FLOATS, z, zAt));
    }

    /** Row 0, columns {@code [j, j + 2 * LANES)}, half-byte levels. */
    private static void oneRowNibbles(float[] x, PackedMatrix w, float[] z, int j) {
        int k = w.rows();
        int n = w.cols();
        int group = w.group();
        byte[] d = w.data();
        float[] scales = w.scales();
        int jb = j + LANES;
        for (int g0 = 0; g0 < k; g0 += group) {
            FloatVector c0 = FloatVector.zero(FLOATS);
            FloatVector c1 = c0;
            for (int r = g0; r < g0 + group; r += 2) {
                int base = (r >> 1) * n;
                FloatVector a = FloatVector.broadcast(FLOATS, x[r]);
                FloatVector b = FloatVector.broadcast(FLOATS, x[r + 1]);
                ByteVector v0 = ByteVector.fromArray(BYTES, d, base + j);
                ByteVector v1 = ByteVector.fromArray(BYTES, d, base + jb);
                c0 = high(v0).fma(b, low(v0).fma(a, c0));
                c1 = high(v1).fma(b, low(v1).fma(a, c1));
            }
            int s = g0 / group * n;
            scaled(scales, s + j, c0, z, j).intoArray(z, j);
            scaled(scales, s + jb, c1, z, jb).intoArray(z, jb);
        }
    }

    /** Row 0, columns {@code [j, j + 4 * LANES)}, byte levels. */
    private static void oneRow(float[] x, PackedMatrix w, float[] z, int j) {
        int k = w.rows();
        int n = w.cols();
        int group = w.group();
        byte[] d = w.data();
        float[] scales = w.scales();
        int j1 = j + LANES;
        int j2 = j1 + LANES;
        int j3 = j2 + LANES;
        for (int g0 = 0; g0 < k; g0 += group) {
            FloatVector c0 = FloatVector.zero(FLOATS);
            FloatVector c1 = c0;
            FloatVector c2 = c0;
            FloatVector c3 = c0;
            for (int r = g0; r < g0 + group; r++) {
                int base = r * n;
                FloatVector a = FloatVector.broadcast(FLOATS, x[r]);
                c0 = widen(ByteVector.fromArray(BYTES, d, base + j)).fma(a, c0);
                c1 = widen(ByteVector.fromArray(BYTES, d, base + j1)).fma(a, c1);
                c2 = widen(ByteVector.fromArray(BYTES, d, base + j2)).fma(a, c2);
                c3 = widen(ByteVector.fromArray(BYTES, d, base + j3)).fma(a, c3);
            }
            int s = g0 / group * n;
            scaled(scales, s + j, c0, z, j).intoArray(z, j);
            scaled(scales, s + j1, c1, z, j1).intoArray(z, j1);
            scaled(scales, s + j2, c2, z, j2).intoArray(z, j2);
            scaled(scales, s + j3, c3, z, j3).intoArray(z, j3);
        }
    }

    /** Rows {@code [i, i + rows)} ({@code rows <= 4}), columns {@code [j, j + 2 * LANES)}. */
    private static void fourRows(float[] x, PackedMatrix w, float[] z, int i, int rows, int j) {
        int k = w.rows();
        int n = w.cols();
        int group = w.group();
        byte[] d = w.data();
        float[] scales = w.scales();
        int jb = j + LANES;
        // rows past the end read row i again; their sums are not stored
        int x0 = i * k;
        int x1 = (rows > 1 ? i + 1 : i) * k;
        int x2 = (rows > 2 ? i + 2 : i) * k;
        int x3 = (rows > 3 ? i + 3 : i) * k;
        for (int g0 = 0; g0 < k; g0 += group) {
            FloatVector c00 = FloatVector.zero(FLOATS);
            FloatVector c01 = c00;
            FloatVector c10 = c00;
            FloatVector c11 = c00;
            FloatVector c20 = c00;
            FloatVector c21 = c00;
            FloatVector c30 = c00;
            FloatVector c31 = c00;
            int step = w.nibbles() ? 2 : 1;
            for (int r = g0; r < g0 + group; r += step) {
                FloatVector b0;
                FloatVector b1;
                if (w.nibbles()) {
                    int base = (r >> 1) * n;
                    ByteVector v0 = ByteVector.fromArray(BYTES, d, base + j);
                    ByteVector v1 = ByteVector.fromArray(BYTES, d, base + jb);
                    FloatVector h0 = high(v0);
                    FloatVector h1 = high(v1);
                    b0 = low(v0);
                    b1 = low(v1);
                    FloatVector a = FloatVector.broadcast(FLOATS, x[x0 + r + 1]);
                    c00 = h0.fma(a, c00);
                    c01 = h1.fma(a, c01);
                    a = FloatVector.broadcast(FLOATS, x[x1 + r + 1]);
                    c10 = h0.fma(a, c10);
                    c11 = h1.fma(a, c11);
                    a = FloatVector.broadcast(FLOATS, x[x2 + r + 1]);
                    c20 = h0.fma(a, c20);
                    c21 = h1.fma(a, c21);
                    a = FloatVector.broadcast(FLOATS, x[x3 + r + 1]);
                    c30 = h0.fma(a, c30);
                    c31 = h1.fma(a, c31);
                } else {
                    int base = r * n;
                    b0 = widen(ByteVector.fromArray(BYTES, d, base + j));
                    b1 = widen(ByteVector.fromArray(BYTES, d, base + jb));
                }
                FloatVector a = FloatVector.broadcast(FLOATS, x[x0 + r]);
                c00 = b0.fma(a, c00);
                c01 = b1.fma(a, c01);
                a = FloatVector.broadcast(FLOATS, x[x1 + r]);
                c10 = b0.fma(a, c10);
                c11 = b1.fma(a, c11);
                a = FloatVector.broadcast(FLOATS, x[x2 + r]);
                c20 = b0.fma(a, c20);
                c21 = b1.fma(a, c21);
                a = FloatVector.broadcast(FLOATS, x[x3 + r]);
                c30 = b0.fma(a, c30);
                c31 = b1.fma(a, c31);
            }
            int s = g0 / group * n;
            int z0 = i * n;
            scaled(scales, s + j, c00, z, z0 + j).intoArray(z, z0 + j);
            scaled(scales, s + jb, c01, z, z0 + jb).intoArray(z, z0 + jb);
            if (rows > 1) {
                scaled(scales, s + j, c10, z, z0 + n + j).intoArray(z, z0 + n + j);
                scaled(scales, s + jb, c11, z, z0 + n + jb).intoArray(z, z0 + n + jb);
            }
            if (rows > 2) {
                scaled(scales, s + j, c20, z, z0 + 2 * n + j).intoArray(z, z0 + 2 * n + j);
                scaled(scales, s + jb, c21, z, z0 + 2 * n + jb).intoArray(z, z0 + 2 * n + jb);
            }
            if (rows > 3) {
                scaled(scales, s + j, c30, z, z0 + 3 * n + j).intoArray(z, z0 + 3 * n + j);
                scaled(scales, s + jb, c31, z, z0 + 3 * n + jb).intoArray(z, z0 + 3 * n + jb);
            }
        }
    }
}
