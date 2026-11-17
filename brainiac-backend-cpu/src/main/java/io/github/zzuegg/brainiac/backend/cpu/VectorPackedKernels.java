package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD version of {@link PackedKernels}: each step loads one level per float lane as bytes,
 * widens them to floats and accumulates with fused multiply-add. Only loaded when the
 * {@code jdk.incubator.vector} module is present, and used where floats have at least 8
 * lanes (so the bytes fill a 64-bit vector).
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

    /** As {@link PackedKernels#matmul}; columns past the last whole vector run scalar. */
    static void matmul(float[] x, PackedMatrix w, float[] z, int m, int j0, int j1) {
        int k = w.rows();
        int n = w.cols();
        int group = w.group();
        int width = j1 - j0;
        int vectorEnd = j0 + (width / LANES) * LANES;
        byte[] d = w.data();
        float[] scales = w.scales();
        boolean nibbles = w.nibbles();
        float[] part = new float[m * width];
        for (int i = 0; i < m; i++) {
            java.util.Arrays.fill(z, i * n + j0, i * n + j1, 0f);
        }
        for (int g0 = 0; g0 < k; g0 += group) {
            int r = g0;
            if (nibbles) {
                for (; r < g0 + group; r += 2) {
                    int base = (r >> 1) * n;
                    for (int j = j0; j < vectorEnd; j += LANES) {
                        ByteVector b = ByteVector.fromArray(BYTES, d, base + j);
                        FloatVector lo = (FloatVector) b.lanewise(VectorOperators.LSHL, 4)
                                .lanewise(VectorOperators.ASHR, 4).convertShape(VectorOperators.B2F, FLOATS, 0);
                        FloatVector hi = (FloatVector) b.lanewise(VectorOperators.ASHR, 4)
                                .convertShape(VectorOperators.B2F, FLOATS, 0);
                        for (int i = 0; i < m; i++) {
                            int row = i * k + r;
                            int p = i * width - j0 + j;
                            FloatVector acc = FloatVector.fromArray(FLOATS, part, p);
                            acc = lo.fma(FloatVector.broadcast(FLOATS, x[row]), acc);
                            acc = hi.fma(FloatVector.broadcast(FLOATS, x[row + 1]), acc);
                            acc.intoArray(part, p);
                        }
                    }
                    for (int j = vectorEnd; j < j1; j++) {
                        byte b = d[base + j];
                        int lo = (byte) (b << 4) >> 4;
                        int hi = b >> 4;
                        for (int i = 0; i < m; i++) {
                            part[i * width - j0 + j] += x[i * k + r] * lo + x[i * k + r + 1] * hi;
                        }
                    }
                }
            } else {
                for (; r < g0 + group; r++) {
                    int base = r * n;
                    for (int j = j0; j < vectorEnd; j += LANES) {
                        FloatVector v = (FloatVector) ByteVector.fromArray(BYTES, d, base + j)
                                .convertShape(VectorOperators.B2F, FLOATS, 0);
                        for (int i = 0; i < m; i++) {
                            int p = i * width - j0 + j;
                            v.fma(FloatVector.broadcast(FLOATS, x[i * k + r]), FloatVector.fromArray(FLOATS, part, p)).intoArray(part, p);
                        }
                    }
                    for (int j = vectorEnd; j < j1; j++) {
                        int level = d[base + j];
                        for (int i = 0; i < m; i++) {
                            part[i * width - j0 + j] += x[i * k + r] * level;
                        }
                    }
                }
            }
            int s = g0 / group * n;
            for (int i = 0; i < m; i++) {
                int p = i * width - j0;
                int o = i * n;
                int j = j0;
                for (; j < vectorEnd; j += LANES) {
                    FloatVector acc = FloatVector.fromArray(FLOATS, part, p + j);
                    acc.fma(FloatVector.fromArray(FLOATS, scales, s + j), FloatVector.fromArray(FLOATS, z, o + j))
                            .intoArray(z, o + j);
                    FloatVector.zero(FLOATS).intoArray(part, p + j);
                }
                for (; j < j1; j++) {
                    z[o + j] += part[p + j] * scales[s + j];
                    part[p + j] = 0f;
                }
            }
        }
    }
}
