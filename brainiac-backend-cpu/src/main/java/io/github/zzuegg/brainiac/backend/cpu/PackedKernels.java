package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;

/**
 * Matrix products with a {@link PackedMatrix} on the right, reading its levels directly:
 * for the few rows of generating text one token at a time, which wait on memory, so
 * reading a quarter (8 bits) or an eighth (4 bits) of float32's bytes makes them faster.
 *
 * <p>Columns are split into chunks; for each chunk the weights are read once, row by row,
 * and every left row accumulates into a partial sum that stays in the L1 cache, scaled into
 * the output at the end of each group.
 */
final class PackedKernels {

    /** Columns per chunk, a multiple of every SIMD width. */
    static final int CHUNK = 128;

    private PackedKernels() {
    }

    /**
     * Columns {@code [j0, j1)} of {@code z = x * w}, {@code x} row-major {@code [m, k]},
     * {@code z} {@code [m, n]}, overwritten.
     */
    static void matmul(float[] x, PackedMatrix w, float[] z, int m, int j0, int j1) {
        int k = w.rows();
        int n = w.cols();
        int group = w.group();
        int width = j1 - j0;
        byte[] d = w.data();
        float[] scales = w.scales();
        boolean nibbles = w.nibbles();
        float[] part = new float[m * width];
        for (int i = 0; i < m; i++) {
            java.util.Arrays.fill(z, i * n + j0, i * n + j1, 0f);
        }
        for (int g0 = 0; g0 < k; g0 += group) {
            for (int r = g0; r < g0 + group; r++) {
                int base = (nibbles ? r >> 1 : r) * n;
                int shift = nibbles && (r & 1) == 0 ? 4 : 0;
                for (int i = 0; i < m; i++) {
                    float xv = x[i * k + r];
                    if (xv == 0f) {
                        continue;
                    }
                    int p = i * width - j0;
                    if (nibbles) {
                        for (int j = j0; j < j1; j++) {
                            part[p + j] += xv * ((byte) (d[base + j] << shift) >> 4);
                        }
                    } else {
                        for (int j = j0; j < j1; j++) {
                            part[p + j] += xv * d[base + j];
                        }
                    }
                }
            }
            int s = g0 / group * n;
            for (int i = 0; i < m; i++) {
                int p = i * width - j0;
                int o = i * n;
                for (int j = j0; j < j1; j++) {
                    z[o + j] += part[p + j] * scales[s + j];
                    part[p + j] = 0f;
                }
            }
        }
    }

    /**
     * Adds a matrix's offsets to columns {@code [j0, j1)} of {@code z = x * w}: each
     * offset times the sum of the {@code x} values of its group.
     */
    static void addOffsets(float[] x, PackedMatrix w, float[] z, int m, int j0, int j1) {
        float[] offsets = w.offsets();
        int k = w.rows();
        int n = w.cols();
        int group = w.group();
        for (int i = 0; i < m; i++) {
            int o = i * n;
            for (int g0 = 0; g0 < k; g0 += group) {
                float sum = 0;
                for (int r = g0; r < g0 + group; r++) {
                    sum += x[i * k + r];
                }
                if (sum == 0f) {
                    continue;
                }
                int s = g0 / group * n;
                for (int j = j0; j < j1; j++) {
                    z[o + j] += sum * offsets[s + j];
                }
            }
        }
    }
}
