package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.GgmlBlocks;

/**
 * Products with a {@link GgmlBlocks} matrix read in place from a GGUF file: each stored row
 * (one output's weights for every input) is decoded once into a buffer and multiplied by
 * every left row, so only the file's 2 to 8.5 bits per weight are read from memory.
 */
final class BlockKernels {

    /** Left rows up to which the formats the SIMD kernels read are multiplied from the blocks for each row. */
    private static final int FUSED_ROWS = 4;

    /** Outputs per parallel task. */
    static final int CHUNK = 16;

    private static final ThreadLocal<float[]> ROW = ThreadLocal.withInitial(() -> new float[0]);

    private BlockKernels() {
    }

    /**
     * Outputs {@code [j0, j1)} of {@code z = x * w}, {@code x} {@code [m, k]} and {@code z}
     * {@code [m, n]}, row-major.
     */
    static void matmul(float[] x, GgmlBlocks w, float[] z, int m, int j0, int j1, boolean simd) {
        int k = w.rows();
        int n = w.cols();
        if (simd && m <= FUSED_ROWS && VectorBlockKernels.reads(w.format())) {
            // multiplied straight from the blocks: no row of floats written and read back
            java.lang.foreign.MemorySegment data = w.data();
            long rowBytes = w.rowBytes();
            String format = w.format();
            for (int j = j0; j < j1; j++) {
                for (int i = 0; i < m; i++) {
                    z[i * n + j] = VectorBlockKernels.dot(format, data, j * rowBytes, x, i * k, k);
                }
            }
            return;
        }
        float[] row = ROW.get();
        if (row.length < k) {
            row = new float[k];
            ROW.set(row);
        }
        for (int j = j0; j < j1; j++) {
            w.decodeRow(j, row);
            for (int i = 0; i < m; i++) {
                z[i * n + j] = simd ? VectorBlockKernels.dot(x, i * k, row, k) : dot(x, i * k, row, k);
            }
        }
    }

    /** The dot product of {@code x[at, at + k)} and {@code y[0, k)}, in four running sums. */
    static float dot(float[] x, int at, float[] y, int k) {
        float s0 = 0;
        float s1 = 0;
        float s2 = 0;
        float s3 = 0;
        int p = 0;
        for (; p + 4 <= k; p += 4) {
            s0 = Math.fma(x[at + p], y[p], s0);
            s1 = Math.fma(x[at + p + 1], y[p + 1], s1);
            s2 = Math.fma(x[at + p + 2], y[p + 2], s2);
            s3 = Math.fma(x[at + p + 3], y[p + 3], s3);
        }
        for (; p < k; p++) {
            s0 = Math.fma(x[at + p], y[p], s0);
        }
        return (s0 + s1) + (s2 + s3);
    }
}
