package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Unstructured sparsity in compressed sparse row form: the weight {@code W [in, out]} keeps,
 * for every input row, the columns and values of its non-zeros. {@code y = x W} streams each
 * row of {@code x} against the non-zeros, a scalar times a sparse row of {@code W}. This is the
 * baseline format; it wins only when most weights are zero.
 */
public final class CsrMatmul {

    private final int in;
    private final int out;
    private final int[] rowStart;
    private final int[] columns;
    private final float[] values;

    /**
     * @param weight {@code [in, out]}, zeros dropped
     */
    public CsrMatmul(Tensor weight) {
        this.in = weight.shape().dim(0);
        this.out = weight.shape().dim(1);
        float[] w = weight.toArray();
        int nnz = 0;
        for (float v : w) {
            nnz += v != 0 ? 1 : 0;
        }
        rowStart = new int[in + 1];
        columns = new int[nnz];
        values = new float[nnz];
        int p = 0;
        for (int i = 0; i < in; i++) {
            rowStart[i] = p;
            for (int j = 0; j < out; j++) {
                float v = w[i * out + j];
                if (v != 0) {
                    columns[p] = j;
                    values[p++] = v;
                }
            }
        }
        rowStart[in] = p;
    }

    /**
     * @param x {@code [b, in]}
     * @return {@code x W}, {@code [b, out]}
     */
    public Tensor multiply(Tensor x) {
        int b = x.shape().dim(0);
        float[] xv = x.toArray();
        float[] y = new float[b * out];
        for (int r = 0; r < b; r++) {
            int xo = r * in;
            int yo = r * out;
            for (int i = 0; i < in; i++) {
                float a = xv[xo + i];
                if (a == 0) {
                    continue;
                }
                for (int p = rowStart[i]; p < rowStart[i + 1]; p++) {
                    y[yo + columns[p]] += a * values[p];
                }
            }
        }
        return Tensor.of(Shape.of(b, out), y);
    }

    /** @return the stored non-zeros */
    public int nonZeros() {
        return values.length;
    }

    /** @return bytes stored: values, column indices and row pointers (4 bytes each) */
    public long bytes() {
        return 4L * (values.length + columns.length + rowStart.length);
    }

    @Override
    public String toString() {
        return "CsrMatmul(" + in + "x" + out + ", nnz " + values.length + ")";
    }
}
