package io.github.zzuegg.brainiac.experimental.tensor;

/**
 * A weight matrix with entries in {-1, 0, +1} times one scale, packed at two bits per
 * weight as a "plus" and a "minus" bit mask per output column. A matrix-vector product
 * then needs no multiplications: each output is the sum of the inputs under its plus mask
 * minus the sum under its minus mask, times the scale. This is the inference format of
 * BitNet b1.58 (Ma et al., <a href="https://arxiv.org/abs/2402.17764">2402.17764</a>); with
 * 8-bit activations the sums are pure integer additions.
 */
public final class TernaryMatrix {
    private final int rows;
    private final int cols;
    private final float scale;
    private final int words;
    private final long[] plus;    // [cols][words]
    private final long[] minus;

    private TernaryMatrix(int rows, int cols, float scale, long[] plus, long[] minus) {
        this.rows = rows;
        this.cols = cols;
        this.scale = scale;
        this.words = (rows + 63) / 64;
        this.plus = plus;
        this.minus = minus;
    }

    /**
     * Quantizes a dense {@code [rows, cols]} matrix with the b1.58 "absmean" rule:
     * {@code scale = mean |w|}, {@code q = clip(round(w / scale), -1, 1)}.
     *
     * @param w the weights, row-major {@code [rows, cols]}
     * @param rows the input size
     * @param cols the output size
     * @return the packed matrix
     */
    public static TernaryMatrix quantize(float[] w, int rows, int cols) {
        double sum = 0;
        for (float v : w) {
            sum += Math.abs(v);
        }
        float scale = (float) Math.max(sum / w.length, 1e-8);
        int words = (rows + 63) / 64;
        long[] plus = new long[cols * words];
        long[] minus = new long[cols * words];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int q = Math.max(-1, Math.min(1, Math.round(w[i * cols + j] / scale)));
                if (q > 0) {
                    plus[j * words + (i >>> 6)] |= 1L << (i & 63);
                } else if (q < 0) {
                    minus[j * words + (i >>> 6)] |= 1L << (i & 63);
                }
            }
        }
        return new TernaryMatrix(rows, cols, scale, plus, minus);
    }

    /**
     * Reads one quantized weight.
     *
     * @param row the input index
     * @param col the output index
     * @return -1, 0 or +1
     */
    public int get(int row, int col) {
        long bit = 1L << (row & 63);
        int w = col * words + (row >>> 6);
        return (plus[w] & bit) != 0 ? 1 : (minus[w] & bit) != 0 ? -1 : 0;
    }

    /** @return the scale every ternary weight is multiplied by */
    public float scale() {
        return scale;
    }

    /** @return the input size */
    public int rows() {
        return rows;
    }

    /** @return the output size */
    public int cols() {
        return cols;
    }

    /** @return the packed size in bytes (two bits per weight) */
    public long bytes() {
        return 2L * cols * words * Long.BYTES;
    }

    /** @return the share of weights that are zero */
    public double sparsity() {
        long nonzero = 0;
        for (int i = 0; i < plus.length; i++) {
            nonzero += Long.bitCount(plus[i]) + Long.bitCount(minus[i]);
        }
        return 1 - (double) nonzero / ((long) rows * cols);
    }

    /**
     * Multiplies integer activations: additions and subtractions only.
     *
     * @param x {@code rows} 8-bit activations
     * @return {@code cols} integer sums, before scaling
     */
    public int[] multiply(byte[] x) {
        int[] y = new int[cols];
        for (int j = 0; j < cols; j++) {
            int acc = 0;
            for (int w = 0; w < words; w++) {
                long p = plus[j * words + w];
                while (p != 0) {
                    acc += x[(w << 6) + Long.numberOfTrailingZeros(p)];
                    p &= p - 1;
                }
                long m = minus[j * words + w];
                while (m != 0) {
                    acc -= x[(w << 6) + Long.numberOfTrailingZeros(m)];
                    m &= m - 1;
                }
            }
            y[j] = acc;
        }
        return y;
    }

    /**
     * The full b1.58 inference product of a batch: per-row absmax 8-bit activation
     * quantization, the integer ternary product, then both scales.
     *
     * @param x {@code [batch, rows]} row-major activations
     * @param batch the batch size
     * @return {@code [batch, cols]} row-major outputs
     */
    public float[] matmul(float[] x, int batch) {
        float[] out = new float[batch * cols];
        byte[] q = new byte[rows];
        for (int b = 0; b < batch; b++) {
            float max = 1e-5f;
            for (int i = 0; i < rows; i++) {
                max = Math.max(max, Math.abs(x[b * rows + i]));
            }
            float s = 127f / max;
            for (int i = 0; i < rows; i++) {
                q[i] = (byte) Math.max(-128, Math.min(127, Math.round(x[b * rows + i] * s)));
            }
            int[] y = multiply(q);
            for (int j = 0; j < cols; j++) {
                out[b * cols + j] = y[j] * scale / s;
            }
        }
        return out;
    }
}
