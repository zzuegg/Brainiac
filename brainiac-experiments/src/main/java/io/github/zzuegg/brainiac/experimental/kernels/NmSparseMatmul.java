package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * N:M structured sparsity (Mishra et al. 2021, "Accelerating Sparse Deep Neural Networks",
 * <a href="https://arxiv.org/abs/2104.08378">arXiv:2104.08378</a>): along the input dimension,
 * every group of {@code m} weights of an output column keeps at most {@code n}. The weight is
 * stored output-major as {@code n} values plus {@code n} small in-group offsets per group, which
 * is 2 bits each for 2:4. {@code y = x W} gathers the {@code n} inputs of each group, so the
 * work is {@code n/m} of dense and has no irregular control flow. {@link #prune} keeps the
 * {@code n} largest magnitudes per group.
 */
public final class NmSparseMatmul {

    private final int in;
    private final int out;
    private final int n;
    private final int m;
    private final float[] values;
    private final byte[] offsets;

    /**
     * @param weight {@code [in, out]} that already satisfies the pattern ({@link #prune})
     * @param n kept per group
     * @param m group size along the input dimension
     */
    public NmSparseMatmul(Tensor weight, int n, int m) {
        this.in = weight.shape().dim(0);
        this.out = weight.shape().dim(1);
        this.n = n;
        this.m = m;
        if (in % m != 0) {
            throw new IllegalArgumentException(m + " must divide " + in);
        }
        float[] w = weight.toArray();
        int groups = in / m;
        values = new float[out * groups * n];
        offsets = new byte[out * groups * n];
        for (int j = 0; j < out; j++) {
            for (int g = 0; g < groups; g++) {
                int slot = (j * groups + g) * n;
                int used = 0;
                for (int t = 0; t < m; t++) {
                    float v = w[(g * m + t) * out + j];
                    if (v != 0) {
                        if (used == n) {
                            throw new IllegalArgumentException("group " + g + " of column " + j + " has more than " + n
                                    + " non-zeros");
                        }
                        values[slot + used] = v;
                        offsets[slot + used] = (byte) t;
                        used++;
                    }
                }
            }
        }
    }

    /**
     * Keeps the {@code n} largest magnitudes of every group of {@code m} inputs per output.
     *
     * @param weight {@code [in, out]}
     * @param n kept
     * @param m group size
     * @return the pruned weight
     */
    public static Tensor prune(Tensor weight, int n, int m) {
        int in = weight.shape().dim(0);
        int out = weight.shape().dim(1);
        float[] w = weight.toArray().clone();
        for (int j = 0; j < out; j++) {
            for (int g = 0; g < in / m; g++) {
                boolean[] dropped = new boolean[m];
                for (int drop = 0; drop < m - n; drop++) {
                    int worst = -1;
                    for (int t = 0; t < m; t++) {
                        int idx = (g * m + t) * out + j;
                        if (!dropped[t] && (worst < 0 || Math.abs(w[idx]) < Math.abs(w[(g * m + worst) * out + j]))) {
                            worst = t;
                        }
                    }
                    dropped[worst] = true;
                    w[(g * m + worst) * out + j] = 0;
                }
            }
        }
        return Tensor.of(Shape.of(in, out), w);
    }

    /**
     * @param x {@code [b, in]}
     * @return {@code x W}, {@code [b, out]}
     */
    public Tensor multiply(Tensor x) {
        int b = x.shape().dim(0);
        float[] xv = x.toArray();
        float[] y = new float[b * out];
        int groups = in / m;
        for (int r = 0; r < b; r++) {
            int xo = r * in;
            for (int j = 0; j < out; j++) {
                int slot = j * groups * n;
                float s = 0;
                for (int g = 0; g < groups; g++) {
                    int base = xo + g * m;
                    for (int t = 0; t < n; t++) {
                        s += values[slot] * xv[base + offsets[slot]];
                        slot++;
                    }
                }
                y[r * out + j] = s;
            }
        }
        return Tensor.of(Shape.of(b, out), y);
    }

    /** @return bytes stored: values (4 bytes) plus 2-bit offsets for {@code m <= 4} */
    public long bytes() {
        long bitsPerOffset = 32 - Integer.numberOfLeadingZeros(m - 1);
        return 4L * values.length + (values.length * bitsPerOffset + 7) / 8;
    }

    @Override
    public String toString() {
        return "NmSparseMatmul(" + n + ":" + m + ", " + in + "x" + out + ")";
    }
}
