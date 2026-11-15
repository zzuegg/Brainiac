package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

/**
 * The CPU loops, each over an index range so callers can split them across threads.
 * Every operation has its own simple loop so the JIT compiler can vectorize it.
 */
final class Kernels {

    /** Rows of the left matrix processed together, so each loaded row of b is reused. */
    private static final int MATMUL_ROWS = 4;
    /** Depth of one block of the shared dimension, to keep a block of b in cache. */
    private static final int MATMUL_DEPTH = 256;

    private Kernels() {
    }

    // ------------------------------------------------------------------ element-wise

    static void unary(UnaryOp op, float[] x, float[] y, int from, int to) {
        switch (op) {
            case NEG -> { for (int i = from; i < to; i++) y[i] = -x[i]; }
            case ABS -> { for (int i = from; i < to; i++) y[i] = Math.abs(x[i]); }
            case EXP -> { for (int i = from; i < to; i++) y[i] = (float) Math.exp(x[i]); }
            case LOG -> { for (int i = from; i < to; i++) y[i] = (float) Math.log(x[i]); }
            case SQRT -> { for (int i = from; i < to; i++) y[i] = (float) Math.sqrt(x[i]); }
            case SQUARE -> { for (int i = from; i < to; i++) y[i] = x[i] * x[i]; }
            case TANH -> { for (int i = from; i < to; i++) y[i] = (float) Math.tanh(x[i]); }
            case SIGMOID -> { for (int i = from; i < to; i++) y[i] = sigmoid(x[i]); }
            case RELU -> { for (int i = from; i < to; i++) y[i] = x[i] > 0f ? x[i] : 0f; }
            case STEP -> { for (int i = from; i < to; i++) y[i] = x[i] > 0f ? 1f : 0f; }
            case SIGN -> { for (int i = from; i < to; i++) y[i] = Math.signum(x[i]); }
        }
    }

    static float sigmoid(float x) {
        // Split by sign so exp never overflows.
        if (x >= 0f) {
            return (float) (1.0 / (1.0 + Math.exp(-x)));
        }
        double e = Math.exp(x);
        return (float) (e / (1.0 + e));
    }

    /** {@code z[i] = x[i] op y[i]}. */
    static void binary(BinaryOp op, float[] x, float[] y, float[] z, int from, int to) {
        switch (op) {
            case ADD -> { for (int i = from; i < to; i++) z[i] = x[i] + y[i]; }
            case SUB -> { for (int i = from; i < to; i++) z[i] = x[i] - y[i]; }
            case MUL -> { for (int i = from; i < to; i++) z[i] = x[i] * y[i]; }
            case DIV -> { for (int i = from; i < to; i++) z[i] = x[i] / y[i]; }
            default -> { for (int i = from; i < to; i++) z[i] = apply(op, x[i], y[i]); }
        }
    }

    /** {@code z[i] = x[i] op s}. */
    static void binaryScalarRight(BinaryOp op, float[] x, float s, float[] z, int from, int to) {
        switch (op) {
            case ADD -> { for (int i = from; i < to; i++) z[i] = x[i] + s; }
            case SUB -> { for (int i = from; i < to; i++) z[i] = x[i] - s; }
            case MUL -> { for (int i = from; i < to; i++) z[i] = x[i] * s; }
            case DIV -> { for (int i = from; i < to; i++) z[i] = x[i] / s; }
            default -> { for (int i = from; i < to; i++) z[i] = apply(op, x[i], s); }
        }
    }

    /** {@code z[i] = s op y[i]}. */
    static void binaryScalarLeft(BinaryOp op, float s, float[] y, float[] z, int from, int to) {
        switch (op) {
            case ADD -> { for (int i = from; i < to; i++) z[i] = s + y[i]; }
            case SUB -> { for (int i = from; i < to; i++) z[i] = s - y[i]; }
            case MUL -> { for (int i = from; i < to; i++) z[i] = s * y[i]; }
            case DIV -> { for (int i = from; i < to; i++) z[i] = s / y[i]; }
            default -> { for (int i = from; i < to; i++) z[i] = apply(op, s, y[i]); }
        }
    }

    /**
     * Rows {@code [from, to)} of {@code z[r, j] = x[r, j] op y[j]} with rows of length
     * {@code n}, the common "add a bias to every row" case.
     */
    static void binaryRows(BinaryOp op, float[] x, float[] y, float[] z, int n, int from, int to) {
        for (int r = from; r < to; r++) {
            int base = r * n;
            switch (op) {
                case ADD -> { for (int j = 0; j < n; j++) z[base + j] = x[base + j] + y[j]; }
                case SUB -> { for (int j = 0; j < n; j++) z[base + j] = x[base + j] - y[j]; }
                case MUL -> { for (int j = 0; j < n; j++) z[base + j] = x[base + j] * y[j]; }
                case DIV -> { for (int j = 0; j < n; j++) z[base + j] = x[base + j] / y[j]; }
                default -> { for (int j = 0; j < n; j++) z[base + j] = apply(op, x[base + j], y[j]); }
            }
        }
    }

    /** Output positions {@code [from, to)} of a broadcast binary op with any strides. */
    static void binaryStrided(BinaryOp op, float[] x, long[] aStrides, float[] y, long[] bStrides, int[] shape,
            float[] z, int from, int to) {
        int rank = shape.length;
        int[] index = new int[rank];
        long ia = 0;
        long ib = 0;
        long rest = from;
        for (int d = rank - 1; d >= 0; d--) {
            index[d] = (int) (rest % shape[d]);
            rest /= shape[d];
            ia += index[d] * aStrides[d];
            ib += index[d] * bStrides[d];
        }
        for (int i = from; i < to; i++) {
            z[i] = apply(op, x[(int) ia], y[(int) ib]);
            for (int d = rank - 1; d >= 0; d--) {
                index[d]++;
                ia += aStrides[d];
                ib += bStrides[d];
                if (index[d] < shape[d]) {
                    break;
                }
                ia -= aStrides[d] * shape[d];
                ib -= bStrides[d] * shape[d];
                index[d] = 0;
            }
        }
    }

    static float apply(BinaryOp op, float a, float b) {
        return switch (op) {
            case ADD -> a + b;
            case SUB -> a - b;
            case MUL -> a * b;
            case DIV -> a / b;
            case POW -> (float) Math.pow(a, b);
            case MAX -> Math.max(a, b);
            case MIN -> Math.min(a, b);
            case GREATER -> a > b ? 1f : 0f;
            case EQUAL -> a == b ? 1f : 0f;
        };
    }

    // ------------------------------------------------------------------ copy

    /** Side of the square tiles a strided copy works in, so both views stay in cache. */
    static final int COPY_TILE = 32;

    /**
     * Copies row tiles {@code [from, to)} between two strided views. The view is seen as
     * matrices over its last two axes (one row for rank 1); each matrix is split into
     * tiles of {@link #COPY_TILE} rows, numbered matrix by matrix.
     */
    static void copyTiles(float[] x, long inOffset, long[] inStrides, int[] shape, float[] z, long outOffset,
            long[] outStrides, int from, int to) {
        int rank = shape.length;
        int last = rank - 1;
        int rows = rank >= 2 ? shape[last - 1] : 1;
        int cols = shape[last];
        long inRow = rank >= 2 ? inStrides[last - 1] : 0;
        long outRow = rank >= 2 ? outStrides[last - 1] : 0;
        long inCol = inStrides[last];
        long outCol = outStrides[last];
        int tilesPerMatrix = Math.ceilDiv(rows, COPY_TILE);
        for (int tile = from; tile < to; tile++) {
            int matrix = tile / tilesPerMatrix;
            int r0 = (tile % tilesPerMatrix) * COPY_TILE;
            int r1 = Math.min(rows, r0 + COPY_TILE);
            long src = inOffset;
            long dst = outOffset;
            int rest = matrix;
            for (int d = rank - 3; d >= 0; d--) {
                int index = rest % shape[d];
                rest /= shape[d];
                src += index * inStrides[d];
                dst += index * outStrides[d];
            }
            if (inCol == 1 && outCol == 1) {
                for (int r = r0; r < r1; r++) {
                    System.arraycopy(x, (int) (src + r * inRow), z, (int) (dst + r * outRow), cols);
                }
                continue;
            }
            for (int c0 = 0; c0 < cols; c0 += COPY_TILE) {
                int c1 = Math.min(cols, c0 + COPY_TILE);
                for (int r = r0; r < r1; r++) {
                    long s = src + r * inRow + c0 * inCol;
                    long t = dst + r * outRow + c0 * outCol;
                    for (int c = c0; c < c1; c++, s += inCol, t += outCol) {
                        z[(int) t] = x[(int) s];
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ matmul

    /**
     * Rows {@code [from, to)} of {@code z = x * y}, with {@code x} of shape {@code [m, k]}
     * and {@code y} of shape {@code [k, n]}. {@code z} must be zero in those rows. Each
     * output value is summed over {@code p} in increasing order, whatever the blocking.
     */
    static void matmulRows(float[] x, float[] y, float[] z, int k, int n, int from, int to) {
        for (int p0 = 0; p0 < k; p0 += MATMUL_DEPTH) {
            int p1 = Math.min(k, p0 + MATMUL_DEPTH);
            int i = from;
            for (; i + MATMUL_ROWS <= to; i += MATMUL_ROWS) {
                int z0 = i * n;
                int z1 = z0 + n;
                int z2 = z1 + n;
                int z3 = z2 + n;
                int x0 = i * k;
                int x1 = x0 + k;
                int x2 = x1 + k;
                int x3 = x2 + k;
                for (int p = p0; p < p1; p++) {
                    float a0 = x[x0 + p];
                    float a1 = x[x1 + p];
                    float a2 = x[x2 + p];
                    float a3 = x[x3 + p];
                    int yRow = p * n;
                    for (int j = 0; j < n; j++) {
                        float b = y[yRow + j];
                        z[z0 + j] += a0 * b;
                        z[z1 + j] += a1 * b;
                        z[z2 + j] += a2 * b;
                        z[z3 + j] += a3 * b;
                    }
                }
            }
            for (; i < to; i++) {
                int zRow = i * n;
                int xRow = i * k;
                for (int p = p0; p < p1; p++) {
                    float a = x[xRow + p];
                    int yRow = p * n;
                    for (int j = 0; j < n; j++) {
                        z[zRow + j] += a * y[yRow + j];
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ reductions

    /**
     * Reduces block {@code o} (of {@code outer}) for inner positions {@code [from, to)}.
     * Values along the reduced axis are visited in increasing order.
     */
    static void reduce(ReduceOp op, float[] x, float[] y, int o, int len, int inner, int from, int to) {
        int base = o * len * inner;
        int outBase = o * inner;
        if (inner == 1) {
            y[outBase] = reduceContiguous(op, x, base, len);
            return;
        }
        int width = to - from;
        switch (op) {
            case SUM, MEAN -> {
                double[] acc = new double[width];
                for (int i = 0; i < len; i++) {
                    int row = base + i * inner + from;
                    for (int j = 0; j < width; j++) {
                        acc[j] += x[row + j];
                    }
                }
                for (int j = 0; j < width; j++) {
                    y[outBase + from + j] = (float) (op == ReduceOp.MEAN ? acc[j] / len : acc[j]);
                }
            }
            case MAX, MIN -> {
                boolean max = op == ReduceOp.MAX;
                float[] acc = new float[width];
                java.util.Arrays.fill(acc, max ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY);
                for (int i = 0; i < len; i++) {
                    int row = base + i * inner + from;
                    for (int j = 0; j < width; j++) {
                        acc[j] = max ? Math.max(acc[j], x[row + j]) : Math.min(acc[j], x[row + j]);
                    }
                }
                System.arraycopy(acc, 0, y, outBase + from, width);
            }
            case ARGMAX -> {
                for (int j = from; j < to; j++) {
                    int start = base + j;
                    int best = 0;
                    for (int i = 1; i < len; i++) {
                        if (x[start + i * inner] > x[start + best * inner]) {
                            best = i;
                        }
                    }
                    y[outBase + j] = best;
                }
            }
        }
    }

    private static float reduceContiguous(ReduceOp op, float[] x, int start, int len) {
        return switch (op) {
            case SUM -> (float) sum(x, start, len);
            case MEAN -> (float) (sum(x, start, len) / len);
            case MAX -> {
                float m = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < len; i++) m = Math.max(m, x[start + i]);
                yield m;
            }
            case MIN -> {
                float m = Float.POSITIVE_INFINITY;
                for (int i = 0; i < len; i++) m = Math.min(m, x[start + i]);
                yield m;
            }
            case ARGMAX -> {
                int best = 0;
                for (int i = 1; i < len; i++) {
                    if (x[start + i] > x[start + best]) best = i;
                }
                yield best;
            }
        };
    }

    private static double sum(float[] x, int start, int len) {
        // Accumulate in double so long sums stay accurate.
        double s = 0;
        for (int i = 0; i < len; i++) {
            s += x[start + i];
        }
        return s;
    }
}
