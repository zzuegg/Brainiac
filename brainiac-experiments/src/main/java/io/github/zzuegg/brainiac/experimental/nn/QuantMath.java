package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** Dense double-precision helpers shared by the post-training quantizers and editors, on core {@link Linalg}. */
final class QuantMath {

    private QuantMath() {
    }

    static double[][] matrix(Tensor t) {
        int rows = t.shape().dim(0);
        int cols = Math.toIntExact(t.shape().size() / rows);
        float[] f = t.toArray();
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m[r][c] = f[r * cols + c];
            }
        }
        return m;
    }

    static Tensor tensor(double[][] m, Device device) {
        int rows = m.length;
        int cols = m[0].length;
        float[] f = new float[rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                f[r * cols + c] = (float) m[r][c];
            }
        }
        return Tensor.of(Shape.of(rows, cols), f, device);
    }

    /** The largest level of symmetric {@code bits}-bit integers. */
    static int levels(int bits) {
        if (bits < 2 || bits > 16) {
            throw new IllegalArgumentException("bits must be in [2, 16], got " + bits);
        }
        return (1 << (bits - 1)) - 1;
    }

    /** Per-column (output channel) scales of {@code [in, out]} weights: {@code max|w| / levels}. */
    static double[] columnScales(double[][] w, int levels) {
        double[] s = new double[w[0].length];
        for (double[] row : w) {
            for (int c = 0; c < row.length; c++) {
                s[c] = Math.max(s[c], Math.abs(row[c]));
            }
        }
        for (int c = 0; c < s.length; c++) {
            s[c] = s[c] == 0 ? 1 : s[c] / levels;
        }
        return s;
    }

    static double quantize(double v, double scale, int levels) {
        return Math.max(-levels, Math.min(levels, Math.rint(v / scale))) * scale;
    }

    /** Round-to-nearest of {@code [in, out]} weights, per output channel. */
    static double[][] roundToNearest(double[][] w, int bits) {
        int levels = levels(bits);
        double[] s = columnScales(w, levels);
        double[][] q = new double[w.length][w[0].length];
        for (int r = 0; r < w.length; r++) {
            for (int c = 0; c < s.length; c++) {
                q[r][c] = quantize(w[r][c], s[c], levels);
            }
        }
        return q;
    }

    /** {@code x^T x} for {@code [n, d]} rows. */
    static double[][] gram(double[][] x) {
        int d = x[0].length;
        double[][] h = new double[d][d];
        for (double[] row : x) {
            for (int i = 0; i < d; i++) {
                double xi = row[i];
                if (xi == 0) {
                    continue;
                }
                for (int j = 0; j < d; j++) {
                    h[i][j] += xi * row[j];
                }
            }
        }
        return h;
    }

    /** A double-precision host tensor of a matrix, for core {@link Linalg}. */
    static Tensor float64(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        double[] flat = new double[rows * cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(a[r], 0, flat, r * cols, cols);
        }
        return Tensor.of(Shape.of(rows, cols), flat, DType.FLOAT64, Device.cpu());
    }

    /** The rows of a 2-D tensor, in double precision. */
    static double[][] rows(Tensor t) {
        int rows = t.shape().dim(0);
        int cols = t.shape().dim(1);
        double[] v = t.toDoubleArray();
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(v, r * cols, m[r], 0, cols);
        }
        return m;
    }

    /** Inverse of a square matrix ({@link Linalg#inverse}). */
    static double[][] inverse(double[][] a) {
        return rows(Linalg.inverse(float64(a)));
    }

    /** Solves {@code a x = b} for square {@code a} ({@link Linalg#solve}). */
    static double[] solve(double[][] a, double[] b) {
        Tensor rhs = Tensor.of(Shape.of(b.length, 1), b, DType.FLOAT64, Device.cpu());
        return Linalg.solve(float64(a), rhs).toDoubleArray();
    }

    /** The upper Cholesky factor {@code U} with {@code a = U^T U} ({@link Linalg#cholesky}). */
    static double[][] choleskyUpper(double[][] a) {
        return rows(Linalg.cholesky(float64(a)).transpose());
    }

    /** {@code ||x a - x b||_F^2 / n}: the mean squared output error of two weights on inputs x. */
    static double outputError(double[][] x, double[][] a, double[][] b) {
        double err = 0;
        int out = a[0].length;
        for (double[] row : x) {
            for (int c = 0; c < out; c++) {
                double s = 0;
                for (int i = 0; i < row.length; i++) {
                    s += row[i] * (a[i][c] - b[i][c]);
                }
                err += s * s;
            }
        }
        return err / (x.length * (double) out);
    }
}
