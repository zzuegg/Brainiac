package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Strassen's algorithm (1969): seven block products instead of eight per 2x2 split, applied
 * recursively down to a cut-over size, below which the core {@link Tensor#matmul} takes over.
 * Square matrices with a power-of-two side times the cut-over.
 */
public final class StrassenMatmul {

    private final int cutover;

    /** @param cutover the side at or below which the core matmul is used */
    public StrassenMatmul(int cutover) {
        this.cutover = cutover;
    }

    /**
     * @param a {@code [n, n]}
     * @param b {@code [n, n]}
     * @return {@code a b}
     */
    public Tensor multiply(Tensor a, Tensor b) {
        int n = a.shape().dim(0);
        return Tensor.of(Shape.of(n, n), multiply(a.toArray(), b.toArray(), n));
    }

    private float[] multiply(float[] a, float[] b, int n) {
        if (n <= cutover || n % 2 != 0) {
            return Tensor.of(Shape.of(n, n), a).matmul(Tensor.of(Shape.of(n, n), b)).toArray();
        }
        int h = n / 2;
        float[] a11 = quad(a, n, 0, 0);
        float[] a12 = quad(a, n, 0, h);
        float[] a21 = quad(a, n, h, 0);
        float[] a22 = quad(a, n, h, h);
        float[] b11 = quad(b, n, 0, 0);
        float[] b12 = quad(b, n, 0, h);
        float[] b21 = quad(b, n, h, 0);
        float[] b22 = quad(b, n, h, h);
        float[] m1 = multiply(add(a11, a22, 1), add(b11, b22, 1), h);
        float[] m2 = multiply(add(a21, a22, 1), b11, h);
        float[] m3 = multiply(a11, add(b12, b22, -1), h);
        float[] m4 = multiply(a22, add(b21, b11, -1), h);
        float[] m5 = multiply(add(a11, a12, 1), b22, h);
        float[] m6 = multiply(add(a21, a11, -1), add(b11, b12, 1), h);
        float[] m7 = multiply(add(a12, a22, -1), add(b21, b22, 1), h);
        float[] c = new float[n * n];
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < h; j++) {
                int q = i * h + j;
                c[i * n + j] = m1[q] + m4[q] - m5[q] + m7[q];
                c[i * n + j + h] = m3[q] + m5[q];
                c[(i + h) * n + j] = m2[q] + m4[q];
                c[(i + h) * n + j + h] = m1[q] - m2[q] + m3[q] + m6[q];
            }
        }
        return c;
    }

    private static float[] quad(float[] x, int n, int r0, int c0) {
        int h = n / 2;
        float[] q = new float[h * h];
        for (int i = 0; i < h; i++) {
            System.arraycopy(x, (r0 + i) * n + c0, q, i * h, h);
        }
        return q;
    }

    private static float[] add(float[] x, float[] y, float sign) {
        float[] z = new float[x.length];
        for (int i = 0; i < z.length; i++) {
            z[i] = x[i] + sign * y[i];
        }
        return z;
    }

    @Override
    public String toString() {
        return "StrassenMatmul(cutover " + cutover + ")";
    }
}
