package io.github.zzuegg.brainiac.experimental.sbi;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/** Small array helpers shared by the simulation-based inference blocks. */
final class SbiArrays {

    private SbiArrays() {
    }

    static double[] mean(double[][] x) {
        double[] m = new double[x[0].length];
        for (double[] r : x) {
            for (int j = 0; j < m.length; j++) {
                m[j] += r[j] / x.length;
            }
        }
        return m;
    }

    static double[] sd(double[][] x, double[] mean) {
        double[] s = new double[mean.length];
        for (double[] r : x) {
            for (int j = 0; j < s.length; j++) {
                s[j] += (r[j] - mean[j]) * (r[j] - mean[j]) / x.length;
            }
        }
        for (int j = 0; j < s.length; j++) {
            s[j] = Math.sqrt(s[j]) + 1e-9;
        }
        return s;
    }

    static Tensor standardised(double[][] rows, double[] mean, double[] sd) {
        int d = mean.length;
        float[] v = new float[rows.length * d];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) ((rows[i][j] - mean[j]) / sd[j]);
            }
        }
        return Tensor.of(Shape.of(rows.length, d), v);
    }

    static Tensor standardisedRows(double[][] rows, int[] index, double[] mean, double[] sd) {
        int d = mean.length;
        float[] v = new float[index.length * d];
        for (int i = 0; i < index.length; i++) {
            double[] r = rows[index[i]];
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) ((r[j] - mean[j]) / sd[j]);
            }
        }
        return Tensor.of(Shape.of(index.length, d), v);
    }

    static Tensor repeated(double[] row, int n, double[] mean, double[] sd) {
        int d = mean.length;
        float[] v = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) ((row[j] - mean[j]) / sd[j]);
            }
        }
        return Tensor.of(Shape.of(n, d), v);
    }

    static double[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        double[] v = t.toDoubleArray();
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    static double[][] unstandardise(double[][] rows, double[] mean, double[] sd) {
        double[][] out = new double[rows.length][mean.length];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                out[i][j] = mean[j] + sd[j] * rows[i][j];
            }
        }
        return out;
    }

    static int[] batch(int n, int size, SplittableRandom r) {
        int[] b = new int[size];
        for (int i = 0; i < size; i++) {
            b[i] = r.nextInt(n);
        }
        return b;
    }

    static double[] concat(double[] a, double[] b) {
        double[] out = new double[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    static double[][] stack(double[][]... parts) {
        int n = 0;
        for (double[][] p : parts) {
            n += p.length;
        }
        double[][] out = new double[n][];
        int k = 0;
        for (double[][] p : parts) {
            for (double[] r : p) {
                out[k++] = r;
            }
        }
        return out;
    }

    static double logSumExp(double[] v) {
        double m = Double.NEGATIVE_INFINITY;
        for (double x : v) {
            m = Math.max(m, x);
        }
        if (m == Double.NEGATIVE_INFINITY) {
            return m;
        }
        double s = 0;
        for (double x : v) {
            s += Math.exp(x - m);
        }
        return m + Math.log(s);
    }
}
