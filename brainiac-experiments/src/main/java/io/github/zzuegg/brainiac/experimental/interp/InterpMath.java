package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/** Small dense double-precision helpers shared by the interpretability tools. */
final class InterpMath {

    private InterpMath() {
    }

    static double[][] toDouble(float[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = x[i][j];
            }
        }
        return out;
    }

    static float[][] toFloat(double[][] x) {
        float[][] out = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new float[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = (float) x[i][j];
            }
        }
        return out;
    }

    static double[] mean(double[][] x) {
        double[] m = new double[x[0].length];
        for (double[] row : x) {
            for (int j = 0; j < m.length; j++) {
                m[j] += row[j] / x.length;
            }
        }
        return m;
    }

    static double[][] centre(double[][] x) {
        double[] m = mean(x);
        double[][] out = new double[x.length][m.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < m.length; j++) {
                out[i][j] = x[i][j] - m[j];
            }
        }
        return out;
    }

    static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    static double[][] matmul(double[][] a, double[][] b) {
        int n = a.length;
        int k = b.length;
        int m = b[0].length;
        double[][] c = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int p = 0; p < k; p++) {
                double v = a[i][p];
                if (v == 0) {
                    continue;
                }
                for (int j = 0; j < m; j++) {
                    c[i][j] += v * b[p][j];
                }
            }
        }
        return c;
    }

    /** {@code a^T b / divisor}. */
    static double[][] crossProduct(double[][] a, double[][] b, double divisor) {
        int p = a[0].length;
        int q = b[0].length;
        double[][] c = new double[p][q];
        for (int i = 0; i < a.length; i++) {
            for (int x = 0; x < p; x++) {
                double v = a[i][x] / divisor;
                for (int y = 0; y < q; y++) {
                    c[x][y] += v * b[i][y];
                }
            }
        }
        return c;
    }

    static double[] matvec(double[][] a, double[] v) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double s = 0;
            for (int j = 0; j < v.length; j++) {
                s += a[i][j] * v[j];
            }
            out[i] = s;
        }
        return out;
    }

    static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    static Tensor tensor(double[][] a) {
        double[] flat = new double[a.length * a[0].length];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, flat, i * a[0].length, a[0].length);
        }
        return Tensor.of(Shape.of(a.length, a[0].length), flat, DType.FLOAT64, Device.cpu());
    }

    static double[][] matrix(Tensor t) {
        int r = t.shape().dim(0);
        int c = t.shape().dim(1);
        double[] flat = t.toDoubleArray();
        double[][] out = new double[r][c];
        for (int i = 0; i < r; i++) {
            System.arraycopy(flat, i * c, out[i], 0, c);
        }
        return out;
    }

    /**
     * A symmetric power {@code V diag(lambda^p) V^T} of a positive semi-definite matrix,
     * with eigenvalues below {@code floor} (relative to the largest) treated as zero.
     */
    static double[][] symmetricPower(double[][] a, double power, double floor) {
        Tensor[] e = Linalg.eigh(tensor(a));
        double[] values = e[0].toDoubleArray();
        double[][] v = matrix(e[1]);
        int n = values.length;
        double max = 0;
        for (double x : values) {
            max = Math.max(max, x);
        }
        double[][] out = new double[n][n];
        for (int k = 0; k < n; k++) {
            if (values[k] <= floor * Math.max(max, 1e-300)) {
                continue;
            }
            double s = Math.pow(values[k], power);
            for (int i = 0; i < n; i++) {
                double vi = v[i][k] * s;
                for (int j = 0; j < n; j++) {
                    out[i][j] += vi * v[j][k];
                }
            }
        }
        return out;
    }

    /** An orthonormal basis (as columns) of the column space of {@code a}. */
    static double[][] columnBasis(double[][] a, double tolerance) {
        ThinSvd svd = ThinSvd.of(a);
        double max = svd.s().length == 0 ? 0 : svd.s()[0];
        int r = 0;
        while (r < svd.s().length && svd.s()[r] > tolerance * Math.max(max, 1e-300)) {
            r++;
        }
        double[][] u = new double[a.length][r];
        for (int i = 0; i < a.length; i++) {
            for (int k = 0; k < r; k++) {
                u[i][k] = svd.u()[i][k];
            }
        }
        return u;
    }

    static double[][] identity(int n) {
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            m[i][i] = 1;
        }
        return m;
    }

    static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /** Spearman rank correlation. */
    static double spearman(double[] a, double[] b) {
        return pearson(ranks(a), ranks(b));
    }

    static double pearson(double[] a, double[] b) {
        int n = a.length;
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < n; i++) {
            ma += a[i] / n;
            mb += b[i] / n;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < n; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return saa == 0 || sbb == 0 ? 0 : sab / Math.sqrt(saa * sbb);
    }

    static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (x, y) -> Double.compare(v[x], v[y]));
        double[] r = new double[v.length];
        int i = 0;
        while (i < v.length) {
            int j = i;
            while (j + 1 < v.length && v[idx[j + 1]] == v[idx[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0;
            for (int k = i; k <= j; k++) {
                r[idx[k]] = rank;
            }
            i = j + 1;
        }
        return r;
    }

    /** ROC-AUC of scores against binary labels (ties count half). */
    static double rocAuc(double[] scores, boolean[] positive) {
        double[] r = ranks(scores);
        double pos = 0;
        double sum = 0;
        for (int i = 0; i < scores.length; i++) {
            if (positive[i]) {
                pos++;
                sum += r[i] + 1;
            }
        }
        double neg = scores.length - pos;
        if (pos == 0 || neg == 0) {
            return 0.5;
        }
        return (sum - pos * (pos + 1) / 2) / (pos * neg);
    }
}
