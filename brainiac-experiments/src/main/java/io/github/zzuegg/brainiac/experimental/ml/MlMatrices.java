package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Host-side {@code double[][]} adapters over core {@link Linalg} for the classic estimators:
 * Cholesky factors with automatic jitter, solves through them, symmetric eigendecompositions
 * sorted by decreasing eigenvalue, and the small vector helpers (squared distances,
 * standardisation) they all need. The decompositions themselves run in core {@code Linalg} on
 * {@link DType#FLOAT64} tensors; this class only converts and orders.
 */
public final class MlMatrices {

    private MlMatrices() {
    }

    /**
     * A symmetric eigendecomposition, largest eigenvalue first.
     *
     * @param values the eigenvalues, descending
     * @param vectors {@code [n][n]}, one eigenvector per column, matching {@code values}
     */
    public record Eigen(double[] values, double[][] vectors) {

        /**
         * Returns one eigenvector.
         *
         * @param k the index, 0 for the largest eigenvalue
         * @return the column as a new array
         */
        public double[] vector(int k) {
            double[] v = new double[vectors.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = vectors[i][k];
            }
            return v;
        }
    }

    /**
     * Wraps a matrix as a double-precision CPU tensor.
     *
     * @param a the matrix {@code [m][n]}
     * @return the tensor {@code [m, n]}
     */
    public static Tensor tensor(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[] flat = new double[m * n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(a[i], 0, flat, i * n, n);
        }
        return Tensor.of(Shape.of(m, n), flat, DType.FLOAT64, Device.cpu());
    }

    /**
     * Reads a rank-2 tensor into a matrix.
     *
     * @param t the tensor {@code [m, n]}
     * @return the matrix
     */
    public static double[][] matrix(Tensor t) {
        int m = t.shape().dim(0);
        int n = t.shape().dim(1);
        double[] flat = t.toDoubleArray();
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(flat, i * n, a[i], 0, n);
        }
        return a;
    }

    /**
     * The lower Cholesky factor of a symmetric positive (semi)definite matrix. If the
     * factorisation fails, a growing multiple of the mean diagonal is added until it succeeds,
     * as GP libraries do.
     *
     * @param a the matrix {@code [n][n]}; not changed
     * @return the lower factor {@code l} with {@code l l^T ~ a}
     */
    public static double[][] cholesky(double[][] a) {
        int n = a.length;
        double mean = 0;
        for (int i = 0; i < n; i++) {
            mean += a[i][i];
        }
        mean = Math.max(mean / n, 1e-12);
        double jitter = 0;
        for (int attempt = 0; attempt < 12; attempt++) {
            double[][] b = a;
            if (jitter > 0) {
                b = new double[n][];
                for (int i = 0; i < n; i++) {
                    b[i] = a[i].clone();
                    b[i][i] += jitter;
                }
            }
            try {
                return matrix(Linalg.cholesky(tensor(b)));
            } catch (IllegalArgumentException notPd) {
                jitter = jitter == 0 ? 1e-10 * mean : jitter * 10;
            }
        }
        throw new IllegalArgumentException("matrix is not positive semidefinite, even with jitter");
    }

    /**
     * Solves {@code l x = b} or {@code l^T x = b} for a lower-triangular {@code l}.
     *
     * @param l the lower factor {@code [n][n]}
     * @param b right-hand sides {@code [n][k]}
     * @param transpose whether to solve with {@code l^T}
     * @return {@code x} {@code [n][k]}
     */
    public static double[][] triangularSolve(double[][] l, double[][] b, boolean transpose) {
        Tensor lt = tensor(l);
        if (transpose) {
            return matrix(Linalg.solveTriangular(lt.transpose(), tensor(b), false));
        }
        return matrix(Linalg.solveTriangular(lt, tensor(b), true));
    }

    /**
     * Solves {@code (l l^T) x = b} from a Cholesky factor.
     *
     * @param l the lower factor {@code [n][n]}
     * @param b right-hand sides {@code [n][k]}
     * @return {@code x}
     */
    public static double[][] choleskySolve(double[][] l, double[][] b) {
        return triangularSolve(l, triangularSolve(l, b, false), true);
    }

    /**
     * Solves {@code (l l^T) x = b} for one right-hand side, on the host.
     *
     * @param l the lower factor {@code [n][n]}
     * @param b the right-hand side {@code [n]}
     * @return {@code x}
     */
    public static double[] choleskySolve(double[][] l, double[] b) {
        int n = b.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double s = b[i];
            for (int k = 0; k < i; k++) {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = y[i];
            for (int k = i + 1; k < n; k++) {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        return x;
    }

    /**
     * Solves {@code l y = b} for one right-hand side.
     *
     * @param l the lower factor
     * @param b the right-hand side
     * @return {@code y}
     */
    public static double[] forwardSolve(double[][] l, double[] b) {
        int n = b.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double s = b[i];
            for (int k = 0; k < i; k++) {
                s -= l[i][k] * y[k];
            }
            y[i] = s / l[i][i];
        }
        return y;
    }

    /**
     * The inverse of {@code l l^T} from its Cholesky factor.
     *
     * @param l the lower factor
     * @return the inverse
     */
    public static double[][] choleskyInverse(double[][] l) {
        return choleskySolve(l, identity(l.length));
    }

    /**
     * {@code log det(l l^T)} from a Cholesky factor.
     *
     * @param l the lower factor
     * @return the log-determinant
     */
    public static double logDet(double[][] l) {
        double s = 0;
        for (int i = 0; i < l.length; i++) {
            s += Math.log(l[i][i]);
        }
        return 2 * s;
    }

    /**
     * The eigendecomposition of a symmetric matrix through core {@link Linalg#eigh}, reordered
     * so the largest eigenvalue comes first.
     *
     * @param a the symmetric matrix {@code [n][n]}
     * @return the eigenvalues (descending) and eigenvectors
     */
    public static Eigen eigh(double[][] a) {
        Tensor[] vd = Linalg.eigh(tensor(a));
        double[] asc = vd[0].toDoubleArray();
        double[][] vec = matrix(vd[1]);
        int n = asc.length;
        double[] values = new double[n];
        double[][] vectors = new double[n][n];
        for (int k = 0; k < n; k++) {
            values[k] = asc[n - 1 - k];
            for (int i = 0; i < n; i++) {
                vectors[i][k] = vec[i][n - 1 - k];
            }
        }
        return new Eigen(values, vectors);
    }

    /**
     * The identity matrix.
     *
     * @param n the size
     * @return {@code I_n}
     */
    public static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            a[i][i] = 1;
        }
        return a;
    }

    /**
     * The product {@code a b}.
     *
     * @param a {@code [m][k]}
     * @param b {@code [k][n]}
     * @return {@code [m][n]}
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int k = b.length;
        int n = b[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                double v = a[i][p];
                if (v == 0) {
                    continue;
                }
                double[] bp = b[p];
                double[] ci = c[i];
                for (int j = 0; j < n; j++) {
                    ci[j] += v * bp[j];
                }
            }
        }
        return c;
    }

    /**
     * The transpose.
     *
     * @param a {@code [m][n]}
     * @return {@code [n][m]}
     */
    public static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    /**
     * The squared Euclidean distance.
     *
     * @param a one point
     * @param b another
     * @return {@code |a - b|^2}
     */
    public static double squaredDistance(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    /**
     * All pairwise squared distances.
     *
     * @param x points {@code [n][d]}
     * @return {@code [n][n]}
     */
    public static double[][] squaredDistances(double[][] x) {
        int n = x.length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                d[i][j] = squaredDistance(x[i], x[j]);
                d[j][i] = d[i][j];
            }
        }
        return d;
    }

    /**
     * The dot product.
     *
     * @param a one vector
     * @param b another
     * @return {@code a . b}
     */
    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * Converts float rows to doubles.
     *
     * @param x rows
     * @return the same values in double precision
     */
    public static double[][] toDouble(float[][] x) {
        double[][] d = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            d[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                d[i][j] = x[i][j];
            }
        }
        return d;
    }

    /**
     * Converts double rows to floats.
     *
     * @param x rows
     * @return the same values in single precision
     */
    public static float[][] toFloat(double[][] x) {
        float[][] f = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            f[i] = new float[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                f[i][j] = (float) x[i][j];
            }
        }
        return f;
    }

    /**
     * The column means.
     *
     * @param x rows {@code [n][d]}
     * @return {@code [d]}
     */
    public static double[] columnMeans(double[][] x) {
        double[] m = new double[x[0].length];
        for (double[] row : x) {
            for (int j = 0; j < m.length; j++) {
                m[j] += row[j];
            }
        }
        for (int j = 0; j < m.length; j++) {
            m[j] /= x.length;
        }
        return m;
    }
}
