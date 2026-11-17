package io.github.zzuegg.brainiac.experimental.tensor;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The thin singular value decomposition {@code A = U diag(s) V^T} of an {@code [m, n]}
 * matrix, from the eigendecomposition of the smaller Gram matrix (core
 * {@link Linalg#eigh}), in double precision on the host. Core {@code Linalg} has no SVD yet. Enough for the weight matrices of small models: low-rank
 * factorization, PiSSA's principal adapters, rank selection by energy.
 *
 * @param u the left singular vectors {@code [m][k]}, {@code k = min(m, n)}
 * @param s the singular values, descending
 * @param v the right singular vectors {@code [n][k]}
 */
public record ThinSvd(double[][] u, double[] s, double[][] v) {

    /**
     * Decomposes a matrix.
     *
     * @param a the matrix {@code [m][n]}
     * @return the decomposition
     */
    public static ThinSvd of(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (n <= m) {
            double[][] g = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    double sum = 0;
                    for (int r = 0; r < m; r++) {
                        sum += a[r][i] * a[r][j];
                    }
                    g[i][j] = sum;
                    g[j][i] = sum;
                }
            }
            double[] flat = new double[n * n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(g[i], 0, flat, i * n, n);
            }
            Tensor[] e = Linalg.eigh(Tensor.of(Shape.of(n, n), flat, DType.FLOAT64, Device.cpu()));
            double[] values = e[0].toDoubleArray();
            double[] vectors = e[1].toDoubleArray();
            double[] s = new double[n];
            double[][] u = new double[m][n];
            double[][] v = new double[n][n];
            for (int k = 0; k < n; k++) {
                int c = n - 1 - k;   // eigh is ascending, singular values descend
                s[k] = Math.sqrt(Math.max(0, values[c]));
                for (int j = 0; j < n; j++) {
                    v[j][k] = vectors[j * n + c];
                }
                for (int r = 0; r < m; r++) {
                    double sum = 0;
                    for (int j = 0; j < n; j++) {
                        sum += a[r][j] * v[j][k];
                    }
                    u[r][k] = s[k] > 1e-12 ? sum / s[k] : 0;
                }
            }
            return new ThinSvd(u, s, v);
        }
        double[][] t = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                t[j][i] = a[i][j];
            }
        }
        ThinSvd tr = of(t);
        return new ThinSvd(tr.v(), tr.s(), tr.u());
    }

    /**
     * @param a a 2-D tensor
     * @return its decomposition
     */
    public static ThinSvd of(Tensor a) {
        return of(toMatrix(a));
    }

    /**
     * @param t a 2-D tensor
     * @return its values as rows
     */
    public static double[][] toMatrix(Tensor t) {
        int rows = t.shape().dim(0);
        int cols = t.shape().dim(1);
        double[] v = t.toDoubleArray();
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(v, i * cols, m[i], 0, cols);
        }
        return m;
    }

    /** @return the number of singular values */
    public int rank() {
        return s.length;
    }

    /**
     * The left factor of the best rank-{@code r} approximation, {@code U_r diag(s_r)^p}.
     *
     * @param r the rank
     * @param power the power of the singular values folded in (1 for all, 0.5 for a
     *     balanced split, 0 for none)
     * @return {@code [m, r]}
     */
    public Tensor left(int r, double power) {
        int m = u.length;
        float[] out = new float[m * r];
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < r; k++) {
                out[i * r + k] = (float) (u[i][k] * Math.pow(s[k], power));
            }
        }
        return Tensor.of(Shape.of(m, r), out);
    }

    /**
     * The right factor of the best rank-{@code r} approximation, {@code diag(s_r)^p V_r^T}.
     *
     * @param r the rank
     * @param power the power of the singular values folded in
     * @return {@code [r, n]}
     */
    public Tensor right(int r, double power) {
        int n = v.length;
        float[] out = new float[r * n];
        for (int k = 0; k < r; k++) {
            double f = Math.pow(s[k], power);
            for (int j = 0; j < n; j++) {
                out[k * n + j] = (float) (v[j][k] * f);
            }
        }
        return Tensor.of(Shape.of(r, n), out);
    }

    /**
     * @param energy the share of the squared singular values to keep, for example 0.9
     * @return the smallest rank that keeps it
     */
    public int rankFor(double energy) {
        double total = 0;
        for (double x : s) {
            total += x * x;
        }
        double cum = 0;
        for (int k = 0; k < s.length; k++) {
            cum += s[k] * s[k];
            if (cum >= energy * total) {
                return k + 1;
            }
        }
        return s.length;
    }
}
