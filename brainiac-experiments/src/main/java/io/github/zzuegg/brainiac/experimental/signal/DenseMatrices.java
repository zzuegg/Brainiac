package io.github.zzuegg.brainiac.experimental.signal;

/**
 * Small dense double-precision matrix helpers for the inner loops of sparse recovery, MIMO
 * detection and dictionary learning, where the problems are tiny (tens of rows) and solved
 * thousands of times, so the per-call overhead of tensor operations (core {@code Linalg})
 * would dominate: products, Gram matrices, Cholesky solves with a ridge, and a thin QR by
 * modified Gram-Schmidt.
 */
public final class DenseMatrices {

    private DenseMatrices() {
    }

    /**
     * {@code a b}.
     *
     * @param a {@code [m][k]}
     * @param b {@code [k][n]}
     * @return {@code [m][n]}
     */
    public static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int k = b.length;
        int n = b[0].length;
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int t = 0; t < k; t++) {
                double v = a[i][t];
                if (v == 0) {
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    out[i][j] += v * b[t][j];
                }
            }
        }
        return out;
    }

    /**
     * {@code a x}.
     *
     * @param a {@code [m][n]}
     * @param x {@code [n]}
     * @return {@code [m]}
     */
    public static double[] multiply(double[][] a, double[] x) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double s = 0;
            for (int j = 0; j < x.length; j++) {
                s += a[i][j] * x[j];
            }
            out[i] = s;
        }
        return out;
    }

    /**
     * {@code a^T x}.
     *
     * @param a {@code [m][n]}
     * @param x {@code [m]}
     * @return {@code [n]}
     */
    public static double[] multiplyTransposed(double[][] a, double[] x) {
        int n = a[0].length;
        double[] out = new double[n];
        for (int i = 0; i < a.length; i++) {
            double xi = x[i];
            if (xi == 0) {
                continue;
            }
            for (int j = 0; j < n; j++) {
                out[j] += a[i][j] * xi;
            }
        }
        return out;
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
     * {@code a^T a}.
     *
     * @param a {@code [m][n]}
     * @return {@code [n][n]}
     */
    public static double[][] gram(double[][] a) {
        int n = a[0].length;
        double[][] g = new double[n][n];
        for (double[] row : a) {
            for (int i = 0; i < n; i++) {
                if (row[i] == 0) {
                    continue;
                }
                for (int j = i; j < n; j++) {
                    g[i][j] += row[i] * row[j];
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                g[i][j] = g[j][i];
            }
        }
        return g;
    }

    /**
     * Solves {@code (g + ridge I) x = b} for a symmetric positive (semi)definite {@code g} by
     * Cholesky.
     *
     * @param g the matrix
     * @param b the right-hand side
     * @param ridge added to the diagonal
     * @return the solution
     */
    public static double[] solveSpd(double[][] g, double[] b, double ridge) {
        int n = b.length;
        double[][] l = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = g[i][j] + (i == j ? ridge : 0);
                for (int k = 0; k < j; k++) {
                    s -= l[i][k] * l[j][k];
                }
                l[i][j] = i == j ? Math.sqrt(Math.max(s, 1e-14)) : s / l[j][j];
            }
        }
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
     * Least squares on a subset of columns: {@code argmin ||y - A_S x_S||}.
     *
     * @param a {@code [m][n]}
     * @param y {@code [m]}
     * @param support the column indices
     * @param count how many of them are used
     * @return the coefficients of the used columns
     */
    public static double[] leastSquares(double[][] a, double[] y, int[] support, int count) {
        double[][] g = new double[count][count];
        double[] b = new double[count];
        for (double[] row : a) {
            int idx = 0;
            for (int i = 0; i < count; i++) {
                idx = support[i];
                if (row[idx] == 0) {
                    continue;
                }
                for (int j = 0; j < count; j++) {
                    g[i][j] += row[idx] * row[support[j]];
                }
            }
        }
        for (int r = 0; r < a.length; r++) {
            for (int i = 0; i < count; i++) {
                b[i] += a[r][support[i]] * y[r];
            }
        }
        return solveSpd(g, b, 1e-10);
    }

    /**
     * Thin QR by modified Gram-Schmidt.
     *
     * @param a {@code [m][n]} with {@code m >= n} and full column rank
     * @return {@code {Q [m][n], R [n][n]}}
     */
    public static double[][][] qr(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] q = new double[m][n];
        double[][] r = new double[n][n];
        double[][] v = new double[n][m];
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                v[j][i] = a[i][j];
            }
        }
        for (int j = 0; j < n; j++) {
            double norm = 0;
            for (int i = 0; i < m; i++) {
                norm += v[j][i] * v[j][i];
            }
            norm = Math.sqrt(norm);
            r[j][j] = norm;
            for (int i = 0; i < m; i++) {
                q[i][j] = v[j][i] / norm;
            }
            for (int k = j + 1; k < n; k++) {
                double dot = 0;
                for (int i = 0; i < m; i++) {
                    dot += q[i][j] * v[k][i];
                }
                r[j][k] = dot;
                for (int i = 0; i < m; i++) {
                    v[k][i] -= dot * q[i][j];
                }
            }
        }
        return new double[][][] {q, r};
    }

    /**
     * The Euclidean norm.
     *
     * @param x a vector
     * @return {@code ||x||}
     */
    public static double norm(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v * v;
        }
        return Math.sqrt(s);
    }
}
