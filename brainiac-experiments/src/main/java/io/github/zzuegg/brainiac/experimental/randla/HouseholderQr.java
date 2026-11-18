package io.github.zzuegg.brainiac.experimental.randla;

/**
 * The thin QR decomposition {@code A = Q R} of a tall matrix by Householder reflections, in
 * double precision: the orthonormalisation step of randomised range finders and the factor of a
 * sketch that preconditions least squares. Householder QR is backward stable, unlike classical
 * Gram-Schmidt, which matters when the columns are nearly dependent (power iterations).
 *
 * @param q the orthonormal factor {@code [m][n]}
 * @param r the upper-triangular factor {@code [n][n]}
 */
public record HouseholderQr(double[][] q, double[][] r) {

    /**
     * Decomposes a matrix with at least as many rows as columns.
     *
     * @param a the matrix {@code [m][n]}, {@code m >= n}; not modified
     * @return the thin factors
     */
    public static HouseholderQr of(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (m < n) {
            throw new IllegalArgumentException("QR needs m >= n: " + m + " x " + n);
        }
        // work column-major for cache-friendly reflections
        double[][] col = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                col[j][i] = a[i][j];
            }
        }
        double[][] v = new double[n][];
        double[] diag = new double[n];
        for (int k = 0; k < n; k++) {
            double[] x = col[k];
            double norm = 0;
            for (int i = k; i < m; i++) {
                norm += x[i] * x[i];
            }
            norm = Math.sqrt(norm);
            double alpha = x[k] > 0 ? -norm : norm;
            double[] vk = new double[m];
            for (int i = k; i < m; i++) {
                vk[i] = x[i];
            }
            vk[k] -= alpha;
            double vn = 0;
            for (int i = k; i < m; i++) {
                vn += vk[i] * vk[i];
            }
            if (vn > 0) {
                for (int j = k; j < n; j++) {
                    double[] c = col[j];
                    double dot = 0;
                    for (int i = k; i < m; i++) {
                        dot += vk[i] * c[i];
                    }
                    double f = 2 * dot / vn;
                    for (int i = k; i < m; i++) {
                        c[i] -= f * vk[i];
                    }
                }
            }
            v[k] = vk;
            diag[k] = vn;
        }
        double[][] r = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                r[i][j] = col[j][i];
            }
        }
        // Q = H_0 H_1 ... H_{n-1} applied to the first n columns of the identity
        double[][] qc = new double[n][m];
        for (int j = 0; j < n; j++) {
            qc[j][j] = 1;
            for (int k = n - 1; k >= 0; k--) {
                if (diag[k] == 0) {
                    continue;
                }
                double[] vk = v[k];
                double[] c = qc[j];
                double dot = 0;
                for (int i = k; i < m; i++) {
                    dot += vk[i] * c[i];
                }
                double f = 2 * dot / diag[k];
                for (int i = k; i < m; i++) {
                    c[i] -= f * vk[i];
                }
            }
        }
        double[][] q = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                q[i][j] = qc[j][i];
            }
        }
        return new HouseholderQr(q, r);
    }

    /**
     * Solves {@code R x = b} by back substitution.
     *
     * @param r an upper-triangular matrix
     * @param b the right-hand side
     * @return {@code x}
     */
    public static double[] solveUpper(double[][] r, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = b[i];
            for (int j = i + 1; j < n; j++) {
                s -= r[i][j] * x[j];
            }
            x[i] = s / r[i][i];
        }
        return x;
    }

    /**
     * Solves {@code R^T x = b} by forward substitution.
     *
     * @param r an upper-triangular matrix
     * @param b the right-hand side
     * @return {@code x}
     */
    public static double[] solveUpperTransposed(double[][] r, double[] b) {
        int n = b.length;
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            double s = b[i];
            for (int j = 0; j < i; j++) {
                s -= r[j][i] * x[j];
            }
            x[i] = s / r[i][i];
        }
        return x;
    }

    /**
     * The least-squares solution {@code argmin |A x - b|} from this factorisation:
     * {@code x = R^-1 Q^T b}.
     *
     * @param b the right-hand side {@code [m]}
     * @return {@code x}
     */
    public double[] solve(double[] b) {
        int m = q.length;
        int n = r.length;
        double[] qtb = new double[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                qtb[j] += q[i][j] * b[i];
            }
        }
        return solveUpper(r, qtb);
    }
}
