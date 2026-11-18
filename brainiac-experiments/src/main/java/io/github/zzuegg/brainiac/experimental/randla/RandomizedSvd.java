package io.github.zzuegg.brainiac.experimental.randla;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Randomised SVD (Halko, Martinsson and Tropp 2011, <a href="https://arxiv.org/abs/0909.4061">
 * arXiv:0909.4061</a>, algorithms 4.4 and 5.1).
 *
 * <p>A rank-{@code k} approximation needs only the range of {@code A}: multiply by a random test
 * matrix {@code Omega} with {@code k + p} columns, orthonormalise the sample {@code Y = A Omega}
 * by QR, optionally sharpen it by {@code q} power iterations {@code (A A^T)^q} (each followed by
 * a QR to stay stable), then take the exact SVD of the small projection {@code B = Q^T A} and lift
 * its left factor back with {@code Q}. The cost is {@code O(m n k)} instead of
 * {@code O(m n min(m, n))}, and with fast spectral decay or two power iterations the error is
 * within a few percent of the optimal rank-{@code k} error.
 *
 * <p>Test matrices are Gaussian or sparse sign (each column has a few {@code +-1} entries), the
 * latter cheaper to apply.
 */
public final class RandomizedSvd {

    /** The random test matrix. */
    public enum TestMatrix {
        /** Independent standard normal entries. */
        GAUSSIAN,
        /** Each row has {@code min(8, columns)} random {@code +-1} entries. */
        SPARSE_SIGN
    }

    private RandomizedSvd() {
    }

    /**
     * A randomised truncated SVD.
     *
     * @param a the matrix {@code [m][n]}
     * @param rank the target rank {@code k}
     * @param oversample extra columns {@code p}, for example 10
     * @param powerIterations the power iterations {@code q}, for example 2
     * @param test the test-matrix family
     * @param seed the seed
     * @return the rank-{@code k} factors ({@code u [m][k]}, {@code s [k]}, {@code v [n][k]})
     */
    public static ThinSvd of(double[][] a, int rank, int oversample, int powerIterations, TestMatrix test,
            long seed) {
        Objects.requireNonNull(test, "test");
        int m = a.length;
        int n = a[0].length;
        int l = Math.min(Math.min(m, n), rank + oversample);
        double[][] omega = testMatrix(n, l, test, new SplittableRandom(seed));
        double[][] q = HouseholderQr.of(multiply(a, omega)).q();
        for (int it = 0; it < powerIterations; it++) {
            double[][] z = HouseholderQr.of(multiplyTransposed(a, q)).q();
            q = HouseholderQr.of(multiply(a, z)).q();
        }
        double[][] b = multiplyTransposed(q, a);   // Q^T A as [l][n]
        ThinSvd small = ThinSvd.of(b);
        int k = Math.min(rank, small.s().length);
        double[][] u = new double[m][k];
        for (int i = 0; i < m; i++) {
            for (int c = 0; c < k; c++) {
                double s = 0;
                for (int j = 0; j < l; j++) {
                    s += q[i][j] * small.u()[j][c];
                }
                u[i][c] = s;
            }
        }
        double[] s = new double[k];
        double[][] v = new double[n][k];
        for (int c = 0; c < k; c++) {
            s[c] = small.s()[c];
            for (int j = 0; j < n; j++) {
                v[j][c] = small.v()[j][c];
            }
        }
        return new ThinSvd(u, s, v);
    }

    /**
     * A test matrix {@code [rows][cols]}.
     *
     * @param rows rows
     * @param cols columns
     * @param test the family
     * @param r the random source
     * @return the matrix
     */
    public static double[][] testMatrix(int rows, int cols, TestMatrix test, SplittableRandom r) {
        double[][] omega = new double[rows][cols];
        if (test == TestMatrix.GAUSSIAN) {
            for (double[] row : omega) {
                for (int j = 0; j < cols; j++) {
                    row[j] = r.nextGaussian();
                }
            }
        } else {
            int nnz = Math.min(8, cols);
            for (double[] row : omega) {
                for (int t = 0; t < nnz; t++) {
                    row[r.nextInt(cols)] = r.nextBoolean() ? 1 : -1;
                }
            }
        }
        return omega;
    }

    /**
     * The Frobenius norm of {@code A - U diag(s) V^T}.
     *
     * @param a the matrix
     * @param svd the factors
     * @param rank how many leading triplets to use
     * @return the approximation error
     */
    public static double residual(double[][] a, ThinSvd svd, int rank) {
        double err = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                double s = a[i][j];
                for (int c = 0; c < rank; c++) {
                    s -= svd.u()[i][c] * svd.s()[c] * svd.v()[j][c];
                }
                err += s * s;
            }
        }
        return Math.sqrt(err);
    }

    /** {@code A B}. */
    static double[][] multiply(double[][] a, double[][] b) {
        int m = a.length;
        int k = b.length;
        int n = b[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++) {
            double[] ci = c[i];
            double[] ai = a[i];
            for (int t = 0; t < k; t++) {
                double v = ai[t];
                if (v == 0) {
                    continue;
                }
                double[] bt = b[t];
                for (int j = 0; j < n; j++) {
                    ci[j] += v * bt[j];
                }
            }
        }
        return c;
    }

    /** {@code A^T B} for {@code A [m][k]}, {@code B [m][n]}. */
    static double[][] multiplyTransposed(double[][] a, double[][] b) {
        int m = a.length;
        int k = a[0].length;
        int n = b[0].length;
        double[][] c = new double[k][n];
        for (int i = 0; i < m; i++) {
            double[] ai = a[i];
            double[] bi = b[i];
            for (int t = 0; t < k; t++) {
                double v = ai[t];
                double[] ct = c[t];
                for (int j = 0; j < n; j++) {
                    ct[j] += v * bi[j];
                }
            }
        }
        return c;
    }
}
