package io.github.zzuegg.brainiac.experimental.tensor;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The eigendecomposition {@code A = V diag(values) V^T} of a symmetric matrix by cyclic
 * Jacobi rotations, in double precision on the host, plus the matrix functions that
 * second-order optimizers need: inverse roots for Shampoo, eigenbases for SOAP and the
 * leading singular subspace for GaLore. Fine for the matrix sizes of preconditioners
 * (up to a few hundred rows); a backend kernel would be the core version.
 *
 * @param values the eigenvalues, descending
 * @param vectors the eigenvectors as columns, {@code vectors[row][k]} for eigenvalue {@code k}
 */
public record SymmetricEigen(double[] values, double[][] vectors) {

    /**
     * Decomposes a symmetric matrix.
     *
     * @param a the matrix; only read, treated as symmetric
     * @return the decomposition, eigenvalues in descending order
     */
    public static SymmetricEigen of(double[][] a) {
        int n = a.length;
        double[][] m = new double[n][];
        for (int i = 0; i < n; i++) {
            m[i] = a[i].clone();
        }
        double[][] v = new double[n][n];
        for (int i = 0; i < n; i++) {
            v[i][i] = 1;
        }
        for (int sweep = 0; sweep < 60; sweep++) {
            double off = 0;
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    off += m[p][q] * m[p][q];
                }
            }
            if (off < 1e-22) {
                break;
            }
            for (int p = 0; p < n; p++) {
                for (int q = p + 1; q < n; q++) {
                    if (Math.abs(m[p][q]) < 1e-300) {
                        continue;
                    }
                    double theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
                    double t = Math.signum(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
                    if (theta == 0) {
                        t = 1;
                    }
                    double c = 1 / Math.sqrt(t * t + 1);
                    double s = t * c;
                    for (int k = 0; k < n; k++) {
                        double mkp = m[k][p];
                        double mkq = m[k][q];
                        m[k][p] = c * mkp - s * mkq;
                        m[k][q] = s * mkp + c * mkq;
                    }
                    for (int k = 0; k < n; k++) {
                        double mpk = m[p][k];
                        double mqk = m[q][k];
                        m[p][k] = c * mpk - s * mqk;
                        m[q][k] = s * mpk + c * mqk;
                    }
                    for (int k = 0; k < n; k++) {
                        double vkp = v[k][p];
                        double vkq = v[k][q];
                        v[k][p] = c * vkp - s * vkq;
                        v[k][q] = s * vkp + c * vkq;
                    }
                }
            }
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (x, y) -> Double.compare(m[y][y], m[x][x]));
        double[] values = new double[n];
        double[][] vectors = new double[n][n];
        for (int k = 0; k < n; k++) {
            values[k] = m[order[k]][order[k]];
            for (int i = 0; i < n; i++) {
                vectors[i][k] = v[i][order[k]];
            }
        }
        return new SymmetricEigen(values, vectors);
    }

    /**
     * @param t a symmetric {@code [n, n]} tensor
     * @return its decomposition
     */
    public static SymmetricEigen of(Tensor t) {
        return of(toMatrix(t));
    }

    /**
     * {@code V diag((max(lambda, 0) + epsilon)^power) V^T}, for example {@code power = -1/4}
     * for Shampoo's preconditioner.
     *
     * @param power the exponent
     * @param epsilon added to every eigenvalue
     * @return the matrix function as a tensor
     */
    public Tensor power(double power, double epsilon) {
        int n = values.length;
        float[] out = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) {
                    s += vectors[i][k] * Math.pow(Math.max(values[k], 0) + epsilon, power) * vectors[j][k];
                }
                out[i * n + j] = (float) s;
            }
        }
        return Tensor.of(Shape.of(n, n), out);
    }

    /**
     * The leading eigenvectors as the columns of a tensor.
     *
     * @param k how many, at most {@code n}
     * @return {@code [n, k]}
     */
    public Tensor leading(int k) {
        int n = values.length;
        float[] out = new float[n * k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                out[i * k + j] = (float) vectors[i][j];
            }
        }
        return Tensor.of(Shape.of(n, k), out);
    }

    /** @return all eigenvectors as the columns of an {@code [n, n]} tensor */
    public Tensor basis() {
        return leading(values.length);
    }

    /**
     * @param t a 2-D tensor
     * @return its values as rows
     */
    public static double[][] toMatrix(Tensor t) {
        int rows = t.shape().dim(0);
        int cols = t.shape().dim(1);
        float[] v = t.toArray();
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = v[i * cols + j];
            }
        }
        return m;
    }
}
