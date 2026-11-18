package io.github.zzuegg.brainiac.experimental.ml;

/**
 * A positive-definite kernel {@code k(a, b)}: the similarity a kernel machine (SVM, kernel
 * PCA, spectral clustering, Gaussian process) works with instead of explicit features.
 * {@link StationaryKernel} adds the RBF and Matérn families with hyperparameter gradients for
 * Gaussian processes.
 */
@FunctionalInterface
public interface KernelFunction {

    /**
     * Evaluates the kernel.
     *
     * @param a one point
     * @param b another point
     * @return {@code k(a, b)}
     */
    double value(double[] a, double[] b);

    /**
     * The kernel matrix between two sets of points.
     *
     * @param a points {@code [m][d]}
     * @param b points {@code [n][d]}
     * @return {@code [m][n]}
     */
    default double[][] matrix(double[][] a, double[][] b) {
        double[][] k = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                k[i][j] = value(a[i], b[j]);
            }
        }
        return k;
    }

    /**
     * The symmetric kernel (Gram) matrix of one set of points.
     *
     * @param x points {@code [n][d]}
     * @return {@code [n][n]}
     */
    default double[][] gram(double[][] x) {
        int n = x.length;
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                k[i][j] = value(x[i], x[j]);
                k[j][i] = k[i][j];
            }
        }
        return k;
    }

    /**
     * The Gaussian (RBF) kernel {@code exp(-gamma |a - b|^2)}.
     *
     * @param gamma the inverse width, positive
     * @return the kernel
     */
    static KernelFunction rbf(double gamma) {
        return (a, b) -> Math.exp(-gamma * MlMatrices.squaredDistance(a, b));
    }

    /**
     * The polynomial kernel {@code (gamma a . b + coef0)^degree}.
     *
     * @param degree the degree, at least 1
     * @param gamma the scale of the dot product
     * @param coef0 the offset
     * @return the kernel
     */
    static KernelFunction polynomial(int degree, double gamma, double coef0) {
        return (a, b) -> Math.pow(gamma * MlMatrices.dot(a, b) + coef0, degree);
    }

    /**
     * The linear kernel {@code a . b}.
     *
     * @return the kernel
     */
    static KernelFunction linear() {
        return MlMatrices::dot;
    }
}
