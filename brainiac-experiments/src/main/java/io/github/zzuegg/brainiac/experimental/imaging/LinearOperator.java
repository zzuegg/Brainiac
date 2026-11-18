package io.github.zzuegg.brainiac.experimental.imaging;

/**
 * A real linear forward model {@code y = A x} of an imaging system together with its adjoint,
 * on flat double arrays: the Radon transform, a masked Fourier transform, a blur, a mask, a set
 * of single-pixel patterns. Iterative solvers ({@link PrimalDual}, {@link ParameterChoice},
 * {@link PnpFlow}, {@link PosteriorSamplers}) only need these two products.
 */
public interface LinearOperator {

    /** @return the length of {@code x} */
    int inputSize();

    /** @return the length of {@code y} */
    int outputSize();

    /**
     * The forward product.
     *
     * @param x the image
     * @return {@code A x}
     */
    double[] apply(double[] x);

    /**
     * The adjoint product.
     *
     * @param y a measurement-space vector
     * @return {@code A^T y}
     */
    double[] adjoint(double[] y);

    /**
     * An upper estimate of {@code |A|^2} by power iteration on {@code A^T A}.
     *
     * @param iterations the iterations
     * @return the estimate, slightly inflated
     */
    default double normSquared(int iterations) {
        double[] x = new double[inputSize()];
        java.util.SplittableRandom r = new java.util.SplittableRandom(7);
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextDouble() - 0.5;
        }
        double lambda = 1;
        for (int it = 0; it < iterations; it++) {
            double n = ImageMath.norm(x);
            for (int i = 0; i < x.length; i++) {
                x[i] /= n;
            }
            x = adjoint(apply(x));
            lambda = ImageMath.norm(x);
        }
        return lambda * 1.02;
    }

    /**
     * This operator times a constant.
     *
     * @param scale the constant
     * @return {@code scale A}
     */
    default LinearOperator scaled(double scale) {
        LinearOperator self = this;
        return new LinearOperator() {
            @Override
            public int inputSize() {
                return self.inputSize();
            }

            @Override
            public int outputSize() {
                return self.outputSize();
            }

            @Override
            public double[] apply(double[] x) {
                double[] y = self.apply(x);
                for (int i = 0; i < y.length; i++) {
                    y[i] *= scale;
                }
                return y;
            }

            @Override
            public double[] adjoint(double[] y) {
                double[] x = self.adjoint(y);
                for (int i = 0; i < x.length; i++) {
                    x[i] *= scale;
                }
                return x;
            }
        };
    }

    /**
     * The identity on {@code n} values.
     *
     * @param n the size
     * @return the operator
     */
    static LinearOperator identity(int n) {
        return mask(null, n);
    }

    /**
     * Keeps the pixels where {@code keep} is true and zeroes the rest (inpainting); the output
     * has the input's size.
     *
     * @param keep the kept pixels, or {@code null} for all
     * @param n the size
     * @return the operator
     */
    static LinearOperator mask(boolean[] keep, int n) {
        return new LinearOperator() {
            @Override
            public int inputSize() {
                return n;
            }

            @Override
            public int outputSize() {
                return n;
            }

            @Override
            public double[] apply(double[] x) {
                double[] y = x.clone();
                if (keep != null) {
                    for (int i = 0; i < n; i++) {
                        if (!keep[i]) {
                            y[i] = 0;
                        }
                    }
                }
                return y;
            }

            @Override
            public double[] adjoint(double[] y) {
                return apply(y);
            }
        };
    }

    /**
     * A circular convolution with a centred square kernel on a {@code side x side} grid.
     *
     * @param kernel the kernel, {@code k x k}
     * @param k its side
     * @param side the image side
     * @return the operator
     */
    static LinearOperator blur(double[] kernel, int k, int side) {
        double[] flipped = new double[kernel.length];
        for (int i = 0; i < kernel.length; i++) {
            flipped[kernel.length - 1 - i] = kernel[i];
        }
        return new LinearOperator() {
            @Override
            public int inputSize() {
                return side * side;
            }

            @Override
            public int outputSize() {
                return side * side;
            }

            @Override
            public double[] apply(double[] x) {
                return ImageMath.convolve(x, side, kernel, k);
            }

            @Override
            public double[] adjoint(double[] y) {
                return ImageMath.convolve(y, side, flipped, k);
            }
        };
    }

    /**
     * Averages {@code factor x factor} blocks (super-resolution); the adjoint spreads each value
     * back over its block.
     *
     * @param side the high-resolution side
     * @param factor the factor
     * @return the operator
     */
    static LinearOperator downsample(int side, int factor) {
        int low = side / factor;
        return new LinearOperator() {
            @Override
            public int inputSize() {
                return side * side;
            }

            @Override
            public int outputSize() {
                return low * low;
            }

            @Override
            public double[] apply(double[] x) {
                double[] y = new double[low * low];
                double w = 1.0 / (factor * factor);
                for (int r = 0; r < side; r++) {
                    for (int c = 0; c < side; c++) {
                        y[(r / factor) * low + c / factor] += w * x[r * side + c];
                    }
                }
                return y;
            }

            @Override
            public double[] adjoint(double[] y) {
                double[] x = new double[side * side];
                double w = 1.0 / (factor * factor);
                for (int r = 0; r < side; r++) {
                    for (int c = 0; c < side; c++) {
                        x[r * side + c] = w * y[(r / factor) * low + c / factor];
                    }
                }
                return x;
            }
        };
    }

    /**
     * A dense matrix.
     *
     * @param a the rows
     * @return the operator
     */
    static LinearOperator matrix(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        return new LinearOperator() {
            @Override
            public int inputSize() {
                return n;
            }

            @Override
            public int outputSize() {
                return m;
            }

            @Override
            public double[] apply(double[] x) {
                double[] y = new double[m];
                for (int i = 0; i < m; i++) {
                    double s = 0;
                    double[] row = a[i];
                    for (int j = 0; j < n; j++) {
                        s += row[j] * x[j];
                    }
                    y[i] = s;
                }
                return y;
            }

            @Override
            public double[] adjoint(double[] y) {
                double[] x = new double[n];
                for (int i = 0; i < m; i++) {
                    double[] row = a[i];
                    double yi = y[i];
                    for (int j = 0; j < n; j++) {
                        x[j] += row[j] * yi;
                    }
                }
                return x;
            }
        };
    }
}
