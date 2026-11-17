package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Closed-form ridge (Tikhonov) regression, {@code W = (X^T X + lambda I)^-1 X^T Y}, solved in
 * double precision with a Cholesky factorization (core {@link Linalg#cholesky}). It is the readout of reservoir computing:
 * echo state networks (Jaeger, 2001) and next generation reservoir computing (Gauthier et
 * al., <a href="https://arxiv.org/abs/2106.07688">2106.07688</a>) train only this linear map.
 *
 * <p>Rows can be added one at a time with {@link #add}, so the feature matrix never has to
 * be kept; only the {@code features x features} and {@code features x outputs} sums are.
 */
public final class RidgeRegression {
    private final int features;
    private final int outputs;
    private final double[][] xtx;
    private final double[][] xty;
    private int rows;

    private RidgeRegression(int features, int outputs) {
        this.features = features;
        this.outputs = outputs;
        this.xtx = new double[features][features];
        this.xty = new double[features][outputs];
    }

    /**
     * Starts an empty accumulation.
     *
     * @param features the input width
     * @param outputs the output width
     * @return the accumulator
     */
    public static RidgeRegression of(int features, int outputs) {
        return new RidgeRegression(features, outputs);
    }

    /**
     * Fits in one go.
     *
     * @param x {@code [rows][features]}
     * @param y {@code [rows][outputs]}
     * @param lambda the ridge penalty
     * @return the weights {@code [features][outputs]}
     */
    public static double[][] fit(double[][] x, double[][] y, double lambda) {
        RidgeRegression r = of(x[0].length, y[0].length);
        for (int i = 0; i < x.length; i++) {
            r.add(x[i], y[i]);
        }
        return r.solve(lambda);
    }

    /**
     * Adds one training row.
     *
     * @param x the features
     * @param y the targets
     */
    public void add(double[] x, double[] y) {
        if (x.length != features || y.length != outputs) {
            throw new IllegalArgumentException("expected " + features + " features and " + outputs + " outputs");
        }
        for (int i = 0; i < features; i++) {
            double xi = x[i];
            if (xi == 0) {
                continue;
            }
            double[] row = xtx[i];
            for (int j = i; j < features; j++) {
                row[j] += xi * x[j];
            }
            for (int k = 0; k < outputs; k++) {
                xty[i][k] += xi * y[k];
            }
        }
        rows++;
    }

    /** @return the number of rows added */
    public int rows() {
        return rows;
    }

    /**
     * Solves for the weights.
     *
     * @param lambda the ridge penalty, at least 0 (0 needs a full-rank design)
     * @return {@code [features][outputs]}
     */
    public double[][] solve(double lambda) {
        int n = features;
        double[] a = new double[n * n];
        double[] b = new double[n * outputs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = (j >= i ? xtx[i][j] : xtx[j][i]) + (i == j ? lambda : 0);
            }
            System.arraycopy(xty[i], 0, b, i * outputs, outputs);
        }
        Tensor l;
        try {
            l = Linalg.cholesky(Tensor.of(Shape.of(n, n), a, DType.FLOAT64, Device.cpu()));
        } catch (IllegalArgumentException e) {
            throw new ArithmeticException("matrix not positive definite; raise lambda");
        }
        Tensor z = Linalg.solveTriangular(l, Tensor.of(Shape.of(n, outputs), b, DType.FLOAT64, Device.cpu()), true);
        double[] x = Linalg.solveTriangular(l.transpose(), z, false).toDoubleArray();
        double[][] w = new double[n][outputs];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x, i * outputs, w[i], 0, outputs);
        }
        return w;
    }

    /**
     * Applies weights to one row.
     *
     * @param w the weights
     * @param x the features
     * @return {@code x W}
     */
    public static double[] predict(double[][] w, double[] x) {
        double[] y = new double[w[0].length];
        for (int i = 0; i < x.length; i++) {
            double xi = x[i];
            for (int k = 0; k < y.length; k++) {
                y[k] += xi * w[i][k];
            }
        }
        return y;
    }
}
