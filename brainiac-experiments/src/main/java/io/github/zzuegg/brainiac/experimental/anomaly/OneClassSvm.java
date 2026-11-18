package io.github.zzuegg.brainiac.experimental.anomaly;

import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * The one-class (nu) support vector machine (Schölkopf, Platt, Shawe-Taylor, Smola and
 * Williamson, Neural Computation 2001,
 * <a href="https://doi.org/10.1162/089976601750264965">doi:10.1162/089976601750264965</a>):
 * it separates the data from the origin in the kernel feature space with maximum margin,
 * which for an RBF kernel draws the smallest region holding all but about a {@code nu} share
 * of the points. The dual {@code min 1/2 a' K a} subject to {@code 0 <= a_i <= 1/(nu n)} and
 * {@code sum a_i = 1} is solved by SMO with the second-order working-set selection of
 * {@code experimental.ml.KernelSvm} (Fan, Chen and Lin 2005), here with every label {@code +1}
 * and no linear term, as in LIBSVM. The anomaly score is {@code rho - sum a_i k(x_i, x)}.
 */
public final class OneClassSvm {

    private final KernelFunction kernel;
    private final double[][] vectors;
    private final double[] weights;
    private final double rho;

    private OneClassSvm(KernelFunction kernel, double[][] vectors, double[] weights, double rho) {
        this.kernel = kernel;
        this.vectors = vectors;
        this.weights = weights;
        this.rho = rho;
    }

    /**
     * Fits the machine.
     *
     * @param x reference points {@code [n][d]}
     * @param kernel the kernel, usually {@link KernelFunction#rbf}
     * @param nu the upper bound on the outlier share and lower bound on the support-vector
     *     share, in {@code (0, 1]}
     * @param tolerance the KKT gap at which SMO stops, for example 1e-4
     * @return the fitted machine
     */
    public static OneClassSvm fit(double[][] x, KernelFunction kernel, double nu, double tolerance) {
        int n = x.length;
        double c = 1.0 / (nu * n);
        double[][] k = kernel.gram(x);
        double[] alpha = new double[n];
        double left = 1;
        for (int i = 0; i < n && left > 0; i++) {
            alpha[i] = Math.min(c, left);
            left -= alpha[i];
        }
        double[] grad = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                grad[i] += k[i][j] * alpha[j];
            }
        }
        for (int iter = 0; iter < 100 * n + 10_000; iter++) {
            int i = -1;
            double gmax = Double.NEGATIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                if (alpha[t] < c && -grad[t] >= gmax) {
                    gmax = -grad[t];
                    i = t;
                }
            }
            int j = -1;
            double gmin = Double.POSITIVE_INFINITY;
            double objMin = Double.POSITIVE_INFINITY;
            for (int t = 0; t < n; t++) {
                if (alpha[t] > 0) {
                    gmin = Math.min(gmin, -grad[t]);
                    double b = gmax + grad[t];
                    if (i >= 0 && b > 0) {
                        double a = k[i][i] + k[t][t] - 2 * k[i][t];
                        double obj = -b * b / (a > 0 ? a : 1e-12);
                        if (obj <= objMin) {
                            objMin = obj;
                            j = t;
                        }
                    }
                }
            }
            if (i < 0 || j < 0 || gmax - gmin < tolerance) {
                break;
            }
            double a = k[i][i] + k[j][j] - 2 * k[i][j];
            double delta = (grad[j] - grad[i]) / (a > 0 ? a : 1e-12);
            delta = Math.min(delta, Math.min(c - alpha[i], alpha[j]));
            alpha[i] += delta;
            alpha[j] -= delta;
            for (int t = 0; t < n; t++) {
                grad[t] += delta * (k[t][i] - k[t][j]);
            }
        }
        double free = 0;
        int freeCount = 0;
        double up = Double.POSITIVE_INFINITY;
        double low = Double.NEGATIVE_INFINITY;
        for (int t = 0; t < n; t++) {
            if (alpha[t] > 1e-12 && alpha[t] < c - 1e-12) {
                free += grad[t];
                freeCount++;
            } else if (alpha[t] <= 1e-12) {
                up = Math.min(up, grad[t]);
            } else {
                low = Math.max(low, grad[t]);
            }
        }
        double rho = freeCount > 0 ? free / freeCount : (Double.isFinite(up) && Double.isFinite(low) ? (up + low) / 2
                : Double.isFinite(up) ? up : low);
        List<double[]> sv = new ArrayList<>();
        List<Double> w = new ArrayList<>();
        for (int t = 0; t < n; t++) {
            if (alpha[t] > 1e-12) {
                sv.add(x[t]);
                w.add(alpha[t]);
            }
        }
        double[] weights = new double[w.size()];
        for (int t = 0; t < weights.length; t++) {
            weights[t] = w.get(t);
        }
        return new OneClassSvm(kernel, sv.toArray(new double[0][]), weights, rho);
    }

    /**
     * The decision value {@code sum a_i k(x_i, x) - rho}: positive inside the learned region.
     *
     * @param x a point
     * @return the decision value
     */
    public double decision(double[] x) {
        double s = 0;
        for (int i = 0; i < vectors.length; i++) {
            s += weights[i] * kernel.value(vectors[i], x);
        }
        return s - rho;
    }

    /**
     * Anomaly scores.
     *
     * @param x points {@code [m][d]}
     * @return {@code -decision}, higher is more anomalous
     */
    public double[] score(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = -decision(x[i]);
        }
        return out;
    }

    /** @return the number of support vectors */
    public int supportVectors() {
        return vectors.length;
    }
}
