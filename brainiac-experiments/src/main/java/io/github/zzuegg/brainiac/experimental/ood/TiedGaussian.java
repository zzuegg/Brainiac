package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Class-conditional Gaussians with one shared (tied) covariance, scored by the negative
 * minimum Mahalanobis distance to a class mean: the density model behind the Mahalanobis OOD
 * score and the likelihood-ratio selective classifier.
 */
public final class TiedGaussian {

    private final int dim;
    private final double[][] means;
    private final boolean[] present;
    private final double[][] precision;

    private TiedGaussian(int dim, double[][] means, boolean[] present, double[][] precision) {
        this.dim = dim;
        this.means = means;
        this.present = present;
        this.precision = precision;
    }

    /**
     * Fits the class means and the tied covariance (with a small ridge).
     *
     * @param features rows {@code [n, d]}
     * @param labels the class of each row
     * @param classes the number of classes; a class without rows is ignored when scoring
     * @return the model
     */
    public static TiedGaussian fit(Tensor features, int[] labels, int classes) {
        int n = labels.length;
        int d = features.shape().dim(1);
        double[] f = features.toDoubleArray();
        double[][] means = new double[classes][d];
        int[] count = new int[classes];
        for (int i = 0; i < n; i++) {
            count[labels[i]]++;
            for (int j = 0; j < d; j++) {
                means[labels[i]][j] += f[i * d + j];
            }
        }
        boolean[] present = new boolean[classes];
        for (int c = 0; c < classes; c++) {
            present[c] = count[c] > 0;
            for (int j = 0; j < d; j++) {
                means[c][j] /= Math.max(1, count[c]);
            }
        }
        double[] cov = new double[d * d];
        for (int i = 0; i < n; i++) {
            double[] m = means[labels[i]];
            for (int a = 0; a < d; a++) {
                double xa = f[i * d + a] - m[a];
                for (int b = 0; b < d; b++) {
                    cov[a * d + b] += xa * (f[i * d + b] - m[b]) / n;
                }
            }
        }
        double trace = 0;
        for (int a = 0; a < d; a++) {
            trace += cov[a * d + a];
        }
        float[] c32 = new float[d * d];
        for (int a = 0; a < d; a++) {
            cov[a * d + a] += 1e-3 * trace / d + 1e-6;
        }
        for (int i = 0; i < c32.length; i++) {
            c32[i] = (float) cov[i];
        }
        double[] inv = Linalg.inverse(Tensor.of(Shape.of(d, d), c32)).toDoubleArray();
        double[][] precision = new double[d][d];
        for (int a = 0; a < d; a++) {
            System.arraycopy(inv, a * d, precision[a], 0, d);
        }
        return new TiedGaussian(d, means, present, precision);
    }

    /**
     * @param features rows {@code [n, d]}
     * @return {@code max_c -(x - mu_c)^T Sigma^-1 (x - mu_c)} per row
     */
    public double[] score(Tensor features) {
        double[] f = features.toDoubleArray();
        int n = f.length / dim;
        double[] out = new double[n];
        double[] diff = new double[dim];
        for (int i = 0; i < n; i++) {
            double best = Double.POSITIVE_INFINITY;
            for (int c = 0; c < means.length; c++) {
                if (!present[c]) {
                    continue;
                }
                for (int j = 0; j < dim; j++) {
                    diff[j] = f[i * dim + j] - means[c][j];
                }
                double q = 0;
                for (int a = 0; a < dim; a++) {
                    double s = 0;
                    double[] row = precision[a];
                    for (int b = 0; b < dim; b++) {
                        s += row[b] * diff[b];
                    }
                    q += diff[a] * s;
                }
                best = Math.min(best, q);
            }
            out[i] = -best;
        }
        return out;
    }
}
