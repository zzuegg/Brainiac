package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;

/**
 * The linear read-out used on fixed series features (random convolution kernels, scattering
 * coefficients, frozen self-supervised representations): features are standardised with the
 * training statistics, a bias column is added, and {@link RidgeRegression} regresses one-hot
 * targets in {@code {-1, 1}}; the class with the largest output wins (as scikit-learn's
 * {@code RidgeClassifier}, which ROCKET uses).
 */
public final class RidgeClassifier {

    private final double[] mean;
    private final double[] scale;
    private final double[][] weights;
    private final int classes;

    private RidgeClassifier(double[] mean, double[] scale, double[][] weights, int classes) {
        this.mean = mean;
        this.scale = scale;
        this.weights = weights;
        this.classes = classes;
    }

    /**
     * Fits the read-out.
     *
     * @param x features {@code [n][d]}
     * @param labels classes in {@code [0, classes)}
     * @param classes the number of classes
     * @param lambda the ridge penalty on standardised features, for example 1 to 100
     * @return the classifier
     */
    public static RidgeClassifier fit(double[][] x, int[] labels, int classes, double lambda) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = new double[d];
        double[] scale = new double[d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                scale[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        for (int j = 0; j < d; j++) {
            scale[j] = Math.sqrt(scale[j]) + 1e-8;
        }
        RidgeRegression r = RidgeRegression.of(d + 1, classes);
        for (int i = 0; i < n; i++) {
            double[] t = new double[classes];
            java.util.Arrays.fill(t, -1);
            t[labels[i]] = 1;
            r.add(row(x[i], mean, scale), t);
        }
        return new RidgeClassifier(mean, scale, r.solve(lambda), classes);
    }

    private static double[] row(double[] x, double[] mean, double[] scale) {
        double[] z = new double[x.length + 1];
        for (int j = 0; j < x.length; j++) {
            z[j] = (x[j] - mean[j]) / scale[j];
        }
        z[x.length] = 1;
        return z;
    }

    /**
     * Predicts classes.
     *
     * @param x features {@code [m][d]}
     * @return the predicted classes
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            double[] y = RidgeRegression.predict(weights, row(x[i], mean, scale));
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (y[c] > y[best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * The accuracy of predictions.
     *
     * @param predicted predicted classes
     * @param truth true classes
     * @return the share predicted right
     */
    public static double accuracy(int[] predicted, int[] truth) {
        int right = 0;
        for (int i = 0; i < truth.length; i++) {
            right += predicted[i] == truth[i] ? 1 : 0;
        }
        return right / (double) truth.length;
    }
}
