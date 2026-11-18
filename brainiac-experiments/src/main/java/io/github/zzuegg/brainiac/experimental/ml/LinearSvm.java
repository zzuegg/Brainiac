package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * A linear soft-margin SVM (hinge loss, L2 regularisation) trained by dual coordinate descent
 * (Hsieh et al. 2008, "A dual coordinate descent method for large-scale linear SVM"), one
 * versus rest for several classes. Each epoch costs {@code O(n d)}, linear in the data, which
 * is what makes {@link RandomFourierFeatures} plus a linear SVM a scalable stand-in for an RBF
 * {@link KernelSvm}.
 */
public final class LinearSvm {

    private final double[][] weights;
    private final double[] bias;

    private LinearSvm(double[][] weights, double[] bias) {
        this.weights = weights;
        this.bias = bias;
    }

    /**
     * Trains one binary machine per class (one versus rest; a single machine for two classes).
     *
     * @param x the training points {@code [n][d]}
     * @param labels their classes
     * @param classes the number of classes, at least 2
     * @param c the box constraint, positive
     * @param epochs passes over the data
     * @param seed shuffles the coordinate order
     * @return the trained machine
     */
    public static LinearSvm fit(double[][] x, int[] labels, int classes, double c, int epochs, long seed) {
        int machines = classes == 2 ? 1 : classes;
        int n = x.length;
        int d = x[0].length;
        double[][] w = new double[machines][d + 1];
        SplittableRandom random = new SplittableRandom(seed);
        double[] qd = new double[n];
        for (int i = 0; i < n; i++) {
            qd[i] = MlMatrices.dot(x[i], x[i]) + 1;   // + 1 for the bias feature
        }
        int[] order = new int[n];
        for (int m = 0; m < machines; m++) {
            double[] alpha = new double[n];
            double[] wm = w[m];
            for (int e = 0; e < epochs; e++) {
                for (int i = 0; i < n; i++) {
                    order[i] = i;
                }
                for (int i = n - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    int t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
                for (int i : order) {
                    double y = labels[i] == m ? 1 : -1;
                    double g = y * (MlMatrices.dot(x[i], wm) + wm[d]) - 1;
                    double next = Math.clamp(alpha[i] - g / qd[i], 0, c);
                    double delta = (next - alpha[i]) * y;
                    if (delta != 0) {
                        for (int j = 0; j < d; j++) {
                            wm[j] += delta * x[i][j];
                        }
                        wm[d] += delta;
                        alpha[i] = next;
                    }
                }
            }
        }
        double[][] weights = new double[machines][d];
        double[] bias = new double[machines];
        for (int m = 0; m < machines; m++) {
            System.arraycopy(w[m], 0, weights[m], 0, d);
            bias[m] = w[m][d];
        }
        return new LinearSvm(weights, bias);
    }

    /**
     * The decision values, one per machine.
     *
     * @param x a point
     * @return {@code w_m . x + b_m}
     */
    public double[] decision(double[] x) {
        double[] f = new double[weights.length];
        for (int m = 0; m < f.length; m++) {
            f[m] = MlMatrices.dot(weights[m], x) + bias[m];
        }
        return f;
    }

    /**
     * Predicts a class: the machine with the largest margin, or the sign for two classes.
     *
     * @param x a point
     * @return the class
     */
    public int predict(double[] x) {
        double[] f = decision(x);
        if (f.length == 1) {
            return f[0] > 0 ? 0 : 1;
        }
        int best = 0;
        for (int m = 1; m < f.length; m++) {
            if (f[m] > f[best]) {
                best = m;
            }
        }
        return best;
    }

    /**
     * Predicts many points.
     *
     * @param x points
     * @return their classes
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }
}
