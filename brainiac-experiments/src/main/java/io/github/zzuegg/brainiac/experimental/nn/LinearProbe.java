package io.github.zzuegg.brainiac.experimental.nn;

import java.util.SplittableRandom;

/**
 * A linear probe: a softmax regression trained on frozen activations to test whether a
 * property is <em>linearly decodable</em> from a layer. High probe accuracy alone can mean
 * the probe learned the task itself, so {@link #selectivity} compares with a control task
 * (the same inputs with labels randomly reassigned per input type): a good probe is
 * accurate on the real task and poor on the control.
 *
 * <p>Alain and Bengio 2016, "Understanding intermediate layers using linear classifier
 * probes", <a href="https://arxiv.org/abs/1610.01644">arXiv:1610.01644</a>; Hewitt and Liang
 * 2019, "Designing and Interpreting Probes with Control Tasks",
 * <a href="https://arxiv.org/abs/1909.03368">arXiv:1909.03368</a>.
 */
public final class LinearProbe {

    private final double[][] w;
    private final double[] b;

    private LinearProbe(double[][] w, double[] b) {
        this.w = w;
        this.b = b;
    }

    /**
     * Trains a probe by full-batch Adam on the cross-entropy, with an L2
     * penalty.
     *
     * @param x activations {@code [samples][features]}
     * @param y labels
     * @param classes the number of classes
     * @param epochs gradient steps
     * @param learningRate the step size
     * @param l2 the weight decay
     * @return the probe
     */
    public static LinearProbe fit(float[][] x, int[] y, int classes, int epochs, double learningRate, double l2) {
        int d = x[0].length;
        double[][] w = new double[d][classes];
        double[] b = new double[classes];
        double[][] m = new double[d][classes];
        double[][] v = new double[d][classes];
        for (int e = 1; e <= epochs; e++) {
            double[][] g = new double[d][classes];
            double[] gb = new double[classes];
            for (int i = 0; i < x.length; i++) {
                double[] p = probs(w, b, x[i]);
                p[y[i]] -= 1;
                for (int c = 0; c < classes; c++) {
                    gb[c] += p[c] / x.length;
                    for (int j = 0; j < d; j++) {
                        g[j][c] += p[c] * x[i][j] / x.length;
                    }
                }
            }
            for (int c = 0; c < classes; c++) {
                b[c] -= learningRate * gb[c];
                for (int j = 0; j < d; j++) {
                    double gj = g[j][c] + l2 * w[j][c];
                    m[j][c] = 0.9 * m[j][c] + 0.1 * gj;
                    v[j][c] = 0.999 * v[j][c] + 0.001 * gj * gj;
                    double mh = m[j][c] / (1 - Math.pow(0.9, e));
                    double vh = v[j][c] / (1 - Math.pow(0.999, e));
                    w[j][c] -= learningRate * mh / (Math.sqrt(vh) + 1e-8);
                }
            }
        }
        return new LinearProbe(w, b);
    }

    private static double[] probs(double[][] w, double[] b, float[] x) {
        int classes = b.length;
        double[] z = b.clone();
        for (int j = 0; j < x.length; j++) {
            if (x[j] == 0) {
                continue;
            }
            for (int c = 0; c < classes; c++) {
                z[c] += w[j][c] * x[j];
            }
        }
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) {
            max = Math.max(max, v);
        }
        double s = 0;
        for (int c = 0; c < classes; c++) {
            z[c] = Math.exp(z[c] - max);
            s += z[c];
        }
        for (int c = 0; c < classes; c++) {
            z[c] /= s;
        }
        return z;
    }

    /**
     * Predicts a class.
     *
     * @param x activations of one sample
     * @return the class
     */
    public int predict(float[] x) {
        double[] p = probs(w, b, x);
        int best = 0;
        for (int c = 1; c < p.length; c++) {
            if (p[c] > p[best]) {
                best = c;
            }
        }
        return best;
    }

    /**
     * The accuracy on labelled activations.
     *
     * @param x activations
     * @param y labels
     * @return the share predicted right
     */
    public double accuracy(float[][] x, int[] y) {
        int ok = 0;
        for (int i = 0; i < x.length; i++) {
            ok += predict(x[i]) == y[i] ? 1 : 0;
        }
        return ok / (double) x.length;
    }

    /**
     * Selectivity: probe accuracy on the task minus on a control task, both measured on
     * held-out samples. The control gives every input type (given by {@code type}) a fixed
     * random label.
     *
     * @param train training activations
     * @param trainY training labels
     * @param trainType the input type of each training sample (such as the word identity)
     * @param test test activations
     * @param testY test labels
     * @param testType test input types
     * @param classes the number of classes
     * @param seed the seed for the control labels
     * @return {@code [task accuracy, control accuracy, selectivity]}
     */
    public static double[] selectivity(float[][] train, int[] trainY, int[] trainType, float[][] test, int[] testY,
            int[] testType, int classes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        java.util.Map<Integer, Integer> control = new java.util.HashMap<>();
        int[] cTrain = new int[trainType.length];
        int[] cTest = new int[testType.length];
        for (int i = 0; i < cTrain.length; i++) {
            cTrain[i] = control.computeIfAbsent(trainType[i], k -> r.nextInt(classes));
        }
        for (int i = 0; i < cTest.length; i++) {
            cTest[i] = control.computeIfAbsent(testType[i], k -> r.nextInt(classes));
        }
        double task = fit(train, trainY, classes, 300, 0.05, 1e-3).accuracy(test, testY);
        double ctl = fit(train, cTrain, classes, 300, 0.05, 1e-3).accuracy(test, cTest);
        return new double[] {task, ctl, task - ctl};
    }
}
