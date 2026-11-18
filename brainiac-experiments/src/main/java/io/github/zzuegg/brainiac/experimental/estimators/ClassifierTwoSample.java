package io.github.zzuegg.brainiac.experimental.estimators;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Classifier two-sample tests (Lopez-Paz and Oquab, ICLR 2017,
 * <a href="https://arxiv.org/abs/1610.06545">arXiv:1610.06545</a>): pool the two samples with
 * labels 0 and 1, train a classifier on one part and measure its accuracy on the rest. Under
 * {@code P = Q} the held-out accuracy is {@code Binomial(n_test, 1/2) / n_test}, so the p-value
 * is the upper tail of its normal approximation; the accuracy itself says how far apart the
 * distributions are, and the trained classifier says where they differ.
 */
public final class ClassifierTwoSample {

    /** The classifier inside the test. */
    public enum Classifier {
        /** L2-regularised logistic regression on standardised linear and quadratic features. */
        LOGISTIC_QUADRATIC,
        /** A {@code k}-nearest-neighbour vote with {@code k ~ sqrt(n_train)}. */
        KNN
    }

    private ClassifierTwoSample() {
    }

    /**
     * Runs the test.
     *
     * @param p the first sample {@code [n][d]}
     * @param q the second sample {@code [m][d]}
     * @param classifier the classifier
     * @param trainFraction the share used for training, for example 0.5
     * @param seed the split seed
     * @return the held-out accuracy and its p-value
     */
    public static KernelTests.Result test(double[][] p, double[][] q, Classifier classifier, double trainFraction,
            long seed) {
        Objects.requireNonNull(classifier, "classifier");
        int n = p.length + q.length;
        double[][] z = new double[n][];
        int[] label = new int[n];
        for (int i = 0; i < p.length; i++) {
            z[i] = p[i];
        }
        for (int i = 0; i < q.length; i++) {
            z[p.length + i] = q[i];
            label[p.length + i] = 1;
        }
        int[] perm = SteinTests.permutation(n, new SplittableRandom(seed));
        int nTrain = (int) Math.round(n * trainFraction);
        double[][] xTrain = new double[nTrain][];
        int[] yTrain = new int[nTrain];
        double[][] xTest = new double[n - nTrain][];
        int[] yTest = new int[n - nTrain];
        for (int i = 0; i < n; i++) {
            if (i < nTrain) {
                xTrain[i] = z[perm[i]];
                yTrain[i] = label[perm[i]];
            } else {
                xTest[i - nTrain] = z[perm[i]];
                yTest[i - nTrain] = label[perm[i]];
            }
        }
        int[] predicted = classifier == Classifier.KNN ? knn(xTrain, yTrain, xTest)
                : logistic(xTrain, yTrain, xTest);
        int correct = 0;
        for (int i = 0; i < predicted.length; i++) {
            if (predicted[i] == yTest[i]) {
                correct++;
            }
        }
        int m = predicted.length;
        double acc = (double) correct / m;
        double zScore = (correct - 0.5 * m) / Math.sqrt(0.25 * m);
        return new KernelTests.Result(acc, Math.exp(NormalTails.logSf(zScore)));
    }

    private static int[] knn(double[][] x, int[] y, double[][] xs) {
        int k = Math.max(1, (int) Math.round(Math.sqrt(x.length)) | 1);
        int[] out = new int[xs.length];
        double[] dist = new double[x.length];
        for (int t = 0; t < xs.length; t++) {
            for (int i = 0; i < x.length; i++) {
                dist[i] = MlMatrices.squaredDistance(x[i], xs[t]);
            }
            Integer[] idx = new Integer[x.length];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = i;
            }
            Arrays.sort(idx, (a, b) -> Double.compare(dist[a], dist[b]));
            int votes = 0;
            for (int i = 0; i < k; i++) {
                votes += y[idx[i]];
            }
            out[t] = 2 * votes > k ? 1 : 0;
        }
        return out;
    }

    private static double[] quadratic(double[] x) {
        int d = x.length;
        double[] f = new double[d + d * (d + 1) / 2];
        int c = 0;
        for (double v : x) {
            f[c++] = v;
        }
        for (int i = 0; i < d; i++) {
            for (int j = i; j < d; j++) {
                f[c++] = x[i] * x[j];
            }
        }
        return f;
    }

    private static int[] logistic(double[][] x, int[] y, double[][] xs) {
        int n = x.length;
        double[][] f = new double[n][];
        for (int i = 0; i < n; i++) {
            f[i] = quadratic(x[i]);
        }
        int p = f[0].length;
        double[] mean = MlMatrices.columnMeans(f);
        double[] sd = new double[p];
        for (double[] r : f) {
            for (int j = 0; j < p; j++) {
                sd[j] += (r[j] - mean[j]) * (r[j] - mean[j]) / n;
            }
        }
        for (int j = 0; j < p; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-12;
        }
        for (double[] r : f) {
            for (int j = 0; j < p; j++) {
                r[j] = (r[j] - mean[j]) / sd[j];
            }
        }
        double[] w = new double[p];
        double b = 0;
        double lambda = 1e-2;
        double lr = 0.5;
        for (int it = 0; it < 300; it++) {
            double[] g = new double[p];
            double gb = 0;
            for (int i = 0; i < n; i++) {
                double s = 1 / (1 + Math.exp(-(MlMatrices.dot(w, f[i]) + b)));
                double e = (s - y[i]) / n;
                for (int j = 0; j < p; j++) {
                    g[j] += e * f[i][j];
                }
                gb += e;
            }
            for (int j = 0; j < p; j++) {
                w[j] -= lr * (g[j] + lambda * w[j]);
            }
            b -= lr * gb;
        }
        int[] out = new int[xs.length];
        for (int t = 0; t < xs.length; t++) {
            double[] q = quadratic(xs[t]);
            double s = b;
            for (int j = 0; j < p; j++) {
                s += w[j] * (q[j] - mean[j]) / sd[j];
            }
            out[t] = s > 0 ? 1 : 0;
        }
        return out;
    }
}
