package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.tensor.Matrices;

/**
 * A ridge classifier for many more features than rows, with the penalty chosen by exact
 * leave-one-out cross-validation, as scikit-learn's {@code RidgeClassifierCV} that ROCKET, Hydra
 * and WEASEL 2.0 use. Features are standardised, targets are one-hot in {@code {-1, 1}}, and the
 * solution is computed in the dual: with the centred Gram matrix {@code K = X X^T = U diag(s) U^T},
 * the fitted values are {@code H Y} with {@code H = U diag(s / (s + lambda)) U^T}, so the
 * leave-one-out residuals {@code (y - H y) / (1 - H_ii)} for every candidate {@code lambda} cost
 * one eigendecomposition of an {@code n x n} matrix. Unlike the primal
 * {@code experimental.timeseries.RidgeClassifier}, the cost does not grow with the square of the
 * feature count, which matters for ten thousand random-kernel or word-count features.
 */
public final class WideRidgeClassifier {

    private final double[] mean;
    private final double[] scale;
    private final double[][] weights;
    private final double[] bias;
    private final double lambda;

    private WideRidgeClassifier(double[] mean, double[] scale, double[][] weights, double[] bias, double lambda) {
        this.mean = mean;
        this.scale = scale;
        this.weights = weights;
        this.bias = bias;
        this.lambda = lambda;
    }

    /**
     * The usual penalty grid, {@code 10^-3 ... 10^3} in 10 log steps.
     *
     * @return the grid
     */
    public static double[] defaultAlphas() {
        double[] a = new double[10];
        for (int i = 0; i < a.length; i++) {
            a[i] = Math.pow(10, -3 + 6.0 * i / (a.length - 1));
        }
        return a;
    }

    /**
     * Fits the classifier, picking the penalty by leave-one-out error.
     *
     * @param x features {@code [n][d]}
     * @param labels classes in {@code [0, classes)}
     * @param classes the number of classes
     * @param alphas candidate penalties (relative to the mean diagonal of the Gram matrix)
     * @return the classifier
     */
    public static WideRidgeClassifier fit(double[][] x, int[] labels, int classes, double[] alphas) {
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
                double v = row[j] - mean[j];
                scale[j] += v * v / n;
            }
        }
        for (int j = 0; j < d; j++) {
            scale[j] = scale[j] > 1e-20 ? Math.sqrt(scale[j]) : 1;
        }
        double[][] z = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                z[i][j] = (x[i][j] - mean[j]) / scale[j];
            }
        }
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double s = 0;
                double[] a = z[i];
                double[] b = z[j];
                for (int f = 0; f < d; f++) {
                    s += a[f] * b[f];
                }
                k[i][j] = s;
                k[j][i] = s;
            }
        }
        double trace = 0;
        for (int i = 0; i < n; i++) {
            trace += k[i][i];
        }
        double unit = Math.max(1e-12, trace / n);
        double[][] y = new double[n][classes];
        double[] yMean = new double[classes];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < classes; c++) {
                y[i][c] = labels[i] == c ? 1 : -1;
                yMean[c] += y[i][c] / n;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < classes; c++) {
                y[i][c] -= yMean[c];
            }
        }
        Matrices.SymmetricEigen eig = Matrices.symmetricEigen(k);
        double[][] u = eig.vectors();
        double[] s = eig.values();
        double[][] uty = new double[n][classes];
        for (int e = 0; e < n; e++) {
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < classes; c++) {
                    uty[e][c] += u[i][e] * y[i][c];
                }
            }
        }
        double best = Double.POSITIVE_INFINITY;
        double bestLambda = alphas[0] * unit;
        for (double alpha : alphas) {
            double lam = alpha * unit;
            double err = 0;
            for (int i = 0; i < n; i++) {
                double hii = 0;
                double[] fit = new double[classes];
                for (int e = 0; e < n; e++) {
                    double sh = Math.max(0, s[e]) / (Math.max(0, s[e]) + lam);
                    hii += u[i][e] * u[i][e] * sh;
                    for (int c = 0; c < classes; c++) {
                        fit[c] += u[i][e] * sh * uty[e][c];
                    }
                }
                double denom = Math.max(1e-9, 1 - hii);
                int arg = 0;
                double argV = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < classes; c++) {
                    double loo = y[i][c] + yMean[c] - (y[i][c] - fit[c]) / denom;
                    double r = (y[i][c] - fit[c]) / denom;
                    err += r * r;
                    if (loo > argV) {
                        argV = loo;
                        arg = c;
                    }
                }
                err += arg == labels[i] ? 0 : 1e-3;
            }
            if (err < best) {
                best = err;
                bestLambda = lam;
            }
        }
        // dual coefficients A = U diag(1 / (s + lambda)) U^T Y, weights W = Z^T A
        double[][] a = new double[n][classes];
        for (int e = 0; e < n; e++) {
            double inv = 1 / (Math.max(0, s[e]) + bestLambda);
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < classes; c++) {
                    a[i][c] += u[i][e] * inv * uty[e][c];
                }
            }
        }
        double[][] w = new double[classes][d];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < classes; c++) {
                double ac = a[i][c];
                if (ac == 0) {
                    continue;
                }
                double[] row = z[i];
                double[] wc = w[c];
                for (int f = 0; f < d; f++) {
                    wc[f] += ac * row[f];
                }
            }
        }
        return new WideRidgeClassifier(mean, scale, w, yMean, bestLambda);
    }

    /**
     * Decision scores, one per class (larger is more likely).
     *
     * @param x features {@code [m][d]}
     * @return {@code [m][classes]}
     */
    public double[][] scores(double[][] x) {
        double[][] out = new double[x.length][weights.length];
        for (int i = 0; i < x.length; i++) {
            for (int c = 0; c < weights.length; c++) {
                double s = bias[c];
                double[] w = weights[c];
                for (int f = 0; f < w.length; f++) {
                    s += w[f] * (x[i][f] - mean[f]) / scale[f];
                }
                out[i][c] = s;
            }
        }
        return out;
    }

    /**
     * Predicts classes.
     *
     * @param x features {@code [m][d]}
     * @return the classes
     */
    public int[] predict(double[][] x) {
        return argmax(scores(x));
    }

    /** @return the penalty chosen by leave-one-out cross-validation */
    public double lambda() {
        return lambda;
    }

    /**
     * The row-wise argmax.
     *
     * @param scores {@code [m][classes]}
     * @return the index of each row's largest entry
     */
    public static int[] argmax(double[][] scores) {
        int[] out = new int[scores.length];
        for (int i = 0; i < scores.length; i++) {
            int b = 0;
            for (int c = 1; c < scores[i].length; c++) {
                if (scores[i][c] > scores[i][b]) {
                    b = c;
                }
            }
            out[i] = b;
        }
        return out;
    }
}
