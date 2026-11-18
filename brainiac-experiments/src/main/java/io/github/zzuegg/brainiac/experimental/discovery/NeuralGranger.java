package io.github.zzuegg.brainiac.experimental.discovery;

import java.util.SplittableRandom;

/**
 * Neural Granger causality with component-wise MLPs (Tank et al., TPAMI 2021,
 * <a href="https://arxiv.org/abs/1802.05842">arXiv:1802.05842</a>). One MLP per target series
 * reads the last {@code K} values of every series; a hierarchical group lasso on its first-layer
 * weights (for every source, the nested groups "lags {@code k..K}" for {@code k = 1..K}) drives
 * the whole input block of a non-causal source to exactly zero, and the hierarchy also selects
 * the lag order. Training is proximal gradient descent: a gradient step on the mean squared
 * error, then the nested group soft-thresholds from the deepest group outwards (the exact
 * proximal operator for tree-structured groups, Jenatton et al. 2011). The Granger score of
 * {@code i -> j} is the norm of {@code i}'s first-layer block in {@code j}'s network.
 */
public final class NeuralGranger {

    private final int lags;
    private final int hidden;
    private final double lambda;
    private final double lr;
    private final int steps;

    /**
     * Creates the learner.
     *
     * @param lags the input lags {@code K}
     * @param hidden the hidden width
     * @param lambda the group-lasso strength
     * @param lr the proximal-gradient step size
     * @param steps full-batch steps per target
     */
    public NeuralGranger(int lags, int hidden, double lambda, double lr, int steps) {
        this.lags = lags;
        this.hidden = hidden;
        this.lambda = lambda;
        this.lr = lr;
        this.steps = steps;
    }

    /**
     * Fits one network per target.
     *
     * @param series {@code [T][d]}
     * @param seed the initialisation seed
     * @return {@code score[i][j]}: the first-layer group norm of source {@code i} for target
     *     {@code j} (0 means not causal)
     */
    public double[][] fit(double[][] series, long seed) {
        double[][] z = LeastSquares.standardise(series);
        int d = z[0].length;
        int[] all = new int[d];
        for (int j = 0; j < d; j++) {
            all[j] = j;
        }
        double[][] x = GrangerCausality.laggedDesign(z, all, lags);
        int n = x.length;
        int in = d * lags;
        double[][] score = new double[d][d];
        SplittableRandom r = new SplittableRandom(seed);
        for (int j = 0; j < d; j++) {
            double[] y = new double[n];
            for (int t = 0; t < n; t++) {
                y[t] = z[t + lags][j];
            }
            double[][] w1 = new double[hidden][in];
            double[] b1 = new double[hidden];
            double[] w2 = new double[hidden];
            double scale = 1 / Math.sqrt(in);
            for (int h = 0; h < hidden; h++) {
                for (int a = 0; a < in; a++) {
                    w1[h][a] = scale * r.nextGaussian();
                }
                w2[h] = r.nextGaussian() / Math.sqrt(hidden);
            }
            double[] b2 = {0};
            double[][] act = new double[n][hidden];
            for (int s = 0; s < steps; s++) {
                double[][] g1 = new double[hidden][in];
                double[] gb1 = new double[hidden];
                double[] g2 = new double[hidden];
                double gb2 = 0;
                for (int t = 0; t < n; t++) {
                    double out = b2[0];
                    for (int h = 0; h < hidden; h++) {
                        double pre = b1[h];
                        for (int a = 0; a < in; a++) {
                            pre += w1[h][a] * x[t][a];
                        }
                        act[t][h] = Math.tanh(pre);
                        out += w2[h] * act[t][h];
                    }
                    double err = 2 * (out - y[t]) / n;
                    gb2 += err;
                    for (int h = 0; h < hidden; h++) {
                        g2[h] += err * act[t][h];
                        double back = err * w2[h] * (1 - act[t][h] * act[t][h]);
                        gb1[h] += back;
                        for (int a = 0; a < in; a++) {
                            g1[h][a] += back * x[t][a];
                        }
                    }
                }
                b2[0] -= lr * gb2;
                for (int h = 0; h < hidden; h++) {
                    w2[h] -= lr * g2[h];
                    b1[h] -= lr * gb1[h];
                    for (int a = 0; a < in; a++) {
                        w1[h][a] -= lr * g1[h][a];
                    }
                }
                prox(w1, d);
            }
            for (int i = 0; i < d; i++) {
                double norm = 0;
                for (int h = 0; h < hidden; h++) {
                    for (int l = 0; l < lags; l++) {
                        norm += w1[h][i * lags + l] * w1[h][i * lags + l];
                    }
                }
                score[i][j] = Math.sqrt(norm);
            }
        }
        return score;
    }

    /** Hierarchical group soft-threshold, deepest group (largest lags) first. */
    private void prox(double[][] w1, int d) {
        double t = lr * lambda;
        for (int i = 0; i < d; i++) {
            for (int k = lags - 1; k >= 0; k--) {
                double norm = 0;
                for (double[] row : w1) {
                    for (int l = k; l < lags; l++) {
                        norm += row[i * lags + l] * row[i * lags + l];
                    }
                }
                norm = Math.sqrt(norm);
                double f = norm <= t ? 0 : 1 - t / norm;
                for (double[] row : w1) {
                    for (int l = k; l < lags; l++) {
                        row[i * lags + l] *= f;
                    }
                }
            }
        }
    }
}
