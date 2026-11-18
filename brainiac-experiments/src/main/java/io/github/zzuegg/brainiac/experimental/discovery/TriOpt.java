package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.Arrays;

/**
 * Linear ordering-based discovery in the spirit of TriOpt (2026,
 * <a href="https://arxiv.org/abs/2605.17465">arXiv:2605.17465</a>): leaf removal with a linear
 * kernel and Sherman-Morrison downdates, then a convex L1 regression restricted by a strictly
 * upper-triangular mask, so no acyclicity constraint is needed.
 *
 * <p>With a linear kernel, the Stein estimate of the score is {@code -Theta (x - mu)} with
 * {@code Theta} the precision matrix, so the score Jacobian's diagonal is {@code -Theta_jj}. In a
 * linear SEM with equal noise variances, {@code Theta_jj = 1/sigma^2 + sum over children of
 * w^2/sigma^2}, so the leaf is the node with the smallest precision diagonal (the
 * equal-variance identifiability result of Peters and Bühlmann 2014, as used by Ghoshal and
 * Honorio 2018). Removing the leaf marginalises it out, and the precision of the remaining
 * nodes is the Schur complement {@code Theta' = Theta_{-j,-j} - theta theta^T / Theta_jj}, a
 * rank-one downdate, so the whole order costs one matrix inversion. The weights are then fitted
 * by an L1-penalised regression of each node on its predecessors only (the triangular mask), the
 * level chosen by BIC.
 *
 * <p>The paper's abstract gives the ingredients (linear kernel, Sherman-Morrison rank-1 downdate,
 * masked convex fit); the precision-diagonal leaf criterion here is the closed form those
 * ingredients give in the linear case, not a transcription of the paper's algorithm.
 */
public final class TriOpt {

    private TriOpt() {
    }

    /**
     * The order by precision-diagonal leaf removal with rank-one downdates.
     *
     * @param x the data {@code [n][d]}, unstandardised (standardising breaks equal variances)
     * @return the order, most upstream first
     */
    public static int[] order(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[] mean = MlMatrices.columnMeans(x);
        double[][] cov = new double[d][d];
        for (double[] r : x) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    cov[a][b] += (r[a] - mean[a]) * (r[b] - mean[b]) / n;
                }
            }
        }
        double[][] theta = MlMatrices.choleskyInverse(MlMatrices.cholesky(cov));
        boolean[] alive = new boolean[d];
        Arrays.fill(alive, true);
        int[] order = new int[d];
        for (int pos = d - 1; pos >= 0; pos--) {
            int leaf = -1;
            for (int j = 0; j < d; j++) {
                if (alive[j] && (leaf < 0 || theta[j][j] < theta[leaf][leaf])) {
                    leaf = j;
                }
            }
            order[pos] = leaf;
            alive[leaf] = false;
            double tl = theta[leaf][leaf];
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    if (alive[a] && alive[b]) {
                        theta[a][b] -= theta[a][leaf] * theta[leaf][b] / tl;
                    }
                }
            }
        }
        return order;
    }

    /**
     * Orders and fits masked lasso weights.
     *
     * @param x the data
     * @return the order and weights
     */
    public static Lingam.Result fit(double[][] x) {
        int[] order = order(x);
        int d = order.length;
        double[][] w = new double[d][d];
        for (int k = 1; k < d; k++) {
            int[] pred = Arrays.copyOf(order, k);
            double[] beta = Lingam.lassoBic(LeastSquares.columns(x, pred), LeastSquares.column(x, order[k]), x.length,
                    false);
            for (int a = 0; a < k; a++) {
                w[pred[a]][order[k]] = beta[a];
            }
        }
        return new Lingam.Result(order, w);
    }
}
