package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * EASE, an embarrassingly shallow linear autoencoder for implicit feedback (Steck, 2019,
 * <a href="https://arxiv.org/abs/1905.03375">arXiv:1905.03375</a>). An item-item weight matrix
 * {@code B} reconstructs each user's interaction row, {@code x_u B}, under the constraint
 * {@code diag(B) = 0} (no item may predict itself). The ridge problem has a closed form:
 * {@code P = (X^T X + lambda I)^-1} and {@code B = I - P diagMat(1 / diag(P))}, that is
 * {@code B_ij = -P_ij / P_jj} off the diagonal. There is no training loop, only one item-by-item
 * matrix inverse (here a Cholesky factorisation in double precision on the host), and it is
 * often state of the art on collaborative filtering benchmarks.
 */
public final class EaseRecommender {

    private final double[][] weights;
    private final double[][] interactions;

    private EaseRecommender(double[][] weights, double[][] interactions) {
        this.weights = weights;
        this.interactions = interactions;
    }

    /**
     * Fits EASE.
     *
     * @param interactions {@code [users][items]} implicit feedback, 1 for an interaction
     * @param lambda the L2 regularisation
     * @return the fitted model
     */
    public static EaseRecommender fit(double[][] interactions, double lambda) {
        int items = interactions[0].length;
        double[][] gram = new double[items][items];
        for (double[] row : interactions) {
            for (int i = 0; i < items; i++) {
                if (row[i] == 0) {
                    continue;
                }
                for (int j = 0; j < items; j++) {
                    gram[i][j] += row[i] * row[j];
                }
            }
        }
        for (int i = 0; i < items; i++) {
            gram[i][i] += lambda;
        }
        double[][] p = MlMatrices.choleskyInverse(MlMatrices.cholesky(gram));
        double[][] b = new double[items][items];
        for (int i = 0; i < items; i++) {
            for (int j = 0; j < items; j++) {
                b[i][j] = i == j ? 0 : -p[i][j] / p[j][j];
            }
        }
        return new EaseRecommender(b, interactions);
    }

    /**
     * @param row one user's interaction row
     * @return the score of every item, {@code row B}
     */
    public float[] scores(double[] row) {
        int items = weights.length;
        float[] out = new float[items];
        for (int i = 0; i < items; i++) {
            if (row[i] == 0) {
                continue;
            }
            double v = row[i];
            double[] w = weights[i];
            for (int j = 0; j < items; j++) {
                out[j] += (float) (v * w[j]);
            }
        }
        return out;
    }

    /**
     * @param user a user of the fitted interactions
     * @return the score of every item for that user
     */
    public float[] scores(int user) {
        return scores(interactions[user]);
    }

    /** @return the item-item weight matrix {@code B} (zero diagonal) */
    public double[][] weights() {
        return weights;
    }
}
