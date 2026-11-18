package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * Deep deterministic uncertainty (DDU, Mukhoti et al. 2023,
 * <a href="https://arxiv.org/abs/2102.11582">arXiv:2102.11582</a>): after training a
 * classifier whose feature extractor is residual and spectrally normalised
 * ({@link SpectralNormDense}), so that distances in feature space reflect distances in input
 * space (bi-Lipschitz), fit one Gaussian per class to the training features (Gaussian
 * discriminant analysis). The log feature density
 * {@code log sum_c pi_c N(phi(x); mu_c, Sigma_c)} is the epistemic score: low density means the
 * input is unlike anything seen. The softmax entropy still measures aleatoric uncertainty. One
 * deterministic forward pass, no ensembles.
 */
public final class FeatureDensityUncertainty {

    private final double[] logPriors;
    private final double[][] means;
    private final double[][][] chols;
    private final double[] logDets;

    private FeatureDensityUncertainty(double[] logPriors, double[][] means, double[][][] chols) {
        this.logPriors = logPriors;
        this.means = means;
        this.chols = chols;
        this.logDets = new double[chols.length];
        for (int c = 0; c < chols.length; c++) {
            logDets[c] = MlMatrices.logDet(chols[c]);
        }
    }

    /**
     * Fits the class-conditional Gaussians.
     *
     * @param features training features {@code [n][d]}
     * @param labels their classes
     * @param classes the number of classes
     * @param jitter added to each covariance diagonal, for example {@code 1e-3}
     * @return the density
     */
    public static FeatureDensityUncertainty fit(double[][] features, int[] labels, int classes, double jitter) {
        int d = features[0].length;
        double[][] means = new double[classes][d];
        int[] counts = new int[classes];
        for (int i = 0; i < features.length; i++) {
            counts[labels[i]]++;
            for (int k = 0; k < d; k++) {
                means[labels[i]][k] += features[i][k];
            }
        }
        double[][][] cov = new double[classes][d][d];
        for (int c = 0; c < classes; c++) {
            for (int k = 0; k < d; k++) {
                means[c][k] /= Math.max(1, counts[c]);
            }
        }
        for (int i = 0; i < features.length; i++) {
            int c = labels[i];
            for (int a = 0; a < d; a++) {
                double da = features[i][a] - means[c][a];
                for (int b = 0; b <= a; b++) {
                    cov[c][a][b] += da * (features[i][b] - means[c][b]);
                }
            }
        }
        double[] logPriors = new double[classes];
        double[][][] chols = new double[classes][][];
        for (int c = 0; c < classes; c++) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b <= a; b++) {
                    cov[c][a][b] /= Math.max(1, counts[c] - 1);
                    cov[c][b][a] = cov[c][a][b];
                }
                cov[c][a][a] += jitter;
            }
            chols[c] = MlMatrices.cholesky(cov[c]);
            logPriors[c] = Math.log(Math.max(1, counts[c]) / (double) features.length);
        }
        return new FeatureDensityUncertainty(logPriors, means, chols);
    }

    /**
     * The log feature density.
     *
     * @param feature one feature vector
     * @return {@code log p(phi)}
     */
    public double logDensity(double[] feature) {
        int d = feature.length;
        double max = Double.NEGATIVE_INFINITY;
        double[] lp = new double[means.length];
        for (int c = 0; c < means.length; c++) {
            double[] diff = new double[d];
            for (int k = 0; k < d; k++) {
                diff[k] = feature[k] - means[c][k];
            }
            double[] y = MlMatrices.forwardSolve(chols[c], diff);
            lp[c] = logPriors[c] - 0.5 * MlMatrices.dot(y, y) - 0.5 * logDets[c] - 0.5 * d * Math.log(2 * Math.PI);
            max = Math.max(max, lp[c]);
        }
        double s = 0;
        for (double v : lp) {
            s += Math.exp(v - max);
        }
        return max + Math.log(s);
    }

    /**
     * Epistemic uncertainty scores, the negative log densities (higher means more uncertain).
     *
     * @param features feature vectors {@code [n][d]}
     * @return {@code -log p(phi)} per row
     */
    public double[] uncertainty(double[][] features) {
        double[] u = new double[features.length];
        for (int i = 0; i < u.length; i++) {
            u[i] = -logDensity(features[i]);
        }
        return u;
    }
}
