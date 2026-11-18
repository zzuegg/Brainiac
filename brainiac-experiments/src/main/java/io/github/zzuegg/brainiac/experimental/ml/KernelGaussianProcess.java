package io.github.zzuegg.brainiac.experimental.ml;

/**
 * Exact Gaussian-process regression with any {@link KernelFunction}, for kernels that are not
 * a {@link StationaryKernel} (which {@link GaussianProcess} is built around): the spectral
 * mixture kernel, the infinite-width NNGP and NTK kernels, or a Nyström-approximated kernel.
 * The targets are centred on their mean; one Cholesky factor of {@code K + noise I} gives the
 * posterior mean, the variances and the log marginal likelihood (Rasmussen and Williams,
 * algorithm 2.1). With a near-zero noise the mean is kernel (ridge) regression.
 */
public final class KernelGaussianProcess {

    private final KernelFunction kernel;
    private final double noise;
    private final double[][] x;
    private final double[] y;
    private final double yMean;
    private final double[][] chol;
    private final double[] alpha;

    private KernelGaussianProcess(KernelFunction kernel, double noise, double[][] x, double[] y, boolean centre) {
        this.kernel = kernel;
        this.noise = noise;
        this.x = x;
        double m = 0;
        if (centre) {
            for (double v : y) {
                m += v / y.length;
            }
        }
        this.yMean = m;
        this.y = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            this.y[i] = y[i] - m;
        }
        double[][] k = kernel.gram(x);
        for (int i = 0; i < k.length; i++) {
            k[i][i] += noise;
        }
        this.chol = MlMatrices.cholesky(k);
        this.alpha = MlMatrices.choleskySolve(chol, this.y);
    }

    /**
     * Conditions the GP on data, centring the targets.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets
     * @param kernel the kernel
     * @param noise the noise variance (the ridge), positive
     * @return the posterior
     */
    public static KernelGaussianProcess fit(double[][] x, double[] y, KernelFunction kernel, double noise) {
        return new KernelGaussianProcess(kernel, noise, x, y, true);
    }

    /**
     * Conditions the GP on data with a zero prior mean (no centring).
     *
     * @param x inputs {@code [n][d]}
     * @param y targets
     * @param kernel the kernel
     * @param noise the noise variance, positive
     * @return the posterior
     */
    public static KernelGaussianProcess fitZeroMean(double[][] x, double[] y, KernelFunction kernel, double noise) {
        return new KernelGaussianProcess(kernel, noise, x, y, false);
    }

    /**
     * The posterior mean.
     *
     * @param xs test inputs {@code [m][d]}
     * @return the mean at each
     */
    public double[] mean(double[][] xs) {
        double[] out = new double[xs.length];
        for (int j = 0; j < xs.length; j++) {
            double s = yMean;
            for (int i = 0; i < x.length; i++) {
                s += kernel.value(x[i], xs[j]) * alpha[i];
            }
            out[j] = s;
        }
        return out;
    }

    /**
     * The posterior variance of the latent function.
     *
     * @param xs test inputs {@code [m][d]}
     * @return the latent variance at each (without the noise)
     */
    public double[] latentVariance(double[][] xs) {
        double[][] ks = kernel.matrix(x, xs);
        double[][] v = MlMatrices.triangularSolve(chol, ks, false);
        double[] out = new double[xs.length];
        for (int j = 0; j < xs.length; j++) {
            double s = 0;
            for (int i = 0; i < x.length; i++) {
                s += v[i][j] * v[i][j];
            }
            out[j] = Math.max(kernel.value(xs[j], xs[j]) - s, 1e-12);
        }
        return out;
    }

    /**
     * The log marginal likelihood of the centred targets.
     *
     * @return {@code log p(y | X)}
     */
    public double logMarginalLikelihood() {
        return -0.5 * MlMatrices.dot(y, alpha) - 0.5 * MlMatrices.logDet(chol) - 0.5 * y.length * Math.log(2 * Math.PI);
    }

    /**
     * {@code alpha alpha^T - (K + noise I)^-1}, whose half trace against {@code dK/dtheta} is
     * the gradient of the log marginal likelihood in a kernel hyperparameter.
     *
     * @return {@code [n][n]}
     */
    public double[][] marginalLikelihoodWeights() {
        double[][] w = MlMatrices.choleskyInverse(chol);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w.length; j++) {
                w[i][j] = alpha[i] * alpha[j] - w[i][j];
            }
        }
        return w;
    }

    /**
     * The noise variance.
     *
     * @return the noise
     */
    public double noise() {
        return noise;
    }
}
