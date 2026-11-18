package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * A sparse variational Gaussian process for big data (Hensman, Fusi and Lawrence 2013,
 * <a href="https://arxiv.org/abs/1309.6835">arXiv:1309.6835</a>): the function is summarised
 * by its values {@code u} at {@code M} inducing inputs {@code Z} with a free Gaussian
 * {@code q(u) = N(mu, S)}, and the evidence lower bound
 * <pre>
 * ELBO = sum_i E_q[log N(y_i | f_i, noise)] - KL(q(u) || p(u))
 * </pre>
 * factorises over the data, so it can be maximised by stochastic natural-gradient steps on
 * minibatches at {@code O(B M^2 + M^3)} per step instead of the exact GP's {@code O(n^3)}. For a
 * Gaussian likelihood the natural-gradient step with rate 1 on the full data lands on the
 * optimal {@code q(u)} of Titsias (2009).
 *
 * <p>The kernel hyperparameters and noise are fixed here; fit them on a subset with
 * {@link GaussianProcess#optimize} and put the inducing inputs at {@link KMeans} centres.
 */
public final class SparseVariationalGp {

    private final StationaryKernel kernel;
    private final double noise;
    private final double[][] z;
    private final double[][] kmmInv;
    private final double[][] kmmChol;
    private final double yMean;
    private double[][] precision;
    private double[] natural;
    private double[] mu;
    private double[][] s;

    private SparseVariationalGp(double[][] z, StationaryKernel kernel, double noise, double yMean) {
        this.kernel = kernel;
        this.noise = noise;
        this.z = z;
        this.yMean = yMean;
        double[][] kmm = kernel.gram(z);
        this.kmmChol = MlMatrices.cholesky(kmm);
        this.kmmInv = MlMatrices.choleskyInverse(kmmChol);
        this.precision = copy(kmmInv);
        this.natural = new double[z.length];
        this.mu = new double[z.length];
        this.s = copy(kmm);
    }

    private static double[][] copy(double[][] a) {
        double[][] c = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            c[i] = a[i].clone();
        }
        return c;
    }

    /**
     * Creates the model with {@code q(u)} equal to the prior.
     *
     * @param z the inducing inputs {@code [M][d]}
     * @param kernel the kernel
     * @param noise the observation noise variance
     * @param yMean the constant prior mean (the targets are modelled as {@code yMean + f})
     * @return the model
     */
    public static SparseVariationalGp create(double[][] z, StationaryKernel kernel, double noise, double yMean) {
        return new SparseVariationalGp(z, kernel, noise, yMean);
    }

    /**
     * Trains by stochastic natural-gradient variational inference.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets {@code [n]}
     * @param z the inducing inputs
     * @param kernel the kernel
     * @param noise the noise variance
     * @param batch the minibatch size
     * @param steps the number of natural-gradient steps
     * @param rate the step size in {@code (0, 1]}; it decays as {@code rate / sqrt(1 + t / 10)}
     * @param seed draws the minibatches
     * @return the trained model
     */
    public static SparseVariationalGp fit(double[][] x, double[] y, double[][] z, StationaryKernel kernel,
            double noise, int batch, int steps, double rate, long seed) {
        double mean = 0;
        for (double v : y) {
            mean += v;
        }
        SparseVariationalGp gp = create(z, kernel, noise, mean / y.length);
        SplittableRandom random = new SplittableRandom(seed);
        int b = Math.min(batch, x.length);
        for (int t = 0; t < steps; t++) {
            double[][] xb = new double[b][];
            double[] yb = new double[b];
            for (int i = 0; i < b; i++) {
                int j = random.nextInt(x.length);
                xb[i] = x[j];
                yb[i] = y[j];
            }
            gp.naturalGradientStep(xb, yb, x.length, rate / Math.sqrt(1 + t / 10.0));
        }
        return gp;
    }

    /**
     * One natural-gradient step on a minibatch: the natural parameters {@code S^-1} and
     * {@code S^-1 mu} move a fraction {@code rho} towards their minibatch estimate of the optimum.
     *
     * @param xb minibatch inputs
     * @param yb minibatch targets
     * @param datasetSize the full data size {@code n}, which rescales the minibatch
     * @param rho the step size in {@code (0, 1]}
     */
    public void naturalGradientStep(double[][] xb, double[] yb, int datasetSize, double rho) {
        int m = z.length;
        double scale = (double) datasetSize / xb.length / noise;
        double[][] kmb = kernel.matrix(z, xb);              // [M][B]
        double[][] akmb = MlMatrices.multiply(kmmInv, kmb); // [M][B]
        double[][] targetP = MlMatrices.multiply(akmb, MlMatrices.transpose(akmb));
        double[] targetH = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                targetP[i][j] = kmmInv[i][j] + scale * targetP[i][j];
            }
            for (int bIdx = 0; bIdx < xb.length; bIdx++) {
                targetH[i] += scale * akmb[i][bIdx] * (yb[bIdx] - yMean);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                precision[i][j] = (1 - rho) * precision[i][j] + rho * targetP[i][j];
            }
            natural[i] = (1 - rho) * natural[i] + rho * targetH[i];
        }
        double[][] pChol = MlMatrices.cholesky(precision);
        s = MlMatrices.choleskyInverse(pChol);
        mu = MlMatrices.choleskySolve(pChol, natural);
    }

    /**
     * Predicts at new inputs.
     *
     * @param xs test inputs
     * @return the predictive mean and variances
     */
    public GaussianProcess.Prediction predict(double[][] xs) {
        double[][] kms = kernel.matrix(z, xs);                 // [M][T]
        double[][] a = MlMatrices.multiply(kmmInv, kms);      // A k*
        double[] aMu = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            aMu[i] = MlMatrices.dot(kmmInv[i], mu);
        }
        double[][] sa = MlMatrices.multiply(s, a);
        double[] mean = new double[xs.length];
        double[] latent = new double[xs.length];
        double[] total = new double[xs.length];
        for (int t = 0; t < xs.length; t++) {
            double mm = yMean;
            double q = 0;
            double r = 0;
            for (int i = 0; i < z.length; i++) {
                mm += kms[i][t] * aMu[i];
                q += kms[i][t] * a[i][t];
                r += a[i][t] * sa[i][t];
            }
            mean[t] = mm;
            latent[t] = Math.max(kernel.value(xs[t], xs[t]) - q + r, 1e-12);
            total[t] = latent[t] + noise;
        }
        return new GaussianProcess.Prediction(mean, total, latent);
    }

    /**
     * The evidence lower bound on a data set.
     *
     * @param x inputs
     * @param y targets
     * @return the ELBO in nats
     */
    public double elbo(double[][] x, double[] y) {
        GaussianProcess.Prediction p = predict(x);
        double fit = 0;
        for (int i = 0; i < x.length; i++) {
            double d = y[i] - p.mean()[i];
            fit += -0.5 * Math.log(2 * Math.PI * noise) - 0.5 * d * d / noise - 0.5 * p.latentVariance()[i] / noise;
        }
        int m = z.length;
        double trace = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                trace += kmmInv[i][j] * s[j][i];
            }
        }
        double quad = 0;
        for (int i = 0; i < m; i++) {
            quad += mu[i] * MlMatrices.dot(kmmInv[i], mu);
        }
        double kl = 0.5 * (trace + quad - m + MlMatrices.logDet(kmmChol) - MlMatrices.logDet(MlMatrices.cholesky(s)));
        return fit - kl;
    }

    /** @return the number of inducing points */
    public int inducingPoints() {
        return z.length;
    }
}
