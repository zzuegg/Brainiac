package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Objects;

/**
 * Exact Gaussian-process regression and Laplace-approximate binary GP classification
 * (Rasmussen and Williams, <a href="http://gaussianprocess.org/gpml/">Gaussian Processes for
 * Machine Learning</a>, algorithms 2.1, 3.1 and 3.2).
 *
 * <p>Regression puts a GP prior with a {@link StationaryKernel} on the function and Gaussian
 * noise on the targets; the posterior mean and variance come from one Cholesky factor of
 * {@code K + noise I}. {@link #optimize} fits the kernel's lengthscale and variance and the
 * noise by maximising the log marginal likelihood (type-II maximum likelihood) with Adam on the
 * log hyperparameters, using the analytic gradient
 * {@code 1/2 tr((alpha alpha^T - K^-1) dK/dtheta)}.
 *
 * <p>Why it matters: the GP is the reference for calibrated regression uncertainty. Its
 * variance grows away from the data by construction, which deep ensembles, MC dropout and
 * last-layer Laplace only approximate; it is also the surrogate of Bayesian optimisation. The
 * price is {@code O(n^3)}, which {@link SparseVariationalGp} removes.
 */
public final class GaussianProcess {

    /**
     * Predictions at test points.
     *
     * @param mean the posterior mean
     * @param variance the predictive variance of a new target (latent variance plus noise)
     * @param latentVariance the posterior variance of the function itself
     */
    public record Prediction(double[] mean, double[] variance, double[] latentVariance) {
    }

    private final StationaryKernel kernel;
    private final double noise;
    private final double[][] x;
    private final double[] y;
    private final double yMean;
    private final double[][] chol;
    private final double[] alpha;

    private GaussianProcess(StationaryKernel kernel, double noise, double[][] x, double[] y) {
        this.kernel = kernel;
        this.noise = noise;
        this.x = x;
        double m = 0;
        for (double v : y) {
            m += v;
        }
        this.yMean = m / y.length;
        this.y = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            this.y[i] = y[i] - yMean;
        }
        double[][] k = kernel.gram(x);
        for (int i = 0; i < k.length; i++) {
            k[i][i] += noise;
        }
        this.chol = MlMatrices.cholesky(k);
        this.alpha = MlMatrices.choleskySolve(chol, this.y);
    }

    /**
     * Conditions a GP with fixed hyperparameters on data. The targets are centred on their
     * mean, which serves as the constant prior mean.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets {@code [n]}
     * @param kernel the kernel
     * @param noise the observation noise variance, positive
     * @return the posterior
     */
    public static GaussianProcess fit(double[][] x, double[] y, StationaryKernel kernel, double noise) {
        Objects.requireNonNull(kernel, "kernel");
        return new GaussianProcess(kernel, noise, x, y);
    }

    /**
     * Fits the hyperparameters by maximising the log marginal likelihood, then conditions on
     * the data.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets {@code [n]}
     * @param initial the starting kernel (its family is kept)
     * @param initialNoise the starting noise variance
     * @param iterations Adam steps, for example 100
     * @return the posterior with fitted hyperparameters
     */
    public static GaussianProcess optimize(double[][] x, double[] y, StationaryKernel initial, double initialNoise,
            int iterations) {
        double[] theta = {Math.log(initial.lengthscale()), Math.log(initial.variance()), Math.log(initialNoise)};
        double[] m = new double[3];
        double[] v = new double[3];
        double lr = 0.05;
        GaussianProcess gp = fit(x, y, initial, initialNoise);
        for (int t = 1; t <= iterations; t++) {
            double[] g = gp.logMarginalGradient();
            for (int p = 0; p < 3; p++) {
                m[p] = 0.9 * m[p] + 0.1 * g[p];
                v[p] = 0.999 * v[p] + 0.001 * g[p] * g[p];
                double mh = m[p] / (1 - Math.pow(0.9, t));
                double vh = v[p] / (1 - Math.pow(0.999, t));
                theta[p] += lr * mh / (Math.sqrt(vh) + 1e-8);   // ascent
            }
            theta[2] = Math.max(theta[2], Math.log(1e-6));
            gp = fit(x, y, initial.with(Math.exp(theta[0]), Math.exp(theta[1])), Math.exp(theta[2]));
        }
        return gp;
    }

    /**
     * The log marginal likelihood {@code log p(y | X, theta)} of the centred targets.
     *
     * @return the log evidence in nats
     */
    public double logMarginalLikelihood() {
        return -0.5 * MlMatrices.dot(y, alpha) - 0.5 * MlMatrices.logDet(chol) - 0.5 * y.length * Math.log(2 * Math.PI);
    }

    /**
     * The gradient of the log marginal likelihood in the log lengthscale, log variance and log
     * noise.
     *
     * @return {@code {d/d log l, d/d log variance, d/d log noise}}
     */
    public double[] logMarginalGradient() {
        int n = y.length;
        double[][] kInv = MlMatrices.choleskyInverse(chol);
        double gl = 0;
        double gv = 0;
        double gn = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double w = alpha[i] * alpha[j] - kInv[i][j];
                double kij = kernel.value(x[i], x[j]);
                gl += w * kernel.dLogLengthscale(x[i], x[j]);
                gv += w * kij;
                if (i == j) {
                    gn += w * noise;
                }
            }
        }
        return new double[] {0.5 * gl, 0.5 * gv, 0.5 * gn};
    }

    /**
     * Twice the gradient of the negative log marginal likelihood with respect to the kernel
     * matrix: {@code K^-1 - alpha alpha^T}, for backpropagating into kernel inputs.
     *
     * @return {@code [n][n]}
     */
    public double[][] nllKernelGradient() {
        double[][] w = MlMatrices.choleskyInverse(chol);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w.length; j++) {
                w[i][j] -= alpha[i] * alpha[j];
            }
        }
        return w;
    }

    /**
     * Predicts at new inputs.
     *
     * @param xs test inputs {@code [m][d]}
     * @return the predictive mean and variances
     */
    public Prediction predict(double[][] xs) {
        double[][] ks = kernel.matrix(x, xs);   // [n][m]
        double[][] v = MlMatrices.triangularSolve(chol, ks, false);
        double[] mean = new double[xs.length];
        double[] latent = new double[xs.length];
        double[] total = new double[xs.length];
        for (int j = 0; j < xs.length; j++) {
            double mu = yMean;
            double vv = 0;
            for (int i = 0; i < x.length; i++) {
                mu += ks[i][j] * alpha[i];
                vv += v[i][j] * v[i][j];
            }
            mean[j] = mu;
            latent[j] = Math.max(kernel.value(xs[j], xs[j]) - vv, 1e-12);
            total[j] = latent[j] + noise;
        }
        return new Prediction(mean, total, latent);
    }

    /** @return the kernel */
    public StationaryKernel kernel() {
        return kernel;
    }

    /** @return the noise variance */
    public double noise() {
        return noise;
    }

    /**
     * Fits a binary GP classifier with the Laplace approximation and a logistic likelihood.
     *
     * @param x inputs {@code [n][d]}
     * @param labels 0 or 1 per input
     * @param kernel the kernel, fixed
     * @return the classifier
     */
    public static Classifier classifier(double[][] x, int[] labels, KernelFunction kernel) {
        return new Classifier(x, labels, kernel);
    }

    /**
     * Binary GP classification by the Laplace approximation (GPML algorithms 3.1 and 3.2): Newton
     * iterations find the mode of the latent posterior, whose curvature gives a Gaussian;
     * predictions average the logistic over the latent Gaussian by the probit approximation.
     */
    public static final class Classifier {
        private final KernelFunction kernel;
        private final double[][] x;
        private final double[] t;
        private final double[] gradLog;
        private final double[] sqrtW;
        private final double[][] chol;
        private final double logMarginal;

        private Classifier(double[][] x, int[] labels, KernelFunction kernel) {
            this.kernel = kernel;
            this.x = x;
            int n = x.length;
            this.t = new double[n];
            for (int i = 0; i < n; i++) {
                t[i] = labels[i];
            }
            double[][] k = kernel.gram(x);
            double[] f = new double[n];
            double[] a = new double[n];
            double[] pi = new double[n];
            double[] sw = new double[n];
            double[][] l = null;
            double previous = Double.NEGATIVE_INFINITY;
            double objective = 0;
            for (int iter = 0; iter < 50; iter++) {
                for (int i = 0; i < n; i++) {
                    pi[i] = sigmoid(f[i]);
                    sw[i] = Math.sqrt(pi[i] * (1 - pi[i]));
                }
                double[][] b = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        b[i][j] = sw[i] * k[i][j] * sw[j] + (i == j ? 1 : 0);
                    }
                }
                l = MlMatrices.cholesky(b);
                double[] bv = new double[n];
                for (int i = 0; i < n; i++) {
                    bv[i] = sw[i] * sw[i] * f[i] + (t[i] - pi[i]);
                }
                double[] kb = matVec(k, bv);
                double[] swkb = new double[n];
                for (int i = 0; i < n; i++) {
                    swkb[i] = sw[i] * kb[i];
                }
                double[] inner = MlMatrices.choleskySolve(l, swkb);
                for (int i = 0; i < n; i++) {
                    a[i] = bv[i] - sw[i] * inner[i];
                }
                f = matVec(k, a);
                objective = -0.5 * MlMatrices.dot(a, f);
                for (int i = 0; i < n; i++) {
                    objective += logLikelihood(t[i], f[i]);
                }
                if (Math.abs(objective - previous) < 1e-8) {
                    break;
                }
                previous = objective;
            }
            for (int i = 0; i < n; i++) {
                pi[i] = sigmoid(f[i]);
                sw[i] = Math.sqrt(pi[i] * (1 - pi[i]));
            }
            double[][] b = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    b[i][j] = sw[i] * k[i][j] * sw[j] + (i == j ? 1 : 0);
                }
            }
            l = MlMatrices.cholesky(b);
            this.chol = l;
            this.sqrtW = sw;
            this.gradLog = new double[n];
            for (int i = 0; i < n; i++) {
                gradLog[i] = t[i] - pi[i];
            }
            this.logMarginal = objective - 0.5 * MlMatrices.logDet(l);
        }

        private static double[] matVec(double[][] k, double[] v) {
            double[] out = new double[v.length];
            for (int i = 0; i < k.length; i++) {
                out[i] = MlMatrices.dot(k[i], v);
            }
            return out;
        }

        private static double sigmoid(double z) {
            return 1 / (1 + Math.exp(-z));
        }

        private static double logLikelihood(double t, double f) {
            // log sigmoid(+-f), stable
            double s = t > 0.5 ? f : -f;
            return -Math.log1p(Math.exp(-Math.abs(s))) + Math.min(s, 0);
        }

        /**
         * The Laplace approximation of the log marginal likelihood, for choosing kernel
         * hyperparameters.
         *
         * @return the approximate log evidence
         */
        public double logMarginalLikelihood() {
            return logMarginal;
        }

        /**
         * The predictive probability of class 1.
         *
         * @param xs test inputs
         * @return {@code p(y = 1 | x)} per input
         */
        public double[] predictProbability(double[][] xs) {
            double[][] ks = kernel.matrix(x, xs);
            double[] out = new double[xs.length];
            double[] col = new double[x.length];
            for (int j = 0; j < xs.length; j++) {
                double mean = 0;
                for (int i = 0; i < x.length; i++) {
                    mean += ks[i][j] * gradLog[i];
                    col[i] = sqrtW[i] * ks[i][j];
                }
                double[] v = MlMatrices.forwardSolve(chol, col);
                double var = Math.max(kernel.value(xs[j], xs[j]) - MlMatrices.dot(v, v), 1e-12);
                out[j] = sigmoid(mean / Math.sqrt(1 + Math.PI * var / 8));
            }
            return out;
        }
    }
}
