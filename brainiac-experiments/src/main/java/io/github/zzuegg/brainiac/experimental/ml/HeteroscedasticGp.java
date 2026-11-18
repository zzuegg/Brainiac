package io.github.zzuegg.brainiac.experimental.ml;

/**
 * Gaussian-process regression with input-dependent noise by the most-likely-heteroscedastic EM of
 * Kersting, Plagemann, Pfaff and Burgard (ICML 2007), a point-estimate version of Goldberg,
 * Williams and Bishop (NeurIPS 1998), who model the log noise variance as a second GP.
 *
 * <p>Starting from a homoscedastic {@link GaussianProcess}, each round
 * <ol>
 *   <li>estimates the noise variance at every training input as the expected squared residual
 *       under the current posterior, {@code (y_i - mu_i)^2 + s_i^2};</li>
 *   <li>fits a homoscedastic GP to the log of these estimates (plus {@code gamma + log 2}, the
 *       bias of {@code log chi^2_1}, so the noise GP tracks {@code log sigma^2} rather than the
 *       mean log squared residual);</li>
 *   <li>refits the mean GP with fixed per-point noise {@code exp(z(x_i))}, optimising its kernel by
 *       the log marginal likelihood.</li>
 * </ol>
 * Predictions add the noise GP's mean variance to the latent variance.
 */
public final class HeteroscedasticGp {

    private static final double LOG_CHI2_BIAS = 0.5772156649015329 + Math.log(2);

    private final double[][] x;
    private final double yMean;
    private final StationaryKernel kernel;
    private final GaussianProcess noiseGp;
    private final double[][] chol;
    private final double[] alpha;

    private HeteroscedasticGp(double[][] x, double yMean, StationaryKernel kernel, GaussianProcess noiseGp,
            double[][] chol, double[] alpha) {
        this.x = x;
        this.yMean = yMean;
        this.kernel = kernel;
        this.noiseGp = noiseGp;
        this.chol = chol;
        this.alpha = alpha;
    }

    /**
     * Fits the model.
     *
     * @param x inputs {@code [n][d]}
     * @param y targets {@code [n]}
     * @param initial the starting mean kernel
     * @param rounds EM rounds, for example 4
     * @param iterations Adam steps per GP fit
     * @return the fitted model
     */
    public static HeteroscedasticGp fit(double[][] x, double[] y, StationaryKernel initial, int rounds,
            int iterations) {
        int n = x.length;
        double ym = 0;
        for (double v : y) {
            ym += v / n;
        }
        double[] yc = new double[n];
        double var = 0;
        for (int i = 0; i < n; i++) {
            yc[i] = y[i] - ym;
            var += yc[i] * yc[i] / n;
        }
        GaussianProcess home = GaussianProcess.optimize(x, y, initial, 0.1 * var, iterations);
        GaussianProcess.Prediction p = home.predict(x);
        double[] mu = p.mean();
        double[] latent = p.latentVariance();
        StationaryKernel k = home.kernel();
        GaussianProcess noise = null;
        double[][] l = null;
        double[] a = null;
        for (int round = 0; round < rounds; round++) {
            double[] z = new double[n];
            for (int i = 0; i < n; i++) {
                double r = y[i] - mu[i];
                z[i] = Math.log(r * r + latent[i] + 1e-12) + LOG_CHI2_BIAS;
            }
            noise = GaussianProcess.optimize(x, z, StationaryKernel.rbf(k.lengthscale() * 2, 1), 1, iterations);
            double[] r = new double[n];
            double[] zm = noise.predict(x).mean();
            for (int i = 0; i < n; i++) {
                r[i] = Math.exp(zm[i]);
            }
            k = optimiseKernel(x, yc, r, k, iterations);
            double[][] cov = gram(x, k, r);
            l = MlMatrices.cholesky(cov);
            a = MlMatrices.choleskySolve(l, yc);
            for (int i = 0; i < n; i++) {
                double[] ki = new double[n];
                for (int j = 0; j < n; j++) {
                    ki[j] = k.value(x[i], x[j]);
                }
                mu[i] = ym + MlMatrices.dot(ki, a);
                double[] col = MlMatrices.forwardSolve(l, ki);
                latent[i] = Math.max(k.variance() - MlMatrices.dot(col, col), 1e-12);
            }
        }
        return new HeteroscedasticGp(x, ym, k, noise, l, a);
    }

    private static double[][] gram(double[][] x, StationaryKernel k, double[] noise) {
        int n = x.length;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double v = k.value(x[i], x[j]);
                c[i][j] = v;
                c[j][i] = v;
            }
            c[i][i] += noise[i];
        }
        return c;
    }

    /** Adam on log lengthscale and log variance with fixed per-point noise. */
    private static StationaryKernel optimiseKernel(double[][] x, double[] y, double[] noise, StationaryKernel start,
            int iterations) {
        int n = x.length;
        double[] theta = {Math.log(start.lengthscale()), Math.log(start.variance())};
        double[] m = new double[2];
        double[] v = new double[2];
        StationaryKernel k = start;
        for (int t = 1; t <= iterations; t++) {
            k = start.with(Math.exp(theta[0]), Math.exp(theta[1]));
            double[][] l = MlMatrices.cholesky(gram(x, k, noise));
            double[] a = MlMatrices.choleskySolve(l, y);
            double[][] inv = MlMatrices.choleskyInverse(l);
            double[] g = new double[2];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double w = 0.5 * (a[i] * a[j] - inv[i][j]);
                    double kv = k.value(x[i], x[j]);
                    g[0] += w * kv * MlMatrices.squaredDistance(x[i], x[j]) / (k.lengthscale() * k.lengthscale());
                    g[1] += w * kv;
                }
            }
            for (int p = 0; p < 2; p++) {
                m[p] = 0.9 * m[p] + 0.1 * g[p];
                v[p] = 0.999 * v[p] + 0.001 * g[p] * g[p];
                theta[p] += 0.05 * (m[p] / (1 - Math.pow(0.9, t))) / (Math.sqrt(v[p] / (1 - Math.pow(0.999, t)))
                        + 1e-8);
            }
        }
        return start.with(Math.exp(theta[0]), Math.exp(theta[1]));
    }

    /**
     * Predictions with input-dependent noise.
     *
     * @param xs test inputs {@code [m][d]}
     * @return mean, total variance (latent plus noise) and latent variance
     */
    public GaussianProcess.Prediction predict(double[][] xs) {
        int n = x.length;
        double[] noise = noiseVariance(xs);
        double[] mu = new double[xs.length];
        double[] var = new double[xs.length];
        double[] lat = new double[xs.length];
        for (int t = 0; t < xs.length; t++) {
            double[] ks = new double[n];
            for (int i = 0; i < n; i++) {
                ks[i] = kernel.value(xs[t], x[i]);
            }
            mu[t] = yMean + MlMatrices.dot(ks, alpha);
            double[] v = MlMatrices.forwardSolve(chol, ks);
            lat[t] = Math.max(kernel.variance() - MlMatrices.dot(v, v), 1e-12);
            var[t] = lat[t] + noise[t];
        }
        return new GaussianProcess.Prediction(mu, var, lat);
    }

    /**
     * The fitted noise variance.
     *
     * @param xs inputs
     * @return {@code exp(z(x))}
     */
    public double[] noiseVariance(double[][] xs) {
        double[] z = noiseGp.predict(xs).mean();
        double[] r = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            r[i] = Math.exp(z[i]);
        }
        return r;
    }

    /**
     * The fitted mean kernel.
     *
     * @return the kernel
     */
    public StationaryKernel kernel() {
        return kernel;
    }
}
