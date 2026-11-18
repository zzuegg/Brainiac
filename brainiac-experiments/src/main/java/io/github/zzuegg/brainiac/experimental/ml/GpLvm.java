package io.github.zzuegg.brainiac.experimental.ml;

/**
 * The Gaussian-process latent variable model (Lawrence, JMLR 2005): a nonlinear probabilistic PCA
 * whose latent points {@code X [n][q]} are optimised (MAP, with a standard normal prior) to
 * maximise the marginal likelihood of the centred data {@code Y [n][D]} under independent GPs
 * per data dimension sharing one RBF kernel:
 * {@code log p(Y | X) = -D/2 log|K| - 1/2 tr(K^-1 Y Y^T) + const},
 * {@code K = sf2 exp(-|x - x'|^2 / 2 l^2) + I / beta}. Latents start from PCA and move with Adam
 * on the analytic gradient {@code dL/dK = 1/2 (K^-1 Y Y^T K^-1 - D K^-1)} together with the
 * kernel's lengthscale, signal variance and noise. Back-constraints and the Bayesian GPLVM
 * (Titsias and Lawrence, AISTATS 2010) are not implemented.
 */
public final class GpLvm {

    private final double[][] latent;
    private final double lengthscale;
    private final double signal;
    private final double noise;
    private final double logLik;

    private GpLvm(double[][] latent, double lengthscale, double signal, double noise, double logLik) {
        this.latent = latent;
        this.lengthscale = lengthscale;
        this.signal = signal;
        this.noise = noise;
        this.logLik = logLik;
    }

    /**
     * Fits the latent points.
     *
     * @param y the data {@code [n][D]}
     * @param dimensions the latent dimension {@code q}
     * @param iterations Adam steps
     * @param learningRate the Adam step, for example 0.05
     * @return the fitted model
     */
    public static GpLvm fit(double[][] y, int dimensions, int iterations, double learningRate) {
        int n = y.length;
        int d = y[0].length;
        double[] mean = MlMatrices.columnMeans(y);
        double[][] yc = new double[n][d];
        double total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                yc[i][j] = y[i][j] - mean[j];
                total += yc[i][j] * yc[i][j] / (n * d);
            }
        }
        double[][] x = ComponentAnalysis.pca(y, dimensions).transform(y);
        for (int c = 0; c < dimensions; c++) {
            double s = 0;
            for (double[] p : x) {
                s += p[c] * p[c] / n;
            }
            double sd = Math.sqrt(Math.max(s, 1e-12));
            for (double[] p : x) {
                p[c] /= sd;
            }
        }
        // parameters: latents, then log lengthscale, log signal variance, log noise variance
        int np = n * dimensions + 3;
        double[] theta = new double[np];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x[i], 0, theta, i * dimensions, dimensions);
        }
        theta[np - 3] = 0;
        theta[np - 2] = Math.log(total);
        theta[np - 1] = Math.log(0.1 * total);
        double[] m = new double[np];
        double[] v = new double[np];
        double ll = 0;
        for (int t = 1; t <= iterations + 1; t++) {
            double[] g = new double[np];
            ll = objective(theta, yc, n, dimensions, g);
            if (t > iterations) {
                break;
            }
            for (int k = 0; k < np; k++) {
                m[k] = 0.9 * m[k] + 0.1 * g[k];
                v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k];
                theta[k] += learningRate * (m[k] / (1 - Math.pow(0.9, t)))
                        / (Math.sqrt(v[k] / (1 - Math.pow(0.999, t))) + 1e-8);
            }
        }
        double[][] out = new double[n][dimensions];
        for (int i = 0; i < n; i++) {
            System.arraycopy(theta, i * dimensions, out[i], 0, dimensions);
        }
        return new GpLvm(out, Math.exp(theta[np - 3]), Math.exp(theta[np - 2]), Math.exp(theta[np - 1]), ll);
    }

    /** The log joint {@code log p(Y | X) + log p(X)} and its gradient. */
    private static double objective(double[] theta, double[][] y, int n, int q, double[] grad) {
        int d = y[0].length;
        int np = theta.length;
        double ls = Math.exp(theta[np - 3]);
        double sf = Math.exp(theta[np - 2]);
        double sn = Math.exp(theta[np - 1]) + 1e-6;
        double[][] kr = new double[n][n];
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double r2 = 0;
                for (int c = 0; c < q; c++) {
                    double dd = theta[i * q + c] - theta[j * q + c];
                    r2 += dd * dd;
                }
                double v = sf * Math.exp(-r2 / (2 * ls * ls));
                kr[i][j] = v;
                kr[j][i] = v;
                k[i][j] = v;
                k[j][i] = v;
            }
            k[i][i] += sn;
        }
        double[][] l = MlMatrices.cholesky(k);
        double[][] a = MlMatrices.choleskySolve(l, y);
        double[][] inv = MlMatrices.choleskyInverse(l);
        double tr = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                tr += y[i][j] * a[i][j];
            }
        }
        double prior = 0;
        for (int i = 0; i < n * q; i++) {
            prior += theta[i] * theta[i];
        }
        double ll = -0.5 * d * MlMatrices.logDet(l) - 0.5 * tr - 0.5 * prior;
        double gl = 0;
        double gs = 0;
        double gn = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double aa = 0;
                for (int c = 0; c < d; c++) {
                    aa += a[i][c] * a[j][c];
                }
                double gij = 0.5 * (aa - d * inv[i][j]);
                if (i == j) {
                    gn += gij * (sn - 1e-6);
                }
                gs += gij * kr[i][j];
                double r2 = 0;
                for (int c = 0; c < q; c++) {
                    double dd = theta[i * q + c] - theta[j * q + c];
                    r2 += dd * dd;
                    // k_ij and k_ji both depend on x_i
                    grad[i * q + c] += 2 * gij * (-kr[i][j] * dd / (ls * ls));
                }
                gl += gij * kr[i][j] * r2 / (ls * ls);
            }
        }
        for (int i = 0; i < n * q; i++) {
            grad[i] -= theta[i];
        }
        grad[np - 3] = gl;
        grad[np - 2] = gs;
        grad[np - 1] = gn;
        return ll;
    }

    /**
     * The fitted latent points.
     *
     * @return {@code X} {@code [n][q]}
     */
    public double[][] latent() {
        double[][] c = new double[latent.length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = latent[i].clone();
        }
        return c;
    }

    /**
     * The fitted kernel and noise.
     *
     * @return {@code {lengthscale, signal variance, noise variance}}
     */
    public double[] hyperparameters() {
        return new double[] {lengthscale, signal, noise};
    }

    /**
     * The final log joint.
     *
     * @return {@code log p(Y | X) + log p(X)} up to a constant
     */
    public double logLikelihood() {
        return logLik;
    }
}
