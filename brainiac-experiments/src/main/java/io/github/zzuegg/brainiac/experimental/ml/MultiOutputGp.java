package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * A multi-output Gaussian process through the linear model of coregionalisation (Alvarez, Rosasco
 * and Lawrence, <i>Kernels for vector-valued functions</i>, 2012,
 * <a href="https://arxiv.org/abs/1106.6251">arXiv:1106.6251</a>).
 *
 * <p>Output {@code p} at input {@code x} is {@code f_p(x) = sum_q sum_r a_{q,p,r} u_{q,r}(x)}
 * with independent latent GPs {@code u_{q,r}} sharing the RBF kernel {@code k_q}. The covariance
 * between observations is {@code sum_q B_q[p][p'] k_q(x, x')} with coregionalisation matrices
 * {@code B_q = W_q W_q^T + diag(kappa_q)} of rank {@code R} plus a diagonal, and each output has
 * its own noise. One latent kernel ({@code Q = 1}) is the intrinsic coregionalisation model (ICM);
 * several kernels with different lengthscales give the LMC. All hyperparameters (the {@code W_q},
 * {@code kappa_q}, lengthscales, noise) maximise the exact log marginal likelihood with Adam and
 * analytic gradients {@code 1/2 tr((alpha alpha^T - K^-1) dK)}. Observations need not share inputs
 * across outputs, so an output missing in a region borrows strength from the others there.
 */
public final class MultiOutputGp {

    private final int outputs;
    private final int latent;
    private final int rank;
    private final double[][] x;
    private final int[] out;
    private final double[] mean;
    private final double[] lengthscale;
    private final double[][][] w;
    private final double[][] kappa;
    private final double[] noise;
    private double[][] chol;
    private double[] alpha;
    private double logLik;

    private MultiOutputGp(double[][] x, int[] out, int outputs, int latent, int rank) {
        this.x = x;
        this.out = out;
        this.outputs = outputs;
        this.latent = latent;
        this.rank = rank;
        this.mean = new double[outputs];
        this.lengthscale = new double[latent];
        this.w = new double[latent][outputs][rank];
        this.kappa = new double[latent][outputs];
        this.noise = new double[outputs];
    }

    /**
     * Fits an ICM ({@code latentKernels = 1}) or LMC model.
     *
     * @param x inputs of all observations {@code [n][d]}
     * @param output the output index of each observation
     * @param y the observed values {@code [n]}
     * @param outputs the number of outputs {@code P}
     * @param latentKernels the number of latent kernels {@code Q}
     * @param rank the rank {@code R} of each {@code W_q}
     * @param iterations Adam steps on the log marginal likelihood
     * @param seed seeds the initial {@code W_q}
     * @return the fitted model
     */
    public static MultiOutputGp fit(double[][] x, int[] output, double[] y, int outputs, int latentKernels, int rank,
            int iterations, long seed) {
        MultiOutputGp gp = new MultiOutputGp(x, output, outputs, latentKernels, rank);
        gp.train(y, iterations, new SplittableRandom(seed));
        return gp;
    }

    private int parameterCount() {
        return latent * (1 + outputs * rank + outputs) + outputs;
    }

    private void unpack(double[] theta) {
        int c = 0;
        for (int q = 0; q < latent; q++) {
            lengthscale[q] = Math.exp(theta[c++]);
            for (int p = 0; p < outputs; p++) {
                for (int r = 0; r < rank; r++) {
                    w[q][p][r] = theta[c++];
                }
            }
            for (int p = 0; p < outputs; p++) {
                kappa[q][p] = Math.exp(theta[c++]);
            }
        }
        for (int p = 0; p < outputs; p++) {
            noise[p] = Math.exp(theta[c++]) + 1e-6;
        }
    }

    private double b(int q, int i, int j) {
        double s = i == j ? kappa[q][i] : 0;
        for (int r = 0; r < rank; r++) {
            s += w[q][i][r] * w[q][j][r];
        }
        return s;
    }

    private double kq(int q, double[] a, double[] c) {
        return Math.exp(-MlMatrices.squaredDistance(a, c) / (2 * lengthscale[q] * lengthscale[q]));
    }

    private void train(double[] yRaw, int iterations, SplittableRandom random) {
        int n = x.length;
        int[] count = new int[outputs];
        double[] var = new double[outputs];
        for (int i = 0; i < n; i++) {
            mean[out[i]] += yRaw[i];
            count[out[i]]++;
        }
        for (int p = 0; p < outputs; p++) {
            mean[p] /= Math.max(count[p], 1);
        }
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = yRaw[i] - mean[out[i]];
            var[out[i]] += y[i] * y[i] / Math.max(count[out[i]], 1);
        }
        double[] theta = new double[parameterCount()];
        int c = 0;
        double span = inputSpan();
        for (int q = 0; q < latent; q++) {
            // spread initial lengthscales so LMC kernels start on different scales
            theta[c++] = Math.log(span * 0.2 * Math.pow(3, q));
            for (int p = 0; p < outputs; p++) {
                for (int r = 0; r < rank; r++) {
                    theta[c++] = Math.sqrt(Math.max(var[p], 1e-3) / (latent * rank)) * (0.5 + random.nextDouble());
                }
            }
            for (int p = 0; p < outputs; p++) {
                theta[c++] = Math.log(0.1 * Math.max(var[p], 1e-3) / latent);
            }
        }
        for (int p = 0; p < outputs; p++) {
            theta[c++] = Math.log(0.1 * Math.max(var[p], 1e-3));
        }
        double[] m = new double[theta.length];
        double[] v = new double[theta.length];
        double lr = 0.05;
        for (int t = 1; t <= iterations; t++) {
            unpack(theta);
            double[] g = gradient(y);
            for (int k = 0; k < theta.length; k++) {
                // ascend the log marginal likelihood
                m[k] = 0.9 * m[k] + 0.1 * g[k];
                v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k];
                double mh = m[k] / (1 - Math.pow(0.9, t));
                double vh = v[k] / (1 - Math.pow(0.999, t));
                theta[k] += lr * mh / (Math.sqrt(vh) + 1e-8);
            }
        }
        unpack(theta);
        factor(y);
    }

    private double inputSpan() {
        double s = 0;
        for (int j = 0; j < x[0].length; j++) {
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (double[] p : x) {
                lo = Math.min(lo, p[j]);
                hi = Math.max(hi, p[j]);
            }
            s = Math.max(s, hi - lo);
        }
        return Math.max(s, 1e-3);
    }

    private double[][][] latentGrams() {
        int n = x.length;
        double[][][] kq = new double[latent][n][n];
        for (int q = 0; q < latent; q++) {
            for (int a = 0; a < n; a++) {
                for (int c = a; c < n; c++) {
                    double v = kq(q, x[a], x[c]);
                    kq[q][a][c] = v;
                    kq[q][c][a] = v;
                }
            }
        }
        return kq;
    }

    private double[][] covariance(double[][][] kq) {
        int n = x.length;
        double[][] k = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int c = 0; c < n; c++) {
                double s = a == c ? noise[out[a]] : 0;
                for (int q = 0; q < latent; q++) {
                    s += b(q, out[a], out[c]) * kq[q][a][c];
                }
                k[a][c] = s;
            }
        }
        return k;
    }

    private void factor(double[] y) {
        double[][] k = covariance(latentGrams());
        chol = MlMatrices.cholesky(k);
        alpha = MlMatrices.choleskySolve(chol, y);
        logLik = -0.5 * MlMatrices.dot(y, alpha) - 0.5 * MlMatrices.logDet(chol)
                - 0.5 * y.length * Math.log(2 * Math.PI);
    }

    private double[] gradient(double[] y) {
        int n = x.length;
        double[][][] kq = latentGrams();
        double[][] k = covariance(kq);
        double[][] l = MlMatrices.cholesky(k);
        double[] a = MlMatrices.choleskySolve(l, y);
        double[][] kInv = MlMatrices.choleskyInverse(l);
        double[][] mm = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mm[i][j] = 0.5 * (a[i] * a[j] - kInv[i][j]);
            }
        }
        double[] g = new double[parameterCount()];
        int c = 0;
        for (int q = 0; q < latent; q++) {
            double ls2 = lengthscale[q] * lengthscale[q];
            double gl = 0;
            double[][] s = new double[outputs][outputs];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double t = mm[i][j] * kq[q][i][j];
                    s[out[i]][out[j]] += t;
                    gl += t * b(q, out[i], out[j]) * MlMatrices.squaredDistance(x[i], x[j]) / ls2;
                }
            }
            g[c++] = gl;
            for (int p = 0; p < outputs; p++) {
                for (int r = 0; r < rank; r++) {
                    double v = 0;
                    for (int j = 0; j < outputs; j++) {
                        v += s[p][j] * w[q][j][r];
                    }
                    g[c++] = 2 * v;
                }
            }
            for (int p = 0; p < outputs; p++) {
                g[c++] = s[p][p] * kappa[q][p];
            }
        }
        for (int p = 0; p < outputs; p++) {
            double v = 0;
            for (int i = 0; i < n; i++) {
                if (out[i] == p) {
                    v += mm[i][i];
                }
            }
            g[c++] = v * (noise[p] - 1e-6);
        }
        return g;
    }

    /**
     * The predictive distribution of one output, noise included.
     *
     * @param xs test inputs {@code [m][d]}
     * @param output the output
     * @return {@code {mean[m], variance[m]}}
     */
    public double[][] predict(double[][] xs, int output) {
        int n = x.length;
        double[] mu = new double[xs.length];
        double[] var = new double[xs.length];
        double prior = noise[output];
        for (int q = 0; q < latent; q++) {
            prior += b(q, output, output);
        }
        for (int t = 0; t < xs.length; t++) {
            double[] ks = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int q = 0; q < latent; q++) {
                    s += b(q, output, out[i]) * kq(q, xs[t], x[i]);
                }
                ks[i] = s;
            }
            mu[t] = mean[output] + MlMatrices.dot(ks, alpha);
            double[] v = MlMatrices.forwardSolve(chol, ks);
            var[t] = Math.max(prior - MlMatrices.dot(v, v), 1e-12);
        }
        return new double[][] {mu, var};
    }

    /**
     * The fitted log marginal likelihood.
     *
     * @return {@code log p(y)}
     */
    public double logMarginalLikelihood() {
        return logLik;
    }

    /**
     * One fitted coregionalisation matrix.
     *
     * @param q the latent kernel
     * @return {@code B_q} {@code [P][P]}
     */
    public double[][] coregionalisation(int q) {
        double[][] bq = new double[outputs][outputs];
        for (int i = 0; i < outputs; i++) {
            for (int j = 0; j < outputs; j++) {
                bq[i][j] = b(q, i, j);
            }
        }
        return bq;
    }

    /**
     * The fitted lengthscales of the latent kernels.
     *
     * @return {@code [Q]}
     */
    public double[] lengthscales() {
        return lengthscale.clone();
    }
}
