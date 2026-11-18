package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;

/**
 * The spectral mixture kernel (Wilson and Adams 2013,
 * <a href="https://arxiv.org/abs/1302.4245">arXiv:1302.4245</a>) for one-dimensional inputs:
 * {@code k(tau) = sum_q w_q exp(-2 pi^2 tau^2 v_q) cos(2 pi tau mu_q)}. By Bochner's theorem
 * every stationary kernel is the Fourier transform of a spectral density; modelling that
 * density as a mixture of Gaussians (at frequencies {@code mu_q} with widths {@code v_q})
 * approximates any stationary kernel, so a single GP can discover quasi-periodic structure and
 * extrapolate it far beyond the data, which an RBF kernel (one Gaussian at frequency 0) cannot.
 *
 * <p>{@link #fit} initialises the frequencies at the periodogram peaks of the data, then
 * maximises the log marginal likelihood of a {@link KernelGaussianProcess} with Adam on the log
 * hyperparameters and the noise, using the analytic kernel derivatives.
 *
 * @param weights the component weights {@code w_q}
 * @param means the component frequencies {@code mu_q} (cycles per unit of input)
 * @param variances the component spectral variances {@code v_q}
 */
public record SpectralMixtureKernel(double[] weights, double[] means, double[] variances) implements KernelFunction {

    /** Copies the arrays. */
    public SpectralMixtureKernel {
        weights = weights.clone();
        means = means.clone();
        variances = variances.clone();
    }

    @Override
    public double value(double[] a, double[] b) {
        double tau = a[0] - b[0];
        double s = 0;
        for (int q = 0; q < weights.length; q++) {
            s += weights[q] * Math.exp(-2 * Math.PI * Math.PI * tau * tau * variances[q])
                    * Math.cos(2 * Math.PI * tau * means[q]);
        }
        return s;
    }

    /**
     * A fitted GP with its kernel.
     *
     * @param kernel the fitted kernel
     * @param gp the GP conditioned on the training data
     */
    public record Fit(SpectralMixtureKernel kernel, KernelGaussianProcess gp) {
    }

    /**
     * Fits the kernel and noise by type-II maximum likelihood.
     *
     * @param x inputs (one column)
     * @param y targets
     * @param components the number of mixture components {@code Q}
     * @param iterations Adam steps, for example 150
     * @return the fitted kernel and GP
     */
    public static Fit fit(double[][] x, double[] y, int components, int iterations) {
        int n = x.length;
        double mean = Arrays.stream(y).average().orElse(0);
        double var = 0;
        for (double v : y) {
            var += (v - mean) * (v - mean) / n;
        }
        double span = x[n - 1][0] - x[0][0];
        double dx = span / (n - 1);
        // periodogram peaks as initial frequencies
        int bins = n / 2;
        double[] power = new double[bins];
        for (int k = 1; k < bins; k++) {
            double f = k / (n * dx);
            double re = 0;
            double im = 0;
            for (int i = 0; i < n; i++) {
                re += (y[i] - mean) * Math.cos(2 * Math.PI * f * x[i][0]);
                im += (y[i] - mean) * Math.sin(2 * Math.PI * f * x[i][0]);
            }
            power[k] = re * re + im * im;
        }
        double[] theta = new double[3 * components + 1];
        boolean[] used = new boolean[bins];
        for (int q = 0; q < components; q++) {
            int best = 1;
            for (int k = 1; k < bins; k++) {
                if (!used[k] && (used[best] || power[k] > power[best])) {
                    best = k;
                }
            }
            used[best] = true;
            for (int k = Math.max(1, best - 1); k <= Math.min(bins - 1, best + 1); k++) {
                used[k] = true;
            }
            theta[q] = Math.log(var / components);
            theta[components + q] = Math.log(best / (n * dx));
            theta[2 * components + q] = Math.log(Math.pow(1 / span, 2));
        }
        theta[3 * components] = Math.log(0.1 * var + 1e-6);
        double[] m = new double[theta.length];
        double[] v = new double[theta.length];
        Fit fit = build(x, y, theta, components);
        Fit best = fit;
        double bestLml = fit.gp().logMarginalLikelihood();
        double lr = 0.05;
        for (int t = 1; t <= iterations; t++) {
            double[] g = gradient(x, fit, theta, components);
            for (int p = 0; p < theta.length; p++) {
                m[p] = 0.9 * m[p] + 0.1 * g[p];
                v[p] = 0.999 * v[p] + 0.001 * g[p] * g[p];
                theta[p] += lr * (m[p] / (1 - Math.pow(0.9, t))) / (Math.sqrt(v[p] / (1 - Math.pow(0.999, t))) + 1e-8);
            }
            theta[3 * components] = Math.max(theta[3 * components], Math.log(1e-6));
            fit = build(x, y, theta, components);
            double lml = fit.gp().logMarginalLikelihood();
            if (lml > bestLml) {
                bestLml = lml;
                best = fit;
            }
        }
        return best;
    }

    private static Fit build(double[][] x, double[] y, double[] theta, int qn) {
        double[] w = new double[qn];
        double[] mu = new double[qn];
        double[] vv = new double[qn];
        for (int q = 0; q < qn; q++) {
            w[q] = Math.exp(theta[q]);
            mu[q] = Math.exp(theta[qn + q]);
            vv[q] = Math.exp(theta[2 * qn + q]);
        }
        SpectralMixtureKernel k = new SpectralMixtureKernel(w, mu, vv);
        return new Fit(k, KernelGaussianProcess.fit(x, y, k, Math.exp(theta[3 * qn])));
    }

    private static double[] gradient(double[][] x, Fit fit, double[] theta, int qn) {
        double[][] w = fit.gp().marginalLikelihoodWeights();
        SpectralMixtureKernel k = fit.kernel();
        int n = x.length;
        double[] g = new double[theta.length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double tau = x[i][0] - x[j][0];
                double wij = 0.5 * w[i][j];
                for (int q = 0; q < qn; q++) {
                    double e = Math.exp(-2 * Math.PI * Math.PI * tau * tau * k.variances[q]);
                    double c = Math.cos(2 * Math.PI * tau * k.means[q]);
                    double kq = k.weights[q] * e * c;
                    g[q] += wij * kq;
                    g[qn + q] += wij * k.weights[q] * e * -Math.sin(2 * Math.PI * tau * k.means[q]) * 2 * Math.PI * tau
                            * k.means[q];
                    g[2 * qn + q] += wij * kq * -2 * Math.PI * Math.PI * tau * tau * k.variances[q];
                }
                if (i == j) {
                    g[3 * qn] += wij * fit.gp().noise();
                }
            }
        }
        return g;
    }
}
