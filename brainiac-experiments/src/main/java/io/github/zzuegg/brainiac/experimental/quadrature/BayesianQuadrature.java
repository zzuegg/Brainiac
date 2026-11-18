package io.github.zzuegg.brainiac.experimental.quadrature;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Bayesian quadrature of {@code int f(x) N(x; 0, I) dx} (O'Hagan 1991; Rasmussen and
 * Ghahramani, "Bayesian Monte Carlo", NeurIPS 2003) with a zero-mean GP prior and the isotropic
 * Gaussian kernel {@code k(x, x') = s^2 exp(-|x - x'|^2 / (2 l^2))}, for which the kernel mean
 * {@code z_i = s^2 (l^2/(l^2+1))^(d/2) exp(-|x_i|^2 / (2(l^2+1)))} and the initial variance
 * {@code s^2 (l^2/(l^2+2))^(d/2)} are closed-form; the posterior of the integral is Gaussian
 * with mean {@code z^T K^-1 f} and variance {@code k_00 - z^T K^-1 z}. The length scale is
 * chosen by marginal likelihood on a grid and {@code s^2} by its closed-form maximiser.
 * {@link #sequential} adds the candidate point that minimises the posterior variance of the
 * integral; {@link #wsabi} is WSABI-L (Gunter et al., NeurIPS 2014) for positive integrands
 * such as likelihoods: a GP on {@code g = sqrt(2 (f - alpha))} with the linearised integral
 * {@code alpha + 1/2 int m_g(x)^2 N(x) dx}, also closed-form.
 */
public final class BayesianQuadrature {

    private BayesianQuadrature() {
    }

    /**
     * A posterior over the integral.
     *
     * @param mean the posterior mean
     * @param sd the posterior standard deviation (NaN for WSABI)
     * @param lengthScale the fitted length scale
     * @param points the evaluation points
     */
    public record Estimate(double mean, double sd, double lengthScale, double[][] points) {
    }

    private static double sq(double[] a, double[] b) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            s += (a[j] - b[j]) * (a[j] - b[j]);
        }
        return s;
    }

    private static double[][] gram(double[][] x, double l) {
        int n = x.length;
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                k[i][j] = Math.exp(-sq(x[i], x[j]) / (2 * l * l)) + (i == j ? 1e-8 : 0);
            }
        }
        return k;
    }

    private static double[] kernelMean(double[][] x, double l) {
        int d = x[0].length;
        double c = Math.pow(l * l / (l * l + 1), d / 2.0);
        double[] z = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            z[i] = c * Math.exp(-sq(x[i], new double[d]) / (2 * (l * l + 1)));
        }
        return z;
    }

    private static double initialVariance(int d, double l) {
        return Math.pow(l * l / (l * l + 2), d / 2.0);
    }

    /** Log marginal likelihood with the closed-form amplitude, up to a constant. */
    private static double evidence(double[][] k, double[] f) {
        int n = f.length;
        double[][] ch = Matrices.cholesky(k);
        double[] a = Matrices.solve(k, f);
        double quad = 0;
        double logDet = 0;
        for (int i = 0; i < n; i++) {
            quad += f[i] * a[i];
            logDet += 2 * Math.log(ch[i][i]);
        }
        double s2 = Math.max(quad / n, 1e-300);
        return -0.5 * n * Math.log(s2) - 0.5 * logDet;
    }

    private static double fitLength(double[][] x, double[] f) {
        double best = 1;
        double bv = Double.NEGATIVE_INFINITY;
        for (int g = 0; g < 25; g++) {
            double l = 0.1 * Math.pow(50, g / 24.0);
            double v;
            try {
                v = evidence(gram(x, l), f);
            } catch (RuntimeException e) {
                v = Double.NEGATIVE_INFINITY;
            }
            if (v > bv) {
                bv = v;
                best = l;
            }
        }
        return best;
    }

    /**
     * Bayesian quadrature on given points.
     *
     * @param x the points {@code [n][d]}
     * @param f the integrand values
     * @return the posterior over the integral
     */
    public static Estimate fit(double[][] x, double[] f) {
        double l = fitLength(x, f);
        double[][] k = gram(x, l);
        double[] a = Matrices.solve(k, f);
        double s2 = 0;
        for (int i = 0; i < f.length; i++) {
            s2 += f[i] * a[i] / f.length;
        }
        double[] z = kernelMean(x, l);
        double[] b = Matrices.solve(k, z);
        double mean = 0;
        double red = 0;
        for (int i = 0; i < f.length; i++) {
            mean += z[i] * a[i];
            red += z[i] * b[i];
        }
        double var = s2 * Math.max(initialVariance(x[0].length, l) - red, 0);
        return new Estimate(mean, Math.sqrt(var), l, x);
    }

    private static double reducedVariance(double[][] x, double l) {
        double[][] k = gram(x, l);
        double[] z = kernelMean(x, l);
        double[] b = Matrices.solve(k, z);
        double red = 0;
        for (int i = 0; i < z.length; i++) {
            red += z[i] * b[i];
        }
        return initialVariance(x[0].length, l) - red;
    }

    /**
     * Sequential Bayesian quadrature: starting from {@code initial} Gaussian draws, each new
     * point minimises the posterior variance of the integral among Gaussian candidates, with
     * the length scale refitted every few points.
     *
     * @param f the integrand
     * @param d the dimension
     * @param evaluations total evaluations
     * @param initial initial random points
     * @param candidates candidates per step
     * @param seed the seed
     * @return the posterior over the integral
     */
    public static Estimate sequential(ToDoubleFunction<double[]> f, int d, int evaluations, int initial,
            int candidates, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[evaluations][];
        double[] y = new double[evaluations];
        for (int i = 0; i < initial; i++) {
            x[i] = gaussian(d, r);
            y[i] = f.applyAsDouble(x[i]);
        }
        double l = fitLength(Arrays.copyOf(x, initial), Arrays.copyOf(y, initial));
        for (int n = initial; n < evaluations; n++) {
            double[] best = null;
            double bv = Double.POSITIVE_INFINITY;
            double[][] trial = Arrays.copyOf(x, n + 1);
            for (int c = 0; c < candidates; c++) {
                double[] cand = gaussian(d, r);
                for (int j = 0; j < d; j++) {
                    cand[j] *= 1.5;
                }
                trial[n] = cand;
                double v;
                try {
                    v = reducedVariance(trial, l);
                } catch (RuntimeException e) {
                    v = Double.POSITIVE_INFINITY;
                }
                if (v < bv) {
                    bv = v;
                    best = cand;
                }
            }
            x[n] = best == null ? gaussian(d, r) : best;
            y[n] = f.applyAsDouble(x[n]);
            if ((n + 1) % 5 == 0) {
                l = fitLength(Arrays.copyOf(x, n + 1), Arrays.copyOf(y, n + 1));
            }
        }
        return fit(x, y);
    }

    private static double[] gaussian(int d, SplittableRandom r) {
        double[] v = new double[d];
        for (int j = 0; j < d; j++) {
            v[j] = r.nextGaussian();
        }
        return v;
    }

    /**
     * WSABI-L for a positive integrand.
     *
     * @param x the points
     * @param f the positive integrand values
     * @return the estimate (the {@code sd} is NaN)
     */
    public static Estimate wsabi(double[][] x, double[] f) {
        int n = f.length;
        int d = x[0].length;
        double min = Arrays.stream(f).min().orElse(0);
        double alpha = 0.8 * min;
        double[] g = new double[n];
        for (int i = 0; i < n; i++) {
            g[i] = Math.sqrt(2 * Math.max(f[i] - alpha, 0));
        }
        double l = fitLength(x, g);
        double[] w = Matrices.solve(gram(x, l), g);
        // int k(x, xi) k(x, xj) N(x) dx with unit amplitude
        double s2 = l * l / 2;
        double c = Math.pow(s2 / (s2 + 1), d / 2.0);
        double integral = 0;
        double[] mid = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double cc = 0;
                for (int t = 0; t < d; t++) {
                    mid[t] = 0.5 * (x[i][t] + x[j][t]);
                    cc += mid[t] * mid[t];
                }
                double kk = Math.exp(-sq(x[i], x[j]) / (4 * l * l)) * c * Math.exp(-cc / (2 * (s2 + 1)));
                integral += w[i] * w[j] * kk;
            }
        }
        return new Estimate(alpha + 0.5 * integral, Double.NaN, l, x);
    }
}
