package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * An unnormalised log density {@code log p(x)} on {@code R^d} with its gradient: the one
 * interface every gradient-based sampler and variational method here works on (Hamiltonian
 * Monte Carlo and NUTS, Stein variational gradient descent, annealed importance sampling, flow
 * variational inference). A probabilistic program, a Bayesian neural network's weight posterior
 * or a closed-form test target all become a {@code LogDensity}.
 *
 * <p>The static factories give the standard sampler test targets with exact samples and a
 * known normaliser (so a variational fit's KL is {@code -ELBO}): the banana (twisted Gaussian,
 * Haario et al. 1999), Neal's funnel (Neal 2003, <a href="https://arxiv.org/abs/physics/0009028">
 * slice sampling</a>) and a Gaussian mixture.
 */
@FunctionalInterface
public interface LogDensity {

    /**
     * Evaluates the log density and, when {@code gradient} is not null, writes its gradient.
     *
     * @param x the point {@code [d]}
     * @param gradient receives {@code d log p / dx}, or {@code null} for the value only
     * @return {@code log p(x)} up to a constant
     */
    double evaluate(double[] x, double[] gradient);

    /**
     * The log density alone.
     *
     * @param x the point
     * @return {@code log p(x)}
     */
    default double value(double[] x) {
        return evaluate(x, null);
    }

    /**
     * A density whose gradient comes from central finite differences: for programs and models
     * without an analytic gradient, in a few dimensions.
     *
     * @param f the log density
     * @param h the difference step, for example {@code 1e-5}
     * @return the density
     */
    static LogDensity numerical(ToDoubleFunction<double[]> f, double h) {
        return (x, g) -> {
            double v = f.applyAsDouble(x);
            if (g != null) {
                double[] y = x.clone();
                for (int i = 0; i < x.length; i++) {
                    y[i] = x[i] + h;
                    double up = f.applyAsDouble(y);
                    y[i] = x[i] - h;
                    double down = f.applyAsDouble(y);
                    y[i] = x[i];
                    g[i] = (up - down) / (2 * h);
                }
            }
            return v;
        };
    }

    /**
     * The geometric path {@code (1 - beta) log p0 + beta log p1} between two densities, the
     * annealing path of AIS and of SMC samplers.
     *
     * @param p0 the start (usually the prior)
     * @param p1 the end (usually the unnormalised posterior)
     * @param beta the inverse temperature in {@code [0, 1]}
     * @return the tempered density
     */
    static LogDensity tempered(LogDensity p0, LogDensity p1, double beta) {
        return (x, g) -> {
            double[] g1 = g == null ? null : new double[x.length];
            double a = beta < 1 ? p0.evaluate(x, g) : 0;
            double b = beta > 0 ? p1.evaluate(x, g1) : 0;
            if (g != null) {
                for (int i = 0; i < x.length; i++) {
                    g[i] = (beta < 1 ? (1 - beta) * g[i] : 0) + (beta > 0 ? beta * g1[i] : 0);
                }
            }
            return (1 - beta) * a + beta * b;
        };
    }

    /**
     * An isotropic Gaussian, normalised.
     *
     * @param dims the dimension
     * @param mean the mean of every coordinate
     * @param std the standard deviation
     * @return the density
     */
    static LogDensity gaussian(int dims, double mean, double std) {
        double c = -dims * (Math.log(std) + 0.5 * Math.log(2 * Math.PI));
        return (x, g) -> {
            double s = 0;
            for (int i = 0; i < dims; i++) {
                double z = (x[i] - mean) / std;
                s += z * z;
                if (g != null) {
                    g[i] = -z / std;
                }
            }
            return c - 0.5 * s;
        };
    }

    /**
     * The banana (twisted Gaussian): {@code x0 ~ N(0, a^2)},
     * {@code x1 - b (x0^2 - a^2) ~ N(0, 1)}; normalised, with curved, thin level sets.
     *
     * @param a the scale of the first coordinate, for example 2
     * @param b the curvature, for example 0.5
     * @return the density
     */
    static LogDensity banana(double a, double b) {
        double c = -Math.log(2 * Math.PI) - Math.log(a);
        return (x, g) -> {
            double u = x[0] / a;
            double w = x[1] - b * (x[0] * x[0] - a * a);
            if (g != null) {
                g[0] = -u / a + w * 2 * b * x[0];
                g[1] = -w;
            }
            return c - 0.5 * u * u - 0.5 * w * w;
        };
    }

    /**
     * Exact samples of {@link #banana}.
     *
     * @param a the scale
     * @param b the curvature
     * @param n the number of samples
     * @param random the random source
     * @return samples {@code [n][2]}
     */
    static double[][] bananaSamples(double a, double b, int n, SplittableRandom random) {
        double[][] s = new double[n][2];
        for (int i = 0; i < n; i++) {
            s[i][0] = a * random.nextGaussian();
            s[i][1] = random.nextGaussian() + b * (s[i][0] * s[i][0] - a * a);
        }
        return s;
    }

    /**
     * Neal's funnel: {@code v ~ N(0, 3^2)}, {@code x_i | v ~ N(0, e^v)} for {@code d - 1}
     * coordinates; normalised. The neck needs tiny steps and the mouth large ones, the classic
     * failure case for a fixed step size.
     *
     * @param dims the dimension, at least 2 ({@code v} is coordinate 0)
     * @return the density
     */
    static LogDensity funnel(int dims) {
        double c = -0.5 * dims * Math.log(2 * Math.PI) - Math.log(3);
        return (x, g) -> {
            double v = x[0];
            double ev = Math.exp(-v);
            double s = 0;
            for (int i = 1; i < dims; i++) {
                s += x[i] * x[i];
                if (g != null) {
                    g[i] = -x[i] * ev;
                }
            }
            if (g != null) {
                g[0] = -v / 9 - 0.5 * (dims - 1) + 0.5 * s * ev;
            }
            return c - v * v / 18 - 0.5 * (dims - 1) * v - 0.5 * s * ev;
        };
    }

    /**
     * Exact samples of {@link #funnel}.
     *
     * @param dims the dimension
     * @param n the number of samples
     * @param random the random source
     * @return samples {@code [n][dims]}
     */
    static double[][] funnelSamples(int dims, int n, SplittableRandom random) {
        double[][] s = new double[n][dims];
        for (int i = 0; i < n; i++) {
            s[i][0] = 3 * random.nextGaussian();
            double sd = Math.exp(s[i][0] / 2);
            for (int j = 1; j < dims; j++) {
                s[i][j] = sd * random.nextGaussian();
            }
        }
        return s;
    }

    /**
     * A normalised mixture of isotropic Gaussians.
     *
     * @param means the component means {@code [k][d]}
     * @param weights the mixing weights, summing to 1
     * @param std the shared standard deviation
     * @return the density
     */
    static LogDensity mixture(double[][] means, double[] weights, double std) {
        int k = means.length;
        int d = means[0].length;
        double c = -d * (Math.log(std) + 0.5 * Math.log(2 * Math.PI));
        return (x, g) -> {
            double[] lp = new double[k];
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) {
                double s = 0;
                for (int i = 0; i < d; i++) {
                    double z = (x[i] - means[j][i]) / std;
                    s += z * z;
                }
                lp[j] = Math.log(weights[j]) + c - 0.5 * s;
                max = Math.max(max, lp[j]);
            }
            double sum = 0;
            for (int j = 0; j < k; j++) {
                sum += Math.exp(lp[j] - max);
            }
            if (g != null) {
                java.util.Arrays.fill(g, 0);
                for (int j = 0; j < k; j++) {
                    double r = Math.exp(lp[j] - max) / sum;
                    for (int i = 0; i < d; i++) {
                        g[i] -= r * (x[i] - means[j][i]) / (std * std);
                    }
                }
            }
            return max + Math.log(sum);
        };
    }
}
