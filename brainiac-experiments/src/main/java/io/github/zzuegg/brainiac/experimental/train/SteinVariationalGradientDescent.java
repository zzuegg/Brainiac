package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.experimental.ml.LogDensity;

import java.util.Arrays;

/**
 * Stein variational gradient descent (Liu and Wang 2016,
 * <a href="https://arxiv.org/abs/1608.04471">arXiv:1608.04471</a>): a deterministic particle
 * method that moves {@code n} particles along
 * {@code phi(x) = 1/n sum_j [k(x_j, x) grad log p(x_j) + grad_{x_j} k(x_j, x)]}, the steepest
 * descent direction of {@code KL(q || p)} in the unit ball of an RBF kernel's RKHS. The first
 * term is a kernel-smoothed gradient that pulls particles to high density; the second pushes
 * them apart, so the set spreads over the posterior instead of collapsing on its modes. With
 * one particle it is gradient ascent; on network weights, each particle a whole network, it is
 * a "repulsive deep ensemble".
 *
 * <p>The bandwidth follows the median heuristic {@code h = med^2 / log(n + 1)} and the step
 * uses AdaGrad with momentum as in the paper. Particles are flat vectors; for a network,
 * {@link HamiltonianMonteCarlo#parameterDensity} gives the log density over its weights.
 */
public final class SteinVariationalGradientDescent {

    private final LogDensity target;
    private final double stepSize;
    private final double[][] particles;
    private final double[][] historical;
    private int iteration;

    private SteinVariationalGradientDescent(LogDensity target, double[][] initial, double stepSize) {
        this.target = target;
        this.stepSize = stepSize;
        this.particles = new double[initial.length][];
        for (int i = 0; i < initial.length; i++) {
            particles[i] = initial[i].clone();
        }
        this.historical = new double[initial.length][initial[0].length];
    }

    /**
     * Starts SVGD from initial particles.
     *
     * @param target the log density
     * @param initial the starting particles {@code [n][d]} (copied)
     * @param stepSize the AdaGrad master step size, for example 0.05
     * @return the sampler
     */
    public static SteinVariationalGradientDescent create(LogDensity target, double[][] initial, double stepSize) {
        return new SteinVariationalGradientDescent(target, initial, stepSize);
    }

    /**
     * The SVGD direction for a particle set, with the median-heuristic RBF kernel.
     *
     * @param x the particles {@code [n][d]}
     * @param gradLogP {@code grad log p} at each particle {@code [n][d]}
     * @param repulsion the weight of the kernel-gradient term, 1 for SVGD and 0 for a
     *     kernel-smoothed ensemble without repulsion
     * @return {@code phi(x_i)} for each particle
     */
    public static double[][] direction(double[][] x, double[][] gradLogP, double repulsion) {
        int n = x.length;
        int d = x[0].length;
        double[][] sq = new double[n][n];
        double[] all = new double[n * (n - 1) / 2];
        int c = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0;
                for (int k = 0; k < d; k++) {
                    double v = x[i][k] - x[j][k];
                    s += v * v;
                }
                sq[i][j] = s;
                sq[j][i] = s;
                if (c < all.length) {
                    all[c++] = s;
                }
            }
        }
        double h = 1;
        if (all.length > 0) {
            Arrays.sort(all);
            double med = all[all.length / 2];
            h = Math.max(med / Math.log(n + 1.0), 1e-8);
        }
        double[][] phi = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double kij = Math.exp(-sq[i][j] / h);
                for (int k = 0; k < d; k++) {
                    // grad_{x_j} k(x_j, x_i) = 2 / h (x_i - x_j) k
                    phi[i][k] += kij * gradLogP[j][k] + repulsion * 2 / h * (x[i][k] - x[j][k]) * kij;
                }
            }
            for (int k = 0; k < d; k++) {
                phi[i][k] /= n;
            }
        }
        return phi;
    }

    /**
     * One SVGD update of every particle.
     *
     * @return the mean log density of the particles before the step
     */
    public double step() {
        int n = particles.length;
        int d = particles[0].length;
        double[][] g = new double[n][d];
        double mean = 0;
        for (int i = 0; i < n; i++) {
            mean += target.evaluate(particles[i], g[i]) / n;
        }
        double[][] phi = direction(particles, g, 1);
        iteration++;
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < d; k++) {
                double v = phi[i][k];
                historical[i][k] = iteration == 1 ? v * v : 0.9 * historical[i][k] + 0.1 * v * v;
                particles[i][k] += stepSize * v / (1e-6 + Math.sqrt(historical[i][k]));
            }
        }
        return mean;
    }

    /**
     * Runs several updates.
     *
     * @param iterations the number of updates
     * @return the particles afterwards (a copy)
     */
    public double[][] run(int iterations) {
        for (int i = 0; i < iterations; i++) {
            step();
        }
        return particles();
    }

    /**
     * The current particles.
     *
     * @return a copy {@code [n][d]}
     */
    public double[][] particles() {
        double[][] out = new double[particles.length][];
        for (int i = 0; i < particles.length; i++) {
            out[i] = particles[i].clone();
        }
        return out;
    }
}
