package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;
import java.util.function.DoubleUnaryOperator;

import io.github.zzuegg.brainiac.experimental.stats.VineCopula;
import io.github.zzuegg.brainiac.experimental.stats.VineCopula.Family;
import io.github.zzuegg.brainiac.tensor.Distributions;
import io.github.zzuegg.brainiac.tensor.Special;

/**
 * Samples with known tails, dependence and functional structure.
 * <ul>
 *   <li>{@link #heavyTail}: i.i.d. samples from GPD, Frechet, Student-t and symmetric
 *       alpha-stable laws (Chambers, Mallows and Stuck 1976) with known tail index and exact (or,
 *       for the stable law, asymptotic) upper quantiles;</li>
 *   <li>{@link #vine}: a 4-D C-vine with planted pair families (Clayton, Gumbel and Gaussian
 *       in the first tree, Frank and Gaussian in the second, Gaussian in the third), so the
 *       first tree, the families and the tail dependence are known;</li>
 *   <li>{@link #curves}: random curves {@code X(t) = mu(t) + sum_k xi_k phi_k(t)} with three
 *       known eigenfunctions, observed densely and at 3 to 5 noisy random times, a scalar
 *       outcome {@code y = int beta(t) X(t) dt + e}, and planted magnitude and shape
 *       outliers.</li>
 * </ul>
 */
public final class DependenceSamples {

    private DependenceSamples() {
    }

    /**
     * A heavy-tailed sample.
     *
     * @param name the law
     * @param tailIndex the tail index {@code alpha} (so the shape is {@code 1/alpha})
     * @param sample the draws
     * @param quantiles a function from an upper-tail probability {@code p} to the quantile
     */
    public record HeavyTail(String name, double tailIndex, double[] sample,
            DoubleUnaryOperator quantiles) {

        /**
         * The true upper quantile.
         *
         * @param p the upper-tail probability
         * @return {@code x} with {@code P(X > x) = p}
         */
        public double quantile(double p) {
            return quantiles.applyAsDouble(p);
        }
    }

    private static double normal(SplittableRandom r) {
        return Special.normalQuantile(Math.min(1 - 1e-16, Math.max(1e-16, r.nextDouble())));
    }

    /**
     * A heavy-tailed sample.
     *
     * @param law "gpd" (shape 0.25), "frechet" (alpha 3), "student-t" (3 degrees of freedom) or
     *     "stable" (alpha 1.7, symmetric)
     * @param n the sample size
     * @param seed the seed
     * @return the sample with its true quantiles
     */
    public static HeavyTail heavyTail(String law, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = new double[n];
        switch (law) {
            case "gpd" -> {
                double xi = 0.25;
                for (int i = 0; i < n; i++) {
                    x[i] = (Math.pow(1 - r.nextDouble(), -xi) - 1) / xi;
                }
                return new HeavyTail(law, 1 / xi, x, p -> (Math.pow(p, -xi) - 1) / xi);
            }
            case "frechet" -> {
                double a = 3;
                for (int i = 0; i < n; i++) {
                    x[i] = Math.pow(-Math.log(Math.max(r.nextDouble(), 1e-300)), -1 / a);
                }
                return new HeavyTail(law, a, x, p -> Math.pow(-Math.log1p(-p), -1 / a));
            }
            case "student-t" -> {
                int nu = 3;
                for (int i = 0; i < n; i++) {
                    double c = 0;
                    for (int k = 0; k < nu; k++) {
                        double z = normal(r);
                        c += z * z;
                    }
                    x[i] = normal(r) / Math.sqrt(c / nu);
                }
                return new HeavyTail(law, nu, x, p -> Distributions.studentTQuantile(1 - p, nu));
            }
            case "stable" -> {
                double a = 1.7;
                for (int i = 0; i < n; i++) {
                    double v = Math.PI * (r.nextDouble() - 0.5);
                    double w = -Math.log(Math.max(r.nextDouble(), 1e-300));
                    x[i] = Math.sin(a * v) / Math.pow(Math.cos(v), 1 / a)
                            * Math.pow(Math.cos((1 - a) * v) / w, (1 - a) / a);
                }
                // P(X > x) ~ C x^-a with C = Gamma(a) sin(pi a / 2) / pi
                double c = Math.exp(Special.lgamma(a)) * Math.sin(Math.PI * a / 2) / Math.PI;
                return new HeavyTail(law, a, x, p -> Math.pow(c / p, 1 / a));
            }
            default -> throw new IllegalArgumentException("unknown law " + law);
        }
    }

    /**
     * A planted 4-D C-vine.
     *
     * @param families {@code families[tree][edge]}
     * @param parameters matching parameters
     * @param firstTree the first-tree edges {@code {0,1}, {0,2}, {0,3}}
     */
    public record Vine(Family[][] families, double[][] parameters, int[][] firstTree) {

        /**
         * Draws uniforms.
         *
         * @param n rows
         * @param seed the seed
         * @return {@code [n][4]}
         */
        public double[][] sample(int n, long seed) {
            return VineCopula.sampleCVine(families, parameters, n, new SplittableRandom(seed));
        }

        /**
         * The true tail-dependence coefficients of a first-tree pair.
         *
         * @param edge the edge index
         * @return {@code {lambda_L, lambda_U}}
         */
        public double[] tailDependence(int edge) {
            return families[0][edge].tailDependence(parameters[0][edge]);
        }
    }

    /**
     * The planted vine: (0,1) Clayton 3, (0,2) Gumbel 2.5, (0,3) Gaussian 0.7; (1,2|0) Frank 1,
     * (1,3|0) Gaussian 0.1; (2,3|0,1) Gaussian 0.1.
     *
     * @return the vine
     */
    public static Vine vine() {
        Family[][] f = {
            {Family.CLAYTON, Family.GUMBEL, Family.GAUSSIAN},
            {Family.FRANK, Family.GAUSSIAN},
            {Family.GAUSSIAN}};
        double[][] p = {{3, 2.5, 0.7}, {1, 0.1}, {0.1}};
        return new Vine(f, p, new int[][] {{0, 1}, {0, 2}, {0, 3}});
    }

    /**
     * Functional data.
     *
     * @param grid the dense grid on {@code [0, 1]}
     * @param truth noise-free curves on the grid {@code [n][grid]}
     * @param dense noisy dense observations on the grid
     * @param times sparse observation times per curve
     * @param values sparse noisy observations
     * @param scores the true scores {@code [n][3]}
     * @param y the scalar outcomes
     * @param outlier 0 for regular curves, 1 for magnitude and 2 for shape outliers
     * @param mean the true mean on the grid
     * @param eigenfunctions the true eigenfunctions {@code [3][grid]}
     * @param beta the true coefficient function on the grid
     */
    public record Curves(double[] grid, double[][] truth, double[][] dense, double[][] times, double[][] values,
            double[][] scores, double[] y, int[] outlier, double[] mean, double[][] eigenfunctions, double[] beta) {
    }

    /** The true eigenvalues of {@link #curves}. */
    public static final double[] EIGENVALUES = {1.0, 0.4, 0.1};

    /** The measurement-noise standard deviation of {@link #curves}. */
    public static final double NOISE = 0.2;

    private static double eigen(int k, double t) {
        return switch (k) {
            case 0 -> Math.sqrt(2) * Math.sin(2 * Math.PI * t);
            case 1 -> Math.sqrt(2) * Math.cos(2 * Math.PI * t);
            default -> Math.sqrt(2) * Math.sin(4 * Math.PI * t);
        };
    }

    private static double meanFunction(double t) {
        return t + Math.sin(Math.PI * t);
    }

    /**
     * Random curves.
     *
     * @param n curves
     * @param gridSize the dense grid size
     * @param minPoints the fewest sparse observations per curve
     * @param maxPoints the most sparse observations per curve
     * @param outlierShare the share of planted outliers (half magnitude, half shape)
     * @param seed the seed
     * @return the curves
     */
    public static Curves curves(int n, int gridSize, int minPoints, int maxPoints, double outlierShare, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] grid = new double[gridSize];
        for (int g = 0; g < gridSize; g++) {
            grid[g] = g / (gridSize - 1.0);
        }
        double[] mean = new double[gridSize];
        double[][] phi = new double[3][gridSize];
        double[] beta = new double[gridSize];
        for (int g = 0; g < gridSize; g++) {
            mean[g] = meanFunction(grid[g]);
            for (int k = 0; k < 3; k++) {
                phi[k][g] = eigen(k, grid[g]);
            }
            beta[g] = phi[0][g] - phi[1][g];
        }
        double[][] truth = new double[n][gridSize];
        double[][] dense = new double[n][gridSize];
        double[][] times = new double[n][];
        double[][] values = new double[n][];
        double[][] scores = new double[n][3];
        double[] y = new double[n];
        int[] outlier = new int[n];
        int planted = (int) Math.round(outlierShare * n);
        for (int i = 0; i < n; i++) {
            int kind = i < planted ? (i % 2 == 0 ? 1 : 2) : 0;
            outlier[i] = kind;
            for (int k = 0; k < 3; k++) {
                scores[i][k] = Math.sqrt(EIGENVALUES[k]) * normal(r);
            }
            double shift = kind == 1 ? (r.nextBoolean() ? 1 : -1) * (5 + r.nextDouble()) : 0;
            double phase = r.nextDouble();
            int m = minPoints + r.nextInt(maxPoints - minPoints + 1);
            times[i] = new double[m];
            values[i] = new double[m];
            for (int g = 0; g < gridSize; g++) {
                truth[i][g] = curveValue(grid[g], scores[i], kind, shift, phase);
                dense[i][g] = truth[i][g] + NOISE * normal(r);
            }
            for (int j = 0; j < m; j++) {
                double t = r.nextDouble();
                times[i][j] = t;
                values[i][j] = curveValue(t, scores[i], kind, shift, phase) + NOISE * normal(r);
            }
            // y = int beta X dt = <beta, mu> + xi_1 - xi_2 (orthonormal eigenfunctions)
            double s = 0;
            for (int g = 0; g < gridSize; g++) {
                double w = g == 0 || g == gridSize - 1 ? 0.5 : 1;
                s += w * beta[g] * truth[i][g] / (gridSize - 1);
            }
            y[i] = s + 0.1 * normal(r);
        }
        // shuffle so outliers are not the first rows
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            swap(truth, i, j);
            swap(dense, i, j);
            swap(times, i, j);
            swap(values, i, j);
            swap(scores, i, j);
            double ty = y[i];
            y[i] = y[j];
            y[j] = ty;
            int to = outlier[i];
            outlier[i] = outlier[j];
            outlier[j] = to;
        }
        return new Curves(grid, truth, dense, times, values, scores, y, outlier, mean, phi, beta);
    }

    private static double curveValue(double t, double[] xi, int kind, double shift, double phase) {
        double v = meanFunction(t);
        if (kind == 2) {
            // a shape outlier: a fast oscillation of typical amplitude around the mean
            return v + 2 * Math.sin(2 * Math.PI * (6 * t + phase));
        }
        for (int k = 0; k < 3; k++) {
            v += xi[k] * eigen(k, t);
        }
        return v + shift;
    }

    private static void swap(double[][] a, int i, int j) {
        double[] t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}
