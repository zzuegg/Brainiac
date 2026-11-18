package io.github.zzuegg.brainiac.experimental.quadrature;

import io.github.zzuegg.brainiac.experimental.sciml.CollocationSampler;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Randomised (quasi-)Monte Carlo estimates of {@code int_[0,1]^d f(u) du} with error bars from
 * independent randomisations (Owen 1997): the standard error is the spread of the replicate
 * means. Point sets: plain {@link Method#MC}, {@link Method#ANTITHETIC} pairs
 * {@code (u, 1 - u)}, {@link Method#LATIN_HYPERCUBE} (wave 12's {@link CollocationSampler}),
 * scrambled {@link Method#HALTON}, {@link Method#SOBOL_OWEN}, {@link Method#SOBOL_LINEAR} and
 * a randomly shifted {@link Method#LATTICE}.
 */
public final class RqmcEstimator {

    private RqmcEstimator() {
    }

    /** Point-set families. */
    public enum Method {
        /** I.i.d. uniforms. */
        MC,
        /** Antithetic pairs. */
        ANTITHETIC,
        /** Latin hypercube. */
        LATIN_HYPERCUBE,
        /** Scrambled Halton. */
        HALTON,
        /** Owen-scrambled Sobol. */
        SOBOL_OWEN,
        /** Linearly scrambled, digitally shifted Sobol. */
        SOBOL_LINEAR,
        /** Randomly shifted Korobov lattice. */
        LATTICE
    }

    /**
     * An estimate.
     *
     * @param mean the mean of the replicate estimates
     * @param standardError the standard error of {@code mean}
     * @param replicates the replicate estimates
     */
    public record Estimate(double mean, double standardError, double[] replicates) {
    }

    /**
     * One randomised point set.
     *
     * @param method the family
     * @param n the number of points
     * @param d the dimension
     * @param seed the randomisation seed
     * @return points {@code [n][d]} in the unit cube
     */
    public static double[][] points(Method method, int n, int d, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return switch (method) {
            case MC -> {
                double[][] u = new double[n][d];
                for (double[] row : u) {
                    for (int j = 0; j < d; j++) {
                        row[j] = r.nextDouble();
                    }
                }
                yield u;
            }
            case ANTITHETIC -> {
                double[][] u = new double[n][d];
                for (int i = 0; i + 1 < n; i += 2) {
                    for (int j = 0; j < d; j++) {
                        u[i][j] = r.nextDouble();
                        u[i + 1][j] = 1 - u[i][j];
                    }
                }
                if (n % 2 == 1) {
                    for (int j = 0; j < d; j++) {
                        u[n - 1][j] = r.nextDouble();
                    }
                }
                yield u;
            }
            case LATIN_HYPERCUBE -> {
                double[] lo = new double[d];
                double[] hi = new double[d];
                Arrays.fill(hi, 1);
                yield CollocationSampler.box(lo, hi).latinHypercube(n, r);
            }
            case HALTON -> LowDiscrepancy.halton(n, d, true, seed);
            case SOBOL_OWEN -> LowDiscrepancy.sobol(n, d, LowDiscrepancy.Scramble.OWEN, seed);
            case SOBOL_LINEAR -> LowDiscrepancy.sobol(n, d, LowDiscrepancy.Scramble.LINEAR, seed);
            case LATTICE -> LowDiscrepancy.lattice(n, LowDiscrepancy.korobov(n, d), seed);
        };
    }

    /**
     * Estimates an integral with independent replicates.
     *
     * @param f the integrand on the unit cube
     * @param d the dimension
     * @param method the point set
     * @param n points per replicate
     * @param replicates independent randomisations, at least 2
     * @param seed the seed
     * @return the estimate with its standard error
     */
    public static Estimate estimate(ToDoubleFunction<double[]> f, int d, Method method, int n, int replicates,
            long seed) {
        double[] reps = new double[replicates];
        long[] z = method == Method.LATTICE ? LowDiscrepancy.korobov(n, d) : null;
        for (int k = 0; k < replicates; k++) {
            double[][] u = z != null ? LowDiscrepancy.lattice(n, z, seed + 7919L * k) : points(method, n, d,
                    seed + 7919L * k);
            double s = 0;
            for (double[] row : u) {
                s += f.applyAsDouble(row);
            }
            reps[k] = s / n;
        }
        double m = Arrays.stream(reps).average().orElse(Double.NaN);
        double v = 0;
        for (double x : reps) {
            v += (x - m) * (x - m) / (replicates - 1);
        }
        return new Estimate(m, Math.sqrt(v / replicates), reps);
    }
}
