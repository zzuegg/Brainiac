package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Collocation points for physics-informed training in a box domain: uniform random, Latin
 * hypercube and grid samples, points on a face of the box (boundary and initial conditions),
 * and residual-based adaptive sampling.
 *
 * <p>{@link #residualAdaptive} implements RAD/RAR-D of Wu et al. (2023): candidate points are
 * drawn uniformly and kept with probability proportional to {@code |r|^k / mean(|r|^k) + c},
 * so new points concentrate where the PDE residual is large (a shock, a front) while
 * {@code c} keeps some uniform coverage. Paper:
 * <a href="https://arxiv.org/abs/2207.10289">arXiv:2207.10289</a>.
 */
public final class CollocationSampler {

    private final double[] low;
    private final double[] high;

    private CollocationSampler(double[] low, double[] high) {
        if (low.length != high.length) {
            throw new IllegalArgumentException("low and high differ in size");
        }
        this.low = low.clone();
        this.high = high.clone();
    }

    /**
     * A box domain.
     *
     * @param low the lower corner
     * @param high the upper corner
     * @return the sampler
     */
    public static CollocationSampler box(double[] low, double[] high) {
        return new CollocationSampler(low, high);
    }

    /** @return the dimension */
    public int dimension() {
        return low.length;
    }

    /**
     * @param axis a coordinate
     * @return its lower bound
     */
    public double low(int axis) {
        return low[axis];
    }

    /**
     * @param axis a coordinate
     * @return its upper bound
     */
    public double high(int axis) {
        return high[axis];
    }

    /**
     * Uniform random points.
     *
     * @param n the count
     * @param random the source
     * @return {@code [n][d]}
     */
    public double[][] uniform(int n, SplittableRandom random) {
        double[][] p = new double[n][low.length];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < low.length; j++) {
                p[i][j] = low[j] + (high[j] - low[j]) * random.nextDouble();
            }
        }
        return p;
    }

    /**
     * A Latin hypercube sample: each axis split into {@code n} strata, one point per stratum.
     *
     * @param n the count
     * @param random the source
     * @return {@code [n][d]}
     */
    public double[][] latinHypercube(int n, SplittableRandom random) {
        double[][] p = new double[n][low.length];
        for (int j = 0; j < low.length; j++) {
            int[] perm = new int[n];
            for (int i = 0; i < n; i++) {
                perm[i] = i;
            }
            for (int i = n - 1; i > 0; i--) {
                int k = random.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[k];
                perm[k] = t;
            }
            for (int i = 0; i < n; i++) {
                p[i][j] = low[j] + (high[j] - low[j]) * (perm[i] + random.nextDouble()) / n;
            }
        }
        return p;
    }

    /**
     * A tensor-product grid including the box corners.
     *
     * @param counts points per axis, each at least 2
     * @return {@code [prod counts][d]}, the first axis varying slowest
     */
    public double[][] grid(int... counts) {
        int total = 1;
        for (int c : counts) {
            total *= c;
        }
        double[][] p = new double[total][low.length];
        for (int i = 0; i < total; i++) {
            int rest = i;
            for (int j = low.length - 1; j >= 0; j--) {
                int idx = rest % counts[j];
                rest /= counts[j];
                p[i][j] = low[j] + (high[j] - low[j]) * idx / (counts[j] - 1);
            }
        }
        return p;
    }

    /**
     * Random points on a face of the box, such as {@code t = 0} or {@code x = low}.
     *
     * @param axis the fixed coordinate
     * @param upper whether it is fixed at its upper bound
     * @param n the count
     * @param random the source
     * @return {@code [n][d]}
     */
    public double[][] face(int axis, boolean upper, int n, SplittableRandom random) {
        double[][] p = latinHypercube(n, random);
        for (double[] q : p) {
            q[axis] = upper ? high[axis] : low[axis];
        }
        return p;
    }

    /**
     * Residual-based adaptive sampling (RAD, the distribution behind RAR-D).
     *
     * @param n the points to return
     * @param candidates the uniform candidates to score, several times {@code n}
     * @param residual the absolute residual at a batch of points
     * @param k the exponent on the residual, 1 in the paper
     * @param c the uniform floor, 1 in the paper
     * @param random the source
     * @return {@code [n][d]} points drawn without replacement from the candidates
     */
    public double[][] residualAdaptive(int n, int candidates, Function<double[][], double[]> residual, double k,
            double c, SplittableRandom random) {
        double[][] cand = uniform(candidates, random);
        double[] r = residual.apply(cand);
        double mean = 0;
        for (int i = 0; i < r.length; i++) {
            r[i] = Math.pow(Math.abs(r[i]), k);
            mean += r[i];
        }
        mean = Math.max(mean / r.length, 1e-30);
        // Efraimidis-Spirakis weighted sampling without replacement: keys u^(1/w)
        double[] key = new double[candidates];
        Integer[] order = new Integer[candidates];
        for (int i = 0; i < candidates; i++) {
            double w = r[i] / mean + c;
            key[i] = Math.log(random.nextDouble() + 1e-300) / w;
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Double.compare(key[b], key[a]));
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = cand[order[i]];
        }
        return out;
    }

    /**
     * Points as a float32 tensor.
     *
     * @param points {@code [n][d]}
     * @return {@code [n, d]}
     */
    public static Tensor tensor(double[][] points) {
        int n = points.length;
        int d = points[0].length;
        float[] f = new float[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                f[i * d + j] = (float) points[i][j];
            }
        }
        return Tensor.of(Shape.of(n, d), f);
    }

    /**
     * Values as a float32 column.
     *
     * @param values {@code [n]}
     * @return {@code [n, 1]}
     */
    public static Tensor column(double[] values) {
        float[] f = new float[values.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) values[i];
        }
        return Tensor.of(Shape.of(values.length, 1), f);
    }
}
