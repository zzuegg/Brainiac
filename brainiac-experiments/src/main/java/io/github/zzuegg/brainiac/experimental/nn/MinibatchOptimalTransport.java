package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Exact optimal transport between two equal-size minibatches under the squared Euclidean
 * cost, by the Hungarian algorithm ({@code O(n^3)}, a few milliseconds for 256 points). Pairing
 * noise with data this way before flow matching (OT-CFM, Tong et al.,
 * <a href="https://arxiv.org/abs/2302.00482">arXiv:2302.00482</a>) gives straighter, non-crossing
 * paths; energy matching and bridge matching use it for the same reason.
 *
 * <p>{@link #pairSinkhorn} is the entropic alternative ({@link EntropicTransport}): a Sinkhorn
 * coupling, {@code O(n^2)} per iteration, sampled into a pairing, which scales to couplings of
 * thousands of points (Klein et al. 2025). The Hungarian {@link #pair} stays the default.
 */
public final class MinibatchOptimalTransport {
    private MinibatchOptimalTransport() {
    }

    /**
     * Finds the assignment minimising {@code sum_i |a_i - b_{p(i)}|^2}.
     *
     * @param a source points {@code [n][d]}
     * @param b target points {@code [n][d]}
     * @return {@code p}: row {@code i} of {@code a} is paired with row {@code p[i]} of {@code b}
     */
    public static int[] assign(float[][] a, float[][] b) {
        int n = a.length;
        if (b.length != n) {
            throw new IllegalArgumentException("batches must have the same size");
        }
        double[][] cost = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0;
                for (int k = 0; k < a[i].length; k++) {
                    double v = a[i][k] - b[j][k];
                    s += v * v;
                }
                cost[i][j] = s;
            }
        }
        // Jonker-Volgenant style shortest augmenting paths with potentials (1-indexed)
        double[] u = new double[n + 1];
        double[] v = new double[n + 1];
        int[] p = new int[n + 1];
        int[] way = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            p[0] = i;
            int j0 = 0;
            double[] minv = new double[n + 1];
            Arrays.fill(minv, Double.POSITIVE_INFINITY);
            boolean[] used = new boolean[n + 1];
            do {
                used[j0] = true;
                int i0 = p[j0];
                double delta = Double.POSITIVE_INFINITY;
                int j1 = 0;
                for (int j = 1; j <= n; j++) {
                    if (!used[j]) {
                        double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }
                for (int j = 0; j <= n; j++) {
                    if (used[j]) {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }
                j0 = j1;
            } while (p[j0] != 0);
            do {
                int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0 != 0);
        }
        int[] out = new int[n];
        for (int j = 1; j <= n; j++) {
            out[p[j] - 1] = j - 1;
        }
        return out;
    }

    /**
     * Reorders {@code b} so its rows are paired with {@code a}'s.
     *
     * @param a source points
     * @param b target points
     * @return {@code b} permuted by {@link #assign}
     */
    public static float[][] pair(float[][] a, float[][] b) {
        int[] p = assign(a, b);
        float[][] out = new float[b.length][];
        for (int i = 0; i < p.length; i++) {
            out[i] = b[p[i]];
        }
        return out;
    }

    /**
     * Pairs {@code b} with {@code a} by sampling from an entropic (Sinkhorn) coupling under the
     * squared Euclidean cost; {@code epsilon} is relative to the mean cost.
     *
     * @param a source points {@code [n][d]}
     * @param b target points {@code [m][d]}
     * @param relativeEpsilon the regularisation as a fraction of the mean cost, for example 0.01
     * @param iterations Sinkhorn iterations
     * @param random draws the pairing
     * @return for each row of {@code a}, its sampled partner from {@code b}
     */
    public static float[][] pairSinkhorn(float[][] a, float[][] b, double relativeEpsilon, int iterations,
            SplittableRandom random) {
        double[][] x = new double[a.length][];
        double[][] y = new double[b.length][];
        for (int i = 0; i < a.length; i++) {
            x[i] = toDouble(a[i]);
        }
        for (int j = 0; j < b.length; j++) {
            y[j] = toDouble(b[j]);
        }
        double[][] c = EntropicTransport.squaredEuclidean(x, y);
        double mean = 0;
        double max = 0;
        for (double[] row : c) {
            for (double v : row) {
                mean += v / ((double) c.length * row.length);
                max = Math.max(max, v);
            }
        }
        double eps = relativeEpsilon * mean;
        double[] wa = EntropicTransport.uniform(a.length);
        double[] wb = EntropicTransport.uniform(b.length);
        EntropicTransport.Plan plan = max / eps < 80 ? EntropicTransport.sinkhornScaling(wa, wb, c, eps, iterations)
                : EntropicTransport.sinkhorn(wa, wb, c, eps, iterations, 1e-4);
        int[] p = EntropicTransport.pairing(plan, random);
        float[][] out = new float[a.length][];
        for (int i = 0; i < p.length; i++) {
            out[i] = b[p[i]];
        }
        return out;
    }

    private static double[] toDouble(float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return d;
    }
}
