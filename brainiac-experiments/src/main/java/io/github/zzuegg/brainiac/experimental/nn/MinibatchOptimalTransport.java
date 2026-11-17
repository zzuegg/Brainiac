package io.github.zzuegg.brainiac.experimental.nn;

import java.util.Arrays;

/**
 * Exact optimal transport between two equal-size minibatches under the squared Euclidean
 * cost, by the Hungarian algorithm ({@code O(n^3)}, a few milliseconds for 256 points). Pairing
 * noise with data this way before flow matching (OT-CFM, Tong et al.,
 * <a href="https://arxiv.org/abs/2302.00482">arXiv:2302.00482</a>) gives straighter, non-crossing
 * paths; energy matching and bridge matching use it for the same reason.
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
}
