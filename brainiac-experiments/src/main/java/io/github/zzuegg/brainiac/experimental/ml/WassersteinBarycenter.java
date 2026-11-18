package io.github.zzuegg.brainiac.experimental.ml;

/**
 * Fixed-support entropic Wasserstein barycenters by iterative Bregman projections (Cuturi and
 * Doucet 2014, <a href="https://arxiv.org/abs/1310.4375">arXiv:1310.4375</a>; Benamou et al.
 * 2015).
 *
 * <p>All histograms live on one support with cost matrix {@code C}; with the Gibbs kernel
 * {@code K = exp(-C / eps)} the barycenter minimising {@code sum_k w_k OT_eps(p, h_k)} is found
 * by alternating the two marginal projections of every coupling:
 * {@code u_k = h_k / (K v_k)}, {@code p = prod_k (K^T u_k)^(w_k)}, {@code v_k = p / (K^T u_k)}.
 * On an image grid this averages shapes in the transport geometry: translated copies of a
 * glyph average to the glyph at the mean position, where the Euclidean mean blurs them.
 */
public final class WassersteinBarycenter {

    private WassersteinBarycenter() {
    }

    /**
     * Squared Euclidean distances between the pixels of a {@code side x side} grid, in pixels.
     *
     * @param side the grid side
     * @return {@code [side^2][side^2]}
     */
    public static double[][] gridCost(int side) {
        int n = side * side;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dy = i / side - j / side;
                double dx = i % side - j % side;
                c[i][j] = dx * dx + dy * dy;
            }
        }
        return c;
    }

    /**
     * The fixed-support barycenter.
     *
     * @param histograms the inputs {@code [K][n]}, each non-negative with sum 1
     * @param weights the barycentric weights {@code [K]}, sum 1
     * @param cost the support cost {@code [n][n]}
     * @param epsilon the entropic regularisation, in the units of the cost
     * @param iterations the projection rounds
     * @return the barycenter {@code [n]}, sum 1
     */
    public static double[] fixedSupport(double[][] histograms, double[] weights, double[][] cost, double epsilon,
            int iterations) {
        int n = cost.length;
        int count = histograms.length;
        double[][] kernel = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                kernel[i][j] = Math.exp(-cost[i][j] / epsilon);
            }
        }
        double[][] v = new double[count][n];
        for (double[] row : v) {
            java.util.Arrays.fill(row, 1.0);
        }
        double[][] ktu = new double[count][n];
        double[] p = new double[n];
        for (int it = 0; it < iterations; it++) {
            double[] logP = new double[n];
            for (int k = 0; k < count; k++) {
                double[] u = new double[n];
                for (int i = 0; i < n; i++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) {
                        s += kernel[i][j] * v[k][j];
                    }
                    u[i] = histograms[k][i] / Math.max(s, 1e-300);
                }
                for (int j = 0; j < n; j++) {
                    double s = 0;
                    for (int i = 0; i < n; i++) {
                        s += kernel[i][j] * u[i];
                    }
                    ktu[k][j] = Math.max(s, 1e-300);
                    logP[j] += weights[k] * Math.log(ktu[k][j]);
                }
            }
            for (int j = 0; j < n; j++) {
                p[j] = Math.exp(logP[j]);
            }
            for (int k = 0; k < count; k++) {
                for (int j = 0; j < n; j++) {
                    v[k][j] = p[j] / ktu[k][j];
                }
            }
        }
        double total = 0;
        for (double x : p) {
            total += x;
        }
        for (int j = 0; j < n; j++) {
            p[j] /= total;
        }
        return p;
    }
}
