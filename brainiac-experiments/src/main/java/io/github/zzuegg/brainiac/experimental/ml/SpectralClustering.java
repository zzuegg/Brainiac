package io.github.zzuegg.brainiac.experimental.ml;

/**
 * Normalised spectral clustering (Ng, Jordan and Weiss 2002; tutorial: von Luxburg 2007,
 * <a href="https://arxiv.org/abs/0711.0189">arXiv:0711.0189</a>). Points become nodes of a
 * similarity graph; the {@code k} leading eigenvectors of the normalised affinity
 * {@code D^-1/2 W D^-1/2} (the smallest of the normalised Laplacian) embed the nodes so that
 * connected components become compact, and k-means on the row-normalised embedding finds them.
 *
 * <p>Why it matters: it clusters by connectivity rather than by distance to a centre, so it
 * separates two moons or concentric rings that k-means and a GMM cut in half.
 */
public final class SpectralClustering {

    private SpectralClustering() {
    }

    /**
     * A symmetric k-nearest-neighbour affinity with self-tuning RBF weights
     * (Zelnik-Manor and Perona 2004): {@code w_ij = exp(-d_ij^2 / (s_i s_j))}, where {@code s_i}
     * is the distance to the {@code k}-th neighbour, kept when either point is among the
     * other's neighbours.
     *
     * @param x the points {@code [n][d]}
     * @param k the neighbourhood size
     * @return the affinity matrix {@code [n][n]}, zero diagonal
     */
    public static double[][] knnAffinity(double[][] x, int k) {
        int n = x.length;
        double[][] d2 = MlMatrices.squaredDistances(x);
        double[] scale = new double[n];
        boolean[][] near = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            int[] nn = MlMetrics.nearest(d2[i], i, k);
            scale[i] = Math.sqrt(d2[i][nn[k - 1]]) + 1e-12;
            for (int j : nn) {
                near[i][j] = true;
            }
        }
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j && (near[i][j] || near[j][i])) {
                    w[i][j] = Math.exp(-d2[i][j] / (scale[i] * scale[j]));
                }
            }
        }
        return w;
    }

    /**
     * A fully connected RBF affinity {@code exp(-gamma |x_i - x_j|^2)}.
     *
     * @param x the points
     * @param gamma the inverse width
     * @return the affinity matrix, zero diagonal
     */
    public static double[][] rbfAffinity(double[][] x, double gamma) {
        double[][] w = KernelFunction.rbf(gamma).gram(x);
        for (int i = 0; i < w.length; i++) {
            w[i][i] = 0;
        }
        return w;
    }

    /**
     * The spectral embedding: the {@code k} leading eigenvectors of the normalised affinity,
     * each row scaled to unit length.
     *
     * @param affinity a symmetric non-negative affinity
     * @param k the embedding dimension
     * @return {@code [n][k]}
     */
    public static double[][] embed(double[][] affinity, int k) {
        int n = affinity.length;
        double[] dInv = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (double v : affinity[i]) {
                s += v;
            }
            dInv[i] = 1 / Math.sqrt(Math.max(s, 1e-12));
        }
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i][j] = dInv[i] * affinity[i][j] * dInv[j];
            }
        }
        MlMatrices.Eigen e = MlMatrices.eigh(m);
        double[][] u = new double[n][k];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int c = 0; c < k; c++) {
                u[i][c] = e.vectors()[i][c];
                norm += u[i][c] * u[i][c];
            }
            norm = Math.sqrt(Math.max(norm, 1e-24));
            for (int c = 0; c < k; c++) {
                u[i][c] /= norm;
            }
        }
        return u;
    }

    /**
     * Clusters points by the spectral embedding of an affinity.
     *
     * @param affinity a symmetric non-negative affinity
     * @param k the number of clusters
     * @param seed seeds k-means
     * @return the cluster of each point
     */
    public static int[] cluster(double[][] affinity, int k, long seed) {
        return KMeans.fit(embed(affinity, k), k, KMeans.Init.PLUS_PLUS, 10, 100, seed).labels();
    }
}
