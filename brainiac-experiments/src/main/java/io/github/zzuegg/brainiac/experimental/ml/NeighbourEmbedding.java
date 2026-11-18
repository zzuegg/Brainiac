package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Non-linear neighbour embeddings into a few dimensions for visualisation, both initialised
 * from {@link ComponentAnalysis#pca PCA}.
 * <ul>
 *   <li>{@link #tsne}: t-SNE (van der Maaten and Hinton 2008,
 *       <a href="https://jmlr.org/papers/v9/vandermaaten08a.html">Visualizing data using
 *       t-SNE</a>), exact {@code O(n^2)} gradients: Gaussian input affinities calibrated to a
 *       perplexity, heavy-tailed Student-t output affinities, KL divergence minimised by
 *       gradient descent with momentum, per-parameter gains and early exaggeration.</li>
 *   <li>{@link #umap}: UMAP (McInnes, Healy and Melville 2018,
 *       <a href="https://arxiv.org/abs/1802.03426">arXiv:1802.03426</a>): a fuzzy k-nearest-neighbour
 *       graph (each point's distances shifted by its nearest neighbour and scaled so that its
 *       memberships sum to {@code log2 k}, symmetrised by fuzzy union), laid out by stochastic
 *       gradient descent on the fuzzy-set cross-entropy with negative sampling and the
 *       {@code 1 / (1 + a d^2b)} output kernel.</li>
 * </ul>
 */
public final class NeighbourEmbedding {

    private NeighbourEmbedding() {
    }

    /**
     * t-SNE settings.
     *
     * @param dimensions the output dimension, usually 2
     * @param perplexity the effective neighbour count, for example 30
     * @param iterations gradient steps, for example 500
     * @param learningRate the step size, for example 200
     * @param exaggeration the early exaggeration factor, for example 12
     * @param seed the seed (for the tiny jitter added to the PCA start)
     */
    public record TsneSettings(int dimensions, double perplexity, int iterations, double learningRate,
            double exaggeration, long seed) {

        /**
         * The usual settings.
         *
         * @param perplexity the perplexity
         * @return 2-D, 500 iterations, rate 200, exaggeration 12
         */
        public static TsneSettings defaults(double perplexity) {
            return new TsneSettings(2, perplexity, 500, 200, 12, 1);
        }
    }

    /**
     * Embeds points with t-SNE.
     *
     * @param x the points {@code [n][d]}
     * @param s the settings
     * @return the embedding {@code [n][dimensions]}
     */
    public static double[][] tsne(double[][] x, TsneSettings s) {
        int n = x.length;
        int dims = s.dimensions();
        double[][] d2 = MlMatrices.squaredDistances(x);
        double[][] p = new double[n][n];
        double target = Math.log(s.perplexity());
        for (int i = 0; i < n; i++) {
            double[] row = conditional(d2[i], i, target);
            System.arraycopy(row, 0, p[i], 0, n);
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double v = Math.max((p[i][j] + p[j][i]) / (2 * n), 1e-12);
                p[i][j] = v;
                p[j][i] = v;
            }
        }
        double[][] y = initial(x, dims, 1e-4, s.seed());
        double[][] velocity = new double[n][dims];
        double[][] gains = new double[n][dims];
        for (double[] g : gains) {
            java.util.Arrays.fill(g, 1);
        }
        double[][] num = new double[n][n];
        double[][] grad = new double[n][dims];
        int stopExaggeration = Math.min(100, s.iterations() / 4);
        for (int it = 0; it < s.iterations(); it++) {
            double exag = it < stopExaggeration ? s.exaggeration() : 1;
            double momentum = it < stopExaggeration ? 0.5 : 0.8;
            double z = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double v = 1 / (1 + MlMatrices.squaredDistance(y[i], y[j]));
                    num[i][j] = v;
                    num[j][i] = v;
                    z += 2 * v;
                }
            }
            for (int i = 0; i < n; i++) {
                java.util.Arrays.fill(grad[i], 0);
                for (int j = 0; j < n; j++) {
                    if (i == j) {
                        continue;
                    }
                    double mult = 4 * (exag * p[i][j] - num[i][j] / z) * num[i][j];
                    for (int c = 0; c < dims; c++) {
                        grad[i][c] += mult * (y[i][c] - y[j][c]);
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < dims; c++) {
                    boolean sameSign = Math.signum(grad[i][c]) == Math.signum(velocity[i][c]);
                    gains[i][c] = Math.max(0.01, sameSign ? gains[i][c] * 0.8 : gains[i][c] + 0.2);
                    velocity[i][c] = momentum * velocity[i][c] - s.learningRate() * gains[i][c] * grad[i][c];
                    y[i][c] += velocity[i][c];
                }
            }
            centre(y);
        }
        return y;
    }

    /** One row of Gaussian conditional affinities whose entropy matches {@code log perplexity}. */
    private static double[] conditional(double[] d2, int self, double targetEntropy) {
        int n = d2.length;
        double beta = 1;
        double lo = 0;
        double hi = Double.POSITIVE_INFINITY;
        double[] p = new double[n];
        for (int iter = 0; iter < 64; iter++) {
            double sum = 0;
            double dot = 0;
            for (int j = 0; j < n; j++) {
                p[j] = j == self ? 0 : Math.exp(-beta * d2[j]);
                sum += p[j];
                dot += p[j] * d2[j];
            }
            sum = Math.max(sum, 1e-300);
            double entropy = Math.log(sum) + beta * dot / sum;
            for (int j = 0; j < n; j++) {
                p[j] /= sum;
            }
            if (Math.abs(entropy - targetEntropy) < 1e-5) {
                break;
            }
            if (entropy > targetEntropy) {
                lo = beta;
                beta = hi == Double.POSITIVE_INFINITY ? beta * 2 : (beta + hi) / 2;
            } else {
                hi = beta;
                beta = (beta + lo) / 2;
            }
        }
        return p;
    }

    /** PCA coordinates scaled to a small standard deviation, plus a little jitter. */
    private static double[][] initial(double[][] x, int dims, double sd, long seed) {
        double[][] y = ComponentAnalysis.pca(x, dims).transform(x);
        double s = 0;
        for (double[] row : y) {
            s += row[0] * row[0];
        }
        s = Math.sqrt(s / y.length) + 1e-12;
        SplittableRandom r = new SplittableRandom(seed);
        for (double[] row : y) {
            for (int c = 0; c < dims; c++) {
                row[c] = row[c] / s * sd + 1e-3 * sd * r.nextGaussian();
            }
        }
        return y;
    }

    private static void centre(double[][] y) {
        double[] m = MlMatrices.columnMeans(y);
        for (double[] row : y) {
            for (int c = 0; c < m.length; c++) {
                row[c] -= m[c];
            }
        }
    }

    /**
     * UMAP settings.
     *
     * @param dimensions the output dimension
     * @param neighbours the k of the k-nearest-neighbour graph, for example 15
     * @param epochs layout epochs, for example 200
     * @param negativeSamples repulsive samples per attractive update, for example 5
     * @param a the output-kernel shape (1.577 for min_dist 0.1)
     * @param b the output-kernel shape (0.895 for min_dist 0.1)
     * @param seed the seed
     */
    public record UmapSettings(int dimensions, int neighbours, int epochs, int negativeSamples, double a, double b,
            long seed) {

        /**
         * The usual settings for {@code min_dist = 0.1}.
         *
         * @param neighbours the k
         * @return 2-D, 200 epochs, 5 negative samples
         */
        public static UmapSettings defaults(int neighbours) {
            return new UmapSettings(2, neighbours, 200, 5, 1.577, 0.895, 1);
        }
    }

    /**
     * Embeds points with UMAP.
     *
     * @param x the points {@code [n][d]}
     * @param s the settings
     * @return the embedding {@code [n][dimensions]}
     */
    public static double[][] umap(double[][] x, UmapSettings s) {
        int n = x.length;
        int k = s.neighbours();
        double[][] d2 = MlMatrices.squaredDistances(x);
        double[][] w = new double[n][n];
        double target = Math.log(k) / Math.log(2);
        for (int i = 0; i < n; i++) {
            int[] nn = MlMetrics.nearest(d2[i], i, k);
            double rho = Math.sqrt(d2[i][nn[0]]);
            double lo = 0;
            double hi = Double.POSITIVE_INFINITY;
            double sigma = 1;
            for (int iter = 0; iter < 64; iter++) {
                double sum = 0;
                for (int j : nn) {
                    sum += Math.exp(-Math.max(0, Math.sqrt(d2[i][j]) - rho) / sigma);
                }
                if (Math.abs(sum - target) < 1e-5) {
                    break;
                }
                if (sum > target) {
                    hi = sigma;
                    sigma = (lo + hi) / 2;
                } else {
                    lo = sigma;
                    sigma = hi == Double.POSITIVE_INFINITY ? sigma * 2 : (lo + hi) / 2;
                }
            }
            for (int j : nn) {
                w[i][j] = Math.exp(-Math.max(0, Math.sqrt(d2[i][j]) - rho) / sigma);
            }
        }
        List<int[]> edges = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        double maxW = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double v = w[i][j] + w[j][i] - w[i][j] * w[j][i];
                if (v > 0) {
                    edges.add(new int[] {i, j});
                    weights.add(v);
                    maxW = Math.max(maxW, v);
                }
            }
        }
        double[][] y = initial(x, s.dimensions(), 3, s.seed());
        SplittableRandom r = new SplittableRandom(s.seed());
        double a = s.a();
        double b = s.b();
        int dims = s.dimensions();
        double[] epochsPerSample = new double[edges.size()];
        double[] nextSample = new double[edges.size()];
        for (int e = 0; e < edges.size(); e++) {
            epochsPerSample[e] = maxW / weights.get(e);
            nextSample[e] = epochsPerSample[e];
        }
        for (int epoch = 1; epoch <= s.epochs(); epoch++) {
            double alpha = 1 - (epoch - 1) / (double) s.epochs();
            for (int e = 0; e < edges.size(); e++) {
                if (nextSample[e] > epoch) {
                    continue;
                }
                nextSample[e] += epochsPerSample[e];
                int i = edges.get(e)[0];
                int j = edges.get(e)[1];
                for (int dir = 0; dir < 2; dir++) {
                    int from = dir == 0 ? i : j;
                    int to = dir == 0 ? j : i;
                    double dist = MlMatrices.squaredDistance(y[from], y[to]);
                    if (dist > 0) {
                        double coef = -2 * a * b * Math.pow(dist, b - 1) / (1 + a * Math.pow(dist, b));
                        for (int c = 0; c < dims; c++) {
                            y[from][c] += alpha * Math.clamp(coef * (y[from][c] - y[to][c]), -4, 4);
                        }
                    }
                    for (int ns = 0; ns < s.negativeSamples(); ns++) {
                        int o = r.nextInt(n);
                        if (o == from) {
                            continue;
                        }
                        double dn = MlMatrices.squaredDistance(y[from], y[o]);
                        double coef = 2 * b / ((0.001 + dn) * (1 + a * Math.pow(dn, b)));
                        for (int c = 0; c < dims; c++) {
                            y[from][c] += alpha * Math.clamp(coef * (y[from][c] - y[o][c]), -4, 4);
                        }
                    }
                }
            }
        }
        return y;
    }
}
