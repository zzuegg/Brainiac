package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

import java.util.SplittableRandom;

/**
 * A linear projection {@code s = W (x - mean)} found by principal component analysis or by
 * FastICA, plus RBF {@linkplain Kernel kernel PCA}.
 * <ul>
 *   <li>{@link #pca}: the directions of largest variance, from the thin SVD of the centred
 *       data ({@link ThinSvd}).</li>
 *   <li>{@link #fastIca}: independent component analysis by Hyvärinen's fixed-point iteration
 *       (<a href="https://ieeexplore.ieee.org/document/761722">Fast and robust fixed-point
 *       algorithms for independent component analysis</a>, 1999) with the log-cosh contrast
 *       and symmetric decorrelation, on PCA-whitened data. Where PCA only decorrelates, ICA
 *       recovers non-Gaussian sources mixed linearly (the cocktail-party problem).</li>
 *   <li>{@link #kernelPca}: PCA in the feature space of a kernel (Schölkopf, Smola and Müller
 *       1998, <a href="https://direct.mit.edu/neco/article/10/5/1299/6193">Nonlinear component
 *       analysis as a kernel eigenvalue problem</a>), from the eigenvectors of the centred
 *       kernel matrix; it unfolds structure such as concentric circles that no linear
 *       projection separates.</li>
 * </ul>
 */
public final class ComponentAnalysis {

    private final double[] mean;
    private final double[][] w;
    private final double[] variances;

    private ComponentAnalysis(double[] mean, double[][] w, double[] variances) {
        this.mean = mean;
        this.w = w;
        this.variances = variances;
    }

    private static double[][] centred(double[][] x, double[] mean) {
        double[][] c = new double[x.length][mean.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                c[i][j] = x[i][j] - mean[j];
            }
        }
        return c;
    }

    /**
     * Principal component analysis.
     *
     * @param x the data {@code [n][d]}
     * @param k the number of components, at most {@code min(n, d)}
     * @return the projection onto the top {@code k} principal directions
     */
    public static ComponentAnalysis pca(double[][] x, int k) {
        double[] mean = MlMatrices.columnMeans(x);
        ThinSvd svd = ThinSvd.of(centred(x, mean));
        double[][] w = new double[k][];
        double[] var = new double[k];
        for (int c = 0; c < k; c++) {
            w[c] = new double[mean.length];
            for (int j = 0; j < mean.length; j++) {
                w[c][j] = svd.v()[j][c];
            }
            var[c] = svd.s()[c] * svd.s()[c] / Math.max(1, x.length - 1);
        }
        return new ComponentAnalysis(mean, w, var);
    }

    /**
     * FastICA with the log-cosh contrast and symmetric decorrelation.
     *
     * @param x the mixed signals {@code [n][d]}, one observation per row
     * @param k the number of sources
     * @param iterations the most fixed-point iterations
     * @param seed draws the initial unmixing matrix
     * @return the unmixing projection; rows of {@link #transform} are the estimated sources
     *     (unit variance, arbitrary sign and order)
     */
    public static ComponentAnalysis fastIca(double[][] x, int k, int iterations, long seed) {
        ComponentAnalysis white = pca(x, k);
        int n = x.length;
        double[][] z = white.transform(x);
        for (double[] row : z) {
            for (int c = 0; c < k; c++) {
                row[c] /= Math.sqrt(white.variances[c]);
            }
        }
        SplittableRandom random = new SplittableRandom(seed);
        double[][] wm = new double[k][k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                wm[a][b] = random.nextGaussian();
            }
        }
        wm = decorrelate(wm);
        for (int it = 0; it < iterations; it++) {
            double[][] next = new double[k][k];
            for (int c = 0; c < k; c++) {
                double gPrimeMean = 0;
                for (double[] zi : z) {
                    double u = MlMatrices.dot(wm[c], zi);
                    double g = Math.tanh(u);
                    gPrimeMean += (1 - g * g) / n;
                    for (int j = 0; j < k; j++) {
                        next[c][j] += zi[j] * g / n;
                    }
                }
                for (int j = 0; j < k; j++) {
                    next[c][j] -= gPrimeMean * wm[c][j];
                }
            }
            next = decorrelate(next);
            double change = 0;
            for (int c = 0; c < k; c++) {
                change = Math.max(change, Math.abs(Math.abs(MlMatrices.dot(next[c], wm[c])) - 1));
            }
            wm = next;
            if (change < 1e-8) {
                break;
            }
        }
        // compose: s = W_ica diag(1/sqrt(var)) W_pca (x - mean)
        double[][] full = new double[k][x[0].length];
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < k; j++) {
                double coef = wm[c][j] / Math.sqrt(white.variances[j]);
                for (int t = 0; t < full[c].length; t++) {
                    full[c][t] += coef * white.w[j][t];
                }
            }
        }
        double[] ones = new double[k];
        java.util.Arrays.fill(ones, 1);
        return new ComponentAnalysis(white.mean, full, ones);
    }

    /** {@code (W W^T)^-1/2 W}: makes the rows orthonormal without favouring any of them. */
    private static double[][] decorrelate(double[][] w) {
        double[][] g = MlMatrices.multiply(w, MlMatrices.transpose(w));
        MlMatrices.Eigen e = MlMatrices.eigh(g);
        int k = w.length;
        double[][] inv = new double[k][k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                double s = 0;
                for (int c = 0; c < k; c++) {
                    s += e.vectors()[a][c] * e.vectors()[b][c] / Math.sqrt(Math.max(e.values()[c], 1e-15));
                }
                inv[a][b] = s;
            }
        }
        return MlMatrices.multiply(inv, w);
    }

    /**
     * Projects rows.
     *
     * @param x the data {@code [n][d]}
     * @return {@code [n][k]}
     */
    public double[][] transform(double[][] x) {
        double[][] out = new double[x.length][w.length];
        for (int i = 0; i < x.length; i++) {
            for (int c = 0; c < w.length; c++) {
                double s = 0;
                for (int j = 0; j < mean.length; j++) {
                    s += w[c][j] * (x[i][j] - mean[j]);
                }
                out[i][c] = s;
            }
        }
        return out;
    }

    /**
     * Maps projected rows back to the input space ({@code x = W^T s + mean}, exact for PCA's
     * orthonormal rows).
     *
     * @param s projected rows {@code [n][k]}
     * @return {@code [n][d]}
     */
    public double[][] inverseTransform(double[][] s) {
        double[][] out = new double[s.length][mean.length];
        for (int i = 0; i < s.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                double v = mean[j];
                for (int c = 0; c < w.length; c++) {
                    v += w[c][j] * s[i][c];
                }
                out[i][j] = v;
            }
        }
        return out;
    }

    /** @return the variance along each component (all 1 for ICA's unit-variance sources) */
    public double[] variances() {
        return variances.clone();
    }

    /**
     * Fits kernel PCA.
     *
     * @param x the data {@code [n][d]}
     * @param kernel the kernel
     * @param k the number of components
     * @return the fitted projection
     */
    public static Kernel kernelPca(double[][] x, KernelFunction kernel, int k) {
        return new Kernel(x, kernel, k);
    }

    /** Kernel PCA: projections onto the leading eigenvectors of the centred kernel matrix. */
    public static final class Kernel {
        private final double[][] x;
        private final KernelFunction kernel;
        private final double[][] alphas;
        private final double[] rowMeans;
        private final double totalMean;

        private Kernel(double[][] x, KernelFunction kernel, int k) {
            this.x = x;
            this.kernel = kernel;
            int n = x.length;
            double[][] km = kernel.gram(x);
            this.rowMeans = new double[n];
            double tm = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    rowMeans[i] += km[i][j] / n;
                }
                tm += rowMeans[i] / n;
            }
            this.totalMean = tm;
            double[][] kc = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    kc[i][j] = km[i][j] - rowMeans[i] - rowMeans[j] + tm;
                }
            }
            MlMatrices.Eigen e = MlMatrices.eigh(kc);
            this.alphas = new double[k][n];
            for (int c = 0; c < k; c++) {
                double norm = Math.sqrt(Math.max(e.values()[c], 1e-12));
                for (int i = 0; i < n; i++) {
                    alphas[c][i] = e.vectors()[i][c] / norm;
                }
            }
        }

        /**
         * Projects points onto the kernel principal components.
         *
         * @param xs points {@code [m][d]}
         * @return {@code [m][k]}
         */
        public double[][] transform(double[][] xs) {
            int n = x.length;
            double[][] out = new double[xs.length][alphas.length];
            double[] kx = new double[n];
            for (int t = 0; t < xs.length; t++) {
                double mean = 0;
                for (int i = 0; i < n; i++) {
                    kx[i] = kernel.value(xs[t], x[i]);
                    mean += kx[i] / n;
                }
                for (int c = 0; c < alphas.length; c++) {
                    double s = 0;
                    for (int i = 0; i < n; i++) {
                        s += alphas[c][i] * (kx[i] - mean - rowMeans[i] + totalMean);
                    }
                    out[t][c] = s;
                }
            }
            return out;
        }
    }
}
