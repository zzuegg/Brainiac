package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;

/**
 * Similarity indices between two layers' representations of the same inputs
 * ({@code [n][p]} and {@code [n][q]}), for asking whether two networks compute the same
 * thing layer by layer.
 *
 * <ul>
 *   <li>{@link #linearCka} (Kornblith et al., ICML 2019): HSIC between the centred Gram
 *       matrices, normalised, {@code |Y^T X|_F^2 / (|X^T X|_F |Y^T Y|_F)} on centred data;
 *       invariant to orthogonal transforms and isotropic scaling, not to arbitrary
 *       invertible linear maps.</li>
 *   <li>{@link #rbfCka}: the same with Gaussian kernels, the bandwidth a multiple of the
 *       median distance.</li>
 *   <li>{@link #svcca} (Raghu et al., NeurIPS 2017): reduce each side to the top singular
 *       directions keeping 99% of the variance, then average the canonical correlations.</li>
 *   <li>{@link #pwcca} (Morcos et al., 2018): canonical correlations weighted by how much
 *       of the first representation each canonical variate accounts for.</li>
 *   <li>{@link #maxCorrelation}: the raw baseline, the mean over units of the first layer of
 *       the best absolute Pearson correlation with a unit of the second.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1905.00414">arXiv:1905.00414</a>,
 * <a href="https://arxiv.org/abs/1706.05806">arXiv:1706.05806</a>,
 * <a href="https://arxiv.org/abs/1806.05759">arXiv:1806.05759</a>.
 */
public final class RepresentationSimilarity {

    private RepresentationSimilarity() {
    }

    /**
     * Linear CKA.
     *
     * @param x {@code [n][p]}
     * @param y {@code [n][q]}
     * @return the index in {@code [0, 1]}
     */
    public static double linearCka(float[][] x, float[][] y) {
        double[][] a = InterpMath.centre(InterpMath.toDouble(x));
        double[][] b = InterpMath.centre(InterpMath.toDouble(y));
        double xy = frobenius2(InterpMath.crossProduct(b, a, 1));
        double xx = Math.sqrt(frobenius2(InterpMath.crossProduct(a, a, 1)));
        double yy = Math.sqrt(frobenius2(InterpMath.crossProduct(b, b, 1)));
        return xx == 0 || yy == 0 ? 0 : xy / (xx * yy);
    }

    private static double frobenius2(double[][] m) {
        double s = 0;
        for (double[] r : m) {
            for (double v : r) {
                s += v * v;
            }
        }
        return s;
    }

    /**
     * CKA with Gaussian kernels.
     *
     * @param x {@code [n][p]}
     * @param y {@code [n][q]}
     * @param bandwidth the kernel width as a multiple of each side's median distance
     * @return the index in {@code [0, 1]}
     */
    public static double rbfCka(float[][] x, float[][] y, double bandwidth) {
        double[][] kx = centredGram(InterpMath.toDouble(x), bandwidth);
        double[][] ky = centredGram(InterpMath.toDouble(y), bandwidth);
        double xy = 0;
        double xx = 0;
        double yy = 0;
        for (int i = 0; i < kx.length; i++) {
            for (int j = 0; j < kx.length; j++) {
                xy += kx[i][j] * ky[i][j];
                xx += kx[i][j] * kx[i][j];
                yy += ky[i][j] * ky[i][j];
            }
        }
        return xx == 0 || yy == 0 ? 0 : xy / Math.sqrt(xx * yy);
    }

    private static double[][] centredGram(double[][] z, double bandwidth) {
        int n = z.length;
        double[][] d2 = new double[n][n];
        double[] all = new double[n * (n - 1) / 2];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0;
                for (int c = 0; c < z[i].length; c++) {
                    double t = z[i][c] - z[j][c];
                    s += t * t;
                }
                d2[i][j] = s;
                d2[j][i] = s;
                all[k++] = s;
            }
        }
        java.util.Arrays.sort(all);
        double median = all.length == 0 ? 1 : Math.max(all[all.length / 2], 1e-12);
        double sigma2 = bandwidth * bandwidth * median;
        double[][] g = new double[n][n];
        double[] rowMean = new double[n];
        double total = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                g[i][j] = Math.exp(-d2[i][j] / (2 * sigma2));
                rowMean[i] += g[i][j] / n;
            }
            total += rowMean[i] / n;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                g[i][j] = g[i][j] - rowMean[i] - rowMean[j] + total;
            }
        }
        return g;
    }

    /** The canonical correlations and variates of two centred, reduced representations. */
    private record Cca(double[] correlations, double[][] variatesX) {
    }

    private static double[][] reduce(double[][] centred, double energy) {
        ThinSvd svd = ThinSvd.of(centred);
        double[] s = svd.s();
        double total = 0;
        for (double v : s) {
            total += v * v;
        }
        int k = 0;
        double kept = 0;
        while (k < s.length && (kept < energy * total || k == 0)) {
            if (s[k] <= 1e-9 * s[0]) {
                break;
            }
            kept += s[k] * s[k];
            k++;
        }
        double[][] u = new double[centred.length][k];
        for (int i = 0; i < centred.length; i++) {
            for (int j = 0; j < k; j++) {
                u[i][j] = svd.u()[i][j];
            }
        }
        return u;
    }

    private static Cca cca(double[][] ux, double[][] uy) {
        double[][] m = InterpMath.crossProduct(ux, uy, 1);
        ThinSvd svd = ThinSvd.of(m);
        double[] rho = new double[svd.s().length];
        for (int i = 0; i < rho.length; i++) {
            rho[i] = Math.min(1, svd.s()[i]);
        }
        double[][] a = new double[ux[0].length][rho.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < rho.length; j++) {
                a[i][j] = svd.u()[i][j];
            }
        }
        return new Cca(rho, InterpMath.matmul(ux, a));
    }

    /**
     * SVCCA: the mean canonical correlation after keeping the top singular directions.
     *
     * @param x {@code [n][p]}
     * @param y {@code [n][q]}
     * @param energy the share of variance to keep on each side, such as 0.99
     * @return the mean canonical correlation
     */
    public static double svcca(float[][] x, float[][] y, double energy) {
        double[][] ux = reduce(InterpMath.centre(InterpMath.toDouble(x)), energy);
        double[][] uy = reduce(InterpMath.centre(InterpMath.toDouble(y)), energy);
        double[] rho = cca(ux, uy).correlations();
        double s = 0;
        for (double r : rho) {
            s += r;
        }
        return rho.length == 0 ? 0 : s / rho.length;
    }

    /**
     * PWCCA: canonical correlations weighted by each variate's summed absolute
     * correlation with the units of {@code x}.
     *
     * @param x {@code [n][p]}
     * @param y {@code [n][q]}
     * @param energy the share of variance to keep before CCA
     * @return the weighted mean canonical correlation
     */
    public static double pwcca(float[][] x, float[][] y, double energy) {
        double[][] xc = InterpMath.centre(InterpMath.toDouble(x));
        double[][] ux = reduce(xc, energy);
        double[][] uy = reduce(InterpMath.centre(InterpMath.toDouble(y)), energy);
        Cca c = cca(ux, uy);
        int k = c.correlations().length;
        double[] weight = new double[k];
        for (int v = 0; v < k; v++) {
            double hn = 0;
            for (double[] row : c.variatesX()) {
                hn += row[v] * row[v];
            }
            for (int j = 0; j < xc[0].length; j++) {
                double dotp = 0;
                double xn = 0;
                for (int i = 0; i < xc.length; i++) {
                    dotp += c.variatesX()[i][v] * xc[i][j];
                    xn += xc[i][j] * xc[i][j];
                }
                if (hn > 0 && xn > 0) {
                    weight[v] += Math.abs(dotp) / Math.sqrt(hn * xn);
                }
            }
        }
        double sw = 0;
        double s = 0;
        for (int v = 0; v < k; v++) {
            sw += weight[v];
            s += weight[v] * c.correlations()[v];
        }
        return sw == 0 ? 0 : s / sw;
    }

    /**
     * The raw baseline: the mean over units of {@code x} of the best absolute correlation
     * with any unit of {@code y} (constant units score 0).
     *
     * @param x {@code [n][p]}
     * @param y {@code [n][q]}
     * @return the mean best correlation
     */
    public static double maxCorrelation(float[][] x, float[][] y) {
        double[][] a = standardise(InterpMath.centre(InterpMath.toDouble(x)));
        double[][] b = standardise(InterpMath.centre(InterpMath.toDouble(y)));
        double[][] c = InterpMath.crossProduct(a, b, x.length);
        double s = 0;
        for (double[] row : c) {
            double best = 0;
            for (double v : row) {
                best = Math.max(best, Math.abs(v));
            }
            s += best;
        }
        return s / c.length;
    }

    private static double[][] standardise(double[][] c) {
        int n = c.length;
        int d = c[0].length;
        double[] sd = new double[d];
        for (double[] row : c) {
            for (int j = 0; j < d; j++) {
                sd[j] += row[j] * row[j] / n;
            }
        }
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = sd[j] > 1e-12 ? c[i][j] / Math.sqrt(sd[j]) : 0;
            }
        }
        return out;
    }

    /**
     * The layer-correspondence test: for each layer of network A, whether the most similar
     * layer of network B (by the given index) is the same layer.
     *
     * @param similarity {@code [layersA][layersB]} similarities
     * @return the share of layers whose best match is the same layer
     */
    public static double correspondence(double[][] similarity) {
        int hits = 0;
        for (int i = 0; i < similarity.length; i++) {
            int best = 0;
            for (int j = 1; j < similarity[i].length; j++) {
                best = similarity[i][j] > similarity[i][best] ? j : best;
            }
            hits += best == i ? 1 : 0;
        }
        return (double) hits / similarity.length;
    }
}
