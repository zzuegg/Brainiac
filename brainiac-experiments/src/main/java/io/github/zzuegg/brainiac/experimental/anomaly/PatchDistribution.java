package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.List;
import java.util.SplittableRandom;

/**
 * PaDiM: a multivariate Gaussian per spatial position, fitted to the patch features of normal
 * images (optionally a random subset of the feature dimensions), with the covariance shrunk by
 * {@code eps I}; test patches are scored by their Mahalanobis distance to their position's
 * Gaussian.
 *
 * <p>Paper: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
 * Detection and Localization", ICPR workshops 2020,
 * <a href="https://arxiv.org/abs/2011.08785">arXiv:2011.08785</a>.
 */
public final class PatchDistribution {

    private final int[] dims;
    private final double[][] mean;
    private final double[][][] precision;

    private PatchDistribution(int[] dims, double[][] mean, double[][][] precision) {
        this.dims = dims;
        this.mean = mean;
        this.precision = precision;
    }

    /**
     * Fits the Gaussians.
     *
     * @param images per normal image its features {@code [positions][dims]}
     * @param keep feature dimensions kept at random (0 or more than available keeps all)
     * @param eps the covariance shrinkage
     * @param seed the seed
     * @return the model
     */
    public static PatchDistribution fit(List<float[][]> images, int keep, double eps, long seed) {
        int positions = images.get(0).length;
        int all = images.get(0)[0].length;
        int[] dims;
        if (keep <= 0 || keep >= all) {
            dims = java.util.stream.IntStream.range(0, all).toArray();
        } else {
            int[] perm = java.util.stream.IntStream.range(0, all).toArray();
            SplittableRandom r = new SplittableRandom(seed);
            for (int i = all - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = perm[i];
                perm[i] = perm[j];
                perm[j] = t;
            }
            dims = java.util.Arrays.copyOf(perm, keep);
            java.util.Arrays.sort(dims);
        }
        int d = dims.length;
        int n = images.size();
        double[][] mean = new double[positions][d];
        double[][][] prec = new double[positions][][];
        for (int p = 0; p < positions; p++) {
            for (float[][] im : images) {
                for (int k = 0; k < d; k++) {
                    mean[p][k] += im[p][dims[k]] / (double) n;
                }
            }
            double[][] cov = new double[d][d];
            for (float[][] im : images) {
                for (int a = 0; a < d; a++) {
                    double xa = im[p][dims[a]] - mean[p][a];
                    for (int b = a; b < d; b++) {
                        cov[a][b] += xa * (im[p][dims[b]] - mean[p][b]) / Math.max(1, n - 1);
                    }
                }
            }
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < a; b++) {
                    cov[a][b] = cov[b][a];
                }
                cov[a][a] += eps;
            }
            prec[p] = invert(cov);
        }
        return new PatchDistribution(dims, mean, prec);
    }

    /** Gauss-Jordan inverse of a symmetric positive definite matrix. */
    private static double[][] invert(double[][] m) {
        int n = m.length;
        double[][] a = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(m[i], 0, a[i], 0, n);
            a[i][n + i] = 1;
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int i = c + 1; i < n; i++) {
                if (Math.abs(a[i][c]) > Math.abs(a[piv][c])) {
                    piv = i;
                }
            }
            double[] t = a[c];
            a[c] = a[piv];
            a[piv] = t;
            double v = a[c][c];
            for (int j = 0; j < 2 * n; j++) {
                a[c][j] /= v;
            }
            for (int i = 0; i < n; i++) {
                if (i != c && a[i][c] != 0) {
                    double f = a[i][c];
                    for (int j = 0; j < 2 * n; j++) {
                        a[i][j] -= f * a[c][j];
                    }
                }
            }
        }
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], n, out[i], 0, n);
        }
        return out;
    }

    /**
     * Mahalanobis distances per position.
     *
     * @param patches {@code [positions][dims]}
     * @return the scores
     */
    public double[] map(float[][] patches) {
        double[] out = new double[patches.length];
        int d = dims.length;
        double[] x = new double[d];
        for (int p = 0; p < patches.length; p++) {
            for (int k = 0; k < d; k++) {
                x[k] = patches[p][dims[k]] - mean[p][k];
            }
            double s = 0;
            for (int a = 0; a < d; a++) {
                double row = 0;
                for (int b = 0; b < d; b++) {
                    row += precision[p][a][b] * x[b];
                }
                s += x[a] * row;
            }
            out[p] = Math.sqrt(Math.max(0, s));
        }
        return out;
    }

    /**
     * @param patches {@code [positions][dims]}
     * @return the largest position score
     */
    public double score(float[][] patches) {
        return java.util.Arrays.stream(map(patches)).max().orElse(0);
    }
}
