package io.github.zzuegg.brainiac.experimental.eval;

import java.util.Arrays;

/**
 * The Kernel Divergence Score (Choi et al., "How Contaminated Is Your Benchmark? Quantifying
 * Dataset Leakage in Large Language Models with Kernel Divergence", ICML 2025,
 * <a href="https://arxiv.org/abs/2502.00678">arXiv:2502.00678</a>): a set-level contamination
 * estimate. Embed every benchmark sample with the model, briefly fine-tune the model on the
 * benchmark, embed again, and compare the two RBF kernel matrices. Fine-tuning moves the
 * embeddings of samples the model has not seen more than those it has memorised, so the
 * similarity-weighted kernel divergence
 * {@code D = Σ_ij K_ij |log K_ij - log K'_ij| / Σ_ij K_ij} falls as the seen fraction rises; the
 * score is {@code -D} (higher means more contaminated).
 */
public final class KernelDivergenceScore {

    private KernelDivergenceScore() {
    }

    /**
     * The RBF kernel matrix {@code exp(-γ ||z_i - z_j||²)} with γ set from the median squared
     * distance of the "before" embeddings (the bandwidth heuristic), so both matrices share it.
     *
     * @param z embeddings {@code [n][d]}
     * @param gamma the kernel scale
     * @return {@code [n][n]}
     */
    public static double[][] kernel(double[][] z, double gamma) {
        int n = z.length;
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double v = Math.exp(-gamma * squaredDistance(z[i], z[j]));
                k[i][j] = v;
                k[j][i] = v;
            }
        }
        return k;
    }

    private static double squaredDistance(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    /**
     * The median-heuristic scale {@code 1 / median ||z_i - z_j||²}.
     *
     * @param z embeddings
     * @return γ
     */
    public static double medianGamma(double[][] z) {
        int n = z.length;
        double[] d = new double[n * (n - 1) / 2];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                d[k++] = squaredDistance(z[i], z[j]);
            }
        }
        Arrays.sort(d);
        double m = d.length == 0 ? 1 : d[d.length / 2];
        return 1 / Math.max(1e-12, m);
    }

    /**
     * The score {@code -D} of embeddings before and after fine-tuning on the benchmark.
     *
     * @param before embeddings before fine-tuning {@code [n][d]}
     * @param after embeddings after fine-tuning, same samples
     * @return the score; higher means a larger seen fraction
     */
    public static double score(double[][] before, double[][] after) {
        double gamma = medianGamma(before);
        double[][] k = kernel(before, gamma);
        double[][] k2 = kernel(after, gamma);
        double num = 0;
        double den = 0;
        for (int i = 0; i < k.length; i++) {
            for (int j = 0; j < k.length; j++) {
                if (i == j) {
                    continue;
                }
                double a = Math.max(1e-300, k[i][j]);
                double b = Math.max(1e-300, k2[i][j]);
                num += a * Math.abs(Math.log(a) - Math.log(b));
                den += a;
            }
        }
        return den == 0 ? 0 : -num / den;
    }
}
