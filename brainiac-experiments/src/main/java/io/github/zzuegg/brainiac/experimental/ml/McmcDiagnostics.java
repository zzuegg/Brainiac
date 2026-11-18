package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Convergence and quality diagnostics for Monte Carlo samples: the effective sample size of a
 * correlated chain (Geyer's initial positive sequence), split R-hat over several chains
 * (Gelman et al., <a href="https://arxiv.org/abs/1903.08008">Vehtari et al. 2021</a>), the
 * importance-sampling ESS, and the squared maximum mean discrepancy and sliced Wasserstein
 * distance between two sample sets.
 */
public final class McmcDiagnostics {

    private McmcDiagnostics() {
    }

    /**
     * The effective sample size of one scalar chain, from its autocorrelations summed in
     * pairs until a pair turns negative.
     *
     * @param chain the draws in order
     * @return the ESS, at most the chain length
     */
    public static double effectiveSampleSize(double[] chain) {
        int n = chain.length;
        double mean = 0;
        for (double v : chain) {
            mean += v / n;
        }
        double var = 0;
        for (double v : chain) {
            var += (v - mean) * (v - mean) / n;
        }
        if (var <= 0) {
            return 1;
        }
        double sum = 0;
        for (int lag = 1; lag + 1 < n; lag += 2) {
            double pair = autocorrelation(chain, mean, var, lag) + autocorrelation(chain, mean, var, lag + 1);
            if (pair < 0) {
                break;
            }
            sum += pair;
        }
        double tau = 1 + 2 * sum;
        return Math.min(n, n / Math.max(tau, 1e-9));
    }

    private static double autocorrelation(double[] x, double mean, double var, int lag) {
        double s = 0;
        for (int i = 0; i + lag < x.length; i++) {
            s += (x[i] - mean) * (x[i + lag] - mean);
        }
        return s / (x.length * var);
    }

    /**
     * The smallest per-coordinate ESS of a multivariate chain.
     *
     * @param samples draws {@code [n][d]} in order
     * @return the minimum ESS over coordinates
     */
    public static double minEffectiveSampleSize(double[][] samples) {
        double min = Double.POSITIVE_INFINITY;
        for (int j = 0; j < samples[0].length; j++) {
            min = Math.min(min, effectiveSampleSize(column(samples, j)));
        }
        return min;
    }

    /**
     * Split R-hat of one scalar over several chains: each chain is halved, and the ratio of the
     * pooled to the within-chain variance is near 1 when the chains agree.
     *
     * @param chains the chains {@code [m][n]}
     * @return R-hat, 1 at convergence
     */
    public static double splitRHat(double[][] chains) {
        int half = chains[0].length / 2;
        double[][] split = new double[2 * chains.length][];
        for (int c = 0; c < chains.length; c++) {
            split[2 * c] = Arrays.copyOfRange(chains[c], 0, half);
            split[2 * c + 1] = Arrays.copyOfRange(chains[c], half, 2 * half);
        }
        int m = split.length;
        double[] means = new double[m];
        double within = 0;
        double grand = 0;
        for (int c = 0; c < m; c++) {
            for (double v : split[c]) {
                means[c] += v / half;
            }
            double s = 0;
            for (double v : split[c]) {
                s += (v - means[c]) * (v - means[c]);
            }
            within += s / (half - 1) / m;
            grand += means[c] / m;
        }
        double between = 0;
        for (double mu : means) {
            between += (mu - grand) * (mu - grand) * half / (m - 1);
        }
        double pooled = (half - 1.0) / half * within + between / half;
        return Math.sqrt(pooled / Math.max(within, 1e-300));
    }

    /**
     * The worst split R-hat over coordinates of several multivariate chains.
     *
     * @param chains the chains {@code [m][n][d]}
     * @return the maximum R-hat
     */
    public static double maxSplitRHat(double[][][] chains) {
        double max = 0;
        for (int j = 0; j < chains[0][0].length; j++) {
            double[][] c = new double[chains.length][];
            for (int k = 0; k < chains.length; k++) {
                c[k] = column(chains[k], j);
            }
            max = Math.max(max, splitRHat(c));
        }
        return max;
    }

    /**
     * The effective sample size of importance weights, {@code (sum w)^2 / sum w^2}.
     *
     * @param logWeights the log weights
     * @return the ESS
     */
    public static double importanceEss(double[] logWeights) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logWeights) {
            max = Math.max(max, v);
        }
        double s = 0;
        double s2 = 0;
        for (double v : logWeights) {
            double w = Math.exp(v - max);
            s += w;
            s2 += w * w;
        }
        return s * s / s2;
    }

    /**
     * {@code log mean exp} of log weights, the log of an importance-sampling estimate.
     *
     * @param logWeights the log weights
     * @return {@code log (1/n sum exp w)}
     */
    public static double logMeanExp(double[] logWeights) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : logWeights) {
            max = Math.max(max, v);
        }
        double s = 0;
        for (double v : logWeights) {
            s += Math.exp(v - max);
        }
        return max + Math.log(s / logWeights.length);
    }

    /**
     * The unbiased squared MMD between two samples with an RBF kernel of bandwidth
     * {@code sigma} ({@code k = exp(-|a-b|^2 / (2 sigma^2))}).
     *
     * @param a one sample {@code [n][d]}
     * @param b another {@code [m][d]}
     * @param sigma the bandwidth
     * @return {@code MMD^2_u}
     */
    public static double mmd2(double[][] a, double[][] b, double sigma) {
        double g = 1 / (2 * sigma * sigma);
        double xx = 0;
        double yy = 0;
        double xy = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a.length; j++) {
                if (i != j) {
                    xx += Math.exp(-g * MlMatrices.squaredDistance(a[i], a[j]));
                }
            }
            for (double[] y : b) {
                xy += Math.exp(-g * MlMatrices.squaredDistance(a[i], y));
            }
        }
        for (int i = 0; i < b.length; i++) {
            for (int j = 0; j < b.length; j++) {
                if (i != j) {
                    yy += Math.exp(-g * MlMatrices.squaredDistance(b[i], b[j]));
                }
            }
        }
        int n = a.length;
        int m = b.length;
        return xx / (n * (n - 1.0)) + yy / (m * (m - 1.0)) - 2 * xy / ((double) n * m);
    }

    /**
     * The sliced 1-Wasserstein distance between two equally sized samples: the mean over
     * random directions of the 1-D distance between the sorted projections.
     *
     * @param a one sample {@code [n][d]}
     * @param b another {@code [n][d]}
     * @param projections the number of directions
     * @param seed the direction seed
     * @return the distance
     */
    public static double slicedWasserstein(double[][] a, double[][] b, int projections, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = a[0].length;
        int n = Math.min(a.length, b.length);
        double total = 0;
        for (int p = 0; p < projections; p++) {
            double[] dir = new double[d];
            double norm = 0;
            for (int i = 0; i < d; i++) {
                dir[i] = r.nextGaussian();
                norm += dir[i] * dir[i];
            }
            norm = Math.sqrt(norm);
            double[] pa = new double[n];
            double[] pb = new double[n];
            for (int i = 0; i < n; i++) {
                pa[i] = MlMatrices.dot(a[i], dir) / norm;
                pb[i] = MlMatrices.dot(b[i], dir) / norm;
            }
            Arrays.sort(pa);
            Arrays.sort(pb);
            for (int i = 0; i < n; i++) {
                total += Math.abs(pa[i] - pb[i]) / n;
            }
        }
        return total / projections;
    }

    /**
     * One coordinate of a sample matrix.
     *
     * @param samples {@code [n][d]}
     * @param j the coordinate
     * @return {@code [n]}
     */
    public static double[] column(double[][] samples, int j) {
        double[] c = new double[samples.length];
        for (int i = 0; i < c.length; i++) {
            c[i] = samples[i][j];
        }
        return c;
    }

    /**
     * The per-coordinate mean of samples.
     *
     * @param samples {@code [n][d]}
     * @return {@code [d]}
     */
    public static double[] mean(double[][] samples) {
        return MlMatrices.columnMeans(samples);
    }
}
