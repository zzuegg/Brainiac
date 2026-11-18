package io.github.zzuegg.brainiac.experimental.tsc;

import java.util.SplittableRandom;

/**
 * HYbrid Dictionary-Rocket Architecture (Hydra; Dempster, Schmidt and Webb, DMKD 2023,
 * <a href="https://arxiv.org/abs/2203.13652">arXiv:2203.13652</a>). Random convolutional kernels
 * (length 9, zero-mean, unit L1 norm) are arranged in {@code groups} of {@code k}. At every time
 * step the kernels of a group compete: the kernel with the largest response "wins" and adds that
 * response to its feature (soft counting), the kernel with the smallest response adds one to
 * another feature (hard counting). The counts say how often each pattern in a group is the
 * closest match, which places Hydra between ROCKET (pooled responses of independent kernels) and
 * dictionary methods (counts of symbolic patterns). Kernels run at dilations {@code 2^0 ... } as
 * in ROCKET, on the series and (half of the groups) on its first difference; for multivariate
 * input every kernel sees the sum of a random subset of channels.
 *
 * <p>Features go through Hydra's sparse scaler ({@code sqrt} of the clamped counts, standardised,
 * zeros kept at zero) and a ridge classifier ({@link WideRidgeClassifier}).
 * {@link #concatenate} appends MiniRocket features for MultiRocket-Hydra.
 */
public final class Hydra {

    /** How the competition is counted. */
    public enum Counting {
        /** The paper's choice: soft counts (summed responses) of winners, hard counts of losers. */
        SOFT_MAX_HARD_MIN,
        /** Hard counts of both winners and losers. */
        HARD
    }

    private static final int KERNEL = 9;

    private final int groups;
    private final int k;
    private final int[] dilations;
    private final boolean difference;
    private final Counting counting;
    private final double[][][][] weights;
    private final int[][][] channels;
    private double[] mean;
    private double[] scale;

    /**
     * Draws the kernels.
     *
     * @param length the series length
     * @param inputChannels the channels per series
     * @param groups groups per dilation, even when {@code difference} (half go to the difference)
     * @param kernelsPerGroup kernels per group, such as 8
     * @param counting how to count
     * @param difference whether half of the groups see the first difference
     * @param seed the seed
     */
    public Hydra(int length, int inputChannels, int groups, int kernelsPerGroup, Counting counting,
            boolean difference, long seed) {
        this.groups = groups;
        this.k = kernelsPerGroup;
        this.difference = difference;
        this.counting = counting;
        int maxExp = (int) Math.floor(Math.log((length - 1) / (double) (KERNEL - 1)) / Math.log(2));
        int nd = Math.max(1, maxExp + 1);
        dilations = new int[nd];
        for (int i = 0; i < nd; i++) {
            dilations[i] = 1 << i;
        }
        SplittableRandom r = new SplittableRandom(seed);
        weights = new double[nd][groups][k][KERNEL];
        channels = new int[nd][groups][];
        for (int d = 0; d < nd; d++) {
            for (int g = 0; g < groups; g++) {
                int count = 1 + r.nextInt(Math.min(2, inputChannels));
                int[] ch = new int[count];
                for (int c = 0; c < count; c++) {
                    ch[c] = r.nextInt(inputChannels);
                }
                channels[d][g] = ch;
                for (int j = 0; j < k; j++) {
                    double[] w = weights[d][g][j];
                    double mu = 0;
                    for (int t = 0; t < KERNEL; t++) {
                        w[t] = r.nextGaussian();
                        mu += w[t] / KERNEL;
                    }
                    double l1 = 0;
                    for (int t = 0; t < KERNEL; t++) {
                        w[t] -= mu;
                        l1 += Math.abs(w[t]);
                    }
                    for (int t = 0; t < KERNEL; t++) {
                        w[t] /= l1;
                    }
                }
            }
        }
    }

    /** @return the number of features per series */
    public int features() {
        return dilations.length * groups * k * 2;
    }

    /**
     * Raw (unscaled) count features.
     *
     * @param x series {@code [n][channels][length]}
     * @return {@code [n][features()]}
     */
    public double[][] counts(float[][][] x) {
        double[][] out = new double[x.length][features()];
        for (int i = 0; i < x.length; i++) {
            int f = 0;
            for (int d = 0; d < dilations.length; d++) {
                int dil = dilations[d];
                for (int g = 0; g < groups; g++) {
                    boolean diff = difference && g >= groups / 2;
                    double[] input = combine(x[i], channels[d][g], diff);
                    int len = input.length;
                    int pad = (KERNEL - 1) * dil / 2;
                    double[] resp = new double[k];
                    for (int t = 0; t < len; t++) {
                        for (int j = 0; j < k; j++) {
                            double s = 0;
                            double[] w = weights[d][g][j];
                            for (int u = 0; u < KERNEL; u++) {
                                int idx = t - pad + u * dil;
                                if (idx >= 0 && idx < len) {
                                    s += w[u] * input[idx];
                                }
                            }
                            resp[j] = s;
                        }
                        int arg = 0;
                        int argMin = 0;
                        for (int j = 1; j < k; j++) {
                            if (resp[j] > resp[arg]) {
                                arg = j;
                            }
                            if (resp[j] < resp[argMin]) {
                                argMin = j;
                            }
                        }
                        out[i][f + arg] += counting == Counting.SOFT_MAX_HARD_MIN ? resp[arg] : 1;
                        out[i][f + k + argMin] += 1;
                    }
                    f += 2 * k;
                }
            }
        }
        return out;
    }

    private static double[] combine(float[][] series, int[] ch, boolean diff) {
        int len = series[0].length - (diff ? 1 : 0);
        double[] out = new double[len];
        for (int c : ch) {
            float[] s = series[c];
            for (int t = 0; t < len; t++) {
                out[t] += diff ? s[t + 1] - s[t] : s[t];
            }
        }
        return out;
    }

    /**
     * Learns the sparse scaler on training features and returns the scaled training features.
     *
     * @param x training series {@code [n][channels][length]}
     * @return scaled features
     */
    public double[][] fitTransform(float[][][] x) {
        double[][] c = counts(x);
        int f = features();
        mean = new double[f];
        scale = new double[f];
        for (double[] row : c) {
            for (int j = 0; j < f; j++) {
                row[j] = Math.sqrt(Math.max(0, row[j]));
                mean[j] += row[j] / c.length;
            }
        }
        for (double[] row : c) {
            for (int j = 0; j < f; j++) {
                scale[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / c.length;
            }
        }
        for (int j = 0; j < f; j++) {
            scale[j] = Math.sqrt(scale[j]) + 1e-8;
        }
        return applyScaler(c);
    }

    /**
     * Scaled features with the training statistics of {@link #fitTransform}.
     *
     * @param x series {@code [n][channels][length]}
     * @return scaled features
     */
    public double[][] transform(float[][][] x) {
        if (mean == null) {
            throw new IllegalStateException("call fitTransform on training series first");
        }
        double[][] c = counts(x);
        for (double[] row : c) {
            for (int j = 0; j < row.length; j++) {
                row[j] = Math.sqrt(Math.max(0, row[j]));
            }
        }
        return applyScaler(c);
    }

    private double[][] applyScaler(double[][] c) {
        for (double[] row : c) {
            for (int j = 0; j < row.length; j++) {
                row[j] = row[j] == 0 ? 0 : (row[j] - mean[j]) / scale[j];
            }
        }
        return c;
    }

    /**
     * Joins feature blocks column-wise (MultiRocket-Hydra: Hydra features and MiniRocket features).
     *
     * @param a one block {@code [n][p]}
     * @param b another {@code [n][q]}
     * @return {@code [n][p + q]}
     */
    public static double[][] concatenate(double[][] a, double[][] b) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = new double[a[i].length + b[i].length];
            System.arraycopy(a[i], 0, out[i], 0, a[i].length);
            System.arraycopy(b[i], 0, out[i], a[i].length, b[i].length);
        }
        return out;
    }
}
