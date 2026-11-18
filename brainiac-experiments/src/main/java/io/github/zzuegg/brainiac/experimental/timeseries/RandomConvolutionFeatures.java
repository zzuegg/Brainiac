package io.github.zzuegg.brainiac.experimental.timeseries;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Random convolutional kernel features for series classification: ROCKET (Dempster, Petitjean
 * and Webb 2020, <a href="https://arxiv.org/abs/1910.13051">arXiv:1910.13051</a>) and
 * MiniRocket (Dempster, Schmidt and Webb 2021, <a href="https://arxiv.org/abs/2012.08791">arXiv:2012.08791</a>).
 * <ul>
 *   <li>{@link #rocket}: kernels of random length (7, 9 or 11), normal mean-centred weights, a
 *       uniform bias in [-1, 1], an exponentially distributed dilation and random "same"
 *       padding; each gives two features, the maximum and the proportion of positive values
 *       (PPV) of its convolution;</li>
 *   <li>{@link #miniRocket}: the 84 length-9 kernels with weights {@code -1} and three
 *       {@code 2}s, a fixed set of dilations, biases taken from quantiles of each kernel's
 *       convolution on random training series, and PPV features only, almost
 *       deterministic and much faster.</li>
 * </ul>
 * Nothing is trained but the linear read-out ({@link RidgeClassifier}). For several channels
 * (for example band energies) each kernel reads one random channel.
 */
public final class RandomConvolutionFeatures {

    /** One kernel: weights, bias, dilation, padding, the channel it reads, and its features. */
    private record Kernel(double[] weights, double bias, int dilation, boolean padding, int channel, boolean max) {
    }

    private final List<Kernel> kernels;

    private RandomConvolutionFeatures(List<Kernel> kernels) {
        this.kernels = kernels;
    }

    /**
     * ROCKET kernels.
     *
     * @param count the number of kernels (features = 2 count)
     * @param length the series length
     * @param channels the input channels
     * @param seed the seed
     * @return the transform
     */
    public static RandomConvolutionFeatures rocket(int count, int length, int channels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Kernel> ks = new ArrayList<>();
        int[] lengths = {7, 9, 11};
        for (int i = 0; i < count; i++) {
            int len = lengths[r.nextInt(3)];
            double[] w = new double[len];
            double mean = 0;
            for (int j = 0; j < len; j++) {
                w[j] = r.nextGaussian();
                mean += w[j] / len;
            }
            for (int j = 0; j < len; j++) {
                w[j] -= mean;
            }
            double maxExp = Math.log((length - 1) / (double) (len - 1)) / Math.log(2);
            int dilation = (int) Math.floor(Math.pow(2, r.nextDouble(0, Math.max(1e-9, maxExp))));
            ks.add(new Kernel(w, r.nextDouble(-1, 1), Math.max(1, dilation), r.nextBoolean(), r.nextInt(channels),
                    true));
        }
        return new RandomConvolutionFeatures(ks);
    }

    /**
     * MiniRocket kernels, with biases fitted to training series.
     *
     * @param train training series {@code [n][channels][length]}
     * @param featuresWanted about how many features (rounded to a multiple of 84)
     * @param seed the seed for the bias examples and quantiles
     * @return the transform
     */
    public static RandomConvolutionFeatures miniRocket(float[][][] train, int featuresWanted, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int length = train[0][0].length;
        int channels = train[0].length;
        List<double[]> bases = new ArrayList<>();
        for (int a = 0; a < 9; a++) {
            for (int b = a + 1; b < 9; b++) {
                for (int c = b + 1; c < 9; c++) {
                    double[] w = new double[9];
                    Arrays.fill(w, -1);
                    w[a] = 2;
                    w[b] = 2;
                    w[c] = 2;
                    bases.add(w);
                }
            }
        }
        int perKernel = Math.max(1, featuresWanted / bases.size());
        int maxExp = (int) Math.floor(Math.log((length - 1) / 8.0) / Math.log(2));
        int dilations = Math.max(1, Math.min(perKernel, maxExp + 1));
        int quantilesPer = Math.max(1, perKernel / dilations);
        List<Kernel> ks = new ArrayList<>();
        int k = 0;
        for (int d = 0; d < dilations; d++) {
            int dilation = (int) Math.pow(2, dilations == 1 ? 0 : d * maxExp / (double) (dilations - 1));
            for (double[] w : bases) {
                for (int q = 0; q < quantilesPer; q++) {
                    float[][] example = train[r.nextInt(train.length)];
                    int channel = r.nextInt(channels);
                    boolean padding = (k++ & 1) == 0;
                    double[] conv = convolve(example[channel], w, 0, dilation, padding);
                    Arrays.sort(conv);
                    // low-discrepancy quantiles, as in the paper
                    double quantile = ((q + 1) * 0.6180339887498949) % 1.0;
                    double bias = conv[(int) Math.min(conv.length - 1, quantile * conv.length)];
                    ks.add(new Kernel(w, -bias, dilation, padding, channel, false));
                }
            }
        }
        return new RandomConvolutionFeatures(ks);
    }

    private static double[] convolve(float[] x, double[] w, double bias, int dilation, boolean padding) {
        int len = w.length;
        int span = (len - 1) * dilation;
        int pad = padding ? span / 2 : 0;
        int out = x.length + 2 * pad - span;
        double[] y = new double[Math.max(0, out)];
        for (int i = 0; i < y.length; i++) {
            double s = bias;
            int start = i - pad;
            for (int j = 0; j < len; j++) {
                int t = start + j * dilation;
                if (t >= 0 && t < x.length) {
                    s += w[j] * x[t];
                }
            }
            y[i] = s;
        }
        return y;
    }

    /** @return the number of features per series */
    public int features() {
        int n = 0;
        for (Kernel k : kernels) {
            n += k.max() ? 2 : 1;
        }
        return n;
    }

    /**
     * Transforms series.
     *
     * @param x series {@code [n][channels][length]}
     * @return features {@code [n][features()]}
     */
    public double[][] transform(float[][][] x) {
        double[][] out = new double[x.length][features()];
        for (int i = 0; i < x.length; i++) {
            int f = 0;
            for (Kernel k : kernels) {
                double[] y = convolve(x[i][k.channel()], k.weights(), k.bias(), k.dilation(), k.padding());
                int positive = 0;
                double max = Double.NEGATIVE_INFINITY;
                for (double v : y) {
                    positive += v > 0 ? 1 : 0;
                    max = Math.max(max, v);
                }
                out[i][f++] = y.length == 0 ? 0 : positive / (double) y.length;
                if (k.max()) {
                    out[i][f++] = y.length == 0 ? 0 : max;
                }
            }
        }
        return out;
    }

    /**
     * Wraps univariate series as one-channel input.
     *
     * @param x {@code [n][length]}
     * @return {@code [n][1][length]}
     */
    public static float[][][] univariate(float[][] x) {
        float[][][] out = new float[x.length][1][];
        for (int i = 0; i < x.length; i++) {
            out[i][0] = x[i];
        }
        return out;
    }
}
