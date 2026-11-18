package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.experimental.ml.RandomForest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * QUANT (Dempster, Schmidt and Webb, DMKD 2024, <a href="https://arxiv.org/abs/2308.00928">arXiv:2308.00928</a>):
 * interval classification reduced to its essentials. The series and three derived
 * representations (first and second differences, the magnitude of the discrete Fourier
 * transform) are cut into fixed dyadic intervals: at depth {@code d} the {@code 2^d} intervals of
 * length {@code L / 2^d} plus the {@code 2^d - 1} intervals shifted by half a length. Each interval
 * of length {@code m} contributes {@code 1 + floor(m / 4)} evenly spaced quantiles, every second
 * one with the interval mean subtracted. An extremely randomised trees classifier
 * ({@link RandomForest.Mode#EXTRA_TREES}, random thresholds, no bootstrap, a tenth of the features
 * per split) reads the features out. No kernels, no learned transform: QUANT is among the most
 * accurate interval methods and among the fastest.
 */
public final class Quant {

    /**
     * Settings.
     *
     * @param maxDepth the deepest interval level, such as 6
     * @param trees extra trees, such as 200
     * @param featureShare features tried per split as a share of all, such as 0.1
     * @param seed the forest seed
     */
    public record Settings(int maxDepth, int trees, double featureShare, long seed) {

        /** @return depth 6, 200 trees, a tenth of the features per split */
        public static Settings defaults() {
            return new Settings(6, 200, 0.1, 1);
        }
    }

    private final Settings settings;
    private final RandomForest forest;

    private Quant(Settings settings, RandomForest forest) {
        this.settings = settings;
        this.forest = forest;
    }

    /**
     * Fits QUANT.
     *
     * @param series training series (one length)
     * @param labels their classes
     * @param classes the number of classes
     * @param settings the settings
     * @return the classifier
     */
    public static Quant fit(float[][] series, int[] labels, int classes, Settings settings) {
        double[][] x = features(series, settings.maxDepth());
        int per = Math.max(1, (int) Math.round(settings.featureShare() * x[0].length));
        RandomForest f = RandomForest.fit(x, labels, classes, new RandomForest.Settings(
                RandomForest.Mode.EXTRA_TREES, settings.trees(), 40, 1, per, settings.seed()));
        return new Quant(settings, f);
    }

    /**
     * The QUANT features of series.
     *
     * @param series the series (one length)
     * @param maxDepth the deepest interval level
     * @return {@code [n][features]}
     */
    public static double[][] features(float[][] series, int maxDepth) {
        double[][] out = new double[series.length][];
        for (int i = 0; i < series.length; i++) {
            List<double[]> reps = representations(series[i]);
            List<Double> f = new ArrayList<>();
            for (double[] r : reps) {
                intervals(r, maxDepth, f);
            }
            out[i] = new double[f.size()];
            for (int j = 0; j < out[i].length; j++) {
                out[i][j] = f.get(j);
            }
        }
        return out;
    }

    /** The series, its first and second differences and its DFT magnitude. */
    static List<double[]> representations(float[] s) {
        int n = s.length;
        double[] x = new double[n];
        for (int t = 0; t < n; t++) {
            x[t] = s[t];
        }
        double[] d1 = new double[n - 1];
        for (int t = 0; t < n - 1; t++) {
            d1[t] = x[t + 1] - x[t];
        }
        double[] d2 = new double[Math.max(1, n - 2)];
        for (int t = 0; t < n - 2; t++) {
            d2[t] = d1[t + 1] - d1[t];
        }
        int m = n / 2 + 1;
        double[] mag = new double[m];
        for (int k = 0; k < m; k++) {
            double re = 0;
            double im = 0;
            for (int t = 0; t < n; t++) {
                double a = 2 * Math.PI * k * t / n;
                re += x[t] * Math.cos(a);
                im -= x[t] * Math.sin(a);
            }
            mag[k] = Math.hypot(re, im);
        }
        return List.of(x, d1, d2, mag);
    }

    private static void intervals(double[] r, int maxDepth, List<Double> out) {
        int n = r.length;
        for (int depth = 0; depth <= maxDepth; depth++) {
            int parts = 1 << depth;
            int m = n / parts;
            if (m < 1) {
                break;
            }
            for (int j = 0; j < parts; j++) {
                quantiles(r, j * m, m, out);
            }
            if (depth > 0) {
                for (int j = 0; j < parts - 1; j++) {
                    quantiles(r, j * m + m / 2, m, out);
                }
            }
        }
    }

    private static void quantiles(double[] r, int start, int m, List<Double> out) {
        double[] v = Arrays.copyOfRange(r, start, Math.min(r.length, start + m));
        Arrays.sort(v);
        double mean = 0;
        for (double a : v) {
            mean += a / v.length;
        }
        int q = 1 + m / 4;
        for (int k = 0; k < q; k++) {
            double p = q == 1 ? 0.5 : k / (double) (q - 1);
            double pos = p * (v.length - 1);
            int lo = (int) Math.floor(pos);
            int hi = Math.min(v.length - 1, lo + 1);
            double val = v[lo] + (pos - lo) * (v[hi] - v[lo]);
            out.add(k % 2 == 1 ? val - mean : val);
        }
    }

    /**
     * Class probabilities.
     *
     * @param series the series
     * @return {@code [n][classes]}
     */
    public double[][] probabilities(float[][] series) {
        double[][] x = features(series, settings.maxDepth());
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = forest.predictProbability(x[i]);
        }
        return out;
    }

    /**
     * Predicts classes.
     *
     * @param series the series
     * @return the classes
     */
    public int[] predict(float[][] series) {
        return WideRidgeClassifier.argmax(probabilities(series));
    }
}
