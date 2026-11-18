package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Synthetic univariate series-classification tasks:
 * <ul>
 *   <li>{@link #cylinderBellFunnel}: the classic cylinder-bell-funnel task (Saito 1994,
 *       "Local feature extraction and its applications using a library of bases"): a plateau,
 *       a rising ramp or a falling ramp over a random interval, plus noise;</li>
 *   <li>{@link #shapelets}: noise with one of three short shapes (a spike pair, a dip or a
 *       square bump) planted at a random position, so only a local, shift-invariant feature
 *       separates the classes;</li>
 *   <li>{@link #frequencies}: sinusoids whose class is the frequency band, at random phase and
 *       amplitude.</li>
 * </ul>
 */
public final class SeriesClassificationTasks {

    private SeriesClassificationTasks() {
    }

    /**
     * A labelled set of series.
     *
     * @param series {@code [n][length]}
     * @param labels the classes
     * @param classes the number of classes
     */
    public record Task(float[][] series, int[] labels, int classes) {
        /** @return the number of series */
        public int size() {
            return labels.length;
        }

        /** @return the series length */
        public int length() {
            return series[0].length;
        }

        /**
         * The first {@code n} series.
         *
         * @param n how many
         * @return a smaller task
         */
        public Task head(int n) {
            float[][] s = java.util.Arrays.copyOf(series, n);
            return new Task(s, java.util.Arrays.copyOf(labels, n), classes);
        }
    }

    /**
     * The cylinder-bell-funnel task (three classes, length 128 in the original).
     *
     * @param n series
     * @param length steps
     * @param seed the seed
     * @return the task
     */
    public static Task cylinderBellFunnel(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            int a = length / 8 + r.nextInt(length / 8);
            int b = a + length / 4 + r.nextInt(length / 2);
            b = Math.min(b, length - 1);
            double eta = 6 + r.nextGaussian();
            for (int t = 0; t < length; t++) {
                double shape = 0;
                if (t >= a && t <= b) {
                    double frac = (t - a) / (double) (b - a);
                    shape = switch (c) {
                        case 0 -> 1;
                        case 1 -> frac;
                        default -> 1 - frac;
                    };
                }
                x[i][t] = (float) (eta * shape + r.nextGaussian());
            }
        }
        return new Task(x, y, 3);
    }

    /**
     * Shapelet classes: noise plus one of three planted 12-step shapes at a random position.
     *
     * @param n series
     * @param length steps, at least 24
     * @param seed the seed
     * @return the task
     */
    public static Task shapelets(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % 3;
            y[i] = c;
            for (int t = 0; t < length; t++) {
                x[i][t] = (float) (0.5 * r.nextGaussian());
            }
            int at = r.nextInt(length - 12);
            for (int k = 0; k < 12; k++) {
                double v = switch (c) {
                    case 0 -> k == 3 || k == 8 ? 3 : 0;
                    case 1 -> -2.0 * Math.sin(Math.PI * k / 11);
                    default -> k >= 2 && k < 10 ? 1.5 : 0;
                };
                x[i][at + k] += (float) v;
            }
        }
        return new Task(x, y, 3);
    }

    /**
     * Frequency-band classes: a sinusoid with {@code 2 + 3 c} to {@code 3 + 3 c} cycles per
     * series (class {@code c} of {@code classes}), random phase and amplitude, plus noise.
     *
     * @param n series
     * @param length steps
     * @param classes the number of bands
     * @param seed the seed
     * @return the task
     */
    public static Task frequencies(int n, int length, int classes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][length];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % classes;
            y[i] = c;
            double cycles = 2 + 3 * c + r.nextDouble();
            double phase = r.nextDouble(0, 2 * Math.PI);
            double amp = 0.5 + r.nextDouble();
            for (int t = 0; t < length; t++) {
                x[i][t] = (float) (amp * Math.sin(2 * Math.PI * cycles * t / length + phase) + 0.5 * r.nextGaussian());
            }
        }
        return new Task(x, y, classes);
    }
}
