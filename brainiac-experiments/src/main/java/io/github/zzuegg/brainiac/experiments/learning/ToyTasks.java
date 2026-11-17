package io.github.zzuegg.brainiac.experiments.learning;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Small, seeded classification tasks for comparing learning rules against backpropagation
 * in a few hundred milliseconds: two-dimensional spirals and an MNIST-like set of noisy,
 * shifted pixel patterns.
 */
public final class ToyTasks {

    private ToyTasks() {
    }

    /**
     * A labelled data set held as plain arrays.
     *
     * @param x the samples, one row each
     * @param y the class of every sample
     * @param classes the number of classes
     */
    public record Data(float[][] x, int[] y, int classes) {

        /** @return the number of samples */
        public int size() {
            return y.length;
        }

        /** @return the number of features per sample */
        public int features() {
            return x[0].length;
        }

        /** @return the samples as a {@code [size, features]} tensor */
        public Tensor inputs() {
            return Tensor.of(x);
        }

        /** @return one-hot targets, {@code [size, classes]} */
        public Tensor oneHot() {
            float[] v = new float[size() * classes];
            for (int i = 0; i < y.length; i++) {
                v[i * classes + y[i]] = 1;
            }
            return Tensor.of(Shape.of(size(), classes), v);
        }

        /** @return the class ids as a {@code [size]} tensor */
        public Tensor ids() {
            float[] v = new float[y.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = y[i];
            }
            return Tensor.of(Shape.of(v.length), v);
        }

        /**
         * @param from the first sample, inclusive
         * @param to the last sample, exclusive
         * @return those samples
         */
        public Data slice(int from, int to) {
            return new Data(java.util.Arrays.copyOfRange(x, from, to), java.util.Arrays.copyOfRange(y, from, to),
                    classes);
        }

        /**
         * @param keep which classes to keep
         * @return only the samples of those classes (labels unchanged)
         */
        public Data onlyClasses(int... keep) {
            java.util.List<Integer> idx = new java.util.ArrayList<>();
            for (int i = 0; i < y.length; i++) {
                for (int k : keep) {
                    if (y[i] == k) {
                        idx.add(i);
                    }
                }
            }
            float[][] nx = new float[idx.size()][];
            int[] ny = new int[idx.size()];
            for (int i = 0; i < ny.length; i++) {
                nx[i] = x[idx.get(i)];
                ny[i] = y[idx.get(i)];
            }
            return new Data(nx, ny, classes);
        }
    }

    /**
     * Interleaved spirals, the classic task a linear model cannot solve. Samples are
     * shuffled.
     *
     * @param perClass samples per arm
     * @param classes the number of arms
     * @param noise the standard deviation of the angle noise, for example {@code 0.2}
     * @param seed the random seed
     * @return the data, features in about {@code [-1, 1]}
     */
    public static Data spiral(int perClass, int classes, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = perClass * classes;
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int c = 0; c < classes; c++) {
            for (int i = 0; i < perClass; i++) {
                double radius = (double) i / perClass;
                double angle = c * 2 * Math.PI / classes + radius * 4 + r.nextGaussian() * noise;
                x[c * perClass + i] = new float[] {(float) (radius * Math.sin(angle)), (float) (radius * Math.cos(angle))};
                y[c * perClass + i] = c;
            }
        }
        shuffle(x, y, r);
        return new Data(x, y, classes);
    }

    /**
     * An MNIST-like task: every class is a random binary prototype image of {@code side x
     * side} pixels; samples flip pixels with probability {@code noise}, are shifted by up to
     * one pixel, and get Gaussian pixel noise. Values are in about {@code [0, 1]}.
     *
     * @param classes the number of classes
     * @param perClass samples per class
     * @param side the image side, for example 8
     * @param noise the pixel flip probability, for example {@code 0.1}
     * @param seed the random seed (the same seed gives the same prototypes)
     * @return the data, shuffled, {@code side * side} features
     */
    public static Data digits(int classes, int perClass, int side, double noise, long seed) {
        return digits(classes, perClass, side, noise, seed, seed * 31 + 7);
    }

    /**
     * The MNIST-like task with separate seeds for the class prototypes and the samples, so
     * a held-out set has the same classes but new samples.
     *
     * @param classes the number of classes
     * @param perClass samples per class
     * @param side the image side
     * @param noise the pixel flip probability
     * @param seed the prototype seed
     * @param sampleSeed the sample seed
     * @return the data, shuffled
     */
    public static Data digits(int classes, int perClass, int side, double noise, long seed, long sampleSeed) {
        SplittableRandom protoRandom = new SplittableRandom(seed);
        boolean[][] proto = new boolean[classes][side * side];
        for (int c = 0; c < classes; c++) {
            for (int p = 0; p < side * side; p++) {
                proto[c][p] = protoRandom.nextDouble() < 0.35;
            }
        }
        SplittableRandom r = new SplittableRandom(sampleSeed);
        int n = classes * perClass;
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int c = 0; c < classes; c++) {
            for (int i = 0; i < perClass; i++) {
                int dx = r.nextInt(3) - 1;
                int dy = r.nextInt(3) - 1;
                float[] img = new float[side * side];
                for (int py = 0; py < side; py++) {
                    for (int px = 0; px < side; px++) {
                        int sy = Math.floorMod(py - dy, side);
                        int sx = Math.floorMod(px - dx, side);
                        boolean on = proto[c][sy * side + sx] ^ (r.nextDouble() < noise);
                        img[py * side + px] = (on ? 1f : 0f) + (float) (r.nextGaussian() * 0.1);
                    }
                }
                x[c * perClass + i] = img;
                y[c * perClass + i] = c;
            }
        }
        shuffle(x, y, r);
        return new Data(x, y, classes);
    }

    /**
     * The share of rows whose largest score is at the label.
     *
     * @param scores {@code [n, classes]} scores (logits or probabilities)
     * @param labels the true classes
     * @return the accuracy in {@code [0, 1]}
     */
    public static double accuracy(Tensor scores, int[] labels) {
        float[] s = scores.toArray();
        int classes = scores.shape().dim(-1);
        int correct = 0;
        for (int i = 0; i < labels.length; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (s[i * classes + c] > s[i * classes + best]) {
                    best = c;
                }
            }
            if (best == labels[i]) {
                correct++;
            }
        }
        return (double) correct / labels.length;
    }

    private static void shuffle(float[][] x, int[] y, SplittableRandom r) {
        for (int i = y.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            float[] tx = x[i];
            x[i] = x[j];
            x[j] = tx;
            int ty = y[i];
            y[i] = y[j];
            y[j] = ty;
        }
    }
}
