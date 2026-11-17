package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Synthetic recommendation and tabular data with known structure:
 * <ul>
 *   <li>{@link #preferences}: users and items with hidden taste vectors, observed through
 *       noisy feature vectors; a user's relevant items are those with the highest hidden
 *       affinity (retrieval);</li>
 *   <li>{@link #clicks}: click-through data over categorical fields, one-hot encoded,
 *       where the click probability depends on a hidden pairwise interaction between
 *       two fields (what factorization machines model);</li>
 *   <li>{@link #tabular}: numeric features whose label depends on products of features
 *       and on a few informative columns among distractors (feature interactions and
 *       feature selection).</li>
 * </ul>
 */
public final class SyntheticRecsys {

    private SyntheticRecsys() {
    }

    /**
     * Users, items and their relevant items.
     *
     * @param userFeatures {@code [users, features]}
     * @param itemFeatures {@code [items, features]}
     * @param relevant {@code [users][k]} each user's most relevant items, best first
     */
    public record Preferences(Tensor userFeatures, Tensor itemFeatures, int[][] relevant) {
    }

    /**
     * A labelled table.
     *
     * @param features {@code [rows, columns]}
     * @param labels {@code [rows, 1]}, 0 or 1
     */
    public record Table(Tensor features, Tensor labels) {
        /** @return the number of rows */
        public int rows() {
            return features.shape().dim(0);
        }

        /**
         * @param from the first row
         * @param to one past the last row
         * @return those rows, for train/test splits (tables drawn with one seed share their
         *     hidden structure, so split one table rather than drawing two)
         */
        public Table rows(int from, int to) {
            return new Table(features.slice(0, from, to), labels.slice(0, from, to));
        }
    }

    /**
     * @param users the number of users
     * @param items the number of items
     * @param latent the hidden taste dimension
     * @param features the observed feature dimension
     * @param noise the observation noise
     * @param k relevant items per user
     * @param seed the seed
     * @return the data
     */
    public static Preferences preferences(int users, int items, int latent, int features, double noise, int k,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] u = gaussian(users, latent, r);
        float[][] v = gaussian(items, latent, r);
        float[][] mixU = gaussian(latent, features, r);
        float[][] mixV = gaussian(latent, features, r);
        int[][] relevant = new int[users][];
        for (int i = 0; i < users; i++) {
            final float[] score = new float[items];
            for (int j = 0; j < items; j++) {
                score[j] = dot(u[i], v[j]);
            }
            relevant[i] = java.util.stream.IntStream.range(0, items).boxed()
                    .sorted((a, b) -> Float.compare(score[b], score[a])).limit(k).mapToInt(Integer::intValue)
                    .toArray();
        }
        return new Preferences(observe(u, mixU, noise, r), observe(v, mixV, noise, r), relevant);
    }

    /**
     * Click-through data over categorical fields.
     *
     * @param rows samples
     * @param fields categorical fields
     * @param categories categories per field
     * @param seed the seed
     * @return one-hot features ({@code fields * categories} columns) and clicks
     */
    public static Table clicks(int rows, int fields, int categories, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] pair = gaussian(categories, categories, r);
        float[] bias = new float[categories];
        for (int c = 0; c < categories; c++) {
            bias[c] = (float) (0.3 * r.nextGaussian());
        }
        int width = fields * categories;
        float[] x = new float[rows * width];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            int[] cat = new int[fields];
            for (int f = 0; f < fields; f++) {
                cat[f] = r.nextInt(categories);
                x[i * width + f * categories + cat[f]] = 1;
            }
            double logit = 2 * pair[cat[0]][cat[1]] + bias[cat[fields - 1]];
            y[i] = r.nextDouble() < 1 / (1 + Math.exp(-logit)) ? 1 : 0;
        }
        return new Table(Tensor.of(Shape.of(rows, width), x), Tensor.of(Shape.of(rows, 1), y));
    }

    /**
     * Numeric table: the label is {@code x0 * x1 + x2 * x3 - 0.5 x4 > 0} (with 2% label
     * noise); every other column is a distractor.
     *
     * @param rows samples
     * @param columns columns, at least 5
     * @param seed the seed
     * @return the table
     */
    public static Table tabular(int rows, int columns, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[rows * columns];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < columns; c++) {
                x[i * columns + c] = (float) r.nextGaussian();
            }
            int o = i * columns;
            double s = x[o] * x[o + 1] + x[o + 2] * x[o + 3] - 0.5 * x[o + 4];
            boolean label = s > 0;
            if (r.nextDouble() < 0.02) {
                label = !label;
            }
            y[i] = label ? 1 : 0;
        }
        return new Table(Tensor.of(Shape.of(rows, columns), x), Tensor.of(Shape.of(rows, 1), y));
    }

    private static float[][] gaussian(int n, int d, SplittableRandom r) {
        float[][] m = new float[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = (float) r.nextGaussian();
            }
        }
        return m;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static Tensor observe(float[][] latent, float[][] mix, double noise, SplittableRandom r) {
        int n = latent.length;
        int f = mix[0].length;
        float[] out = new float[n * f];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < f; j++) {
                float s = 0;
                for (int k = 0; k < latent[i].length; k++) {
                    s += latent[i][k] * mix[k][j];
                }
                out[i * f + j] = (float) (s / Math.sqrt(latent[i].length) + noise * r.nextGaussian());
            }
        }
        return Tensor.of(Shape.of(n, f), out);
    }
}
