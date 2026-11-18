package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Small, seeded distribution-shift suites on 8x8 images (flattened to 64 features) and toy
 * tabular data, for out-of-distribution detection, open-set recognition, label shift, domain
 * adaptation and domain generalisation experiments on a CPU:
 *
 * <ul>
 *   <li>{@link #ood}: in-distribution classes with near-OOD (held-out classes of the same
 *       family) and far-OOD sets (scattered bright pixels and faint stripes, of about the
 *       in-distribution pixel mass); {@link #auxiliaryOutliers}
 *       are blobs and checkerboards, disjoint from the far-OOD test sets, for outlier
 *       exposure.</li>
 *   <li>{@link #shapes} and {@link #digits} under a {@link Style} (contrast reversal, thick
 *       strokes, a textured background, noise, blur, low contrast) or in one of two
 *       {@link Font}s: source/target pairs for adaptation and several training domains plus a
 *       held-out one for generalisation.</li>
 *   <li>{@link #withPriors}, {@link #dirichlet}, {@link #tweakOne} and {@link #longTailed}:
 *       label-shifted test sets.</li>
 *   <li>{@link #backgrounds}: a Waterbirds-like split where a background texture agrees with
 *       the class with a chosen probability (spurious correlation, four groups).</li>
 *   <li>{@link #twoMoons}: two-moons with a rotated target domain; {@link #shimodaira}: the
 *       classic covariate-shift regression; {@link #covariateClassification}: a 2-D task whose
 *       label noise grows where the target density lives.</li>
 * </ul>
 *
 * <p>The shapes are {@link ShapeImages}' six classes; the digits are drawn from two 3x5 pixel
 * fonts (the first is {@code DigitGlyphs}' font, the second a rounder one), at a random
 * position, brightness 0.7-1.3 on a dark background, with pixel noise. Every generator is a
 * pure function of its seed.
 */
public final class ShiftSuites {

    /** Image side. */
    public static final int SIDE = 8;
    /** Pixels (features) per image. */
    public static final int PIXELS = SIDE * SIDE;

    /** An appearance shift applied to a whole image set. */
    public enum Style {
        /** unchanged */
        PLAIN,
        /** contrast reversal: {@code x -> 1.2 - x} ("colour") */
        INVERTED,
        /** strokes dilated by one pixel */
        THICK,
        /** a diagonal stripe texture added to the background */
        TEXTURED,
        /** strong additive Gaussian noise */
        NOISY,
        /** a 3x3 box blur */
        BLURRED,
        /** contrast halved and brightness raised */
        LOW_CONTRAST
    }

    /** The glyph font of {@link #digits}. */
    public enum Font {
        /** the blocky font of {@code DigitGlyphs} */
        BLOCK,
        /** a rounder font with open corners */
        ROUND
    }

    /** The image family of an {@link #ood} suite. */
    public enum Family {
        /** shapes 0-3 in distribution, 4-5 near OOD */
        SHAPES,
        /** digits 0-5 in distribution, 6-9 near OOD */
        DIGITS
    }

    /** Far-OOD and auxiliary outlier kinds. */
    public enum Outlier {
        /** 5-10 random bright pixels on pixel noise, the mass of a shape (test far OOD) */
        NOISE,
        /** faint stripes at a random angle, frequency and phase (test far OOD) */
        STRIPES,
        /** smooth Gaussian blobs (auxiliary outliers) */
        BLOBS,
        /** checkerboards of random cell size (auxiliary outliers) */
        CHECKERS
    }

    /**
     * Labelled rows.
     *
     * @param x the inputs {@code [n, features]}
     * @param y the labels
     */
    public record Split(Tensor x, int[] y) {

        /** @return the number of rows */
        public int size() {
            return y.length;
        }

        /** @return the labels as a float tensor {@code [n]} */
        public Tensor ids() {
            float[] v = new float[y.length];
            for (int i = 0; i < v.length; i++) {
                v[i] = y[i];
            }
            return Tensor.of(Shape.of(v.length), v);
        }

        /** @return the inputs as images {@code [n, 1, 8, 8]} (only for image splits) */
        public Tensor images() {
            return x.reshape(y.length, 1, SIDE, SIDE);
        }

        /**
         * Rows by index.
         *
         * @param rows the rows to keep, in order
         * @return the subset
         */
        public Split rows(int[] rows) {
            float[] v = x.toArray();
            int d = x.shape().dim(1);
            float[] out = new float[rows.length * d];
            int[] yy = new int[rows.length];
            for (int i = 0; i < rows.length; i++) {
                System.arraycopy(v, rows[i] * d, out, i * d, d);
                yy[i] = y[rows[i]];
            }
            return new Split(Tensor.of(Shape.of(rows.length, d), out), yy);
        }

        /**
         * Rows {@code [from, to)}.
         *
         * @param from the first row
         * @param to one past the last row
         * @return the subset
         */
        public Split range(int from, int to) {
            return new Split(x.slice(0, from, to), Arrays.copyOfRange(y, from, to));
        }

        /**
         * Both splits, this one first.
         *
         * @param other the rows to append
         * @return the concatenation
         */
        public Split concat(Split other) {
            int[] yy = Arrays.copyOf(y, y.length + other.y.length);
            System.arraycopy(other.y, 0, yy, y.length, other.y.length);
            return new Split(Tensor.concat(0, x, other.x), yy);
        }
    }

    /**
     * Rows with a group label (class and spurious attribute).
     *
     * @param split the inputs and class labels
     * @param group {@code 2 * class + attribute}
     */
    public record Grouped(Split split, int[] group) {
    }

    /**
     * An out-of-distribution suite.
     *
     * @param train in-distribution training rows
     * @param validation in-distribution held-out rows (for calibration and fitting scores)
     * @param test in-distribution test rows
     * @param near held-out classes of the same family {@code [n, 64]}
     * @param far scattered pixels and faint stripes {@code [n, 64]}
     * @param classes the number of in-distribution classes
     */
    public record OodSuite(Split train, Split validation, Split test, Tensor near, Tensor far, int classes) {
    }

    private static final String[] BLOCK_FONT = {
        "111101101101111", "010110010010111", "111001111100111", "111001111001111", "101101111001001",
        "111100111001111", "111100111101111", "111001010010010", "111101111101111", "111101111001111"};
    private static final String[] ROUND_FONT = {
        "010101101101010", "010110010010010", "110001010100111", "110001010001110", "001011101111001",
        "111100110001110", "011100110101010", "111001001010100", "010101010101010", "010101011001110"};

    private ShiftSuites() {
    }

    // ------------------------------------------------------------------ image families

    /**
     * Shape images of chosen classes under a style.
     *
     * @param n the number of images
     * @param classes the {@link ShapeImages} classes to draw; label {@code i} is {@code classes[i]}
     * @param style the appearance
     * @param noise the pixel noise, such as 0.2
     * @param seed the seed
     * @return the split
     */
    public static Split shapes(int n, int[] classes, Style style, double noise, long seed) {
        ShapeImages gen = new ShapeImages(SIDE, (float) noise);
        float[] out = new float[n * PIXELS];
        int[] y = new int[n];
        int[] map = new int[ShapeImages.CLASSES];
        Arrays.fill(map, -1);
        for (int i = 0; i < classes.length; i++) {
            map[classes[i]] = i;
        }
        SplittableRandom r = new SplittableRandom(seed);
        int k = 0;
        while (k < n) {
            Tensor[] g = gen.generate(4 * n, r.nextLong());
            float[] img = g[0].toArray();
            float[] ids = g[1].toArray();
            for (int i = 0; i < ids.length && k < n; i++) {
                int c = map[(int) ids[i]];
                if (c >= 0) {
                    System.arraycopy(img, i * PIXELS, out, k * PIXELS, PIXELS);
                    y[k++] = c;
                }
            }
        }
        applyStyle(out, n, style, r);
        return new Split(Tensor.of(Shape.of(n, PIXELS), out), y);
    }

    /**
     * Digit glyphs of chosen classes in a font and style.
     *
     * @param n the number of images
     * @param classes the digits to draw; label {@code i} is {@code classes[i]}
     * @param font the font
     * @param style the appearance
     * @param noise the pixel noise, such as 0.2
     * @param seed the seed
     * @return the split
     */
    public static Split digits(int n, int[] classes, Font font, Style style, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] out = new float[n * PIXELS];
        int[] y = new int[n];
        String[] glyphs = font == Font.BLOCK ? BLOCK_FONT : ROUND_FONT;
        for (int i = 0; i < n; i++) {
            int c = r.nextInt(classes.length);
            y[i] = c;
            String g = glyphs[classes[c]];
            int x0 = r.nextInt(SIDE - 2);
            int y0 = r.nextInt(SIDE - 4);
            float b = 0.7f + 0.6f * r.nextFloat();
            for (int rr = 0; rr < 5; rr++) {
                for (int cc = 0; cc < 3; cc++) {
                    if (g.charAt(rr * 3 + cc) == '1') {
                        out[i * PIXELS + (y0 + rr) * SIDE + x0 + cc] = b;
                    }
                }
            }
            for (int p = 0; p < PIXELS; p++) {
                out[i * PIXELS + p] += (float) (noise * r.nextGaussian());
            }
        }
        applyStyle(out, n, style, r);
        return new Split(Tensor.of(Shape.of(n, PIXELS), out), y);
    }

    /**
     * Applies a style to a set of images.
     *
     * @param x images {@code [n, 64]} (or {@code [n, 1, 8, 8]})
     * @param style the appearance
     * @param seed the seed of the style's randomness
     * @return the styled images, same shape
     */
    public static Tensor apply(Tensor x, Style style, long seed) {
        float[] v = x.toArray();
        applyStyle(v, v.length / PIXELS, style, new SplittableRandom(seed));
        return Tensor.of(x.shape(), v);
    }

    private static void applyStyle(float[] v, int n, Style style, RandomGenerator r) {
        switch (style) {
            case PLAIN -> {
            }
            case INVERTED -> {
                for (int i = 0; i < v.length; i++) {
                    v[i] = 1.2f - v[i];
                }
            }
            case THICK -> {
                for (int i = 0; i < n; i++) {
                    float[] img = Arrays.copyOfRange(v, i * PIXELS, (i + 1) * PIXELS);
                    for (int y = 0; y < SIDE; y++) {
                        for (int x = 0; x < SIDE; x++) {
                            float m = img[y * SIDE + x];
                            if (x > 0) {
                                m = Math.max(m, img[y * SIDE + x - 1]);
                            }
                            if (y > 0) {
                                m = Math.max(m, img[(y - 1) * SIDE + x]);
                            }
                            v[i * PIXELS + y * SIDE + x] = m;
                        }
                    }
                }
            }
            case TEXTURED -> {
                for (int i = 0; i < n; i++) {
                    int phase = r.nextInt(3);
                    for (int y = 0; y < SIDE; y++) {
                        for (int x = 0; x < SIDE; x++) {
                            if ((x + y + phase) % 3 == 0) {
                                v[i * PIXELS + y * SIDE + x] += 0.5f;
                            }
                        }
                    }
                }
            }
            case NOISY -> {
                for (int i = 0; i < v.length; i++) {
                    v[i] += (float) (0.4 * r.nextGaussian());
                }
            }
            case BLURRED -> {
                for (int i = 0; i < n; i++) {
                    float[] img = Arrays.copyOfRange(v, i * PIXELS, (i + 1) * PIXELS);
                    for (int y = 0; y < SIDE; y++) {
                        for (int x = 0; x < SIDE; x++) {
                            float s = 0;
                            int c = 0;
                            for (int dy = -1; dy <= 1; dy++) {
                                for (int dx = -1; dx <= 1; dx++) {
                                    int yy = y + dy;
                                    int xx = x + dx;
                                    if (yy >= 0 && yy < SIDE && xx >= 0 && xx < SIDE) {
                                        s += img[yy * SIDE + xx];
                                        c++;
                                    }
                                }
                            }
                            v[i * PIXELS + y * SIDE + x] = s / c;
                        }
                    }
                }
            }
            case LOW_CONTRAST -> {
                for (int i = 0; i < v.length; i++) {
                    v[i] = 0.4f * v[i] + 0.35f;
                }
            }
        }
    }

    // ------------------------------------------------------------------ OOD

    /**
     * An out-of-distribution suite.
     *
     * @param family shapes (0-3 in, 4-5 near OOD) or digits (0-5 in, 6-9 near OOD)
     * @param train training rows
     * @param test rows of each test set (validation, test, near and far each)
     * @param seed the seed
     * @return the suite
     */
    public static OodSuite ood(Family family, int train, int test, long seed) {
        int[] in = family == Family.SHAPES ? new int[] {0, 1, 2, 3} : new int[] {0, 1, 2, 3, 4, 5};
        int[] out = family == Family.SHAPES ? new int[] {4, 5} : new int[] {6, 7, 8, 9};
        Split all = family == Family.SHAPES ? shapes(train + 2 * test, in, Style.PLAIN, 0.2, seed)
                : digits(train + 2 * test, in, Font.BLOCK, Style.PLAIN, 0.2, seed);
        Split near = family == Family.SHAPES ? shapes(test, out, Style.PLAIN, 0.2, seed + 1)
                : digits(test, out, Font.BLOCK, Style.PLAIN, 0.2, seed + 1);
        Tensor far = Tensor.concat(0, outliers(test / 2, Outlier.NOISE, seed + 2),
                outliers(test - test / 2, Outlier.STRIPES, seed + 3));
        return new OodSuite(all.range(0, train), all.range(train, train + test),
                all.range(train + test, train + 2 * test), near.x(), far, in.length);
    }

    /**
     * Auxiliary outliers for outlier exposure: blobs and checkerboards, disjoint in kind from
     * {@link #ood}'s far-OOD test sets.
     *
     * @param n the number of images
     * @param seed the seed
     * @return images {@code [n, 64]}
     */
    public static Tensor auxiliaryOutliers(int n, long seed) {
        return Tensor.concat(0, outliers(n / 2, Outlier.BLOBS, seed), outliers(n - n / 2, Outlier.CHECKERS, seed + 1));
    }

    /**
     * Synthetic outlier images.
     *
     * @param n the number of images
     * @param kind the kind
     * @param seed the seed
     * @return images {@code [n, 64]}
     */
    public static Tensor outliers(int n, Outlier kind, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] v = new float[n * PIXELS];
        for (int i = 0; i < n; i++) {
            int o = i * PIXELS;
            switch (kind) {
                case NOISE -> {
                    int lit = 5 + r.nextInt(6);
                    for (int k = 0; k < lit; k++) {
                        v[o + r.nextInt(PIXELS)] = (float) (0.7 + 0.6 * r.nextDouble());
                    }
                    for (int p = 0; p < PIXELS; p++) {
                        v[o + p] += (float) (0.2 * r.nextGaussian());
                    }
                }
                case STRIPES -> {
                    double a = r.nextDouble() * Math.PI;
                    double f = 0.8 + r.nextDouble() * 1.5;
                    double ph = r.nextDouble() * 6.28;
                    for (int y = 0; y < SIDE; y++) {
                        for (int x = 0; x < SIDE; x++) {
                            double t = Math.cos(a) * x + Math.sin(a) * y;
                            v[o + y * SIDE + x] = (float) (0.2 + 0.2 * Math.sin(f * t + ph) + 0.2 * r.nextGaussian());
                        }
                    }
                }
                case BLOBS -> {
                    int blobs = 1 + r.nextInt(3);
                    for (int b = 0; b < blobs; b++) {
                        double cy = r.nextDouble() * SIDE;
                        double cx = r.nextDouble() * SIDE;
                        double s = 0.8 + 1.5 * r.nextDouble();
                        double h = 0.6 + 0.6 * r.nextDouble();
                        for (int y = 0; y < SIDE; y++) {
                            for (int x = 0; x < SIDE; x++) {
                                double d2 = (y - cy) * (y - cy) + (x - cx) * (x - cx);
                                v[o + y * SIDE + x] += (float) (h * Math.exp(-d2 / (2 * s * s)));
                            }
                        }
                    }
                    for (int p = 0; p < PIXELS; p++) {
                        v[o + p] += (float) (0.1 * r.nextGaussian());
                    }
                }
                case CHECKERS -> {
                    int cell = 1 + r.nextInt(3);
                    float h = (float) (0.7 + 0.5 * r.nextDouble());
                    for (int y = 0; y < SIDE; y++) {
                        for (int x = 0; x < SIDE; x++) {
                            v[o + y * SIDE + x] = ((x / cell + y / cell) % 2 == 0 ? h : 0)
                                    + (float) (0.1 * r.nextGaussian());
                        }
                    }
                }
            }
        }
        return Tensor.of(Shape.of(n, PIXELS), v);
    }

    // ------------------------------------------------------------------ label shift

    /**
     * Resamples a pool to given class priors (with replacement).
     *
     * @param pool the labelled pool, every class present
     * @param priors the target class probabilities
     * @param n the number of rows
     * @param seed the seed
     * @return the label-shifted rows
     */
    public static Split withPriors(Split pool, double[] priors, int n, long seed) {
        List<List<Integer>> byClass = new ArrayList<>();
        for (int c = 0; c < priors.length; c++) {
            byClass.add(new ArrayList<>());
        }
        for (int i = 0; i < pool.size(); i++) {
            byClass.get(pool.y()[i]).add(i);
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[] rows = new int[n];
        for (int i = 0; i < n; i++) {
            int c = categorical(priors, r);
            List<Integer> list = byClass.get(c);
            rows[i] = list.get(r.nextInt(list.size()));
        }
        return pool.rows(rows);
    }

    /**
     * Class priors drawn from a symmetric Dirichlet.
     *
     * @param alpha the concentration (0.1 = very skewed, 10 = near uniform)
     * @param k the number of classes
     * @param seed the seed
     * @return the priors
     */
    public static double[] dirichlet(double alpha, int k, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] p = new double[k];
        double s = 0;
        for (int i = 0; i < k; i++) {
            p[i] = gamma(alpha, r) + 1e-12;
            s += p[i];
        }
        for (int i = 0; i < k; i++) {
            p[i] /= s;
        }
        return p;
    }

    /**
     * Tweak-one priors: one class has probability {@code rho}, the others share the rest.
     *
     * @param k the number of classes
     * @param cls the tweaked class
     * @param rho its probability
     * @return the priors
     */
    public static double[] tweakOne(int k, int cls, double rho) {
        double[] p = new double[k];
        Arrays.fill(p, (1 - rho) / (k - 1));
        p[cls] = rho;
        return p;
    }

    /**
     * Long-tailed priors decaying geometrically from class 0 to class {@code k - 1}.
     *
     * @param k the number of classes
     * @param imbalance the ratio of the largest to the smallest prior
     * @return the priors
     */
    public static double[] longTailed(int k, double imbalance) {
        double[] p = new double[k];
        double s = 0;
        for (int i = 0; i < k; i++) {
            p[i] = Math.pow(imbalance, -i / (double) (k - 1));
            s += p[i];
        }
        for (int i = 0; i < k; i++) {
            p[i] /= s;
        }
        return p;
    }

    private static int categorical(double[] p, RandomGenerator r) {
        double u = r.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    /** Marsaglia–Tsang gamma sampling, with the boost for shape below 1. */
    private static double gamma(double shape, RandomGenerator r) {
        if (shape < 1) {
            return gamma(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = r.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }

    // ------------------------------------------------------------------ spurious backgrounds

    /**
     * A Waterbirds-like split: squares (class 0) against plus signs (class 1) on a horizontal
     * or vertical stripe background; the background equals the class with probability
     * {@code agreement}.
     *
     * @param n the number of images
     * @param agreement how often background and class agree (0.5 = no correlation)
     * @param seed the seed
     * @return the images, labels and groups {@code 2 * class + background}
     */
    public static Grouped backgrounds(int n, double agreement, long seed) {
        Split s = shapes(n, new int[] {2, 5}, Style.PLAIN, 0.25, seed);
        float[] v = s.x().toArray();
        SplittableRandom r = new SplittableRandom(seed + 17);
        int[] group = new int[n];
        for (int i = 0; i < n; i++) {
            int bg = r.nextDouble() < agreement ? s.y()[i] : 1 - s.y()[i];
            group[i] = 2 * s.y()[i] + bg;
            int phase = r.nextInt(2);
            for (int y = 0; y < SIDE; y++) {
                for (int x = 0; x < SIDE; x++) {
                    int line = bg == 0 ? y : x;
                    if ((line + phase) % 2 == 0) {
                        v[i * PIXELS + y * SIDE + x] += 0.45f;
                    }
                }
            }
        }
        return new Grouped(new Split(Tensor.of(Shape.of(n, PIXELS), v), s.y()), group);
    }

    // ------------------------------------------------------------------ low-dimensional tasks

    /**
     * Two interleaved moons, rotated about their centre.
     *
     * @param n the number of points
     * @param degrees the rotation (0 for the source domain)
     * @param noise the coordinate noise, such as 0.1
     * @param seed the seed
     * @return the points {@code [n, 2]} and labels
     */
    public static Split twoMoons(int n, double degrees, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] v = new float[2 * n];
        int[] y = new int[n];
        double a = Math.toRadians(degrees);
        for (int i = 0; i < n; i++) {
            int c = r.nextInt(2);
            double t = Math.PI * r.nextDouble();
            double px = c == 0 ? Math.cos(t) : 1 - Math.cos(t);
            double py = c == 0 ? Math.sin(t) : 0.5 - Math.sin(t);
            px += noise * r.nextGaussian() - 0.5;
            py += noise * r.nextGaussian() - 0.25;
            v[2 * i] = (float) (Math.cos(a) * px - Math.sin(a) * py);
            v[2 * i + 1] = (float) (Math.sin(a) * px + Math.cos(a) * py);
            y[i] = c;
        }
        return new Split(Tensor.of(Shape.of(n, 2), v), y);
    }

    /**
     * The covariate-shift regression of Shimodaira (2000), in {@code dim} dimensions: inputs
     * {@code N(0.5, 0.5^2)} per coordinate in the source and {@code N(0, 0.3^2)} in the
     * target, and {@code y = sum_j (-x_j + x_j^3) + N(0, 0.3^2)}; a linear model is
     * misspecified, so importance weighting matters.
     *
     * @param n the number of points
     * @param dim the input dimension
     * @param target draw from the target density
     * @param seed the seed
     * @return the inputs {@code [n, dim]} and targets {@code [n]}
     */
    public static Tensor[] shimodaira(int n, int dim, boolean target, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[n * dim];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < dim; j++) {
                double v = target ? 0.3 * r.nextGaussian() : 0.5 + 0.5 * r.nextGaussian();
                x[i * dim + j] = (float) v;
                s += -v + v * v * v;
            }
            y[i] = (float) (s + 0.3 * r.nextGaussian());
        }
        return new Tensor[] {Tensor.of(Shape.of(n, dim), x), Tensor.of(Shape.of(n), y)};
    }

    /**
     * The log density ratio {@code log q(x) / p(x)} of {@link #shimodaira}'s target over its
     * source (the oracle importance weight).
     *
     * @param x one input
     * @return the log ratio
     */
    public static double shimodairaLogRatio(float[] x) {
        double l = 0;
        for (float v : x) {
            l += -v * v / (2 * 0.09) - Math.log(0.3) + (v - 0.5) * (v - 0.5) / (2 * 0.25) + Math.log(0.5);
        }
        return l;
    }

    /**
     * A 2-D classification task under covariate shift: the label is the sign of
     * {@code x1 + 0.3 sin(2 x2)}, flipped with a probability that grows with {@code |x1|}
     * (from 2% to 30%); the source has {@code x ~ N(0, I)}, the target {@code x ~ N((1.5, 0),
     * 0.7^2 I)}, where labels are noisier.
     *
     * @param n the number of points
     * @param target draw from the target density
     * @param seed the seed
     * @return the split ({@code [n, 2]})
     */
    public static Split covariateClassification(int n, boolean target, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[2 * n];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            double a = target ? 1.5 + 0.7 * r.nextGaussian() : r.nextGaussian();
            double b = target ? 0.7 * r.nextGaussian() : r.nextGaussian();
            x[2 * i] = (float) a;
            x[2 * i + 1] = (float) b;
            int c = a + 0.3 * Math.sin(2 * b) > 0 ? 1 : 0;
            double flip = Math.min(0.3, 0.02 + 0.12 * Math.abs(a));
            y[i] = r.nextDouble() < flip ? 1 - c : c;
        }
        return new Split(Tensor.of(Shape.of(n, 2), x), y);
    }

    /**
     * The log density ratio {@code log q(x) / p(x)} of {@link #covariateClassification}.
     *
     * @param x one input {@code (x1, x2)}
     * @return the log ratio
     */
    public static double covariateLogRatio(float[] x) {
        double a = x[0];
        double b = x[1];
        double logQ = -((a - 1.5) * (a - 1.5) + b * b) / (2 * 0.49) - 2 * Math.log(0.7);
        double logP = -(a * a + b * b) / 2;
        return logQ - logP;
    }
}
