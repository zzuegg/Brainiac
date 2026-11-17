package io.github.zzuegg.brainiac.experiments.generative;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Synthetic 8x8 images of four simple shape classes, with pixels in {-1, +1}: horizontal
 * bars, vertical bars, hollow 4x4 squares and 3x3 plus signs, each at every position. The
 * whole data set is small enough to enumerate, so a generated image can be scored exactly
 * by its distance to the nearest real image, and its class read from that neighbour.
 */
public final class ToyImages {
    /** Image side length. */
    public static final int SIDE = 8;
    /** Pixels per image. */
    public static final int PIXELS = SIDE * SIDE;
    /** The number of shape classes. */
    public static final int CLASSES = 4;

    private static final List<List<float[]>> ALL = build();

    private ToyImages() {
    }

    /**
     * Draws a random image of a class.
     *
     * @param label the class, 0..3
     * @param random the random source
     * @return a fresh copy of the image, row-major
     */
    public static float[] sample(int label, RandomGenerator random) {
        List<float[]> c = ALL.get(label);
        return c.get(random.nextInt(c.size())).clone();
    }

    /**
     * Returns every image of a class.
     *
     * @param label the class
     * @return the images; do not modify them
     */
    public static List<float[]> all(int label) {
        return ALL.get(label);
    }

    /**
     * Finds the nearest real image.
     *
     * @param image a (generated) image
     * @return {@code {class, squared distance}} of the nearest real image
     */
    public static double[] nearest(float[] image) {
        double best = Double.MAX_VALUE;
        int label = -1;
        for (int c = 0; c < CLASSES; c++) {
            for (float[] real : ALL.get(c)) {
                double d = Nets.distance2(image, real);
                if (d < best) {
                    best = d;
                    label = c;
                }
            }
        }
        return new double[] {label, best};
    }

    private static List<List<float[]>> build() {
        List<List<float[]>> all = new ArrayList<>();
        for (int c = 0; c < CLASSES; c++) {
            all.add(new ArrayList<>());
        }
        for (int r = 1; r < SIDE - 1; r++) {
            float[] h = blank();
            float[] v = blank();
            for (int k = 1; k < SIDE - 1; k++) {
                h[r * SIDE + k] = 1;
                v[k * SIDE + r] = 1;
            }
            all.get(0).add(h);
            all.get(1).add(v);
        }
        for (int y = 0; y + 4 <= SIDE; y++) {
            for (int x = 0; x + 4 <= SIDE; x++) {
                float[] s = blank();
                for (int k = 0; k < 4; k++) {
                    s[y * SIDE + x + k] = 1;
                    s[(y + 3) * SIDE + x + k] = 1;
                    s[(y + k) * SIDE + x] = 1;
                    s[(y + k) * SIDE + x + 3] = 1;
                }
                all.get(2).add(s);
            }
        }
        for (int y = 1; y < SIDE - 1; y++) {
            for (int x = 1; x < SIDE - 1; x++) {
                float[] p = blank();
                p[y * SIDE + x] = 1;
                p[(y - 1) * SIDE + x] = 1;
                p[(y + 1) * SIDE + x] = 1;
                p[y * SIDE + x - 1] = 1;
                p[y * SIDE + x + 1] = 1;
                all.get(3).add(p);
            }
        }
        return all;
    }

    private static float[] blank() {
        float[] b = new float[PIXELS];
        java.util.Arrays.fill(b, -1);
        return b;
    }
}
