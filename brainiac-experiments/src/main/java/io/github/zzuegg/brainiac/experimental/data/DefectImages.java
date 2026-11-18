package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic industrial inspection images: five "product" categories (a disc, a square frame, a
 * cross, a striped texture and a pair of blocks) with small pose, size and brightness jitter and
 * pixel noise, and four defect types with pixel masks:
 *
 * <ul>
 *   <li>{@link Defect#SCRATCH}: a thin bright line across the object;
 *   <li>{@link Defect#HOLE}: a small disc of background inside the object;
 *   <li>{@link Defect#STAIN}: a soft low-contrast blob;
 *   <li>{@link Defect#MISPLACED}: a block of the object cut out and pasted a few pixels away
 *       (the moved block of the pair; a phase break in the texture).
 * </ul>
 *
 * <p>Masks mark the pixels the defect changed by more than 0.1 before noise.
 */
public final class DefectImages {

    /** The number of categories. */
    public static final int CATEGORIES = 5;

    /** Defect types. */
    public enum Defect {
        /** a thin bright line */
        SCRATCH,
        /** a missing disc */
        HOLE,
        /** a soft blob */
        STAIN,
        /** a moved part */
        MISPLACED
    }

    /**
     * An image.
     *
     * @param pixels {@code side * side} values, row-major
     * @param mask the defect pixels
     * @param category the product category
     * @param defect the defect, or {@code null} for a normal image
     */
    public record Sample(float[] pixels, boolean[] mask, int category, Defect defect) {

        /** @return whether the image is anomalous */
        public boolean anomalous() {
            return defect != null;
        }
    }

    private final int side;
    private final float noise;

    /**
     * Creates the generator.
     *
     * @param side the image side, at least 16
     * @param noise the pixel noise standard deviation
     */
    public DefectImages(int side, float noise) {
        if (side < 16) {
            throw new IllegalArgumentException("side must be at least 16");
        }
        this.side = side;
        this.noise = noise;
    }

    /** @return the image side */
    public int side() {
        return side;
    }

    private float[] product(int category, SplittableRandom r) {
        int s = side;
        float[] img = new float[s * s];
        float b = 0.7f + 0.2f * r.nextFloat();
        double cx = s / 2.0 - 0.5 + r.nextInt(3) - 1;
        double cy = s / 2.0 - 0.5 + r.nextInt(3) - 1;
        double scale = s / 24.0 * (0.92 + 0.16 * r.nextDouble());
        for (int y = 0; y < s; y++) {
            for (int x = 0; x < s; x++) {
                double dx = (x - cx) / scale;
                double dy = (y - cy) / scale;
                boolean on = switch (category) {
                    case 0 -> dx * dx + dy * dy <= 49;
                    case 1 -> Math.max(Math.abs(dx), Math.abs(dy)) <= 7 && Math.max(Math.abs(dx), Math.abs(dy)) > 5;
                    case 2 -> (Math.abs(dx) <= 1.5 && Math.abs(dy) <= 8) || (Math.abs(dy) <= 1.5 && Math.abs(dx) <= 8);
                    case 3 -> Math.floorMod((int) Math.floor(x + cx), 4) < 2 && Math.abs(dy) <= 9 && Math.abs(dx) <= 9;
                    default -> (Math.abs(dx + 4) <= 2.5 && Math.abs(dy + 4) <= 2.5)
                            || (Math.abs(dx - 4) <= 2.5 && Math.abs(dy - 4) <= 2.5);
                };
                if (on) {
                    img[y * s + x] = b;
                }
            }
        }
        return img;
    }

    private int[] objectPixel(float[] img, SplittableRandom r) {
        for (int t = 0; t < 1000; t++) {
            int p = r.nextInt(img.length);
            if (img[p] > 0.3f) {
                return new int[] {p % side, p / side};
            }
        }
        return new int[] {side / 2, side / 2};
    }

    private void defect(float[] img, Defect d, SplittableRandom r) {
        int s = side;
        int[] c = objectPixel(img, r);
        switch (d) {
            case SCRATCH -> {
                double a = r.nextDouble() * Math.PI;
                int len = 8 + r.nextInt(7);
                for (int t = -len / 2; t <= len / 2; t++) {
                    int x = (int) Math.round(c[0] + t * Math.cos(a));
                    int y = (int) Math.round(c[1] + t * Math.sin(a));
                    if (x >= 0 && y >= 0 && x < s && y < s) {
                        img[y * s + x] = 1.25f;
                    }
                }
            }
            case HOLE -> {
                for (int y = c[1] - 2; y <= c[1] + 2; y++) {
                    for (int x = c[0] - 2; x <= c[0] + 2; x++) {
                        if (x >= 0 && y >= 0 && x < s && y < s && (x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1]) <= 5) {
                            img[y * s + x] = 0;
                        }
                    }
                }
            }
            case STAIN -> {
                float amp = (r.nextBoolean() ? 1 : -1) * 0.35f;
                for (int y = 0; y < s; y++) {
                    for (int x = 0; x < s; x++) {
                        double q = ((x - c[0]) * (x - c[0]) + (y - c[1]) * (y - c[1])) / 6.0;
                        img[y * s + x] += (float) (amp * Math.exp(-q));
                    }
                }
            }
            default -> {
                int bs = 5;
                int x0 = Math.max(0, Math.min(s - bs, c[0] - 2));
                int y0 = Math.max(0, Math.min(s - bs, c[1] - 2));
                int dx = (r.nextBoolean() ? 1 : -1) * (3 + r.nextInt(2));
                int dy = (r.nextBoolean() ? 1 : -1) * r.nextInt(3);
                float[] block = new float[bs * bs];
                for (int y = 0; y < bs; y++) {
                    for (int x = 0; x < bs; x++) {
                        block[y * bs + x] = img[(y0 + y) * s + x0 + x];
                        img[(y0 + y) * s + x0 + x] = 0;
                    }
                }
                for (int y = 0; y < bs; y++) {
                    for (int x = 0; x < bs; x++) {
                        int tx = x0 + x + dx;
                        int ty = y0 + y + dy;
                        if (tx >= 0 && ty >= 0 && tx < s && ty < s && block[y * bs + x] > 0) {
                            img[ty * s + tx] = block[y * bs + x];
                        }
                    }
                }
            }
        }
    }

    private Sample sample(int category, Defect d, SplittableRandom r) {
        while (true) {
            float[] clean = product(category, r);
            float[] img = clean.clone();
            boolean[] mask = new boolean[img.length];
            int changed = 0;
            if (d != null) {
                defect(img, d, r);
                for (int p = 0; p < img.length; p++) {
                    mask[p] = Math.abs(img[p] - clean[p]) > 0.1f;
                    changed += mask[p] ? 1 : 0;
                }
                if (changed < 3) {
                    continue;
                }
            }
            for (int p = 0; p < img.length; p++) {
                img[p] += (float) (noise * r.nextGaussian());
            }
            return new Sample(img, mask, category, d);
        }
    }

    /**
     * Normal training images.
     *
     * @param category the category
     * @param n how many
     * @param seed the seed
     * @return the images
     */
    public List<Sample> normal(int category, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed * 31 + category);
        List<Sample> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(sample(category, null, r));
        }
        return out;
    }

    /**
     * A test set: normal images and every defect type.
     *
     * @param category the category
     * @param normals normal images
     * @param perDefect images per defect type
     * @param seed the seed
     * @return the images
     */
    public List<Sample> test(int category, int normals, int perDefect, long seed) {
        SplittableRandom r = new SplittableRandom(seed * 37 + category + 1000);
        List<Sample> out = new ArrayList<>();
        for (int i = 0; i < normals; i++) {
            out.add(sample(category, null, r));
        }
        for (Defect d : Defect.values()) {
            for (int i = 0; i < perDefect; i++) {
                out.add(sample(category, d, r));
            }
        }
        return out;
    }
}
