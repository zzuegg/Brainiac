package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A synthetic image dataset for fast vision and self-supervised experiments on a CPU: small
 * grayscale images {@code [1, side, side]} showing one of six shapes (horizontal bar,
 * vertical bar, square, diagonal, anti-diagonal, plus sign) at a random position with random
 * brightness and pixel noise. The class is the shape, so a good representation must be
 * invariant to position and brightness; {@link #augment} produces the random views that
 * self-supervised methods learn that invariance from (random wrap-around shifts, brightness
 * and noise).
 */
public final class ShapeImages {

    /** The number of shape classes. */
    public static final int CLASSES = 6;

    private final int side;
    private final float noise;

    /**
     * Creates the generator.
     *
     * @param side the image side, at least 5
     * @param noise the standard deviation of the pixel noise, such as 0.1
     */
    public ShapeImages(int side, float noise) {
        if (side < 5) {
            throw new IllegalArgumentException("side must be at least 5, got " + side);
        }
        this.side = side;
        this.noise = noise;
    }

    /** @return the image side */
    public int side() {
        return side;
    }

    /**
     * Generates labelled images.
     *
     * @param n the number of images
     * @param seed the random seed
     * @return {@code {images [n, 1, side, side], ids [n]}}
     */
    public Tensor[] generate(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int s = side;
        float[] img = new float[n * s * s];
        float[] ids = new float[n];
        for (int i = 0; i < n; i++) {
            int c = r.nextInt(CLASSES);
            ids[i] = c;
            float b = 0.7f + 0.6f * r.nextFloat();
            int y = r.nextInt(s - 3);
            int x = r.nextInt(s - 3);
            int o = i * s * s;
            for (int k = 0; k < 3; k++) {
                switch (c) {
                    case 0 -> img[o + (y + 1) * s + x + k] = b;
                    case 1 -> img[o + (y + k) * s + x + 1] = b;
                    case 2 -> {
                        img[o + y * s + x + k] = b;
                        img[o + (y + 2) * s + x + k] = b;
                        img[o + (y + k) * s + x] = b;
                        img[o + (y + k) * s + x + 2] = b;
                    }
                    case 3 -> img[o + (y + k) * s + x + k] = b;
                    case 4 -> img[o + (y + k) * s + x + 2 - k] = b;
                    default -> {
                        img[o + (y + 1) * s + x + k] = b;
                        img[o + (y + k) * s + x + 1] = b;
                    }
                }
            }
            for (int p = 0; p < s * s; p++) {
                img[o + p] += (float) (noise * r.nextGaussian());
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 1, s, s), img), Tensor.of(Shape.of(n), ids)};
    }

    /**
     * Generates a core {@link Dataset} of flattened images with one-hot targets.
     *
     * @param n the number of images
     * @param seed the random seed
     * @return the dataset, {@code side * side} features and {@link #CLASSES} targets
     */
    public Dataset dataset(int n, long seed) {
        Tensor[] g = generate(n, seed);
        float[] ids = g[1].toArray();
        float[] targets = new float[n * CLASSES];
        for (int i = 0; i < n; i++) {
            targets[i * CLASSES + (int) ids[i]] = 1f;
        }
        return Dataset.ofRows(g[0].toArray(), side * side, targets, CLASSES);
    }

    /**
     * A random view of each image: a wrap-around shift of up to {@code maxShift} pixels per
     * axis, a brightness factor in {@code [0.6, 1.4]} and extra noise.
     *
     * @param images {@code [n, 1, side, side]}
     * @param maxShift the largest shift
     * @param random the random source
     * @return the augmented images, same shape
     */
    public Tensor augment(Tensor images, int maxShift, RandomGenerator random) {
        int n = images.shape().dim(0);
        int s = side;
        float[] in = images.toArray();
        float[] out = new float[in.length];
        for (int i = 0; i < n; i++) {
            int dy = random.nextInt(2 * maxShift + 1) - maxShift;
            int dx = random.nextInt(2 * maxShift + 1) - maxShift;
            float b = 0.6f + 0.8f * random.nextFloat();
            int o = i * s * s;
            for (int y = 0; y < s; y++) {
                for (int x = 0; x < s; x++) {
                    int sy = Math.floorMod(y - dy, s);
                    int sx = Math.floorMod(x - dx, s);
                    out[o + y * s + x] = in[o + sy * s + sx] * b + (float) (0.5 * noise * random.nextGaussian());
                }
            }
        }
        return Tensor.of(images.shape(), out, images.device());
    }
}
