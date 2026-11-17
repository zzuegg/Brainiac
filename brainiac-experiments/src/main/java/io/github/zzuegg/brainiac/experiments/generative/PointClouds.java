package io.github.zzuegg.brainiac.experiments.generative;

import java.util.random.RandomGenerator;

/**
 * Classic 2-D density-estimation and transport targets beside the {@link ToyPoints} ring:
 * two interleaved moons and two concentric rings. Both have thin curved supports that a
 * single Gaussian or an affine map cannot fit, so they separate flexible flows from
 * simple ones, and moons to rings is a natural pair for learning a transport between two
 * data distributions.
 */
public final class PointClouds {
    private PointClouds() {
    }

    /**
     * Two interleaved half circles (the scikit-learn "make_moons" set), centred near the origin.
     *
     * @param n how many points
     * @param noise the standard deviation of the Gaussian jitter, such as 0.05
     * @param random the random source
     * @return {@code [n][2]}
     */
    public static float[][] moons(int n, double noise, RandomGenerator random) {
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            double a = Math.PI * random.nextDouble();
            double x;
            double y;
            if (random.nextBoolean()) {
                x = Math.cos(a);
                y = Math.sin(a);
            } else {
                x = 1 - Math.cos(a);
                y = 0.5 - Math.sin(a);
            }
            out[i] = new float[] {(float) (1.2 * (x - 0.5) + noise * random.nextGaussian()),
                (float) (1.2 * (y - 0.25) + noise * random.nextGaussian())};
        }
        return out;
    }

    /**
     * Two concentric circles of radius 0.75 and 1.5, half of the points on each.
     *
     * @param n how many points
     * @param noise the standard deviation of the Gaussian jitter, such as 0.05
     * @param random the random source
     * @return {@code [n][2]}
     */
    public static float[][] rings(int n, double noise, RandomGenerator random) {
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * random.nextDouble();
            double r = random.nextBoolean() ? 0.75 : 1.5;
            out[i] = new float[] {(float) (r * Math.cos(a) + noise * random.nextGaussian()),
                (float) (r * Math.sin(a) + noise * random.nextGaussian())};
        }
        return out;
    }

    /**
     * Points from a {@link ToyPoints} distribution, blobs chosen uniformly.
     *
     * @param points the distribution
     * @param n how many points
     * @param random the random source
     * @return {@code [n][2]}
     */
    public static float[][] blobs(ToyPoints points, int n, RandomGenerator random) {
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            out[i] = points.sample(random.nextInt(points.modes()), random);
        }
        return out;
    }

    /**
     * Draws a batch of random {@link ToyImages}.
     *
     * @param n how many images
     * @param random the random source
     * @return {@code [n][64]} in {-1, +1}
     */
    public static float[][] images(int n, RandomGenerator random) {
        float[][] out = new float[n][];
        for (int i = 0; i < n; i++) {
            out[i] = ToyImages.sample(random.nextInt(ToyImages.CLASSES), random);
        }
        return out;
    }
}
