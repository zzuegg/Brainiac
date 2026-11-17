package io.github.zzuegg.brainiac.experiments.generative;

import java.util.random.RandomGenerator;

/**
 * A 2-D toy distribution for generative models: a ring of {@code modes} tight Gaussian
 * blobs, labelled by blob. Simple enough to score exactly (distance to the nearest blob
 * centre, and whether it is the requested blob), hard enough that a model must learn
 * several separated modes.
 */
public final class ToyPoints {
    private final int modes;
    private final double radius;
    private final double spread;

    private ToyPoints(int modes, double radius, double spread) {
        this.modes = modes;
        this.radius = radius;
        this.spread = spread;
    }

    /**
     * Creates the distribution.
     *
     * @param modes the number of blobs
     * @param radius the ring radius
     * @param spread the standard deviation of each blob
     * @return the distribution
     */
    public static ToyPoints ring(int modes, double radius, double spread) {
        return new ToyPoints(modes, radius, spread);
    }

    /** @return the number of blobs (classes) */
    public int modes() {
        return modes;
    }

    /**
     * Returns a blob centre.
     *
     * @param mode the blob
     * @return {@code {x, y}}
     */
    public float[] centre(int mode) {
        double a = 2 * Math.PI * mode / modes;
        return new float[] {(float) (radius * Math.cos(a)), (float) (radius * Math.sin(a))};
    }

    /**
     * Draws a point from a blob.
     *
     * @param mode the blob
     * @param random the random source
     * @return {@code {x, y}}
     */
    public float[] sample(int mode, RandomGenerator random) {
        float[] c = centre(mode);
        c[0] += (float) (spread * random.nextGaussian());
        c[1] += (float) (spread * random.nextGaussian());
        return c;
    }

    /**
     * Finds the nearest blob.
     *
     * @param point a point
     * @return {@code {mode, distance}}
     */
    public double[] nearest(float[] point) {
        double best = Double.MAX_VALUE;
        int mode = -1;
        for (int m = 0; m < modes; m++) {
            double d = Math.sqrt(Nets.distance2(point, centre(m)));
            if (d < best) {
                best = d;
                mode = m;
            }
        }
        return new double[] {mode, best};
    }

    /**
     * Scores samples.
     *
     * @param points generated points
     * @param labels the blob each point was meant to come from, or {@code null}
     * @return {@code {mean distance to nearest centre, share in the requested blob}}
     */
    public double[] score(float[][] points, int[] labels) {
        double dist = 0;
        int hit = 0;
        for (int i = 0; i < points.length; i++) {
            double[] n = nearest(points[i]);
            dist += n[1];
            hit += labels != null && (int) n[0] == labels[i] ? 1 : 0;
        }
        return new double[] {dist / points.length, (double) hit / points.length};
    }
}
