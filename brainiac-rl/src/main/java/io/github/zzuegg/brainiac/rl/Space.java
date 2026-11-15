package io.github.zzuegg.brainiac.rl;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * The set of valid observations or actions: a number of choices ({@link Discrete}) or a
 * vector of values within bounds ({@link Box}).
 */
public sealed interface Space permits Space.Discrete, Space.Box {

    /** @return how many values describe one element: 1 for discrete, the vector length for a box */
    int size();

    /**
     * Draws a uniformly random element.
     *
     * @param random the random source
     * @return the element
     */
    float[] sample(SplittableRandom random);

    /**
     * @param n the number of choices
     * @return the space {@code {0, ..., n - 1}}
     */
    static Discrete discrete(int n) {
        return new Discrete(n);
    }

    /**
     * @param low the lowest value of every component
     * @param high the highest value of every component
     * @return the box
     */
    static Box box(float[] low, float[] high) {
        return new Box(low, high);
    }

    /**
     * @param size the vector length
     * @param low the lowest value of every component
     * @param high the highest value of every component
     * @return a box with the same bounds for every component
     */
    static Box box(int size, float low, float high) {
        float[] lo = new float[size];
        float[] hi = new float[size];
        Arrays.fill(lo, low);
        Arrays.fill(hi, high);
        return new Box(lo, hi);
    }

    /**
     * A choice of one of {@code n} options, {@code 0} to {@code n - 1}; an element is a
     * one-value array holding the index.
     *
     * @param n the number of choices
     */
    record Discrete(int n) implements Space {
        /** Checks the count. */
        public Discrete {
            if (n < 1) {
                throw new IllegalArgumentException("a discrete space needs at least one choice, got " + n);
            }
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public float[] sample(SplittableRandom random) {
            return new float[] {random.nextInt(n)};
        }

        @Override
        public String toString() {
            return "Discrete(" + n + ")";
        }
    }

    /**
     * A vector of values, each within its bounds (which may be infinite).
     *
     * @param low the lowest value of every component
     * @param high the highest value of every component
     */
    record Box(float[] low, float[] high) implements Space {
        /** Checks and copies the bounds. */
        public Box {
            if (low.length != high.length || low.length == 0) {
                throw new IllegalArgumentException("bounds must have the same, positive length");
            }
            for (int i = 0; i < low.length; i++) {
                if (!(low[i] <= high[i])) {
                    throw new IllegalArgumentException("low " + low[i] + " above high " + high[i] + " at " + i);
                }
            }
            low = low.clone();
            high = high.clone();
        }

        @Override
        public float[] low() {
            return low.clone();
        }

        @Override
        public float[] high() {
            return high.clone();
        }

        @Override
        public int size() {
            return low.length;
        }

        @Override
        public float[] sample(SplittableRandom random) {
            float[] x = new float[low.length];
            for (int i = 0; i < x.length; i++) {
                double lo = Double.isInfinite(low[i]) ? -1 : low[i];
                double hi = Double.isInfinite(high[i]) ? 1 : high[i];
                x[i] = (float) random.nextDouble(lo, Math.nextUp(hi));
            }
            return x;
        }

        /**
         * Clips a vector into the box.
         *
         * @param x the vector
         * @return a clipped copy
         */
        public float[] clip(float[] x) {
            float[] y = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                y[i] = Math.max(low[i], Math.min(high[i], x[i]));
            }
            return y;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Box b && Arrays.equals(low, b.low) && Arrays.equals(high, b.high);
        }

        @Override
        public int hashCode() {
            return 31 * Arrays.hashCode(low) + Arrays.hashCode(high);
        }

        @Override
        public String toString() {
            return "Box(" + Arrays.toString(low) + ", " + Arrays.toString(high) + ")";
        }
    }
}
