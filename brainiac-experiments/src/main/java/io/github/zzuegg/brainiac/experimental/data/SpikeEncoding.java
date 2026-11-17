package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Turns real-valued features in [0, 1] into spike trains {@code [batch, steps, features]}
 * for spiking networks.
 *
 * <ul>
 * <li>{@link #rate}: each step a feature spikes with probability equal to its value
 *     (Poisson/Bernoulli rate coding). Robust, but needs many steps.</li>
 * <li>{@link #latency}: each feature spikes exactly once, earlier for larger values
 *     (time-to-first-spike coding). Sparse: one spike per feature.</li>
 * </ul>
 */
public final class SpikeEncoding {
    private SpikeEncoding() {
    }

    /**
     * Rate coding.
     *
     * @param x {@code [batch][features]} values in [0, 1]
     * @param steps the number of time steps
     * @param random the random source
     * @return {@code [batch, steps, features]} 0/1 spikes
     */
    public static Tensor rate(float[][] x, int steps, RandomGenerator random) {
        int b = x.length;
        int f = x[0].length;
        float[] s = new float[b * steps * f];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < steps; t++) {
                for (int j = 0; j < f; j++) {
                    s[(i * steps + t) * f + j] = random.nextDouble() < x[i][j] ? 1 : 0;
                }
            }
        }
        return Tensor.of(Shape.of(b, steps, f), s);
    }

    /**
     * Latency coding: value 1 spikes at step 0, value near 0 at the last step, value 0 never.
     *
     * @param x {@code [batch][features]} values in [0, 1]
     * @param steps the number of time steps
     * @return {@code [batch, steps, features]} 0/1 spikes
     */
    public static Tensor latency(float[][] x, int steps) {
        int b = x.length;
        int f = x[0].length;
        float[] s = new float[b * steps * f];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < f; j++) {
                if (x[i][j] > 0) {
                    int t = Math.min(steps - 1, Math.round((1 - x[i][j]) * (steps - 1)));
                    s[(i * steps + t) * f + j] = 1;
                }
            }
        }
        return Tensor.of(Shape.of(b, steps, f), s);
    }
}
