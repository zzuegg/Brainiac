package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.data.Dataset;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Curriculum learning (Bengio et al. 2009; survey Wang et al. 2021,
 * <a href="https://arxiv.org/abs/2101.10382">arXiv:2101.10382</a>): train on the easiest
 * samples first and let a <em>pacing function</em> widen the pool to the whole data set.
 * Samples are ranked by a difficulty score (from a heuristic, a teacher model's loss, or
 * the sample's length); at step {@code t} only the easiest {@code g(t)} share is drawn from.
 */
public final class Curriculum {

    /** How fast the pool grows from the starting share to everything. */
    public enum Pacing {
        /** {@code start + (1 - start) t / T} */
        LINEAR,
        /** {@code sqrt(start^2 + (1 - start^2) t / T)}: grows fast first */
        ROOT,
        /** jumps to everything at {@code T/2} */
        STEP
    }

    private final int[] order;
    private final double start;
    private final long rampSteps;
    private final Pacing pacing;

    /**
     * @param difficulty a score per sample, larger is harder
     * @param start the share of the data available at step 0, in {@code (0, 1]}
     * @param rampSteps the step from which all data is available
     * @param pacing the pacing function
     */
    public Curriculum(float[] difficulty, double start, long rampSteps, Pacing pacing) {
        Integer[] idx = new Integer[difficulty.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Float.compare(difficulty[a], difficulty[b]));
        this.order = Arrays.stream(idx).mapToInt(Integer::intValue).toArray();
        this.start = start;
        this.rampSteps = rampSteps;
        this.pacing = pacing;
    }

    /**
     * @param step the training step
     * @return the share of the data available
     */
    public double share(long step) {
        double t = Math.min(1, (double) step / Math.max(1, rampSteps));
        return switch (pacing) {
            case LINEAR -> start + (1 - start) * t;
            case ROOT -> Math.sqrt(start * start + (1 - start * start) * t);
            case STEP -> t < 0.5 ? start : 1;
        };
    }

    /**
     * @param step the training step
     * @return how many of the easiest samples are available
     */
    public int available(long step) {
        return Math.max(1, (int) Math.ceil(share(step) * order.length - 1e-9));
    }

    /**
     * Draws a batch from the available pool, uniformly with replacement.
     *
     * @param step the training step
     * @param size the batch size
     * @param random the random source
     * @return sample indices
     */
    public int[] batch(long step, int size, SplittableRandom random) {
        int pool = available(step);
        int[] out = new int[size];
        for (int i = 0; i < size; i++) {
            out[i] = order[random.nextInt(pool)];
        }
        return out;
    }

    /**
     * The available samples as a core data set.
     *
     * @param data the full data set, in the order the difficulties were given
     * @param step the training step
     * @return the easiest samples available at that step
     */
    public Dataset subset(Dataset data, long step) {
        int pool = available(step);
        float[][] x = new float[pool][];
        float[][] y = new float[pool][];
        for (int i = 0; i < pool; i++) {
            x[i] = data.featuresOf(order[i]);
            y[i] = data.targetsOf(order[i]);
        }
        return Dataset.of(x, y);
    }
}
