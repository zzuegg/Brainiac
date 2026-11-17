package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;

/**
 * Sequence-length warmup (Li et al. 2022, "The Stability-Efficiency Dilemma: Investigating
 * Sequence Length Warmup for Training GPT Models", <a href="https://arxiv.org/abs/2108.06084">arXiv:2108.06084</a>):
 * train on short sequences first and grow them to the full context, linearly over the
 * warmup steps and rounded to a multiple (for kernels). Early training is then cheaper and
 * more stable (fewer loss spikes at large learning rates and batches); the same schedule
 * is the "progressive" stage of long-context recipes that extend the context in steps.
 */
public final class SequenceLengthWarmup {

    private final int start;
    private final int end;
    private final long steps;
    private final int multiple;

    /**
     * @param start the length at step 0
     * @param end the full length
     * @param steps the warmup steps
     * @param multiple lengths are rounded down to a multiple of this (at least {@code start})
     */
    public SequenceLengthWarmup(int start, int end, long steps, int multiple) {
        if (start < 1 || end < start || steps < 1 || multiple < 1) {
            throw new IllegalArgumentException("bad warmup settings");
        }
        this.start = start;
        this.end = end;
        this.steps = steps;
        this.multiple = multiple;
    }

    /**
     * @param step the training step
     * @return the sequence length to use
     */
    public int length(long step) {
        if (step >= steps) {
            return end;
        }
        double l = start + (end - start) * (double) step / steps;
        int rounded = (int) (Math.floor(l / multiple) * multiple);
        return Math.max(start, Math.min(end, rounded));
    }

    /**
     * Cuts each row of a batch to the current length (keeping the start).
     *
     * @param batch {@code [rows][full length]}
     * @param step the training step
     * @return the cropped rows
     */
    public int[][] crop(int[][] batch, long step) {
        int l = length(step);
        int[][] out = new int[batch.length][];
        for (int i = 0; i < batch.length; i++) {
            out[i] = Arrays.copyOf(batch[i], Math.min(l, batch[i].length));
        }
        return out;
    }

    /**
     * @return the tokens per row summed over the warmup, as a share of training at full
     *     length for as many steps
     */
    public double warmupCost() {
        double sum = 0;
        for (long s = 0; s < steps; s++) {
            sum += length(s);
        }
        return sum / ((double) end * steps);
    }
}
