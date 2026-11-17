package io.github.zzuegg.brainiac.experimental.train;

/**
 * A batch size that changes during training, as a function of the examples seen so far:
 * the batch-size counterpart of the core {@code LearningRateSchedule}. Growing the batch
 * instead of decaying the learning rate (Smith et al. 2018, "Don't Decay the Learning
 * Rate, Increase the Batch Size", <a href="https://arxiv.org/abs/1711.00489">arXiv:1711.00489</a>)
 * lowers the gradient noise the same way, since SGD's noise scale is about
 * {@code lr * N / B}, but takes fewer, larger, more parallel steps for the same number of
 * examples. Later work ties the right batch size to the <em>critical batch size</em>,
 * which grows as the loss falls (McCandlish et al. 2018,
 * <a href="https://arxiv.org/abs/1812.06162">arXiv:1812.06162</a>), hence a ramp.
 */
@FunctionalInterface
public interface BatchSizeSchedule {

    /**
     * Returns the batch size to use next.
     *
     * @param examplesSeen the examples trained on so far
     * @return the batch size, at least 1
     */
    int batchSize(long examplesSeen);

    /**
     * @param size the batch size
     * @return the same batch size throughout
     */
    static BatchSizeSchedule constant(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("size must be positive");
        }
        return seen -> size;
    }

    /**
     * Grows linearly from {@code start} to {@code end} over the first {@code rampExamples}
     * examples, then stays at {@code end}.
     *
     * @param start the first batch size
     * @param end the final batch size
     * @param rampExamples the examples over which it grows
     * @return the schedule
     */
    static BatchSizeSchedule linearRamp(int start, int end, long rampExamples) {
        if (start < 1 || end < start || rampExamples < 1) {
            throw new IllegalArgumentException("need 1 <= start <= end and a positive ramp");
        }
        return seen -> seen >= rampExamples ? end
                : Math.max(1, Math.round(start + (end - start) * (float) seen / rampExamples));
    }

    /**
     * Multiplies the batch size by {@code factor} at each of the given fractions of the
     * budget, the step-wise schedule of Smith et al. (where a step schedule would divide the
     * learning rate by {@code factor}).
     *
     * @param start the first batch size
     * @param factor the growth at each milestone, for example 2 or 5
     * @param budget the total examples of the run
     * @param milestones fractions of the budget, ascending, in {@code (0, 1)}
     * @return the schedule
     */
    static BatchSizeSchedule steps(int start, int factor, long budget, double... milestones) {
        if (start < 1 || factor < 1 || budget < 1) {
            throw new IllegalArgumentException("bad batch-size steps");
        }
        double[] at = milestones.clone();
        return seen -> {
            int size = start;
            for (double m : at) {
                if (seen >= m * budget) {
                    size *= factor;
                }
            }
            return size;
        };
    }

    /**
     * The batch sizes of a whole run: the batches this schedule takes until
     * {@code budget} examples are used (the last batch is cut to fit).
     *
     * @param budget the examples to train on
     * @return the size of every batch, in order
     */
    default int[] plan(long budget) {
        java.util.List<Integer> sizes = new java.util.ArrayList<>();
        long seen = 0;
        while (seen < budget) {
            int b = (int) Math.min(Math.max(1, batchSize(seen)), budget - seen);
            sizes.add(b);
            seen += b;
        }
        return sizes.stream().mapToInt(Integer::intValue).toArray();
    }
}
