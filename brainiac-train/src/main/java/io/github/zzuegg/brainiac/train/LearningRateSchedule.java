package io.github.zzuegg.brainiac.train;

/**
 * Chooses the learning rate for each epoch. Starting high and lowering the rate later
 * often trains faster and ends closer to the best weights.
 *
 * <pre>{@code
 * Trainer.of(model)
 *         .optimizer(Optimizer.adam(0.01f))
 *         .schedule(LearningRateSchedule.cosine(200, 0.0001f))
 *         .epochs(200)
 *         .fit(data);
 * }</pre>
 */
@FunctionalInterface
public interface LearningRateSchedule {

    /**
     * Returns the learning rate for an epoch.
     *
     * @param epoch the epoch, starting at 1
     * @param baseRate the optimizer's initial learning rate
     * @return the rate to use, positive
     */
    float rate(int epoch, float baseRate);

    /**
     * Multiplies the rate by {@code factor} every {@code everyEpochs} epochs.
     *
     * @param factor for example {@code 0.5f} to halve
     * @param everyEpochs how often to apply it
     * @return the schedule
     */
    static LearningRateSchedule step(float factor, int everyEpochs) {
        if (everyEpochs < 1) {
            throw new IllegalArgumentException("everyEpochs must be at least 1, got " + everyEpochs);
        }
        return (epoch, base) -> base * (float) Math.pow(factor, (epoch - 1) / everyEpochs);
    }

    /**
     * Multiplies the rate by {@code decay} every epoch.
     *
     * @param decay for example {@code 0.99f}
     * @return the schedule
     */
    static LearningRateSchedule exponential(float decay) {
        return (epoch, base) -> base * (float) Math.pow(decay, epoch - 1);
    }

    /**
     * Lowers the rate along a half cosine from the base rate to {@code minRate} over
     * {@code totalEpochs}, then stays at {@code minRate}.
     *
     * @param totalEpochs the length of the curve
     * @param minRate the final rate
     * @return the schedule
     */
    static LearningRateSchedule cosine(int totalEpochs, float minRate) {
        if (totalEpochs < 1) {
            throw new IllegalArgumentException("totalEpochs must be at least 1, got " + totalEpochs);
        }
        return (epoch, base) -> {
            double t = Math.min(1.0, (epoch - 1) / (double) Math.max(1, totalEpochs - 1));
            return (float) (minRate + (base - minRate) * 0.5 * (1 + Math.cos(Math.PI * t)));
        };
    }

    /**
     * Raises the rate linearly from a small value to the base over the first epochs, then
     * hands over to another schedule. Helps unstable starts.
     *
     * @param epochs the warm-up length
     * @param then the schedule after warm-up, counted from the first epoch
     * @return the schedule
     */
    static LearningRateSchedule warmup(int epochs, LearningRateSchedule then) {
        if (epochs < 1) {
            throw new IllegalArgumentException("epochs must be at least 1, got " + epochs);
        }
        return (epoch, base) -> epoch <= epochs ? base * epoch / epochs : then.rate(epoch, base);
    }
}
