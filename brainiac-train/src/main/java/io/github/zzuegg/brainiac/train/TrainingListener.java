package io.github.zzuegg.brainiac.train;

/**
 * Called after every epoch, for example to print progress or plot a curve, and optionally
 * after every batch, for example to rewire a {@link io.github.zzuegg.brainiac.nn.Brain}.
 */
@FunctionalInterface
public interface TrainingListener {

    /**
     * Receives the result of a finished epoch.
     *
     * @param result the epoch result
     */
    void onEpoch(EpochResult result);

    /**
     * Called after every optimizer step. Does nothing unless overridden.
     *
     * @param epoch the epoch, from 1
     * @param step the optimizer steps taken in this trainer so far, from 1
     * @param loss the batch loss
     */
    default void onBatch(int epoch, long step, float loss) {
    }
}
