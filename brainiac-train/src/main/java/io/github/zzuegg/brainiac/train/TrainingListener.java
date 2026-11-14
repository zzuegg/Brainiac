package io.github.zzuegg.brainiac.train;

/** Called after every epoch, for example to print progress or plot a curve. */
@FunctionalInterface
public interface TrainingListener {

    /**
     * Receives the result of a finished epoch.
     *
     * @param result the epoch result
     */
    void onEpoch(EpochResult result);
}
