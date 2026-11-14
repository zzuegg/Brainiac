package io.github.zzuegg.brainiac.train;

import java.util.List;

/**
 * The outcome of {@link Trainer#fit}.
 *
 * @param history one entry per epoch that ran
 * @param bestEpoch the epoch whose weights the model ends with (the last one, unless
 *     early stopping restored an earlier one)
 * @param stoppedEarly whether early stopping ended training before the epoch limit
 */
public record TrainingResult(List<EpochResult> history, int bestEpoch, boolean stoppedEarly) {

    /** @return the last epoch that ran */
    public EpochResult last() {
        return history.getLast();
    }

    /** @return the epoch whose weights the model ends with */
    public EpochResult best() {
        return history.get(bestEpoch - 1);
    }

    /** @return the training loss of the last epoch */
    public float finalLoss() {
        return last().loss();
    }
}
