package io.github.zzuegg.brainiac.experimental.calibrate;

/**
 * A fitted post-hoc recalibration map: turns a model's logits into recalibrated class
 * probabilities. Every map of this package is fitted on held-out validation logits and labels
 * (see {@link Recalibrators#fit}) and then applied unchanged to test logits, so maps can be
 * swapped in one line.
 */
@FunctionalInterface
public interface Recalibrator {

    /**
     * Recalibrated probabilities.
     *
     * @param logits the model's logits {@code [sample][class]}
     * @return probabilities {@code [sample][class]}, each row summing to one
     */
    double[][] apply(double[][] logits);
}
