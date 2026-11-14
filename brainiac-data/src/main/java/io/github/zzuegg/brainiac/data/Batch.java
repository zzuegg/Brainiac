package io.github.zzuegg.brainiac.data;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A slice of a dataset, ready to feed to a model.
 *
 * @param features shape {@code [size, featureCount]}
 * @param targets shape {@code [size, targetCount]}
 */
public record Batch(Tensor features, Tensor targets) {

    /** @return the number of samples in this batch */
    public int size() {
        return features.shape().dim(0);
    }
}
