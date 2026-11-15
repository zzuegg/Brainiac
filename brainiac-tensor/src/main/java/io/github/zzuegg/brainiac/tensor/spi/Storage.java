package io.github.zzuegg.brainiac.tensor.spi;

import io.github.zzuegg.brainiac.tensor.DType;

/**
 * A block of values of one {@link DType} owned by a {@link Backend}, such as a Java array or a GPU
 * buffer. Tensors share storage freely because Brainiac never changes a tensor's values
 * after it is created.
 */
public interface Storage {

    /**
     * Returns the number of floats in this storage.
     *
     * @return the size
     */
    long size();

    /**
     * Returns the number format of the values.
     *
     * @return the type; {@link DType#FLOAT32} unless overridden
     */
    default DType dtype() {
        return DType.FLOAT32;
    }
}
