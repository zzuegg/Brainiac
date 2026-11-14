package io.github.zzuegg.brainiac.tensor.spi;

/**
 * A block of float values owned by a {@link Backend}, such as a Java array or a GPU
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
}
