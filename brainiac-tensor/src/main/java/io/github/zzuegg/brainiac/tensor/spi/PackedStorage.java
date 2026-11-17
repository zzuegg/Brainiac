package io.github.zzuegg.brainiac.tensor.spi;

/**
 * Storage of a matrix kept as quantized levels with scales, in a backend's own form (see
 * {@link PackedMatrix} for the layout). Only {@link Backend#matmulPacked} reads it
 * directly; every other operation sees the values through {@link Backend#unpack}.
 */
public interface PackedStorage extends Storage {

    /** @return the number of rows, the inputs {@code k} of a matrix product */
    int rows();

    /** @return the number of columns, the outputs {@code n} */
    int cols();

    /** @return the bytes this storage takes on its device */
    long bytes();
}
