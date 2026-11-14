package io.github.zzuegg.brainiac.tensor.spi;

/**
 * A compute backend that owns tensor storage and runs every tensor operation.
 *
 * <p>All math in Brainiac, including the backward pass of autograd and optimizer updates,
 * is expressed as calls on a backend, so a GPU backend can run a whole training step
 * without copying data back to the CPU. Operations are added here as milestone 1 grows.
 */
public interface Backend {

    /**
     * Returns a short, unique name for this backend, such as {@code "cpu"}.
     *
     * @return the backend name
     */
    String name();
}
