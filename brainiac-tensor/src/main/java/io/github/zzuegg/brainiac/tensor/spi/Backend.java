package io.github.zzuegg.brainiac.tensor.spi;

import io.github.zzuegg.brainiac.tensor.Device;

/**
 * A compute backend that owns tensor storage and runs every tensor operation.
 *
 * <p>All math in Brainiac, including the backward pass of autograd and optimizer updates,
 * is expressed as calls on a backend, so a GPU backend can run a whole training step
 * without copying data back to the CPU.
 *
 * <p>All storage is dense, contiguous and row-major. Every method receives storages that
 * were created by this backend; callers guarantee sizes and shapes are consistent, so
 * implementations may skip validation. Output storages never alias inputs.
 *
 * <p>Implementations are found with {@link java.util.ServiceLoader}; they must have a
 * public no-argument constructor and be cheap to construct.
 */
public interface Backend {

    /**
     * Returns a short, unique name for this backend, such as {@code "cpu"}. It is used as
     * the device name.
     *
     * @return the backend name
     */
    String name();

    /**
     * Returns what kind of hardware this backend runs on.
     *
     * @return the device kind
     */
    Device.Kind kind();

    /**
     * Returns whether this backend can run on this machine, for example whether a GPU
     * driver is present. Unavailable backends are ignored.
     *
     * @return {@code true} if the backend can be used
     */
    default boolean isAvailable() {
        return true;
    }

    /**
     * Allocates zero-filled storage.
     *
     * @param size the number of floats
     * @return the new storage
     */
    Storage allocate(long size);

    /**
     * Copies values from the Java heap into new storage.
     *
     * @param data the values; not retained
     * @return the new storage
     */
    Storage upload(float[] data);

    /**
     * Copies storage contents to the Java heap.
     *
     * @param src the storage to read
     * @return a new array with all values
     */
    float[] download(Storage src);

    /**
     * Sets every value of {@code dst} to {@code value}.
     *
     * @param dst the storage to fill
     * @param value the value
     */
    void fill(Storage dst, float value);

    /**
     * Applies an element-wise function: {@code out[i] = op(in[i])}.
     *
     * @param op the function
     * @param in the input
     * @param out the output, same size as the input
     */
    void unary(UnaryOp op, Storage in, Storage out);

    /**
     * Applies an element-wise binary function with broadcasting:
     * {@code out[idx] = op(a[idx . aStrides], b[idx . bStrides])} for every index of
     * {@code shape}. A stride of 0 repeats a value along that axis.
     *
     * @param op the function
     * @param a the left input
     * @param aStrides the left input's strides, one per axis of {@code shape}
     * @param b the right input
     * @param bStrides the right input's strides, one per axis of {@code shape}
     * @param shape the output shape
     * @param out the contiguous output
     */
    void binary(BinaryOp op, Storage a, long[] aStrides, Storage b, long[] bStrides, int[] shape, Storage out);

    /**
     * Reduces one axis. The input is viewed as {@code [outer, length, inner]} and the
     * output as {@code [outer, inner]}.
     *
     * @param op the reduction
     * @param in the input
     * @param outer the product of the dimensions before the axis
     * @param length the size of the reduced axis
     * @param inner the product of the dimensions after the axis
     * @param out the output
     */
    void reduce(ReduceOp op, Storage in, long outer, long length, long inner, Storage out);

    /**
     * Multiplies matrices: {@code out[m, n] = a[m, k] x b[k, n]}.
     *
     * @param a the left matrix, row-major
     * @param b the right matrix, row-major
     * @param m rows of {@code a}
     * @param k columns of {@code a} and rows of {@code b}
     * @param n columns of {@code b}
     * @param out the output matrix, row-major
     */
    void matmul(Storage a, Storage b, int m, int k, int n, Storage out);

    /**
     * Copies a strided view into contiguous storage: {@code out[idx] = in[idx . strides]}
     * for every index of {@code shape}, in row-major order. Used for transposes,
     * permutations and materializing broadcasts.
     *
     * @param in the input
     * @param strides the input strides, one per axis of {@code shape}
     * @param shape the shape of the view
     * @param out the contiguous output
     */
    default void copy(Storage in, long[] strides, int[] shape, Storage out) {
        long[] outStrides = new long[shape.length];
        long step = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            outStrides[i] = step;
            step *= shape[i];
        }
        copy(in, 0, strides, shape, out, 0, outStrides);
    }

    /**
     * Copies between two strided views: {@code out[outOffset + idx . outStrides] =
     * in[inOffset + idx . inStrides]} for every index of {@code shape}. Used for slices
     * and concatenation. The views never overlap.
     *
     * @param in the input
     * @param inOffset where the input view starts
     * @param inStrides the input strides, one per axis of {@code shape}
     * @param shape the shape of both views
     * @param out the output, which is written in place
     * @param outOffset where the output view starts
     * @param outStrides the output strides, one per axis of {@code shape}
     */
    void copy(Storage in, long inOffset, long[] inStrides, int[] shape, Storage out, long outOffset, long[] outStrides);
}
