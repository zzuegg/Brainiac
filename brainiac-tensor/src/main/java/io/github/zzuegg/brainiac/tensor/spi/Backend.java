package io.github.zzuegg.brainiac.tensor.spi;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
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
     * Applies an activation in one pass: {@code out[i] = f(in[i])}.
     *
     * @param op the activation
     * @param param its parameter, ignored by activations without one
     * @param in the inputs
     * @param out where to write the results; same size
     */
    void activation(ActivationOp op, float param, Storage in, Storage out);

    /**
     * Computes an activation's input gradient in one pass:
     * {@code out[i] = grad[i] * f'(in[i])}.
     *
     * @param op the activation
     * @param param its parameter, ignored by activations without one
     * @param in the activation's inputs
     * @param grad the gradient of the activation's outputs; same size
     * @param out where to write the input gradient; same size
     */
    void activationBackward(ActivationOp op, float param, Storage in, Storage grad, Storage out);

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
    // ---------------------------------------------------------------- number formats
    //
    // Every operation receives storages that carry their DType. Outputs are allocated by
    // the caller with the type the result should have; a backend computes in at least the
    // precision of its inputs and rounds into the output's type. The defaults support only
    // FLOAT32.

    /**
     * Returns whether this backend can store and compute with a number format.
     *
     * @param dtype the type
     * @return {@code true} if storages of that type can be allocated
     */
    default boolean supports(DType dtype) {
        return dtype == DType.FLOAT32;
    }

    /**
     * Allocates zero-filled storage of a type.
     *
     * @param size the number of values
     * @param dtype the type
     * @return the new storage
     */
    default Storage allocate(long size, DType dtype) {
        if (dtype == DType.FLOAT32) {
            return allocate(size);
        }
        throw unsupported(dtype + " tensors");
    }

    /**
     * Copies values into new storage of a type, rounding them to its precision.
     *
     * @param data the values; not retained
     * @param dtype the type
     * @return the new storage
     */
    default Storage upload(double[] data, DType dtype) {
        if (dtype != DType.FLOAT32) {
            throw unsupported(dtype + " tensors");
        }
        float[] f = new float[data.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) data[i];
        }
        return upload(f);
    }

    /**
     * Copies storage contents to the Java heap without losing precision.
     *
     * @param src the storage to read
     * @return a new array with all values
     */
    default double[] downloadDouble(Storage src) {
        float[] f = download(src);
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    /**
     * Converts values to the type of {@code out}, which has the same size.
     *
     * @param in the source
     * @param out the destination
     */
    default void convert(Storage in, Storage out) {
        if (in.dtype() != DType.FLOAT32 || out.dtype() != DType.FLOAT32) {
            throw unsupported("converting " + in.dtype() + " to " + out.dtype());
        }
        long[] one = {1};
        copy(in, 0, one, new int[] {Math.toIntExact(in.size())}, out, 0, one);
    }

    // ---------------------------------------------------------------- optional operations
    //
    // The operations below have default implementations that throw, so a new backend can
    // start small; the layers that need them fail with a clear message until it adds them.

    /**
     * Multiplies a batch of matrices: {@code out[b] = a[b] x b[b]} for each of
     * {@code batch} pairs of row-major matrices stored one after another.
     *
     * @param a the left matrices, {@code [batch, m, k]}
     * @param b the right matrices, {@code [batch, k, n]}
     * @param batch the number of pairs
     * @param m rows of each left matrix
     * @param k columns of each left matrix and rows of each right matrix
     * @param n columns of each right matrix
     * @param out the output, {@code [batch, m, n]}
     */
    default void matmulBatched(Storage a, Storage b, int batch, int m, int k, int n, Storage out) {
        throw unsupported("batched matmul");
    }

    /**
     * Unrolls every window of a sliding-window geometry into a row: {@code out} is
     * {@code [batch * positions, channels * kernel]}, with rows ordered by image then
     * position and columns by channel then window cell. Cells outside the image read
     * {@code padValue}.
     *
     * @param in the images, {@code [batch, channels, image...]}
     * @param geometry the windows
     * @param padValue the value of padding cells
     * @param out the unrolled windows
     */
    default void im2col(Storage in, ConvGeometry geometry, float padValue, Storage out) {
        throw unsupported("im2col (convolution and pooling)");
    }

    /**
     * The adjoint of {@link #im2col}: adds every row of {@code cols} back onto the image
     * cells its window covers, dropping padding cells. {@code out} is overwritten.
     *
     * @param cols the unrolled windows, {@code [batch * positions, channels * kernel]}
     * @param geometry the windows
     * @param out the images, {@code [batch, channels, image...]}
     */
    default void col2im(Storage cols, ConvGeometry geometry, Storage out) {
        throw unsupported("col2im (convolution and pooling)");
    }

    /**
     * Picks slices along one axis by index. The input is viewed as
     * {@code [outer, length, inner]}, the output as {@code [outer, count, inner]}, and
     * {@code out[o, i, j] = in[o, indices[i], j]}. Indices are stored as floats holding
     * whole numbers in {@code [0, length)}.
     *
     * @param in the input
     * @param outer the product of the dimensions before the axis
     * @param length the size of the axis
     * @param inner the product of the dimensions after the axis
     * @param indices the {@code count} indices
     * @param count the number of indices
     * @param out the output
     */
    default void gather(Storage in, long outer, long length, long inner, Storage indices, int count, Storage out) {
        throw unsupported("gather");
    }

    /**
     * The adjoint of {@link #gather}: {@code out[o, indices[i], j] += src[o, i, j]}, with
     * {@code out} viewed as {@code [outer, length, inner]} and overwritten first with
     * zeros. Repeated indices add up.
     *
     * @param src the values, {@code [outer, count, inner]}
     * @param outer the product of the dimensions before the axis
     * @param count the number of indices
     * @param inner the product of the dimensions after the axis
     * @param indices the {@code count} indices
     * @param length the size of the axis in the output
     * @param out the output
     */
    default void scatterAdd(Storage src, long outer, int count, long inner, Storage indices, long length,
            Storage out) {
        throw unsupported("scatterAdd");
    }

    private UnsupportedOperationException unsupported(String what) {
        return new UnsupportedOperationException("the " + name() + " backend does not support " + what);
    }
}
