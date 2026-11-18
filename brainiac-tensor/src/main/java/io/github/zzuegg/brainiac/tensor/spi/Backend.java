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
     * Further devices this backend found, each with a backend of its own, such as the other
     * GPUs of a machine with several. Called once, after {@link #isAvailable()} returned
     * {@code true}; each becomes a {@link Device} after this one. The default has none.
     *
     * @return the other devices' backends
     */
    default java.util.List<Backend> siblings() {
        return java.util.List.of();
    }

    /**
     * Sums storages of one size and type that live on different devices, giving each device
     * the sum, for data-parallel training (an all-reduce, such as NCCL's). Called on the first
     * backend of {@code group}; the default returns {@code null}, and the caller then sums on
     * the host.
     *
     * @param group the backends, all different devices, this one first
     * @param storages one storage per backend, in the same order; not changed
     * @return a new storage per backend, in the same order, holding the sum; or {@code null}
     *         if this backend cannot sum across these devices
     */
    default java.util.List<Storage> allReduceSum(java.util.List<Backend> group, java.util.List<Storage> storages) {
        return null;
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
     * Starts copying a storage to the host and returns its values when asked, so the host can
     * queue more work for the device meanwhile. The supplier must be called from the thread
     * that uses this backend. By default the copy happens now.
     */
    default java.util.function.Supplier<float[]> downloadLater(Storage src) {
        float[] now = download(src);
        return () -> now;
    }

    /**
     * New float32 storage of {@code size} values uniform in {@code [0, 1)}: element {@code i} is
     * {@link CounterRandom#uniform(long, long) CounterRandom.uniform(seed, i)}, so every backend
     * gives the same values. A backend may compute them on the device, or only where they are used.
     *
     * @param size the number of values
     * @param seed the seed
     * @return the new storage
     */
    default Storage random(long size, long seed) {
        float[] v = new float[Math.toIntExact(size)];
        for (int i = 0; i < v.length; i++) {
            v[i] = CounterRandom.uniform(seed, i);
        }
        return upload(v);
    }

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
     * Says that the values of these storages are about to be needed. A backend that records
     * operations instead of running them (such as the Vulkan backend's fused shaders) can
     * compute them all together, for example in one kernel; others ignore it, which is the
     * default.
     *
     * @param storages storages of this backend
     */
    default void evaluate(java.util.List<Storage> storages) {
    }

    /**
     * Says that a storage will not be read again, so its memory can be reused now rather than
     * when it is garbage collected. Only a hint; the default ignores it.
     *
     * @param storage a storage of this backend that nothing reads afterwards
     */
    default void release(Storage storage) {
    }

    /**
     * The device memory this backend holds for tensors, or {@code null} when it does not count
     * it (the CPU's memory is the Java heap's).
     *
     * @return the use, or {@code null}
     */
    default Device.Memory memory() {
        return null;
    }

    /** Starts measuring {@link Device.Memory#peak()} again from the current use. */
    default void resetPeakMemory() {
    }

    /**
     * Allocates storage whose contents are undefined, for results that an operation
     * overwrites completely. Backends where clearing costs a device command (such as a GPU)
     * skip it; the default clears anyway.
     *
     * @param size the number of values
     * @param dtype the type
     * @return the new storage
     */
    default Storage allocateUninitialized(long size, DType dtype) {
        return allocate(size, dtype);
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
     * Copies raw values into new storage of the same type, without rounding. The values
     * are laid out as the type stores them, little-endian: four bytes per float32, eight
     * per float64, two per float16 or bfloat16. This is how weights in a memory-mapped
     * file reach a device without passing through the Java heap: the default decodes them
     * into an array and calls {@link #upload(double[], DType)}; backends override it to copy
     * the bytes straight into their own memory.
     *
     * @param data the values; not retained, and only read during this call
     * @param dtype their type
     * @return the new storage, of type {@code dtype}
     */
    default Storage upload(java.lang.foreign.MemorySegment data, DType dtype) {
        long n = data.byteSize() / dtype.bytes();
        if (dtype == DType.FLOAT32) {
            return upload(data.toArray(RawValues.FLOAT));
        }
        return upload(RawValues.decode(data, dtype, n), dtype);
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

    // ---------------------------------------------------------------- packed (quantized) matrices

    /**
     * Takes a quantized matrix into this backend's own form. The default keeps the
     * {@link PackedMatrix} itself, in Java memory, which every backend can read through the
     * default {@link #unpack} and {@link #matmulPacked}.
     *
     * @param matrix the levels and scales
     * @return storage for it
     */
    default PackedStorage pack(PackedMatrix matrix) {
        return matrix;
    }

    /**
     * Takes a matrix kept in a GGUF file's blocks (see {@link GgmlBlocks}). On the CPU the
     * default keeps it where it is, in the file's mapping, which the default {@link #unpack}
     * and {@link #matmulPacked} read and the CPU backend multiplies by directly. Elsewhere it
     * requantizes the values to 8 bits and {@linkplain #pack packs} them.
     *
     * @param blocks the stored blocks
     * @return storage for it
     */
    default PackedStorage packBlocks(GgmlBlocks blocks) {
        return kind() == Device.Kind.CPU ? blocks : pack(unpackToHost(blocks));
    }

    /**
     * As {@link #pack}, for a matrix that may be joined with others ({@link #unpackToHost})
     * before it is used, such as a projection {@code Llama.fuseProjections} joins: a backend
     * that moves matrices to its device in the background can leave this one until its first
     * use, or until {@link #prefetch}.
     *
     * @param matrix the levels and scales
     * @return storage for it
     */
    default PackedStorage packLater(PackedMatrix matrix) {
        return pack(matrix);
    }

    /**
     * Starts moving a matrix from {@link #packLater} to the device, if it is not there yet.
     * The default does nothing.
     *
     * @param packed the matrix
     */
    default void prefetch(PackedStorage packed) {
    }

    /**
     * Quantizes a {@code [rows, cols]} float32 matrix on this backend, as
     * {@link PackedMatrix#quantize} does. The default downloads the values and quantizes them
     * in Java; backends override it to keep the values on the device.
     *
     * @param values {@code rows * cols} float32 values, row-major
     * @param rows the rows
     * @param cols the columns
     * @param bits the bits per level, 2 to 8
     * @param group the rows per scale; must divide {@code rows}
     * @return the packed storage
     */
    default PackedStorage quantize(Storage values, int rows, int cols, int bits, int group) {
        return pack(PackedMatrix.quantize(download(values), rows, cols, bits, group));
    }

    /**
     * Returns a packed storage of this backend in Java memory, to move it to another backend.
     *
     * @param packed storage from {@link #pack}
     * @return the matrix
     */
    default PackedMatrix unpackToHost(PackedStorage packed) {
        if (packed instanceof PackedMatrix m) {
            return m;
        }
        if (packed instanceof GgmlBlocks blocks) {
            // to move it to a backend without a kernel for the blocks: requantized to 8 bits
            float[] values = new float[Math.toIntExact(blocks.size())];
            blocks.dequantize(values);
            return PackedMatrix.quantize(values, blocks.rows(), blocks.cols(), 8, 32);
        }
        throw unsupported("reading back " + packed);
    }

    /**
     * Computes a packed matrix's values into ordinary storage.
     *
     * @param packed storage from {@link #pack}
     * @param out {@code rows * cols} values of any type this backend supports
     */
    default void unpack(PackedStorage packed, Storage out) {
        float[] values = new float[Math.toIntExact(packed.size())];
        if (packed instanceof QuantizedRows rows) {
            rows.dequantize(values);
        } else if (packed instanceof GgmlBlocks blocks) {
            blocks.dequantize(values);
        } else {
            unpackToHost(packed).dequantize(values);
        }
        Storage host = upload(values);
        convert(host, out);
        release(host);
    }

    /**
     * Multiplies by a packed matrix: {@code out = a x b} with {@code a} row-major
     * {@code [m, k]} and {@code b} a packed {@code [k, n]}. The default unpacks {@code b}
     * into temporary storage and calls {@link #matmul}; backends override it to read the
     * levels directly, which moves a quarter (8 bits) or an eighth (4 bits) of the bytes.
     *
     * @param a the left matrix
     * @param b the packed right matrix
     * @param m rows of {@code a}
     * @param k columns of {@code a}, rows of {@code b}
     * @param n columns of {@code b}
     * @param out the output, {@code [m, n]}, of {@code a}'s type
     */
    default void matmulPacked(Storage a, PackedStorage b, int m, int k, int n, Storage out) {
        Storage w = allocateUninitialized((long) k * n, out.dtype());
        unpack(b, w);
        matmul(a, w, m, k, n, out);
        // not released here: a backend that defers work may not have read it yet
    }

    // ---------------------------------------------------------------- rows quantized to 8 bits

    /**
     * Allocates rows kept as 8-bit levels with a scale per row (see {@link QuantizedRows}),
     * all zero. The default keeps them in Java memory.
     *
     * @param rows the rows
     * @param cols the values per row
     * @param dtype the type the values are read as
     * @return the storage
     */
    default RowQuantizedStorage allocateQuantizedRows(long rows, int cols, DType dtype) {
        return new QuantizedRows(rows, cols, dtype);
    }

    /**
     * Quantizes rows into storage from {@link #allocateQuantizedRows}: for each of
     * {@code outer} blocks of {@code blockRows} rows there, rows {@code [o * sourceRows,
     * (o + 1) * sourceRows)} of {@code values} go to rows {@code start ..} of block {@code o}.
     *
     * @param target the quantized rows
     * @param values the source rows, of any type
     * @param outer the blocks
     * @param blockRows rows per block in {@code target}
     * @param sourceRows rows per block in {@code values}
     * @param start the first row written in each block
     */
    default void writeQuantizedRows(RowQuantizedStorage target, Storage values, long outer, int blockRows,
            int sourceRows, int start) {
        if (!(target instanceof QuantizedRows rows)) {
            throw unsupported("writing " + target);
        }
        rows.write(download(values), outer, blockRows, sourceRows, start);
    }

    /**
     * Whether {@link #attentionQuantized} runs for these sizes.
     *
     * @param shape {@code batch, heads, kvHeads, tq, tk, dk, dv}
     * @param type the queries' type
     * @return whether to call it
     */
    default boolean hasQuantizedAttention(int[] shape, DType type) {
        return false;
    }

    /**
     * {@link #attention(Storage, Storage, Storage, int[], float, boolean, Storage, long[], Storage, Storage)}
     * reading keys and values from {@linkplain #allocateQuantizedRows quantized rows} directly.
     *
     * @param q the queries
     * @param k the keys, rows of {@code dk}
     * @param v the values, rows of {@code dv}
     * @param shape {@code batch, heads, kvHeads, tq, tk, dk, dv}
     * @param scale the score scale
     * @param causal whether to mask later keys as well
     * @param bias the bias, of the output's type
     * @param strides the bias strides, in elements
     * @param out {@code [batch, heads, tq, dv]}
     * @param lse {@code [batch, heads, tq]}
     */
    default void attentionQuantized(Storage q, RowQuantizedStorage k, RowQuantizedStorage v, int[] shape,
            float scale, boolean causal, Storage bias, long[] strides, Storage out, Storage lse) {
        throw unsupported("attention over quantized rows");
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
     * Runs a user {@link io.github.zzuegg.brainiac.tensor.Kernel}. The default runs its Java
     * version on the host; a backend that compiles GLSL runs the shader.
     *
     * @param kernel the kernel
     * @param inputs its inputs
     * @param out the output, of the type the kernel computes in
     * @param ints the integer parameters
     * @param floats the float parameters
     */
    default void custom(io.github.zzuegg.brainiac.tensor.Kernel kernel, Storage[] inputs, Storage out, int[] ints,
            float[] floats) {
        double[][] in = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            in[i] = downloadDouble(inputs[i]);
        }
        double[] y = new double[Math.toIntExact(out.size())];
        kernel.reference().run(in, y, ints.clone(), floats.clone());
        Storage host = upload(y, out.dtype());
        convert(host, out);
        release(host);
    }

    /**
     * Sorts each row of {@code [rows, length]} values, stably: equal values keep their
     * order, and NaN counts as larger than every number. The default sorts on the host;
     * backends override it to sort on the device.
     *
     * @param in the values, row-major
     * @param rows the rows
     * @param length the values per row
     * @param descending largest first instead of smallest first
     * @param values the sorted values, of {@code in}'s type
     * @param indices where each sorted value was in its row, as float32 whole numbers
     */
    default void sort(Storage in, long rows, int length, boolean descending, Storage values, Storage indices) {
        double[] x = downloadDouble(in);
        double[] v = new double[x.length];
        float[] at = new float[x.length];
        Integer[] order = new Integer[length];
        for (long r = 0; r < rows; r++) {
            int base = Math.toIntExact(r * length);
            for (int i = 0; i < length; i++) {
                order[i] = i;
            }
            java.util.Comparator<Integer> c = (a, b) -> x[base + a] == x[base + b] ? 0   // -0 equals 0
                    : Double.compare(x[base + a], x[base + b]);
            java.util.Arrays.sort(order, descending ? c.reversed() : c);   // merge sort: stable
            for (int i = 0; i < length; i++) {
                v[base + i] = x[base + order[i]];
                at[base + i] = order[i];
            }
        }
        Storage hv = upload(v, in.dtype());
        convert(hv, values);
        release(hv);
        Storage hi = upload(at);
        convert(hi, indices);
        release(hi);
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

    /**
     * Discrete Fourier transforms of rows of complex values stored as (real, imaginary)
     * pairs: {@code X[k] = sum_j x[j] e^(-2 pi i jk / n)}; the inverse uses the conjugate
     * exponent and divides by {@code n}. By default computed on the host with {@link Fft}.
     *
     * @param in {@code rows * n * 2} contiguous values
     * @param rows the number of rows
     * @param n the complex values per row
     * @param inverse whether to compute the inverse transform
     * @return new float32 storage of the same size
     */
    default Storage fft(Storage in, long rows, int n, boolean inverse) {
        float[] data = download(in);
        Fft.transform(data, rows, n, inverse);
        return upload(data);
    }

    /**
     * Whether {@link #rmsNorm} has a kernel for this type.
     *
     * @param type the element type
     * @return whether to call it
     */
    default boolean hasRmsNorm(DType type) {
        return false;
    }

    /**
     * RMSNorm of each row in one pass, for inference:
     * {@code out = round(x / sqrt(mean(x^2) + eps)) * weight}, computed in float32, the
     * normalized value rounded to the storage type before the scale. All arrays are
     * contiguous and of one type.
     *
     * @param x {@code [rows, cols]}
     * @param weight {@code [cols]}
     * @param rows the rows
     * @param cols the values per row
     * @param eps added to the mean square
     * @param out {@code [rows, cols]}
     */
    default void rmsNorm(Storage x, Storage weight, long rows, int cols, float eps, Storage out) {
        throw unsupported("fused RMSNorm");
    }

    /**
     * Whether {@link #addRmsNorm} has a kernel for this type.
     *
     * @param type the element type
     * @return whether to call it
     */
    default boolean hasAddRmsNorm(DType type) {
        return false;
    }

    /**
     * A residual add and the {@link #rmsNorm} after it in one pass, for inference:
     * {@code sum = x + r} as an add stores it, and {@code out} the RMSNorm of {@code sum}.
     *
     * @param x {@code [rows, cols]}
     * @param r {@code [rows, cols]}, added to {@code x}
     * @param weight {@code [cols]}
     * @param rows the rows
     * @param cols the values per row
     * @param eps added to the mean square
     * @param sum {@code [rows, cols]}, receives {@code x + r}
     * @param out {@code [rows, cols]}, receives the normalized sum
     */
    default void addRmsNorm(Storage x, Storage r, Storage weight, long rows, int cols, float eps, Storage sum,
            Storage out) {
        throw unsupported("fused add and RMSNorm");
    }

    /**
     * Whether {@link #attention} (and, when {@code backward}, {@link #attentionBackward}) are
     * worth using for these sizes; callers take another path when not.
     *
     * @param shape {@code batch, heads, kvHeads, tq, tk, dk, dv}
     * @param type the element type
     * @param backward whether the gradient will be needed too
     * @return whether to use the fused kernels
     */
    default boolean hasAttention(int[] shape, DType type, boolean backward) {
        return false;
    }

    /**
     * Scaled dot-product attention in one pass, without storing the weights:
     * {@code out = softmax(scale * q k^T [+ causal mask]) v}, with each row's log-sum-exp of
     * the scaled scores in {@code lse} for {@link #attentionBackward}. Query head {@code h}
     * uses key/value head {@code h / (heads / kvHeads)}. Causal: query {@code i} sees keys
     * up to {@code i + tk - tq}. All arrays are contiguous.
     *
     * @param q {@code [batch, heads, tq, dk]}
     * @param k {@code [batch, kvHeads, tk, dk]}
     * @param v {@code [batch, kvHeads, tk, dv]}
     * @param shape {@code batch, heads, kvHeads, tq, tk, dk, dv}
     * @param scale the score scale
     * @param causal whether to mask later keys
     * @param out {@code [batch, heads, tq, dv]}
     * @param lse {@code [batch, heads, tq]}
     */
    default void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal, Storage out,
            Storage lse) {
        throw unsupported("fused attention");
    }

    /**
     * Whether {@link #attention(Storage, Storage, Storage, int[], float, boolean, Storage, long[], Storage, Storage)}
     * takes an additive score bias for these sizes, for inference.
     *
     * @param shape {@code batch, heads, kvHeads, tq, tk, dk, dv}
     * @param type the element type
     * @return whether to use it
     */
    default boolean hasAttentionWithBias(int[] shape, DType type) {
        return false;
    }

    /**
     * {@link #attention} with a bias added to the scaled scores before the softmax, such as
     * masks for padding and for unused cache positions. The bias value for batch {@code b},
     * head {@code h}, query {@code i} and key {@code j} is at
     * {@code b * strides[0] + h * strides[1] + i * strides[2] + j * strides[3]}; a stride of
     * 0 broadcasts. Without gradients: {@code lse} is still written.
     *
     * @param q the queries
     * @param k the keys
     * @param v the values
     * @param shape {@code batch, heads, kvHeads, tq, tk, dk, dv}
     * @param scale the score scale
     * @param causal whether to mask later keys as well
     * @param bias the bias, of the output's type
     * @param strides the bias strides, in elements
     * @param out {@code [batch, heads, tq, dv]}
     * @param lse {@code [batch, heads, tq]}
     */
    default void attention(Storage q, Storage k, Storage v, int[] shape, float scale, boolean causal, Storage bias,
            long[] strides, Storage out, Storage lse) {
        throw unsupported("fused attention with a bias");
    }

    /**
     * The gradients of {@link #attention}, recomputing the weights from {@code lse}.
     *
     * @param q the queries
     * @param k the keys
     * @param v the values
     * @param out the forward output
     * @param lse the forward log-sum-exp
     * @param grad the gradient of the output
     * @param shape as for {@link #attention}
     * @param scale the score scale
     * @param causal whether later keys were masked
     * @param dq the gradient of the queries, overwritten
     * @param dk the gradient of the keys, overwritten
     * @param dv the gradient of the values, overwritten
     */
    default void attentionBackward(Storage q, Storage k, Storage v, Storage out, Storage lse, Storage grad,
            int[] shape, float scale, boolean causal, Storage dq, Storage dk, Storage dv) {
        throw unsupported("fused attention");
    }

    private UnsupportedOperationException unsupported(String what) {
        return new UnsupportedOperationException("the " + name() + " backend does not support " + what);
    }
}
