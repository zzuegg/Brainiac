package io.github.zzuegg.brainiac.tensor;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * An immutable n-dimensional array of float values on a {@link Device}.
 *
 * <p>A tensor is a handle: its values live in storage owned by the device's backend, and
 * every operation runs there. Operations return new tensors and never change their
 * inputs. Binary operations broadcast like NumPy: shapes are aligned at their last
 * dimension, and a dimension of size 1 repeats to match the other side.
 *
 * <pre>{@code
 * Tensor x = Tensor.of(Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
 * Tensor b = Tensor.of(10, 20, 30);
 * Tensor y = x.add(b).relu();        // [[11, 22, 33], [14, 25, 36]]
 * float total = y.sum().item();      // 141
 * }</pre>
 *
 * <p>Values only cross between devices through {@link #to(Device)}, {@link #toArray()}
 * and {@link #item()}.
 */
public final class Tensor {

    private final Shape shape;
    private final Device device;
    private final Storage storage;

    /** The tensors created on this thread while {@link #created} runs, innermost call last. */
    private static final ThreadLocal<java.util.ArrayList<Tensor>> CREATED = new ThreadLocal<>();

    private Tensor(Shape shape, Device device, Storage storage) {
        this.shape = shape;
        this.device = device;
        this.storage = storage;
        java.util.ArrayList<Tensor> created = CREATED.get();
        if (created != null) {
            created.add(this);
        }
    }

    /**
     * Runs {@code action} and returns every tensor it created on this thread, so that the
     * caller can {@link #release(java.util.Collection, java.util.Collection) release} the
     * ones it no longer needs. Tensors created by a nested call are returned by that call
     * only.
     *
     * @param action the computation
     * @return the tensors created, in order
     */
    public static java.util.List<Tensor> created(Runnable action) {
        java.util.ArrayList<Tensor> outer = CREATED.get();
        java.util.ArrayList<Tensor> list = new java.util.ArrayList<>();
        CREATED.set(list);
        try {
            action.run();
        } finally {
            CREATED.set(outer);
        }
        return list;
    }

    // ---------------------------------------------------------------- creation

    /**
     * Creates a tensor on the default device from values in row-major order.
     *
     * @param shape the shape
     * @param values the values; their count must equal {@code shape.size()}
     * @return the tensor
     */
    public static Tensor of(Shape shape, float... values) {
        return of(shape, values, Device.getDefault());
    }

    /**
     * Creates a tensor on the given device from values in row-major order.
     *
     * @param shape the shape
     * @param values the values; their count must equal {@code shape.size()}
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor of(Shape shape, float[] values, Device device) {
        Objects.requireNonNull(shape, "shape");
        if (values.length != shape.size()) {
            throw new IllegalArgumentException(
                    "shape " + shape + " needs " + shape.size() + " values, got " + values.length);
        }
        return new Tensor(shape, device, device.backend().upload(values));
    }

    /**
     * Creates a tensor of a given type from values in row-major order, rounding each value
     * to the type's precision.
     *
     * @param shape the shape
     * @param values the values; their count must equal {@code shape.size()}
     * @param dtype the number format
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor of(Shape shape, double[] values, DType dtype, Device device) {
        Objects.requireNonNull(shape, "shape");
        if (values.length != shape.size()) {
            throw new IllegalArgumentException(
                    "shape " + shape + " needs " + shape.size() + " values, got " + values.length);
        }
        return new Tensor(shape, device, device.backend().upload(values, dtype));
    }

    /**
     * Creates a one-dimensional tensor on the default device.
     *
     * @param values the values
     * @return a tensor of shape {@code [values.length]}
     */
    public static Tensor of(float... values) {
        return of(Shape.of(values.length), values);
    }

    /**
     * Creates a two-dimensional tensor on the default device from rows of equal length.
     *
     * @param rows the rows
     * @return a tensor of shape {@code [rows.length, rows[0].length]}
     */
    public static Tensor of(float[][] rows) {
        if (rows.length == 0) {
            throw new IllegalArgumentException("need at least one row");
        }
        int cols = rows[0].length;
        float[] flat = new float[rows.length * cols];
        for (int r = 0; r < rows.length; r++) {
            if (rows[r].length != cols) {
                throw new IllegalArgumentException(
                        "row " + r + " has " + rows[r].length + " values, expected " + cols);
            }
            System.arraycopy(rows[r], 0, flat, r * cols, cols);
        }
        return of(Shape.of(rows.length, cols), flat);
    }

    /**
     * Creates a scalar (rank 0) tensor on the default device.
     *
     * @param value the value
     * @return the scalar tensor
     */
    public static Tensor scalar(float value) {
        return of(Shape.of(), value);
    }

    /**
     * Creates a tensor of zeros on the default device.
     *
     * @param shape the shape
     * @return the tensor
     */
    public static Tensor zeros(Shape shape) {
        return zeros(shape, Device.getDefault());
    }

    /**
     * Creates a tensor of zeros on the given device.
     *
     * @param shape the shape
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor zeros(Shape shape, Device device) {
        return zeros(shape, DType.FLOAT32, device);
    }

    /**
     * Creates a tensor of zeros of a given type.
     *
     * @param shape the shape
     * @param dtype the number format
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor zeros(Shape shape, DType dtype, Device device) {
        return new Tensor(shape, device, device.backend().allocate(shape.size(), dtype));
    }

    /**
     * Creates a tensor of ones on the default device.
     *
     * @param shape the shape
     * @return the tensor
     */
    public static Tensor ones(Shape shape) {
        return full(shape, 1f, Device.getDefault());
    }

    /**
     * Creates a tensor filled with one value, on the default device.
     *
     * @param shape the shape
     * @param value the value of every element
     * @return the tensor
     */
    public static Tensor full(Shape shape, float value) {
        return full(shape, value, Device.getDefault());
    }

    /**
     * Creates a tensor filled with one value.
     *
     * @param shape the shape
     * @param value the value of every element
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor full(Shape shape, float value, Device device) {
        return full(shape, value, DType.FLOAT32, device);
    }

    /**
     * Creates a tensor of a given type filled with one value.
     *
     * @param shape the shape
     * @param value the value of every element
     * @param dtype the number format
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor full(Shape shape, float value, DType dtype, Device device) {
        if (shape.rank() == 0) {
            // Tensors are immutable, so single values are shared; on a GPU each new one
            // would otherwise cost a device command.
            ScalarKey key = new ScalarKey(device, dtype, Float.floatToRawIntBits(value));
            synchronized (SCALARS) {
                Tensor t = SCALARS.get(key);
                if (t == null) {
                    t = filled(shape, value, dtype, device);
                    SCALARS.put(key, t);
                    SHARED.put(t.storage, Boolean.TRUE);
                }
                return t;
            }
        }
        return filled(shape, value, dtype, device);
    }

    private record ScalarKey(Device device, DType dtype, int bits) {
    }

    private static final Map<ScalarKey, Tensor> SCALARS = new LinkedHashMap<>(64, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(Map.Entry<ScalarKey, Tensor> eldest) {
            return size() > 1_024;
        }
    };

    /** Storages handed out to many callers (the shared scalars): never released early. */
    private static final Map<Storage, Boolean> SHARED =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    private static Tensor filled(Shape shape, float value, DType dtype, Device device) {
        Backend backend = device.backend();
        Storage s;
        if (value == 0f) {
            s = backend.allocate(shape.size(), dtype);
        } else {
            s = backend.allocateUninitialized(shape.size(), dtype);
            backend.fill(s, value);
        }
        return new Tensor(shape, device, s);
    }

    /**
     * Creates a tensor of values uniform in {@code [0, 1)} on a device, computed from the seed
     * and each element's index ({@link io.github.zzuegg.brainiac.tensor.spi.CounterRandom}):
     * the same values on every device, and on a GPU they are generated there (inside the kernel
     * that uses them), not on the CPU and uploaded. Meant for masks such as dropout's.
     *
     * @param shape the shape
     * @param seed the seed; a new one for every new mask
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor random(Shape shape, long seed, Device device) {
        return new Tensor(shape, device, device.backend().random(checkedSize(shape), seed));
    }

    /**
     * Creates a tensor of values drawn uniformly from {@code [low, high)} on the default
     * device. Values are generated on the CPU, so a seeded generator gives the same values
     * on every device.
     *
     * @param shape the shape
     * @param low the inclusive lower bound
     * @param high the exclusive upper bound
     * @param random the random source, for example {@code new SplittableRandom(42)}
     * @return the tensor
     */
    public static Tensor uniform(Shape shape, float low, float high, RandomGenerator random) {
        float[] v = new float[checkedSize(shape)];
        for (int i = 0; i < v.length; i++) {
            v[i] = low + (high - low) * random.nextFloat();
        }
        return of(shape, v);
    }

    /**
     * Creates a tensor of normally distributed values on the default device. Values are
     * generated on the CPU, so a seeded generator gives the same values on every device.
     *
     * @param shape the shape
     * @param mean the mean
     * @param std the standard deviation
     * @param random the random source, for example {@code new SplittableRandom(42)}
     * @return the tensor
     */
    public static Tensor normal(Shape shape, float mean, float std, RandomGenerator random) {
        float[] v = new float[checkedSize(shape)];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) random.nextGaussian(mean, std);
        }
        return of(shape, v);
    }

    /**
     * Creates {@code [0, 1, ..., n - 1]}.
     *
     * @param n how many values
     * @return the vector
     */
    public static Tensor arange(int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
        return of(Shape.of(n), v);
    }

    private static int checkedSize(Shape shape) {
        if (shape.size() > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("shape " + shape + " is too large for host memory");
        }
        return (int) shape.size();
    }

    // ---------------------------------------------------------------- properties

    /**
     * Returns the shape.
     *
     * @return the shape
     */
    public Shape shape() {
        return shape;
    }

    /**
     * Returns the device this tensor lives on.
     *
     * @return the device
     */
    public Device device() {
        return device;
    }

    /**
     * Returns the number of dimensions.
     *
     * @return the rank
     */
    public int rank() {
        return shape.rank();
    }

    /**
     * Returns the number of values.
     *
     * @return the size
     */
    public long size() {
        return shape.size();
    }

    // ---------------------------------------------------------------- packed matrices

    /**
     * Creates a matrix kept as quantized levels and scales (see {@link PackedMatrix}): a
     * quarter of float32's memory at 8 bits, an eighth at 4. It behaves like any other
     * {@code [rows, cols]} float32 tensor, but only a matrix product with it on the right,
     * {@code x.matmul(packed)}, reads the levels directly; every other operation, and a
     * transpose, first computes the values into a temporary tensor.
     *
     * @param matrix the levels and scales
     * @param device where to keep it
     * @return the tensor
     */
    public static Tensor packed(PackedMatrix matrix, Device device) {
        return new Tensor(Shape.of(matrix.rows(), matrix.cols()), device, device.backend().pack(matrix));
    }

    /** @return whether this tensor is a {@linkplain #packed packed} matrix */
    public boolean isPacked() {
        return storage instanceof PackedStorage;
    }

    /**
     * Returns the bytes this tensor's values take on its device.
     *
     * @return the size in bytes
     */
    public long bytes() {
        return storage instanceof PackedStorage p ? p.bytes() : size() * dtype().bytes();
    }

    /**
     * Returns an ordinary tensor with this one's values: for a {@linkplain #packed packed}
     * matrix its computed values, otherwise this tensor.
     *
     * @return the values
     */
    public Tensor unpacked() {
        return storage instanceof PackedStorage ? new Tensor(shape, device, storage()) : this;
    }

    /** The values as ordinary storage: a packed matrix's are computed into a new one. */
    private Storage storage() {
        if (storage instanceof PackedStorage p) {
            Storage out = backend().allocateUninitialized(size(), DType.FLOAT32);
            backend().unpack(p, out);
            return out;
        }
        return storage;
    }

    // ---------------------------------------------------------------- data transfer

    /**
     * Returns this tensor on another device, copying the values if needed.
     *
     * @param target the device
     * @return a tensor on {@code target}; {@code this} if it is already there
     */
    public Tensor to(Device target) {
        if (target.equals(device)) {
            return this;
        }
        if (storage instanceof PackedStorage p) {
            return new Tensor(shape, target, target.backend().pack(backend().unpackToHost(p)));
        }
        if (dtype() == DType.FLOAT32) {
            return new Tensor(shape, target, target.backend().upload(toArray()));
        }
        return new Tensor(shape, target, target.backend().upload(toDoubleArray(), dtype()));
    }

    /**
     * Returns the number format of this tensor's values.
     *
     * @return the type
     */
    public DType dtype() {
        return storage.dtype();
    }

    /**
     * Returns this tensor converted to another number format, rounding where needed.
     *
     * @param type the new type
     * @return a tensor of {@code type}; {@code this} if it already is
     */
    public Tensor to(DType type) {
        if (type == dtype()) {
            return this;
        }
        Storage out = backend().allocateUninitialized(size(), type);
        backend().convert(storage(), out);
        return new Tensor(shape, device, out);
    }

    /**
     * Copies all values to a new Java array of doubles, without rounding them to float.
     *
     * @return the values
     */
    public double[] toDoubleArray() {
        return backend().downloadDouble(storage());
    }

    /**
     * Computes these tensors now, together where the device can. Results are the same
     * without it: on devices that record operations instead of running them, it lets
     * related results (such as a parameter and its optimizer state) share one kernel instead
     * of being computed one by one later.
     *
     * @param tensors the tensors, on any devices
     */
    public static void evaluate(java.util.List<Tensor> tensors) {
        Map<Device, java.util.List<Storage>> byDevice = new LinkedHashMap<>();
        for (Tensor t : tensors) {
            if (!(t.storage instanceof PackedStorage)) {
                byDevice.computeIfAbsent(t.device, d -> new java.util.ArrayList<>()).add(t.storage);
            }
        }
        byDevice.forEach((d, storages) -> d.backend().evaluate(storages));
    }

    /**
     * Says that this tensor's values will not be used again, so a device can reuse its memory
     * at once instead of when the tensor is garbage collected. The tensor, its views (such as
     * reshapes) and results computed from it but not yet computed (see {@link #evaluate})
     * must not be used afterwards; a GPU backend reports such a use as an error.
     */
    public void release() {
        if (!SHARED.containsKey(storage)) {
            backend().release(storage);
        }
    }

    /**
     * Releases every tensor in {@code dead} (see {@link #release()}) except those sharing
     * memory with a tensor in {@code keep}, such as a view of a value that is still used.
     * Whatever still reads a released value must be computed first (see {@link #evaluate}).
     *
     * @param dead tensors whose values will not be used again
     * @param keep tensors that stay in use
     */
    public static void release(java.util.Collection<Tensor> dead, java.util.Collection<Tensor> keep) {
        Map<Storage, Boolean> skip = new java.util.IdentityHashMap<>();
        for (Tensor t : keep) {
            skip.put(t.storage, Boolean.TRUE);
        }
        for (Tensor t : dead) {
            if (skip.put(t.storage, Boolean.TRUE) == null && !SHARED.containsKey(t.storage)) {
                t.backend().release(t.storage);
            }
        }
    }

    /**
     * Copies all values to a new Java array in row-major order.
     *
     * @return the values
     */
    public float[] toArray() {
        return backend().download(storage());
    }

    /**
     * Returns the only value of a tensor with one element.
     *
     * @return the value
     * @throws IllegalStateException if the tensor has more than one element
     */
    public float item() {
        if (size() != 1) {
            throw new IllegalStateException("item() needs a tensor with one value, got shape " + shape);
        }
        return toArray()[0];
    }

    // ---------------------------------------------------------------- shape changes

    /**
     * Returns a tensor with the same values in a new shape. No values are copied.
     *
     * @param dims the new dimensions; their product must equal {@link #size()}
     * @return the reshaped tensor
     */
    public Tensor reshape(int... dims) {
        return reshape(Shape.of(dims));
    }

    /**
     * Returns a tensor with the same values in a new shape. No values are copied.
     *
     * @param newShape the new shape; its size must equal {@link #size()}
     * @return the reshaped tensor
     */
    public Tensor reshape(Shape newShape) {
        if (newShape.size() != shape.size()) {
            throw new IllegalArgumentException("cannot reshape " + shape + " to " + newShape);
        }
        if (storage instanceof PackedStorage && newShape.equals(shape)) {
            return this;
        }
        return new Tensor(newShape, device, storage());
    }

    /**
     * Returns the values as a one-dimensional tensor.
     *
     * @return a tensor of shape {@code [size()]}
     */
    public Tensor flatten() {
        return reshape(Math.toIntExact(size()));
    }

    /**
     * Swaps the last two dimensions; for a matrix this is the transpose.
     *
     * @return the transposed tensor
     */
    public Tensor transpose() {
        if (rank() < 2) {
            throw new IllegalStateException("transpose() needs at least 2 dimensions, got shape " + shape);
        }
        int[] axes = new int[rank()];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i;
        }
        axes[rank() - 2] = rank() - 1;
        axes[rank() - 1] = rank() - 2;
        return permute(axes);
    }

    /**
     * Reorders the dimensions.
     *
     * @param axes for each output dimension, the input dimension it takes
     * @return the permuted tensor
     */
    public Tensor permute(int... axes) {
        if (axes.length != rank()) {
            throw new IllegalArgumentException("permute needs " + rank() + " axes, got " + Arrays.toString(axes));
        }
        int[] dims = shape.toArray();
        long[] strides = shape.strides();
        int[] outDims = new int[axes.length];
        long[] outStrides = new long[axes.length];
        boolean[] seen = new boolean[axes.length];
        for (int i = 0; i < axes.length; i++) {
            int a = axis(axes[i]);
            if (seen[a]) {
                throw new IllegalArgumentException("axis " + a + " repeated in " + Arrays.toString(axes));
            }
            seen[a] = true;
            outDims[i] = dims[a];
            outStrides[i] = strides[a];
        }
        return strided(outStrides, Shape.of(outDims));
    }

    /**
     * Repeats values along dimensions of size 1 so the tensor gets a larger shape, copying
     * the values. Binary operations broadcast automatically, so this is rarely needed.
     *
     * @param target a shape this tensor broadcasts to
     * @return the expanded tensor
     */
    public Tensor expand(Shape target) {
        if (!Shape.broadcast(shape, target).equals(target)) {
            throw new IllegalArgumentException("cannot expand " + shape + " to " + target);
        }
        return strided(shape.broadcastStrides(target), target);
    }

    /**
     * Sums values over broadcast dimensions so the result has {@code target} shape: the
     * reverse of broadcasting. Autograd uses this to give a broadcast input its gradient.
     *
     * @param target a shape that broadcasts to this tensor's shape
     * @return the reduced tensor
     */
    public Tensor sumTo(Shape target) {
        if (!Shape.broadcast(target, shape).equals(shape)) {
            throw new IllegalArgumentException("cannot sum " + shape + " down to " + target);
        }
        Tensor t = this;
        int extra = rank() - target.rank();
        for (int i = 0; i < extra; i++) {
            t = t.sum(0);
        }
        for (int i = 0; i < target.rank(); i++) {
            if (target.dim(i) == 1 && t.shape.dim(i) != 1) {
                t = t.sum(i, true);
            }
        }
        return t.reshape(target);
    }

    /**
     * Returns the part of this tensor from {@code start} (inclusive) to {@code end}
     * (exclusive) along one axis, copying the values. The slice must not be empty.
     *
     * @param axis the axis; negative counts from the end
     * @param start the first index to keep
     * @param end one past the last index to keep
     * @return the slice
     */
    public Tensor slice(int axis, int start, int end) {
        int a = axis(axis);
        int length = shape.dim(a);
        if (start < 0 || end > length || start >= end) {
            throw new IllegalArgumentException("slice [" + start + ", " + end + ") out of range for axis " + a
                    + " of " + shape);
        }
        int[] dims = shape.toArray();
        dims[a] = end - start;
        Shape outShape = Shape.of(dims);
        long[] strides = shape.strides();
        Storage out = backend().allocateUninitialized(outShape.size(), dtype());
        backend().copy(storage(), start * strides[a], strides, dims, out, 0, outShape.strides());
        return new Tensor(outShape, device, out);
    }

    /**
     * Joins tensors along one axis. All other dimensions must match, and all tensors must
     * be on the same device.
     *
     * @param axis the axis to join along; negative counts from the end
     * @param parts the tensors, at least one
     * @return the joined tensor
     */
    public static Tensor concat(int axis, Tensor... parts) {
        if (parts.length == 0) {
            throw new IllegalArgumentException("concat needs at least one tensor");
        }
        Tensor first = parts[0];
        int a = first.axis(axis);
        int[] dims = first.shape.toArray();
        int total = 0;
        for (Tensor t : parts) {
            first.sameDevice(t);
            int[] d = t.shape.toArray();
            boolean compatible = d.length == dims.length;
            for (int i = 0; compatible && i < d.length; i++) {
                compatible = i == a || d[i] == dims[i];
            }
            if (!compatible) {
                throw new IllegalArgumentException("cannot concat " + t.shape + " with " + first.shape + " along axis " + a);
            }
            total += d[a];
        }
        dims[a] = total;
        DType type = first.dtype();
        for (Tensor t : parts) {
            type = DType.promote(type, t.dtype());
        }
        Shape outShape = Shape.of(dims);
        long[] outStrides = outShape.strides();
        Storage out = first.backend().allocateUninitialized(outShape.size(), type);
        long offset = 0;
        for (Tensor part : parts) {
            Tensor t = part.to(type);
            first.backend().copy(t.storage(), 0, t.shape.strides(), t.shape.toArray(), out, offset, outStrides);
            offset += t.shape.dim(a) * outStrides[a];
        }
        return new Tensor(outShape, first.device, out);
    }

    private Tensor strided(long[] strides, Shape outShape) {
        Storage out = backend().allocateUninitialized(outShape.size(), dtype());
        backend().copy(storage(), strides, outShape.toArray(), out);
        return new Tensor(outShape, device, out);
    }

    // ---------------------------------------------------------------- element-wise

    /**
     * Adds element-wise with broadcasting.
     *
     * @param other the tensor to add
     * @return {@code this + other}
     */
    public Tensor add(Tensor other) {
        return binary(BinaryOp.ADD, other);
    }

    /**
     * Adds a constant to every value.
     *
     * @param value the constant
     * @return {@code this + value}
     */
    public Tensor add(float value) {
        return binary(BinaryOp.ADD, constant(value));
    }

    /**
     * Subtracts element-wise with broadcasting.
     *
     * @param other the tensor to subtract
     * @return {@code this - other}
     */
    public Tensor sub(Tensor other) {
        return binary(BinaryOp.SUB, other);
    }

    /**
     * Subtracts a constant from every value.
     *
     * @param value the constant
     * @return {@code this - value}
     */
    public Tensor sub(float value) {
        return binary(BinaryOp.SUB, constant(value));
    }

    /**
     * Multiplies element-wise with broadcasting (not matrix multiplication; see
     * {@link #matmul(Tensor)}).
     *
     * @param other the tensor to multiply by
     * @return {@code this * other}
     */
    public Tensor mul(Tensor other) {
        return binary(BinaryOp.MUL, other);
    }

    /**
     * Multiplies every value by a constant.
     *
     * @param value the constant
     * @return {@code this * value}
     */
    public Tensor mul(float value) {
        return binary(BinaryOp.MUL, constant(value));
    }

    /**
     * Divides element-wise with broadcasting.
     *
     * @param other the divisor
     * @return {@code this / other}
     */
    public Tensor div(Tensor other) {
        return binary(BinaryOp.DIV, other);
    }

    /**
     * Divides every value by a constant.
     *
     * @param value the divisor
     * @return {@code this / value}
     */
    public Tensor div(float value) {
        return binary(BinaryOp.DIV, constant(value));
    }

    /**
     * Raises every value to a power element-wise with broadcasting.
     *
     * @param exponent the exponents
     * @return {@code this ^ exponent}
     */
    public Tensor pow(Tensor exponent) {
        return binary(BinaryOp.POW, exponent);
    }

    /**
     * Raises every value to a constant power.
     *
     * @param exponent the exponent
     * @return {@code this ^ exponent}
     */
    public Tensor pow(float exponent) {
        return binary(BinaryOp.POW, constant(exponent));
    }

    /**
     * Returns the element-wise maximum with broadcasting.
     *
     * @param other the other tensor
     * @return {@code max(this, other)}
     */
    public Tensor maximum(Tensor other) {
        return binary(BinaryOp.MAX, other);
    }

    /**
     * Returns the element-wise minimum with broadcasting.
     *
     * @param other the other tensor
     * @return {@code min(this, other)}
     */
    public Tensor minimum(Tensor other) {
        return binary(BinaryOp.MIN, other);
    }

    /**
     * Compares element-wise with broadcasting.
     *
     * @param other the other tensor
     * @return {@code 1} where {@code this > other}, else {@code 0}
     */
    public Tensor greater(Tensor other) {
        return binary(BinaryOp.GREATER, other);
    }

    /**
     * Compares element-wise with broadcasting.
     *
     * @param other the other tensor
     * @return {@code 1} where {@code this == other}, else {@code 0}
     */
    public Tensor equalTo(Tensor other) {
        return binary(BinaryOp.EQUAL, other);
    }

    /**
     * Keeps values where {@code keep} is non-zero and sets the rest to {@code 0}, even
     * infinite or NaN ones (multiplying by a 0/1 mask would give NaN there).
     *
     * @param keep the mask; broadcasts
     * @return the masked values
     */
    public Tensor mask(Tensor keep) {
        return binary(BinaryOp.MASK, keep);
    }

    /**
     * Picks values from {@code ifTrue} where {@code condition} is non-zero and from
     * {@code ifFalse} elsewhere. Values that are not picked never leak into the result,
     * even if infinite or NaN.
     *
     * @param condition the condition, usually from {@link #greater(Tensor)}; broadcasts
     * @param ifTrue values where the condition holds
     * @param ifFalse values elsewhere
     * @return the picked values, in the broadcast shape
     */
    public static Tensor where(Tensor condition, Tensor ifTrue, Tensor ifFalse) {
        return ifTrue.mask(condition).add(ifFalse.mask(condition.equalTo(condition.constant(0f))));
    }

    /** @return {@code -this} */
    public Tensor neg() {
        return unary(UnaryOp.NEG);
    }

    /** @return the absolute values */
    public Tensor abs() {
        return unary(UnaryOp.ABS);
    }

    /** @return {@code e} raised to each value */
    public Tensor exp() {
        return unary(UnaryOp.EXP);
    }

    /** @return the natural logarithm of each value */
    public Tensor log() {
        return unary(UnaryOp.LOG);
    }

    /** @return the square root of each value */
    public Tensor sqrt() {
        return unary(UnaryOp.SQRT);
    }

    /** @return each value squared */
    public Tensor square() {
        return unary(UnaryOp.SQUARE);
    }

    /** @return the hyperbolic tangent of each value */
    public Tensor tanh() {
        return unary(UnaryOp.TANH);
    }

    /** @return the logistic sigmoid {@code 1 / (1 + e^-x)} of each value */
    public Tensor sigmoid() {
        return unary(UnaryOp.SIGMOID);
    }

    /** @return {@code max(0, x)} for each value */
    public Tensor relu() {
        return unary(UnaryOp.RELU);
    }

    /** @return {@code 1} where a value is positive, else {@code 0} */
    public Tensor step() {
        return unary(UnaryOp.STEP);
    }

    /** @return {@code -1}, {@code 0} or {@code 1} for each value */
    public Tensor sign() {
        return unary(UnaryOp.SIGN);
    }

    /** @return the Gauss error function of each value */
    public Tensor erf() {
        return unary(UnaryOp.ERF);
    }

    /** @return {@code log(1 + e^x)} for each value, without overflow */
    public Tensor softplus() {
        return unary(UnaryOp.SOFTPLUS);
    }

    /** @return each value rounded to the nearest whole number, ties to even */
    public Tensor round() {
        return unary(UnaryOp.ROUND);
    }

    /**
     * Applies an activation function in one pass, instead of as a chain of simpler
     * operations.
     *
     * @param op the activation
     * @param param its parameter (such as the leaky ReLU slope); ignored by activations
     *     without one
     * @return {@code f(this)}
     */
    public Tensor activation(ActivationOp op, float param) {
        Storage out = backend().allocateUninitialized(size(), dtype());
        backend().activation(op, param, storage(), out);
        return new Tensor(shape, device, out);
    }

    /**
     * Computes an activation's input gradient in one pass: {@code grad * f'(this)}.
     *
     * @param op the activation
     * @param param its parameter
     * @param grad the gradient of {@code f(this)}; same shape as this tensor
     * @return the gradient with respect to this tensor
     */
    public Tensor activationBackward(ActivationOp op, float param, Tensor grad) {
        sameDevice(grad);
        if (!grad.shape.equals(shape)) {
            throw new IllegalArgumentException("gradient shape " + grad.shape + " does not match " + shape);
        }
        grad = grad.to(dtype());
        Storage out = backend().allocateUninitialized(size(), dtype());
        backend().activationBackward(op, param, storage(), grad.storage(), out);
        return new Tensor(shape, device, out);
    }

    private Tensor unary(UnaryOp op) {
        Storage out = backend().allocateUninitialized(size(), dtype());
        backend().unary(op, storage(), out);
        return new Tensor(shape, device, out);
    }

    private Tensor binary(BinaryOp op, Tensor other) {
        sameDevice(other);
        Shape outShape = Shape.broadcast(shape, other.shape);
        DType type = resultType(this, other);
        Tensor a = to(type);
        Tensor b = other.to(type);
        Storage out = backend().allocateUninitialized(outShape.size(), type);
        backend().binary(op, a.storage(), shape.broadcastStrides(outShape),
                b.storage(), other.shape.broadcastStrides(outShape), outShape.toArray(), out);
        return new Tensor(outShape, device, out);
    }

    private Tensor constant(float value) {
        return full(Shape.of(), value, dtype(), device);
    }

    /** Promotes, except that a single value takes the other side's type. */
    private static DType resultType(Tensor a, Tensor b) {
        if (a.dtype() == b.dtype()) {
            return a.dtype();
        }
        if (b.size() == 1 && a.size() > 1) {
            return a.dtype();
        }
        if (a.size() == 1 && b.size() > 1) {
            return b.dtype();
        }
        return DType.promote(a.dtype(), b.dtype());
    }

    // ---------------------------------------------------------------- linear algebra

    /**
     * Multiplies matrices. Two matrices {@code [m, k] x [k, n]} give {@code [m, n]}. With
     * more dimensions the last two are the matrices and the leading ones are a batch,
     * broadcast like element-wise operations: {@code [b, m, k] x [k, n]} multiplies every
     * matrix of the batch by the same right matrix, and {@code [b, h, m, k] x [b, h, k, n]}
     * gives {@code [b, h, m, n]}.
     *
     * @param other the right operand
     * @return the product
     * @throws IllegalArgumentException if the inner dimensions do not match
     */
    public Tensor matmul(Tensor other) {
        sameDevice(other);
        if (other.dtype() != dtype()) {
            DType type = DType.promote(dtype(), other.dtype());
            return to(type).matmul(other.to(type));
        }
        if (rank() < 2 || other.rank() < 2) {
            throw new IllegalArgumentException(
                    "matmul needs at least two dimensions on both sides, got shapes " + shape + " and " + other.shape);
        }
        int m = shape.dim(-2);
        int k = shape.dim(-1);
        int n = other.shape.dim(-1);
        if (other.shape.dim(-2) != k) {
            throw new IllegalArgumentException(
                    "matmul shapes " + shape + " and " + other.shape + " do not fit: " + k + " != " + other.shape.dim(-2));
        }
        if (rank() == 2 && other.rank() == 2) {
            Shape outShape = Shape.of(m, n);
            Storage out = backend().allocateUninitialized(outShape.size(), dtype());
            if (other.storage instanceof PackedStorage packed) {
                backend().matmulPacked(storage(), packed, m, k, n, out);
            } else {
                backend().matmul(storage(), other.storage(), m, k, n, out);
            }
            return new Tensor(outShape, device, out);
        }
        if (other.rank() == 2) {
            // the same right matrix for the whole batch: one big matmul
            int[] dims = shape.toArray();
            Tensor flat = reshape(Math.toIntExact(size() / k), k).matmul(other);
            dims[dims.length - 1] = n;
            return flat.reshape(dims);
        }
        Shape batchA = Shape.of(Arrays.copyOf(shape.toArray(), rank() - 2));
        Shape batchB = Shape.of(Arrays.copyOf(other.shape.toArray(), other.rank() - 2));
        Shape batch = Shape.broadcast(batchA, batchB);
        Tensor a = withBatch(this, batch, m, k);
        Tensor b = withBatch(other, batch, k, n);
        int count = Math.toIntExact(batch.size());
        int[] outDims = Arrays.copyOf(batch.toArray(), batch.rank() + 2);
        outDims[batch.rank()] = m;
        outDims[batch.rank() + 1] = n;
        Shape outShape = Shape.of(outDims);
        Storage out = backend().allocateUninitialized(outShape.size(), dtype());
        backend().matmulBatched(a.storage(), b.storage(), count, m, k, n, out);
        return new Tensor(outShape, device, out);
    }

    private static Tensor withBatch(Tensor t, Shape batch, int rows, int cols) {
        int[] dims = Arrays.copyOf(batch.toArray(), batch.rank() + 2);
        dims[batch.rank()] = rows;
        dims[batch.rank() + 1] = cols;
        Shape target = Shape.of(dims);
        return t.shape.equals(target) ? t : t.expand(target);
    }

    // ---------------------------------------------------------------- sliding windows and indexing

    /**
     * Unrolls every window of a sliding-window geometry into a row: the building block
     * of convolution and pooling. This tensor holds the images,
     * {@code [batch, channels, image...]}.
     *
     * @param geometry the windows
     * @param padValue the value read from cells outside the image
     * @return {@code [batch * positions, channels * kernel]}, columns ordered by channel
     *     then window cell
     */
    public Tensor im2col(ConvGeometry geometry, float padValue) {
        if (!shape.equals(geometry.imageShape())) {
            throw new IllegalArgumentException("im2col expects images " + geometry.imageShape() + ", got " + shape);
        }
        Shape outShape = geometry.columnShape();
        Storage out = backend().allocateUninitialized(outShape.size(), dtype());
        backend().im2col(storage(), geometry, padValue, out);
        return new Tensor(outShape, device, out);
    }

    /**
     * Adds unrolled windows back into images: the adjoint of {@link #im2col}, used for
     * gradients and transposed convolution. Values from padding cells are dropped.
     *
     * @param geometry the windows
     * @return the images, {@code [batch, channels, image...]}
     */
    public Tensor col2im(ConvGeometry geometry) {
        if (!shape.equals(geometry.columnShape())) {
            throw new IllegalArgumentException("col2im expects columns " + geometry.columnShape() + ", got " + shape);
        }
        Shape outShape = geometry.imageShape();
        Storage out = backend().allocateUninitialized(outShape.size(), dtype());
        backend().col2im(storage(), geometry, out);
        return new Tensor(outShape, device, out);
    }

    /**
     * Whether {@link #attention} is worth using on the backend of {@code q} for these
     * arrays; where not, it may throw {@link UnsupportedOperationException}.
     *
     * @param q the queries
     * @param k the keys
     * @param v the values
     * @param backward whether {@link #attentionBackward} will be needed too
     * @return whether to call it
     */
    public static boolean hasAttention(Tensor q, Tensor k, Tensor v, boolean backward) {
        return q.backend().hasAttention(attentionShape(q, k, v), q.dtype(), backward);
    }

    /**
     * Scaled dot-product attention in one pass, without storing the {@code [tq, tk]}
     * weights: {@code softmax(scale * q k^T) v}, optionally causal (query {@code i} sees keys
     * up to {@code i + tk - tq}). Groups of query heads may share key/value heads.
     *
     * @param q queries {@code [batch, heads, tq, dk]}
     * @param k keys {@code [batch, kvHeads, tk, dk]}
     * @param v values {@code [batch, kvHeads, tk, dv]}
     * @param scale the score scale, usually {@code 1 / sqrt(dk)}
     * @param causal whether each query ignores later keys
     * @return the output {@code [batch, heads, tq, dv]} and each row's log-sum-exp
     *     {@code [batch, heads, tq]}, which {@link #attentionBackward} needs
     * @throws UnsupportedOperationException if the backend has no fused attention
     */
    public static Tensor[] attention(Tensor q, Tensor k, Tensor v, float scale, boolean causal) {
        int[] d = attentionShape(q, k, v);
        Backend backend = q.backend();
        Storage out = backend.allocateUninitialized((long) d[0] * d[1] * d[3] * d[6], q.dtype());
        Storage lse = backend.allocateUninitialized((long) d[0] * d[1] * d[3], q.dtype());
        backend.attention(q.storage(), k.storage(), v.storage(), d, scale, causal, out, lse);
        return new Tensor[] {new Tensor(Shape.of(d[0], d[1], d[3], d[6]), q.device, out),
                new Tensor(Shape.of(d[0], d[1], d[3]), q.device, lse)};
    }

    /**
     * The gradients of {@link #attention}.
     *
     * @param q the queries
     * @param k the keys
     * @param v the values
     * @param out the output of {@link #attention}
     * @param lse the log-sum-exp from {@link #attention}
     * @param grad the gradient of the output
     * @param scale the score scale
     * @param causal whether it was causal
     * @return the gradients of {@code q}, {@code k} and {@code v}
     */
    public static Tensor[] attentionBackward(Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor grad,
            float scale, boolean causal) {
        int[] d = attentionShape(q, k, v);
        Backend backend = q.backend();
        Storage dq = backend.allocateUninitialized(q.shape.size(), q.dtype());
        Storage dk = backend.allocateUninitialized(k.shape.size(), q.dtype());
        Storage dv = backend.allocateUninitialized(v.shape.size(), q.dtype());
        backend.attentionBackward(q.storage(), k.storage(), v.storage(), out.storage(), lse.storage(),
                grad.to(q.dtype()).storage(), d, scale, causal, dq, dk, dv);
        return new Tensor[] {new Tensor(q.shape, q.device, dq), new Tensor(k.shape, q.device, dk),
                new Tensor(v.shape, q.device, dv)};
    }

    private static int[] attentionShape(Tensor q, Tensor k, Tensor v) {
        if (q.shape.rank() != 4 || k.shape.rank() != 4 || v.shape.rank() != 4) {
            throw new IllegalArgumentException("attention takes [batch, heads, steps, size] arrays, got " + q.shape
                    + ", " + k.shape + ", " + v.shape);
        }
        q.sameDevice(k);
        q.sameDevice(v);
        int batch = q.shape.dim(0);
        int heads = q.shape.dim(1);
        int kvHeads = k.shape.dim(1);
        if (k.shape.dim(0) != batch || v.shape.dim(0) != batch || v.shape.dim(1) != kvHeads
                || heads % kvHeads != 0 || k.shape.dim(3) != q.shape.dim(3) || v.shape.dim(2) != k.shape.dim(2)
                || k.dtype() != q.dtype() || v.dtype() != q.dtype()) {
            throw new IllegalArgumentException("attention needs matching queries " + q.shape + ", keys " + k.shape
                    + " and values " + v.shape + " of one type");
        }
        return new int[] {batch, heads, kvHeads, q.shape.dim(2), k.shape.dim(2), q.shape.dim(3), v.shape.dim(3)};
    }

    /**
     * Picks entries along an axis by index, like a table lookup. The axis is replaced by
     * the dimensions of {@code indices}: a weight table {@code [vocab, dim]} gathered on
     * axis 0 with ids {@code [batch, time]} gives {@code [batch, time, dim]}.
     *
     * @param axis the axis to index; negative values count from the end
     * @param indices whole numbers in {@code [0, dim(axis))}, on the same device
     * @return the picked entries
     * @throws IllegalArgumentException if an index is out of range (checked on the CPU)
     */
    public Tensor gather(int axis, Tensor indices) {
        sameDevice(indices);
        int a = axis(axis);
        int[] dims = shape.toArray();
        long outer = 1;
        long inner = 1;
        for (int i = 0; i < a; i++) {
            outer *= dims[i];
        }
        for (int i = a + 1; i < dims.length; i++) {
            inner *= dims[i];
        }
        int count = Math.toIntExact(indices.size());
        int[] idx = indices.shape.toArray();
        int[] outDims = new int[dims.length - 1 + idx.length];
        System.arraycopy(dims, 0, outDims, 0, a);
        System.arraycopy(idx, 0, outDims, a, idx.length);
        System.arraycopy(dims, a + 1, outDims, a + idx.length, dims.length - a - 1);
        Shape outShape = Shape.of(outDims);
        Storage out = backend().allocateUninitialized(outShape.size(), dtype());
        backend().gather(storage(), outer, dims[a], inner, indices.storage(), count, out);
        return new Tensor(outShape, device, out);
    }

    /**
     * The adjoint of {@link #gather}: adds entries into a zero tensor at the given
     * indices along an axis; repeated indices add up. This tensor has the shape a gather
     * with {@code indices} would produce.
     *
     * @param axis the axis of the result that is indexed
     * @param indices the indices, as passed to {@code gather}
     * @param length the size of that axis in the result
     * @return the sums, with {@code indices}' dimensions replaced by {@code length}
     */
    public Tensor scatterAdd(int axis, Tensor indices, int length) {
        sameDevice(indices);
        int[] dims = shape.toArray();
        int idxRank = indices.rank();
        int outRank = dims.length - idxRank + 1;
        int a = axis < 0 ? outRank + axis : axis;
        if (a < 0 || a >= outRank || length < 1) {
            throw new IllegalArgumentException("bad scatterAdd axis " + axis + " or length " + length + " for " + shape);
        }
        for (int i = 0; i < idxRank; i++) {
            if (dims[a + i] != indices.shape.dim(i)) {
                throw new IllegalArgumentException("scatterAdd values " + shape + " do not fit indices "
                        + indices.shape + " at axis " + a);
            }
        }
        long outer = 1;
        long inner = 1;
        for (int i = 0; i < a; i++) {
            outer *= dims[i];
        }
        for (int i = a + idxRank; i < dims.length; i++) {
            inner *= dims[i];
        }
        int[] outDims = new int[outRank];
        System.arraycopy(dims, 0, outDims, 0, a);
        outDims[a] = length;
        System.arraycopy(dims, a + idxRank, outDims, a + 1, dims.length - a - idxRank);
        Shape outShape = Shape.of(outDims);
        Storage out = backend().allocate(outShape.size(), dtype());
        backend().scatterAdd(storage(), outer, Math.toIntExact(indices.size()), inner, indices.storage(), length, out);
        return new Tensor(outShape, device, out);
    }

    // ---------------------------------------------------------------- reductions

    /** @return the sum of all values, as a scalar tensor */
    public Tensor sum() {
        return reduceAll(ReduceOp.SUM);
    }

    /**
     * Sums along one axis, removing it.
     *
     * @param axis the axis; negative values count from the end
     * @return the sums
     */
    public Tensor sum(int axis) {
        return reduce(ReduceOp.SUM, axis, false);
    }

    /**
     * Sums along one axis.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the sums
     */
    public Tensor sum(int axis, boolean keepDims) {
        return reduce(ReduceOp.SUM, axis, keepDims);
    }

    /** @return the mean of all values, as a scalar tensor */
    public Tensor mean() {
        return reduceAll(ReduceOp.MEAN);
    }

    /**
     * Averages along one axis, removing it.
     *
     * @param axis the axis; negative values count from the end
     * @return the means
     */
    public Tensor mean(int axis) {
        return reduce(ReduceOp.MEAN, axis, false);
    }

    /**
     * Averages along one axis.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the means
     */
    public Tensor mean(int axis, boolean keepDims) {
        return reduce(ReduceOp.MEAN, axis, keepDims);
    }

    /** @return the largest value, as a scalar tensor */
    public Tensor max() {
        return reduceAll(ReduceOp.MAX);
    }

    /**
     * Takes the maximum along one axis.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the maxima
     */
    public Tensor max(int axis, boolean keepDims) {
        return reduce(ReduceOp.MAX, axis, keepDims);
    }

    /** @return the smallest value, as a scalar tensor */
    public Tensor min() {
        return reduceAll(ReduceOp.MIN);
    }

    /**
     * Takes the minimum along one axis.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the minima
     */
    public Tensor min(int axis, boolean keepDims) {
        return reduce(ReduceOp.MIN, axis, keepDims);
    }

    /**
     * Finds the index of the largest value along one axis, removing it. Ties pick the
     * first index.
     *
     * @param axis the axis; negative values count from the end
     * @return the indices, as floats
     */
    public Tensor argmax(int axis) {
        return reduce(ReduceOp.ARGMAX, axis, false);
    }

    private Tensor reduceAll(ReduceOp op) {
        Storage out = backend().allocateUninitialized(1, op == ReduceOp.ARGMAX ? DType.FLOAT32 : dtype());
        backend().reduce(op, storage(), 1, size(), 1, out);
        return new Tensor(Shape.of(), device, out);
    }

    private Tensor reduce(ReduceOp op, int axis, boolean keepDims) {
        int a = axis(axis);
        int[] dims = shape.toArray();
        long outer = 1;
        long inner = 1;
        for (int i = 0; i < a; i++) {
            outer *= dims[i];
        }
        for (int i = a + 1; i < dims.length; i++) {
            inner *= dims[i];
        }
        int[] outDims;
        if (keepDims) {
            outDims = dims.clone();
            outDims[a] = 1;
        } else {
            outDims = new int[dims.length - 1];
            System.arraycopy(dims, 0, outDims, 0, a);
            System.arraycopy(dims, a + 1, outDims, a, dims.length - a - 1);
        }
        Storage out = backend().allocateUninitialized(outer * inner, op == ReduceOp.ARGMAX ? DType.FLOAT32 : dtype());
        backend().reduce(op, storage(), outer, dims[a], inner, out);
        return new Tensor(Shape.of(outDims), device, out);
    }

    // ---------------------------------------------------------------- helpers

    private Backend backend() {
        return device.backend();
    }

    private int axis(int axis) {
        int a = axis < 0 ? rank() + axis : axis;
        if (a < 0 || a >= rank()) {
            throw new IllegalArgumentException("axis " + axis + " out of range for shape " + shape);
        }
        return a;
    }

    private void sameDevice(Tensor other) {
        if (!other.device.equals(device)) {
            throw new IllegalArgumentException("tensor is on " + device + ", argument is on " + other.device
                    + "; move one with to(device)");
        }
    }

    /**
     * Returns a short description with the shape, device and, for small tensors, the
     * values. Copies values from the device.
     *
     * @return the description
     */
    @Override
    public String toString() {
        String values = size() <= 16 ? " " + Arrays.toString(toArray()) : "";
        return "Tensor" + shape + "@" + device + values;
    }
}
