package io.github.zzuegg.brainiac.tensor;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.GgmlBlocks;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.RowQuantizedStorage;
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

    /** For a view such as a reshape, the tensor that owns the memory; {@code null} if this one does. */
    private final Tensor base;

    private Tensor(Shape shape, Device device, Storage storage) {
        this(shape, device, storage, null);
    }

    private Tensor(Shape shape, Device device, Storage storage, Tensor base) {
        this.shape = shape;
        this.device = device;
        this.storage = storage;
        this.base = base;
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
     * Creates a tensor of whole numbers, such as token ids or indices, exactly.
     *
     * @param shape the shape
     * @param values the values; their count must equal {@code shape.size()}
     * @param dtype an integer or boolean type, such as {@link DType#INT64}
     * @param device where the tensor lives; a device without integer types (a GPU) takes
     *     {@code to(DType.FLOAT32)} first
     * @return the tensor
     */
    public static Tensor of(Shape shape, long[] values, DType dtype, Device device) {
        Objects.requireNonNull(shape, "shape");
        if (values.length != shape.size()) {
            throw new IllegalArgumentException(
                    "shape " + shape + " needs " + shape.size() + " values, got " + values.length);
        }
        requireSupported(dtype, device);
        java.lang.foreign.MemorySegment raw = java.lang.foreign.MemorySegment.ofArray(new byte[values.length * 8]);
        for (int i = 0; i < values.length; i++) {
            raw.setAtIndex(io.github.zzuegg.brainiac.tensor.spi.RawValues.LONG, i, values[i]);
        }
        Storage s = device.backend().upload(raw, DType.INT64);
        Tensor t = new Tensor(shape, device, s);
        return dtype == DType.INT64 ? t : t.to(dtype);
    }

    /**
     * Creates a tensor of 32-bit integers.
     *
     * @param shape the shape
     * @param values the values; their count must equal {@code shape.size()}
     * @param device where the tensor lives
     * @return an {@link DType#INT32} tensor
     */
    public static Tensor ofInts(Shape shape, int[] values, Device device) {
        long[] l = new long[values.length];
        for (int i = 0; i < l.length; i++) {
            l[i] = values[i];
        }
        return of(shape, l, DType.INT32, device);
    }

    private static void requireSupported(DType dtype, Device device) {
        if (!device.backend().supports(dtype)) {
            throw new UnsupportedOperationException(device + " holds no " + dtype + " tensors; create them on the CPU"
                    + " and convert with to(DType.FLOAT32) to move them");
        }
    }

    /**
     * Creates a tensor from raw values laid out as {@code dtype} stores them, little-endian
     * (see {@link Backend#upload(java.lang.foreign.MemorySegment, DType)}). The values go
     * straight from {@code data} to the device, so a tensor read from a memory-mapped file
     * never passes through the Java heap on a backend that supports it.
     *
     * @param shape the shape
     * @param data {@code shape.size() * dtype.bytes()} bytes; only read during this call
     * @param dtype the type of the values, and of the tensor
     * @param device where the tensor lives
     * @return the tensor
     */
    public static Tensor of(Shape shape, java.lang.foreign.MemorySegment data, DType dtype, Device device) {
        Objects.requireNonNull(shape, "shape");
        if (data.byteSize() != shape.size() * dtype.bytes()) {
            throw new IllegalArgumentException("shape " + shape + " of " + dtype + " needs "
                    + shape.size() * dtype.bytes() + " bytes, got " + data.byteSize());
        }
        return new Tensor(shape, device, device.backend().upload(data, dtype));
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
     * An identity matrix of 32-bit floats on the default device.
     *
     * @param n its size
     * @return {@code [n, n]} with ones on the diagonal
     */
    public static Tensor eye(int n) {
        return Linalg.identity(n, DType.FLOAT32, Device.getDefault());
    }

    /**
     * An identity matrix.
     *
     * @param n its size
     * @param dtype its type
     * @param device its device
     * @return {@code [n, n]} with ones on the diagonal
     */
    public static Tensor eye(int n, DType dtype, Device device) {
        return Linalg.identity(n, dtype, device);
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
        if (WITHOUT_VALUES.get()[0] > 0) {
            return placeholder(shape, dtype, device);
        }
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
     * Creates a tensor of ones.
     *
     * @param shape the shape
     * @param device where it lives
     * @return the tensor
     */
    public static Tensor ones(Shape shape, Device device) {
        return full(shape, 1f, device);
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
        if (WITHOUT_VALUES.get()[0] > 0) {
            return placeholder(shape, dtype, device);
        }
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

    /** How many {@link #withoutValues} calls are running on this thread. */
    private static final ThreadLocal<int[]> WITHOUT_VALUES = ThreadLocal.withInitial(() -> new int[1]);

    /** One value per device and type that stands in for every tensor made by {@link #withoutValues}. */
    private static final Map<ScalarKey, Storage> PLACEHOLDERS = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Runs {@code body} with tensor creation ({@code zeros}, {@code full}, {@code ones},
     * {@code normal}, {@code uniform}) giving tensors that have a shape but no values: they
     * cost no memory and no random numbers. It is meant for building a model whose
     * parameters are all replaced right after, such as by pretrained weights; the values of
     * such a tensor must never be read.
     *
     * @param body the code to run
     * @param <T> what it returns
     * @return what {@code body} returned
     */
    public static <T> T withoutValues(java.util.function.Supplier<T> body) {
        int[] depth = WITHOUT_VALUES.get();
        depth[0]++;
        try {
            return body.get();
        } finally {
            depth[0]--;
        }
    }

    private static Tensor placeholder(Shape shape, DType dtype, Device device) {
        Storage s = PLACEHOLDERS.computeIfAbsent(new ScalarKey(device, dtype, 0), k -> {
            Storage made = device.backend().allocate(1, dtype);
            SHARED.put(made, Boolean.TRUE);
            return made;
        });
        return new Tensor(shape, device, s);
    }

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
        if (WITHOUT_VALUES.get()[0] > 0) {
            return placeholder(shape, DType.FLOAT32, Device.getDefault());
        }
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
        if (WITHOUT_VALUES.get()[0] > 0) {
            return placeholder(shape, DType.FLOAT32, Device.getDefault());
        }
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
        return packed(matrix, device, false);
    }

    /**
     * As {@link #packed(PackedMatrix, Device)}; {@code later} says the matrix may be joined
     * with others ({@link #concat} along columns) before it is used, so a GPU can leave it in
     * host memory until its first use or {@link #evaluate}, instead of uploading it at once.
     *
     * @param matrix the levels and scales
     * @param device where to keep it
     * @param later whether it may wait for its first use
     * @return the tensor
     */
    public static Tensor packed(PackedMatrix matrix, Device device, boolean later) {
        Backend b = device.backend();
        return new Tensor(Shape.of(matrix.rows(), matrix.cols()), device, later ? b.packLater(matrix) : b.pack(matrix));
    }

    /**
     * Wraps a weight matrix kept in a GGUF file's blocks (see {@link GgmlBlocks}) as a
     * {@linkplain #packed packed} {@code [in, out]} tensor, without copying it: on the CPU
     * the product reads the file's mapping directly.
     *
     * @param blocks the stored blocks
     * @param device where to keep it; on a GPU without a kernel for the blocks, products
     *     compute the values first
     * @return the tensor
     */
    public static Tensor packed(GgmlBlocks blocks, Device device) {
        return new Tensor(Shape.of(blocks.rows(), blocks.cols()), device, device.backend().packBlocks(blocks));
    }

    /**
     * Quantizes this {@code [rows, cols]} float32 matrix into a {@linkplain #packed packed} one
     * on the same device (see {@link PackedMatrix#quantize}): each column gets a scale for
     * every {@code group} rows. On a GPU the values stay on the device.
     *
     * @param bits the bits per level, 2 to 8
     * @param group the rows per scale; must divide the rows
     * @return the packed matrix
     */
    public Tensor quantized(int bits, int group) {
        if (shape.rank() != 2 || dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("quantizes a float32 matrix, not " + dtype() + " " + shape);
        }
        if (isPacked()) {
            throw new IllegalArgumentException("already packed");
        }
        return new Tensor(shape, device, backend().quantize(storage(), shape.dim(0), shape.dim(1), bits, group));
    }

    /** @return whether this tensor is a {@linkplain #packed packed} matrix or {@linkplain #quantizedRows quantized rows} */
    public boolean isPacked() {
        return storage instanceof PackedStorage;
    }

    /**
     * Creates zeros kept as 8-bit levels with a scale for every row (the last axis), as
     * {@link io.github.zzuegg.brainiac.tensor.spi.QuantizedRows} describes: half of
     * float16's memory, for a key/value cache. {@link #write} along the second-to-last axis
     * quantizes the rows written; fused attention reads the levels directly where the
     * backend has a kernel for it, and every other operation sees the values, of type
     * {@code dtype}, computed into a temporary tensor.
     *
     * @param shape the shape, of rank 2 or more
     * @param dtype the type the values are read as
     * @param device where to keep it
     * @return the tensor
     */
    public static Tensor quantizedRows(Shape shape, DType dtype, Device device) {
        if (shape.rank() < 2) {
            throw new IllegalArgumentException("quantized rows need at least two axes, got " + shape);
        }
        int cols = shape.dim(shape.rank() - 1);
        return new Tensor(shape, device, device.backend().allocateQuantizedRows(shape.size() / cols, cols, dtype));
    }

    /** @return whether this tensor is kept as {@linkplain #quantizedRows quantized rows} */
    public boolean isQuantizedRows() {
        return storage instanceof RowQuantizedStorage;
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
            Storage out = backend().allocateUninitialized(size(), dtype());
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
        if (storage instanceof RowQuantizedStorage) {
            Tensor t = quantizedRows(shape, dtype(), target);
            t.write(-2, 0, unpacked().to(target));
            return t;
        }
        if (storage instanceof GgmlBlocks blocks) {
            return new Tensor(shape, target, target.backend().packBlocks(blocks));
        }
        if (storage instanceof PackedStorage p) {
            return new Tensor(shape, target, target.backend().pack(backend().unpackToHost(p)));
        }
        if (dtype() == DType.FLOAT32) {
            return new Tensor(shape, target, target.backend().upload(toArray()));
        }
        requireSupported(dtype(), target);
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
     * Copies all values to a new Java array of whole numbers: exact for integer tensors up
     * to 2^53, and truncated toward zero for floating point ones.
     *
     * @return the values
     */
    public long[] toLongArray() {
        double[] d = toDoubleArray();
        long[] l = new long[d.length];
        for (int i = 0; i < l.length; i++) {
            l[i] = (long) d[i];
        }
        return l;
    }

    /**
     * {@link #toLongArray()} as ints.
     *
     * @return the values
     */
    public int[] toIntArray() {
        double[] d = toDoubleArray();
        int[] r = new int[d.length];
        for (int i = 0; i < r.length; i++) {
            r[i] = (int) d[i];
        }
        return r;
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
            if (t.storage instanceof PackedStorage p) {
                t.device.backend().prefetch(p);   // on its way to the device, if it waited
            } else {
                byDevice.computeIfAbsent(t.device, d -> new java.util.ArrayList<>()).add(t.storage);
            }
        }
        byDevice.forEach((d, storages) -> d.backend().evaluate(storages));
    }

    /**
     * Moves packed matrices that wait for their first use (see {@link #packed(PackedMatrix,
     * Device, boolean)}) to their devices now, on this thread, and returns once they are there,
     * keeping no host copy where the backend can: for copies used only on their device, such
     * as a GPU's cache of experts, since reading them back may then be unsupported. A thread
     * that makes many such copies so holds only a few in host memory at once.
     *
     * @param tensors the tensors, on any devices; others are left as they are
     */
    public static void prefetchNow(java.util.List<Tensor> tensors) {
        for (Tensor t : tensors) {
            if (t.storage instanceof PackedStorage p) {
                t.device.backend().prefetchNow(p);
            }
        }
    }

    /**
     * Sums tensors that live on different devices, such as the gradients of data-parallel
     * training, and gives each device the sum: element {@code i} of the result is the sum on
     * the device of {@code tensors.get(i)}. Devices that can exchange values directly (CUDA
     * GPUs with NCCL) do so; otherwise the values are summed on the host.
     *
     * @param tensors one tensor per device, all of one shape and type, on different devices
     * @return the sum on each tensor's device, in the same order
     */
    public static java.util.List<Tensor> sumAcross(java.util.List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            return java.util.List.of();
        }
        Tensor first = tensors.get(0);
        java.util.Set<Device> seen = new java.util.HashSet<>();
        for (Tensor t : tensors) {
            if (!t.shape.equals(first.shape) || t.dtype() != first.dtype()) {
                throw new IllegalArgumentException("sumAcross needs one shape and type: " + t.shape + " " + t.dtype()
                        + " and " + first.shape + " " + first.dtype());
            }
            if (!seen.add(t.device)) {
                throw new IllegalArgumentException("sumAcross needs different devices, " + t.device + " repeats");
            }
        }
        if (tensors.size() == 1) {
            return java.util.List.of(first);
        }
        java.util.List<Backend> group = new java.util.ArrayList<>();
        java.util.List<Storage> storages = new java.util.ArrayList<>();
        for (Tensor t : tensors) {
            group.add(t.backend());
            storages.add(t.storage());
        }
        java.util.List<Storage> sums = group.get(0).allReduceSum(group, storages);
        java.util.List<Tensor> out = new java.util.ArrayList<>();
        if (sums != null) {
            for (int i = 0; i < tensors.size(); i++) {
                out.add(new Tensor(first.shape, tensors.get(i).device, sums.get(i)));
            }
            return out;
        }
        Device cpu = Device.cpu();
        Tensor sum = first.to(cpu);
        for (int i = 1; i < tensors.size(); i++) {
            sum = sum.add(tensors.get(i).to(cpu));
        }
        for (Tensor t : tensors) {
            out.add(sum.to(t.device));
        }
        return out;
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
     * Releases what a computation made (from {@link #created}) except {@code keep}: like
     * {@link #release(java.util.Collection, java.util.Collection)}, but a view made in the
     * computation of a tensor made before it, such as a reshape of a weight, is left alone,
     * as its memory is not the computation's.
     *
     * @param made the tensors {@link #created} returned
     * @param keep tensors that stay in use, such as the result
     */
    public static void releaseCreated(java.util.List<Tensor> made, java.util.Collection<Tensor> keep) {
        Map<Tensor, Boolean> mine = new java.util.IdentityHashMap<>();
        for (Tensor t : made) {
            mine.put(t, Boolean.TRUE);
        }
        java.util.List<Tensor> dead = new java.util.ArrayList<>(made.size());
        for (Tensor t : made) {
            if (t.base == null || mine.containsKey(t.base)) {
                dead.add(t);
            }
        }
        release(dead, keep);
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
     * As {@link #toArray}, without waiting now: the device copies the values after the work
     * queued so far, and the supplier waits for that copy only. Meanwhile more work can be
     * queued, such as the next decoding step from tokens still on the device.
     *
     * @return the values, as {@link #toArray} would give them, when asked
     */
    public java.util.function.Supplier<float[]> toArrayLater() {
        return backend().downloadLater(storage());
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
     * @param dims the new dimensions; their product must equal {@link #size()}, and one
     *     may be {@code -1} to have it inferred
     * @return the reshaped tensor
     */
    public Tensor reshape(int... dims) {
        return reshape(Shape.inferred(shape.size(), dims));
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
        if (storage instanceof PackedStorage) {
            return newShape.equals(shape) ? this : new Tensor(newShape, device, storage());
        }
        return new Tensor(newShape, device, storage, base != null ? base : this);
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
     * be on the same device. {@linkplain #packed Packed} matrices of the same layout joined
     * along their columns stay packed.
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
        if (a == 1 && dims.length == 2) {
            Tensor joined = concatPacked(parts);
            if (joined != null) {
                return joined;
            }
        }
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

    /**
     * Joins {@linkplain #packed packed} matrices side by side into one packed matrix, as
     * {@code concat(1, parts)} does when it can, without ever computing their values: null if
     * they are not all packed with the same bits, group and level layout.
     *
     * @param parts {@code [rows, cols_i]} matrices on one device
     * @return the joined {@code [rows, sum of cols_i]} matrix, or null
     */
    public static Tensor concatPacked(Tensor... parts) {
        Tensor first = parts[0];
        PackedMatrix[] host = new PackedMatrix[parts.length];
        for (int i = 0; i < parts.length; i++) {
            Tensor t = parts[i];
            if (!(t.storage instanceof PackedStorage p) || t.isQuantizedRows() || t.shape.rank() != 2
                    || !t.device.equals(first.device) || t.shape.dim(0) != first.shape.dim(0)) {
                return null;
            }
            host[i] = t.backend().unpackToHost(p);
            if (host[i].bits() != host[0].bits() || host[i].group() != host[0].group()
                    || host[i].nibbles() != host[0].nibbles() || host[i].hasOffsets() != host[0].hasOffsets()) {
                return null;
            }
        }
        return packed(PackedMatrix.concatColumns(host), first.device);
    }

    /**
     * Overwrites part of this tensor in place: along {@code axis}, positions {@code start} to
     * {@code start + values.dim(axis)} take {@code values}. Tensors are otherwise never
     * changed; this is for buffers filled step by step, such as a key/value cache, so a
     * step writes only its new part instead of copying the whole buffer.
     *
     * <p>Results computed from this tensor must be computed first (see {@link #evaluate}),
     * or they may see the new values.
     *
     * @param axis the axis; negative counts from the end
     * @param start where the values go along it
     * @param values the new values: this tensor's shape except along {@code axis}; converted to its type
     */
    public void write(int axis, int start, Tensor values) {
        sameDevice(values);
        int a = axis(axis);
        if (isPacked() && !(storage instanceof RowQuantizedStorage && a == shape.rank() - 2)) {
            throw new UnsupportedOperationException(isQuantizedRows() ? "quantized rows are written along axis -2"
                    : "cannot write into a packed matrix");
        }
        int[] dims = shape.toArray();
        int[] v = values.shape.toArray();
        boolean fits = v.length == dims.length && start >= 0 && start + v[a] <= dims[a];
        for (int i = 0; fits && i < v.length; i++) {
            fits = i == a || v[i] == dims[i];
        }
        if (!fits) {
            throw new IllegalArgumentException("cannot write " + values.shape + " at " + start + " along axis " + a
                    + " of " + shape);
        }
        if (storage instanceof RowQuantizedStorage rows) {
            long outer = size() / ((long) dims[a] * dims[a + 1]);
            backend().writeQuantizedRows(rows, values.unpacked().storage, outer, dims[a], v[a], start);
            return;
        }
        Tensor t = values.to(dtype());
        long[] strides = shape.strides();
        backend().copy(t.storage(), 0, t.shape.strides(), v, storage, start * strides[a], strides);
    }

    /**
     * Reverses the order along one axis. A GPU reads it in place as part of what uses it.
     *
     * @param axis the axis; negative counts from the end
     * @return the reversed tensor
     */
    public Tensor flip(int axis) {
        int a = axis(axis);
        long[] strides = shape.strides();
        long offset = (shape.dim(a) - 1) * strides[a];
        strides[a] = -strides[a];
        Storage out = backend().allocateUninitialized(size(), dtype());
        backend().copy(storage(), offset, strides, shape.toArray(), out, 0, shape.strides());
        return new Tensor(shape, device, out);
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
     * @param c the lower bound
     * @return {@code max(this, c)} element-wise
     */
    public Tensor maximum(float c) {
        return maximum(full(Shape.of(), c, dtype(), device()));
    }

    /**
     * @param c the upper bound
     * @return {@code min(this, c)} element-wise
     */
    public Tensor minimum(float c) {
        return minimum(full(Shape.of(), c, dtype(), device()));
    }

    /**
     * @param low the lower bound
     * @param high the upper bound, at least {@code low}
     * @return each value limited to {@code [low, high]}
     */
    public Tensor clamp(float low, float high) {
        if (!(low <= high)) {
            throw new IllegalArgumentException("clamp needs low <= high, got " + low + " and " + high);
        }
        return maximum(low).minimum(high);
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

    /** @return the sine of each value, in radians */
    public Tensor sin() {
        return unary(UnaryOp.SIN);
    }

    /** @return the cosine of each value, in radians */
    public Tensor cos() {
        return unary(UnaryOp.COS);
    }

    /** @return the arctangent of each value, in radians, from {@code -pi/2} to {@code pi/2} */
    public Tensor atan() {
        return unary(UnaryOp.ATAN);
    }

    /**
     * The angle of the point {@code (x, this)}, from {@code -pi} to {@code pi}, as
     * {@link Math#atan2}: with this tensor as {@code y}, broadcast together.
     *
     * @param x the other coordinate
     * @return the angles in radians; {@code (0, 0)} gives 0
     */
    public Tensor atan2(Tensor x) {
        Tensor zero = x.equalTo(Tensor.zeros(Shape.of(), x.dtype(), x.device()));
        Tensor base = div(x.add(zero)).atan().mul(zero.neg().add(1));
        // x < 0: half a turn towards y's side (y = 0 counts as positive)
        Tensor turn = x.neg().step().mul(neg().step().mul(-2).add(1)).mul((float) Math.PI);
        // x = 0: a quarter turn by y's sign
        Tensor quarter = zero.mul(sign()).mul((float) (Math.PI / 2));
        return base.add(turn).add(quarter);
    }

    /**
     * Compares element-wise with broadcasting.
     *
     * @param other the values to compare with
     * @return 1 where this value is less than the other's, else 0
     */
    public Tensor less(Tensor other) {
        return other.greater(this);
    }

    /** @return a copy of this tensor's values in new memory, on the same device */
    public Tensor copy() {
        if (rank() == 0) {
            return reshape(1).copy().reshape(shape);
        }
        Tensor out = new Tensor(shape, device, backend().allocateUninitialized(size(), dtype()));
        out.write(0, 0, this);
        return out;
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
        // packed weights multiply in the type of what they multiply: their levels have none
        boolean packedRight = other.storage instanceof PackedStorage && rank() >= 2 && other.rank() == 2;
        if (other.dtype() != dtype() && !packedRight) {
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
            if (other.storage instanceof PackedStorage packed && !(packed instanceof RowQuantizedStorage)) {
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
     * The discrete Fourier transform along the second-to-last axis, of complex values stored as
     * (real, imaginary) pairs in a last axis of size 2:
     * {@code X[k] = sum_j x[j] e^(-2 pi i jk / n)}. Any length; powers of two are fastest.
     *
     * @param inverse whether to compute the inverse, {@code x[j] = 1/n sum_k X[k] e^(2 pi i jk / n)}
     * @return the transform, of this shape and type
     */
    public Tensor fft(boolean inverse) {
        int rank = shape.rank();
        if (rank < 2 || shape.dim(rank - 1) != 2) {
            throw new IllegalArgumentException("fft needs complex values [..., n, 2], got " + shape);
        }
        int n = shape.dim(rank - 2);
        if (size() == 0) {
            return this;
        }
        Tensor out = new Tensor(shape, device, backend().fft(storage(), size() / (2L * n), n, inverse));
        return out.dtype() == dtype() ? out : out.to(dtype());
    }

    /**
     * The two-dimensional discrete Fourier transform of complex values {@code [..., h, w, 2]}:
     * {@link #fft} along the columns, then along the rows, as {@code numpy.fft.fft2}.
     *
     * @param inverse whether to compute the inverse
     * @return the transform, of this shape and type
     */
    public Tensor fft2(boolean inverse) {
        int r = shape.rank();
        if (r < 3 || shape.dim(r - 1) != 2) {
            throw new IllegalArgumentException("fft2 takes complex values [..., h, w, 2], not " + shape);
        }
        int[] swap = new int[r];
        for (int i = 0; i < r; i++) {
            swap[i] = i;
        }
        swap[r - 3] = r - 2;
        swap[r - 2] = r - 3;
        return fft(inverse).permute(swap).fft(inverse).permute(swap);
    }

    /**
     * The transform of real values along the last axis, keeping the {@code n / 2 + 1}
     * frequencies that determine the rest.
     *
     * @return complex values {@code [..., n / 2 + 1, 2]}
     */
    public Tensor rfft() {
        int rank = shape.rank();
        int n = shape.dim(rank - 1);
        int[] dims = java.util.Arrays.copyOf(shape.toArray(), rank + 1);
        dims[rank] = 1;
        Tensor re = reshape(dims);
        Tensor complex = concat(rank, re, zeros(Shape.of(dims), dtype(), device));
        return complex.fft(false).slice(rank - 1, 0, n / 2 + 1);
    }

    /**
     * The inverse of {@link #rfft}: real values of length {@code n} from their
     * {@code n / 2 + 1} lowest frequencies, the others being their complex conjugates.
     *
     * @param n the length of the result
     * @return real values {@code [..., n]}
     */
    public Tensor irfft(int n) {
        return fullSpectrum(this, n).fft(true)
                .slice(shape.rank() - 1, 0, 1).reshape(realShape(shape, n));
    }

    /** Indices of the full spectrum's frequencies in the half one: k, then n - k mirrored. */
    static int[] mirrored(int n) {
        int[] idx = new int[n];
        for (int k = 0; k < n; k++) {
            idx[k] = k <= n / 2 ? k : n - k;
        }
        return idx;
    }

    /** +1 for frequencies kept, -1 (conjugated imaginary parts) for the mirrored ones: [n, 2]. */
    static float[] conjugation(int n) {
        float[] c = new float[2 * n];
        for (int k = 0; k < n; k++) {
            c[2 * k] = 1;
            c[2 * k + 1] = k <= n / 2 ? 1 : -1;
        }
        return c;
    }

    private static Tensor fullSpectrum(Tensor half, int n) {
        int rank = half.shape.rank();
        if (rank < 2 || half.shape.dim(rank - 1) != 2 || half.shape.dim(rank - 2) != n / 2 + 1) {
            throw new IllegalArgumentException("irfft of length " + n + " needs [..., " + (n / 2 + 1) + ", 2], got "
                    + half.shape);
        }
        int[] idx = mirrored(n);
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = idx[i];
        }
        Tensor indices = Tensor.of(Shape.of(n), f, half.device);
        return half.gather(rank - 2, indices).mul(Tensor.of(Shape.of(n, 2), conjugation(n), half.device)
                .to(half.dtype()));
    }

    private static int[] realShape(Shape complex, int n) {
        int rank = complex.rank();
        int[] dims = java.util.Arrays.copyOf(complex.toArray(), rank - 1);
        dims[rank - 2] = n;
        return dims;
    }

    /**
     * Whether {@link #rmsNorm} runs as one kernel on this tensor's backend.
     *
     * @return whether to call it
     */
    public boolean hasRmsNorm() {
        return backend().hasRmsNorm(dtype());
    }

    /**
     * RMSNorm over the last axis in one kernel, for inference:
     * {@code (x / sqrt(mean(x^2) + eps)) * weight}, computed in float32 and rounded to this
     * tensor's type before the scale, as the unfused operations do.
     *
     * @param weight the scale {@code [last]}, converted to this tensor's type if needed
     * @param eps added to the mean square
     * @return the normalized tensor, of this shape and type
     * @throws UnsupportedOperationException if the backend has no such kernel
     */
    public Tensor rmsNorm(Tensor weight, float eps) {
        sameDevice(weight);
        int cols = shape.dim(shape.rank() - 1);
        if (weight.size() != cols) {
            throw new IllegalArgumentException("weight " + weight.shape + " for rows of " + cols);
        }
        Tensor w = weight.dtype() == dtype() ? weight : weight.to(dtype());
        Storage out = backend().allocateUninitialized(size(), dtype());
        backend().rmsNorm(storage(), w.storage(), size() / cols, cols, eps, out);
        return new Tensor(shape, device, out);
    }

    /**
     * A residual add and the RMSNorm after it, for inference: {@code this + r} and
     * {@code rmsNorm(this + r, weight, eps)}, in one kernel where the backend has one.
     * Requires {@link #hasRmsNorm}.
     *
     * @param r added to this tensor: the same shape and type
     * @param weight the scale {@code [last]}
     * @param eps added to the mean square
     * @return the sum and the normalized sum
     */
    public Tensor[] addRmsNorm(Tensor r, Tensor weight, float eps) {
        sameDevice(r);
        sameDevice(weight);
        int cols = shape.dim(shape.rank() - 1);
        if (!backend().hasAddRmsNorm(dtype()) || !r.shape.equals(shape) || r.dtype() != dtype()
                || weight.size() != cols) {
            Tensor sum = add(r);
            return new Tensor[] {sum, sum.rmsNorm(weight, eps)};
        }
        Tensor w = weight.dtype() == dtype() ? weight : weight.to(dtype());
        Storage sum = backend().allocateUninitialized(size(), dtype());
        Storage out = backend().allocateUninitialized(size(), dtype());
        backend().addRmsNorm(storage(), r.storage(), w.storage(), size() / cols, cols, eps, sum, out);
        return new Tensor[] {new Tensor(shape, device, sum), new Tensor(shape, device, out)};
    }

    /**
     * Whether {@link #attention(Tensor, Tensor, Tensor, Tensor, float, boolean)} runs fused on
     * the backend of {@code q} for these arrays.
     *
     * @param q the queries
     * @param k the keys
     * @param v the values
     * @return whether to call it
     */
    public static boolean hasAttentionWithBias(Tensor q, Tensor k, Tensor v) {
        int[] d = attentionShape(q, k, v);
        return k.isQuantizedRows() && v.isQuantizedRows() && q.backend().hasQuantizedAttention(d, q.dtype())
                || q.backend().hasAttentionWithBias(d, q.dtype());
    }

    /**
     * Fused attention for inference with a bias added to the scaled scores:
     * {@code softmax(scale * q k^T + bias) v}, the bias broadcast to
     * {@code [batch, heads, tq, tk]} as for {@link #add}.
     *
     * @param q queries {@code [batch, heads, tq, dk]}
     * @param k keys {@code [batch, kvHeads, tk, dk]}
     * @param v values {@code [batch, kvHeads, tk, dv]}
     * @param bias broadcast to {@code [batch, heads, tq, tk]}
     * @param scale the score scale
     * @param causal whether each query also ignores later keys
     * @return the output {@code [batch, heads, tq, dv]} and each row's log-sum-exp
     * @throws UnsupportedOperationException if the backend has no such kernel
     */
    public static Tensor[] attention(Tensor q, Tensor k, Tensor v, Tensor bias, float scale, boolean causal) {
        int[] d = attentionShape(q, k, v);
        q.sameDevice(bias);
        int rank = bias.shape.rank();
        if (rank > 4) {
            throw new IllegalArgumentException("an attention bias has at most 4 axes, got " + bias.shape);
        }
        int[] full = {d[0], d[1], d[3], d[4]};
        long[] strides = new long[4];
        long stride = 1;
        for (int a = 3; a >= 0; a--) {
            int axis = a - (4 - rank);
            int size = axis < 0 ? 1 : bias.shape.dim(axis);
            if (size != 1 && size != full[a]) {
                throw new IllegalArgumentException("bias " + bias.shape + " does not broadcast to "
                        + Shape.of(full));
            }
            strides[a] = size == 1 ? 0 : stride;
            stride *= size;
        }
        Tensor b = bias.dtype() == q.dtype() ? bias : bias.to(q.dtype());
        Backend backend = q.backend();
        Storage out = backend.allocateUninitialized((long) d[0] * d[1] * d[3] * d[6], q.dtype());
        Storage lse = backend.allocateUninitialized((long) d[0] * d[1] * d[3], q.dtype());
        if (k.storage instanceof RowQuantizedStorage qk && v.storage instanceof RowQuantizedStorage qv
                && backend.hasQuantizedAttention(d, q.dtype())) {
            backend.attentionQuantized(q.storage(), qk, qv, d, scale, causal, b.storage(), strides, out, lse);
        } else {
            backend.attention(q.storage(), k.storage(), v.storage(), d, scale, causal, b.storage(), strides, out, lse);
        }
        return new Tensor[] {new Tensor(Shape.of(d[0], d[1], d[3], d[6]), q.device, out),
                new Tensor(Shape.of(d[0], d[1], d[3]), q.device, lse)};
    }

    /**
     * Whether {@link #latentAttention} runs fused on the backend of {@code qLatent} for these
     * arrays.
     *
     * @param qLatent the queries' latent part {@code [batch, heads, tq, latent]}
     * @param qRope the queries' rotary part {@code [batch, heads, tq, rope]}
     * @param c the cached latents {@code [batch, 1, tk, latent]}
     * @param r the cached rotary keys {@code [batch, 1, tk, rope]}
     * @return whether to call it
     */
    public static boolean hasLatentAttention(Tensor qLatent, Tensor qRope, Tensor c, Tensor r) {
        return qLatent.backend().hasLatentAttention(latentShape(qLatent, qRope, c, r), qLatent.dtype());
    }

    /**
     * DeepSeek's multi-head latent attention over a cache of latents, for inference: every
     * query head attends to the same keys and values, the cached latents {@code c} (with the
     * shared rotary keys {@code r} adding to the scores), so the output stays in latent space:
     * {@code softmax(scale (qLatent c^T + qRope r^T) + bias) c}. The bias broadcasts to
     * {@code [batch, heads, tq, tk]} as for {@link #add}.
     *
     * @param qLatent the queries' latent part {@code [batch, heads, tq, latent]}
     * @param qRope the queries' rotary part {@code [batch, heads, tq, rope]}
     * @param c the cached latents {@code [batch, 1, tk, latent]}
     * @param r the cached rotary keys {@code [batch, 1, tk, rope]}
     * @param bias the score bias, such as a mask
     * @param scale the score scale
     * @return {@code [batch, heads, tq, latent]}
     * @throws UnsupportedOperationException if the backend has no such kernel
     */
    public static Tensor latentAttention(Tensor qLatent, Tensor qRope, Tensor c, Tensor r, Tensor bias, float scale) {
        int[] d = latentShape(qLatent, qRope, c, r);
        qLatent.sameDevice(bias);
        int rank = bias.shape.rank();
        if (rank > 4) {
            throw new IllegalArgumentException("an attention bias has at most 4 axes, got " + bias.shape);
        }
        int[] full = {d[0], d[1], d[2], d[3]};
        long[] strides = new long[4];
        long stride = 1;
        for (int a = 3; a >= 0; a--) {
            int axis = a - (4 - rank);
            int size = axis < 0 ? 1 : bias.shape.dim(axis);
            if (size != 1 && size != full[a]) {
                throw new IllegalArgumentException("bias " + bias.shape + " does not broadcast to " + Shape.of(full));
            }
            strides[a] = size == 1 ? 0 : stride;
            stride *= size;
        }
        Tensor b = bias.dtype() == qLatent.dtype() ? bias : bias.to(qLatent.dtype());
        Backend backend = qLatent.backend();
        Storage out = backend.allocateUninitialized((long) d[0] * d[1] * d[2] * d[4], qLatent.dtype());
        backend.latentAttention(qLatent.storage(), qRope.storage(), c.storage(), r.storage(), d, scale, b.storage(),
                strides, out);
        return new Tensor(Shape.of(d[0], d[1], d[2], d[4]), qLatent.device, out);
    }

    /** {@code batch, heads, tq, tk, latent, rope} of a latent attention, checked. */
    private static int[] latentShape(Tensor qLatent, Tensor qRope, Tensor c, Tensor r) {
        if (qLatent.shape.rank() != 4 || qRope.shape.rank() != 4 || c.shape.rank() != 4 || r.shape.rank() != 4) {
            throw new IllegalArgumentException("latent attention takes [batch, heads, steps, size] arrays");
        }
        qLatent.sameDevice(qRope);
        qLatent.sameDevice(c);
        qLatent.sameDevice(r);
        int batch = qLatent.shape.dim(0);
        int heads = qLatent.shape.dim(1);
        int tq = qLatent.shape.dim(2);
        int tk = c.shape.dim(2);
        int latent = qLatent.shape.dim(3);
        int rope = qRope.shape.dim(3);
        if (!qRope.shape.equals(Shape.of(batch, heads, tq, rope)) || !c.shape.equals(Shape.of(batch, 1, tk, latent))
                || !r.shape.equals(Shape.of(batch, 1, tk, rope)) || qRope.dtype() != qLatent.dtype()
                || c.dtype() != qLatent.dtype() || r.dtype() != qLatent.dtype()) {
            throw new IllegalArgumentException("latent attention needs matching queries " + qLatent.shape + " and "
                    + qRope.shape + ", latents " + c.shape + " and rotary keys " + r.shape + " of one type");
        }
        return new int[] {batch, heads, tq, tk, latent, rope};
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
     * Picks entries of each row by its own indices: for {@code [..., n]} values and
     * {@code [..., k]} indices with the same leading dimensions, entry {@code j} of a row is
     * the row's value at its index {@code j}. For the log-probability of each row's chosen
     * action or token.
     *
     * @param indices whole numbers in {@code [0, n)}, {@code [..., k]}
     * @return {@code [..., k]}
     */
    public Tensor takeAlong(Tensor indices) {
        return reshape(Math.toIntExact(size())).gather(0, rowIndices(shape, indices)).reshape(indices.shape());
    }

    /**
     * Where {@link #takeAlong} reads in the flattened values: each index plus its row's
     * start. Also used by the autograd version.
     *
     * @param values the shape of the values {@code [..., n]}
     * @param indices {@code [..., k]}
     * @return flat indices {@code [rows * k]}
     */
    public static Tensor rowIndices(Shape values, Tensor indices) {
        Shape is = indices.shape();
        if (is.rank() != values.rank()) {
            throw new IllegalArgumentException("indices " + is + " do not fit values " + values);
        }
        for (int i = 0; i < is.rank() - 1; i++) {
            if (is.dim(i) != values.dim(i)) {
                throw new IllegalArgumentException("indices " + is + " do not fit values " + values);
            }
        }
        int n = values.dim(-1);
        int k = is.dim(-1);
        int rows = Math.toIntExact(values.size() / n);
        if (values.size() > 1 << 24) {
            throw new IllegalArgumentException("takeAlong supports up to 2^24 values, got " + values.size());
        }
        float[] starts = new float[rows];
        for (int r = 0; r < rows; r++) {
            starts[r] = (float) r * n;
        }
        Tensor start = of(Shape.of(rows, 1), starts).to(indices.device()).to(indices.dtype());
        return indices.reshape(rows, k).add(start).reshape(rows * k);
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
     * {@code log(sum(exp(x)))} along one axis, computed stably by shifting by the maximum.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the log of the summed exponentials
     */
    public Tensor logsumexp(int axis, boolean keepDims) {
        Tensor m = max(axis, true);
        Tensor s = sub(m).exp().sum(axis, keepDims).log();
        return s.add(m.reshape(s.shape()));
    }

    /**
     * The variance along one axis, {@code sum((x - mean)^2) / (n - correction)}, as
     * {@code torch.var}: correction 1 is the sample variance, 0 the population variance.
     *
     * @param axis the axis; negative values count from the end
     * @param correction subtracted from the count, usually 0 or 1
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the variances
     */
    public Tensor variance(int axis, int correction, boolean keepDims) {
        float n = shape.dim(axis) - correction;
        return sub(mean(axis, true)).square().sum(axis, keepDims).div(n);
    }

    /**
     * The standard deviation along one axis, the square root of {@link #variance}.
     *
     * @param axis the axis; negative values count from the end
     * @param correction subtracted from the count, usually 0 or 1
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the standard deviations
     */
    public Tensor std(int axis, int correction, boolean keepDims) {
        return variance(axis, correction, keepDims).sqrt();
    }

    /**
     * Whether any value along one axis is non-zero.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return 1 where some value is non-zero, else 0
     */
    public Tensor any(int axis, boolean keepDims) {
        return abs().max(axis, keepDims).sign();
    }

    /**
     * Whether every value along one axis is non-zero.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return 1 where all values are non-zero, else 0
     */
    public Tensor all(int axis, boolean keepDims) {
        return abs().min(axis, keepDims).sign();
    }

    /**
     * Repeats each entry along an axis, as {@code torch.repeat_interleave}:
     * {@code [a, b]} repeated twice is {@code [a, a, b, b]}.
     *
     * @param repeats how often each entry appears, at least 1
     * @param axis the axis; negative values count from the end
     * @return the tensor with that axis {@code repeats} times longer
     */
    public Tensor repeatInterleave(int repeats, int axis) {
        return gather(axis, repeatIndices(shape.dim(axis), repeats, device()));
    }

    /**
     * The indices {@code [0, 0, 1, 1, ...]} that {@link #repeatInterleave} gathers.
     *
     * @param n the axis length
     * @param repeats how often each index appears
     * @param device where the indices live
     * @return {@code n * repeats} indices
     */
    public static Tensor repeatIndices(int n, int repeats, Device device) {
        if (repeats < 1) {
            throw new IllegalArgumentException("repeats must be at least 1, got " + repeats);
        }
        float[] idx = new float[n * repeats];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i / repeats;
        }
        return of(Shape.of(idx.length), idx).to(device);
    }

    /**
     * The {@code k} largest values along the last axis and where they are, largest first;
     * ties pick the first index. Computed as {@code k} rounds of {@link #argmax}, so meant
     * for small {@code k}.
     *
     * @param k how many, at most the last axis' length
     * @return the values and their indices (as floats), each of this shape with the last
     *     axis {@code k} long
     */
    public Tensor[] topk(int k) {
        int last = shape.rank() - 1;
        int n = shape.dim(last);
        if (k < 1 || k > n) {
            throw new IllegalArgumentException("top " + k + " of " + n);
        }
        int[] keep = shape.toArray();
        keep[last] = 1;
        Tensor positions = arange(n).to(device);
        Tensor work = this;
        Tensor[] values = new Tensor[k];
        Tensor[] indices = new Tensor[k];
        for (int i = 0; i < k; i++) {
            indices[i] = work.argmax(last).reshape(keep);
            Tensor picked = indices[i].equalTo(positions).to(dtype());
            values[i] = mul(picked).sum(last, true);
            // the picked entry falls to the lowest finite value
            work = work.mul(picked.neg().add(1)).add(picked.mul(-Float.MAX_VALUE));
        }
        return new Tensor[] {concat(last, values), concat(last, indices)};
    }

    /**
     * Runs a user {@link Kernel}: its shader on Vulkan, its Java version elsewhere.
     *
     * @param kernel the kernel
     * @param shape the output's shape
     * @param dtype the output's type, which the inputs are converted to
     * @param ints up to {@link Kernel#MAX_INTS} integer parameters
     * @param floats up to {@link Kernel#MAX_FLOATS} float parameters
     * @param inputs the inputs, as many as the kernel reads, on one device
     * @return the output, on the inputs' device (the default device when there are none)
     */
    public static Tensor custom(Kernel kernel, Shape shape, DType dtype, int[] ints, float[] floats,
            Tensor... inputs) {
        if (inputs.length != kernel.inputs()) {
            throw new IllegalArgumentException(kernel + " takes " + kernel.inputs() + " inputs, got " + inputs.length);
        }
        if (ints.length > Kernel.MAX_INTS || floats.length > Kernel.MAX_FLOATS) {
            throw new IllegalArgumentException("at most " + Kernel.MAX_INTS + " ints and " + Kernel.MAX_FLOATS
                    + " floats, got " + ints.length + " and " + floats.length);
        }
        Device device = inputs.length > 0 ? inputs[0].device : Device.getDefault();
        Storage[] in = new Storage[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            inputs[0].sameDevice(inputs[i]);
            in[i] = inputs[i].storage();
        }
        Backend backend = device.backend();
        Storage out = backend.allocateUninitialized(shape.size(), dtype);
        if (shape.size() > 0) {
            backend.custom(kernel, in, out, ints, floats);
        }
        return new Tensor(shape, device, out);
    }

    /**
     * Sorts along one axis, stably: equal values keep their order, and NaN counts as larger
     * than every number (last ascending, first descending).
     *
     * @param axis the axis; negative values count from the end
     * @param descending largest first instead of smallest first
     * @return the sorted values and where each was along the axis (as floats), both of this
     *     shape
     */
    public Tensor[] sort(int axis, boolean descending) {
        if (rank() == 0) {
            Tensor[] one = reshape(1).sort(0, descending);
            return new Tensor[] {one[0].reshape(shape), one[1].reshape(shape)};
        }
        int a = axis(axis);
        int last = rank() - 1;
        int[] swap = new int[rank()];
        for (int i = 0; i < swap.length; i++) {
            swap[i] = i;
        }
        swap[a] = last;
        swap[last] = a;
        Tensor x = a == last ? this : permute(swap);
        int n = x.shape.dim(last);
        if (n > 1 << 24) {
            throw new IllegalArgumentException("sort supports axes of up to 2^24 values, got " + n);
        }
        Storage values = backend().allocateUninitialized(x.size(), dtype());
        Storage indices = backend().allocateUninitialized(x.size(), DType.FLOAT32);
        if (n > 0 && x.size() > 0) {
            backend().sort(x.storage(), x.size() / n, n, descending, values, indices);
        }
        Tensor v = new Tensor(x.shape, device, values);
        Tensor i = new Tensor(x.shape, device, indices);
        return a == last ? new Tensor[] {v, i} : new Tensor[] {v.permute(swap), i.permute(swap)};
    }

    /**
     * Where each entry of the {@linkplain #sort sorted} axis comes from.
     *
     * @param axis the axis; negative values count from the end
     * @param descending largest first instead of smallest first
     * @return the indices along the axis, as floats, of this shape
     */
    public Tensor argsort(int axis, boolean descending) {
        return sort(axis, descending)[1];
    }

    /**
     * Running sums along one axis: entry {@code i} is the sum of entries {@code 0 .. i}.
     * Computed as a product with a triangle of ones, so the work grows with the square of
     * the axis' length.
     *
     * @param axis the axis; negative values count from the end
     * @return the running sums, of this shape
     */
    public Tensor cumsum(int axis) {
        int a = axis(axis);
        int n = shape.dim(a);
        int[] order = moveLast(shape.rank(), a);
        Tensor moved = a == shape.rank() - 1 ? this : permute(order);
        Tensor sums = moved.matmul(upperTriangle(n, dtype(), device));
        return a == shape.rank() - 1 ? sums : sums.permute(inverse(order));
    }

    /** @return {@code [n, n]} ones on and above the diagonal */
    public static Tensor upperTriangle(int n, DType dtype, Device device) {
        float[] t = new float[n * n];
        for (int j = 0; j < n; j++) {
            for (int i = j; i < n; i++) {
                t[j * n + i] = 1;
            }
        }
        return of(Shape.of(n, n), t, device).to(dtype);
    }

    /** @return the axis order moving axis {@code a} of {@code rank} to the end */
    public static int[] moveLast(int rank, int a) {
        int[] order = new int[rank];
        for (int i = 0, j = 0; i < rank; i++) {
            if (i != a) {
                order[j++] = i;
            }
        }
        order[rank - 1] = a;
        return order;
    }

    /** @return the permutation undoing {@code order} */
    public static int[] inverse(int[] order) {
        int[] back = new int[order.length];
        for (int i = 0; i < order.length; i++) {
            back[order[i]] = i;
        }
        return back;
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
