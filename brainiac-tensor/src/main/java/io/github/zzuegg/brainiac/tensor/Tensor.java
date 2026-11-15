package io.github.zzuegg.brainiac.tensor;

import io.github.zzuegg.brainiac.tensor.spi.Backend;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.util.Arrays;
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

    private Tensor(Shape shape, Device device, Storage storage) {
        this.shape = shape;
        this.device = device;
        this.storage = storage;
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
        return new Tensor(shape, device, device.backend().allocate(shape.size()));
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
        Backend backend = device.backend();
        Storage s = backend.allocate(shape.size());
        if (value != 0f) {
            backend.fill(s, value);
        }
        return new Tensor(shape, device, s);
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
        return new Tensor(shape, target, target.backend().upload(toArray()));
    }

    /**
     * Copies all values to a new Java array in row-major order.
     *
     * @return the values
     */
    public float[] toArray() {
        return backend().download(storage);
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
        return new Tensor(newShape, device, storage);
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
        Storage out = backend().allocate(outShape.size());
        backend().copy(storage, start * strides[a], strides, dims, out, 0, outShape.strides());
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
        Shape outShape = Shape.of(dims);
        long[] outStrides = outShape.strides();
        Storage out = first.backend().allocate(outShape.size());
        long offset = 0;
        for (Tensor t : parts) {
            first.backend().copy(t.storage, 0, t.shape.strides(), t.shape.toArray(), out, offset, outStrides);
            offset += t.shape.dim(a) * outStrides[a];
        }
        return new Tensor(outShape, first.device, out);
    }

    private Tensor strided(long[] strides, Shape outShape) {
        Storage out = backend().allocate(outShape.size());
        backend().copy(storage, strides, outShape.toArray(), out);
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

    private Tensor unary(UnaryOp op) {
        Storage out = backend().allocate(size());
        backend().unary(op, storage, out);
        return new Tensor(shape, device, out);
    }

    private Tensor binary(BinaryOp op, Tensor other) {
        sameDevice(other);
        Shape outShape = Shape.broadcast(shape, other.shape);
        Storage out = backend().allocate(outShape.size());
        backend().binary(op, storage, shape.broadcastStrides(outShape),
                other.storage, other.shape.broadcastStrides(outShape), outShape.toArray(), out);
        return new Tensor(outShape, device, out);
    }

    private Tensor constant(float value) {
        return full(Shape.of(), value, device);
    }

    // ---------------------------------------------------------------- linear algebra

    /**
     * Multiplies two matrices: {@code [m, k] x [k, n] -> [m, n]}.
     *
     * @param other the right-hand matrix
     * @return the matrix product
     */
    public Tensor matmul(Tensor other) {
        sameDevice(other);
        if (rank() != 2 || other.rank() != 2) {
            throw new IllegalArgumentException(
                    "matmul needs two matrices, got shapes " + shape + " and " + other.shape);
        }
        int m = shape.dim(0);
        int k = shape.dim(1);
        int n = other.shape.dim(1);
        if (other.shape.dim(0) != k) {
            throw new IllegalArgumentException(
                    "matmul shapes " + shape + " and " + other.shape + " do not fit: " + k + " != " + other.shape.dim(0));
        }
        Shape outShape = Shape.of(m, n);
        Storage out = backend().allocate(outShape.size());
        backend().matmul(storage, other.storage, m, k, n, out);
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
        Storage out = backend().allocate(1);
        backend().reduce(op, storage, 1, size(), 1, out);
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
        Storage out = backend().allocate(outer * inner);
        backend().reduce(op, storage, outer, dims[a], inner, out);
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
