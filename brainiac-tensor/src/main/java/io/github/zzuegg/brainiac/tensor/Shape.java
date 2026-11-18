package io.github.zzuegg.brainiac.tensor;

import java.util.Arrays;

/**
 * The dimensions of a tensor, such as {@code [4, 2]} for four rows of two values.
 *
 * <p>Shapes are immutable. Every dimension must be at least 1; a shape with no
 * dimensions describes a scalar.
 */
public final class Shape {

    private final int[] dims;
    private final long size;

    private Shape(int[] dims) {
        long n = 1;
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] < 1) {
                throw new IllegalArgumentException(
                        "dimension " + i + " must be at least 1, got " + dims[i] + " in " + Arrays.toString(dims));
            }
            n = Math.multiplyExact(n, dims[i]);
        }
        this.dims = dims;
        this.size = n;
    }

    /**
     * Creates a shape from its dimensions.
     *
     * @param dims the size of each dimension, outermost first
     * @return the shape
     * @throws IllegalArgumentException if a dimension is smaller than 1
     */
    public static Shape of(int... dims) {
        return new Shape(dims.clone());
    }

    /**
     * Creates a shape in which one dimension may be {@code -1}, standing for whatever
     * makes the shape hold {@code size} values, as {@code reshape(-1)} in PyTorch.
     *
     * @param size the number of values the shape must hold
     * @param dims the dimensions, at most one of them {@code -1}
     * @return the shape
     * @throws IllegalArgumentException if no size fits or two dimensions are {@code -1}
     */
    public static Shape inferred(long size, int... dims) {
        int unknown = -1;
        long known = 1;
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] == -1) {
                if (unknown >= 0) {
                    throw new IllegalArgumentException("only one dimension can be -1 in " + Arrays.toString(dims));
                }
                unknown = i;
            } else {
                known *= dims[i];
            }
        }
        if (unknown < 0) {
            return of(dims);
        }
        if (known <= 0 || size % known != 0) {
            throw new IllegalArgumentException(size + " values do not fit " + Arrays.toString(dims));
        }
        int[] d = dims.clone();
        d[unknown] = Math.toIntExact(size / known);
        return new Shape(d);
    }

    /**
     * Returns this shape with one dimension changed.
     *
     * @param axis the dimension; negative values count from the end
     * @param size its new size
     * @return the new shape
     */
    public Shape withDim(int axis, int size) {
        int a = axis < 0 ? dims.length + axis : axis;
        dim(axis);
        int[] d = dims.clone();
        d[a] = size;
        return new Shape(d);
    }

    /**
     * Returns the number of dimensions.
     *
     * @return the rank, 0 for a scalar
     */
    public int rank() {
        return dims.length;
    }

    /**
     * Returns the size of one dimension.
     *
     * @param axis the dimension index; negative values count from the end
     * @return the size of that dimension
     * @throws IndexOutOfBoundsException if the axis does not exist
     */
    public int dim(int axis) {
        int a = axis < 0 ? dims.length + axis : axis;
        if (a < 0 || a >= dims.length) {
            throw new IndexOutOfBoundsException("axis " + axis + " out of range for shape " + this);
        }
        return dims[a];
    }

    /**
     * Returns the total number of values in a tensor of this shape.
     *
     * @return the product of all dimensions
     */
    public long size() {
        return size;
    }

        /**
     * The dimensions, as {@link #toArray()}; named as PyTorch's {@code size()} tuple is read.
     *
     * @return a copy of the dimensions
     */
    public int[] dims() {
        return toArray();
    }

    /**
     * Returns a copy of the dimensions.
     *
     * @return the dimensions, outermost first
     */
    public int[] toArray() {
        return dims.clone();
    }

    /**
     * Returns the row-major strides: how many values to skip to move one step along each
     * dimension.
     *
     * @return the strides, outermost first
     */
    public long[] strides() {
        long[] strides = new long[dims.length];
        long step = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            strides[i] = step;
            step *= dims[i];
        }
        return strides;
    }

    /**
     * Returns the shape that results from broadcasting two shapes together. Shapes are
     * aligned at their last dimension; each pair of dimensions must be equal or one of
     * them must be 1.
     *
     * @param a the first shape
     * @param b the second shape
     * @return the broadcast shape
     * @throws IllegalArgumentException if the shapes are not compatible
     */
    public static Shape broadcast(Shape a, Shape b) {
        int rank = Math.max(a.rank(), b.rank());
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int da = i < rank - a.rank() ? 1 : a.dims[i - (rank - a.rank())];
            int db = i < rank - b.rank() ? 1 : b.dims[i - (rank - b.rank())];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("shapes " + a + " and " + b + " cannot be broadcast together");
            }
            out[i] = Math.max(da, db);
        }
        return new Shape(out);
    }

    /**
     * Returns the strides to read a tensor of this shape as if it had the larger
     * {@code target} shape: broadcast dimensions get stride 0.
     *
     * @param target a shape this shape broadcasts to
     * @return strides with one entry per dimension of {@code target}
     */
    long[] broadcastStrides(Shape target) {
        long[] own = strides();
        long[] out = new long[target.rank()];
        int offset = target.rank() - rank();
        for (int i = 0; i < rank(); i++) {
            out[i + offset] = dims[i] == 1 ? 0 : own[i];
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Shape other && Arrays.equals(dims, other.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}
