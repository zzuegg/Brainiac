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
     * Returns a copy of the dimensions.
     *
     * @return the dimensions, outermost first
     */
    public int[] toArray() {
        return dims.clone();
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
