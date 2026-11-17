package io.github.zzuegg.brainiac.tensor.spi;

import java.util.Objects;

/**
 * A {@code [rows, cols]} matrix as whole-number levels times per-group scales, in Java
 * memory: the form backends receive packed weights in, and the CPU backend's own storage.
 *
 * <p>The value at {@code (r, c)} is {@code level(r, c) * scale(r / group, c)}: each column
 * (an output unit) has a scale for every {@code group} consecutive rows (inputs). Levels
 * take one signed byte each, row-major; with {@code bits <= 4} and an even group size they
 * take half a byte: the byte at {@code (r / 2) * cols + c} holds row {@code r} (even) in
 * its low four bits and row {@code r + 1} in its high four bits, two's complement, so a
 * SIMD lane or GPU work item for column {@code c} reads both from one byte.
 *
 * <p>A matrix may also have an offset per scale, as GGUF's k-quants and {@code Q4_1} do:
 * then the value is {@code level(r, c) * scale(r / group, c) + offset(r / group, c)}.
 */
public final class PackedMatrix implements PackedStorage {

    private final int rows;
    private final int cols;
    private final int bits;
    private final int group;
    private final boolean nibbles;
    private final byte[] data;
    private final float[] scales;
    private final float[] offsets;

    private PackedMatrix(int rows, int cols, int bits, int group, boolean nibbles, byte[] data, float[] scales,
            float[] offsets) {
        this.rows = rows;
        this.cols = cols;
        this.bits = bits;
        this.group = group;
        this.nibbles = nibbles;
        this.data = data;
        this.scales = scales;
        this.offsets = offsets;
    }

    /**
     * Packs levels.
     *
     * @param levels one level per value, row-major, each within {@code bits} bits signed
     * @param scales {@code (rows / group) * cols} scales, row-major
     * @param rows the rows
     * @param cols the columns
     * @param bits the bits per level, 2 to 8
     * @param group the rows per scale; must divide {@code rows}
     * @return the packed matrix
     */
    public static PackedMatrix of(byte[] levels, float[] scales, int rows, int cols, int bits, int group) {
        return of(levels, scales, null, rows, cols, bits, group);
    }

    /**
     * Packs levels with an offset per scale.
     *
     * @param levels one level per value, row-major, each within {@code bits} bits signed
     * @param scales {@code (rows / group) * cols} scales, row-major
     * @param offsets as many offsets as scales, or null for none
     * @param rows the rows
     * @param cols the columns
     * @param bits the bits per level, 2 to 8
     * @param group the rows per scale; must divide {@code rows}
     * @return the packed matrix
     */
    public static PackedMatrix of(byte[] levels, float[] scales, float[] offsets, int rows, int cols, int bits,
            int group) {
        Objects.requireNonNull(levels, "levels");
        Objects.requireNonNull(scales, "scales");
        if (rows < 1 || cols < 1 || bits < 2 || bits > 8 || group < 1 || rows % group != 0) {
            throw new IllegalArgumentException("bad packed matrix: " + rows + "x" + cols + ", " + bits + " bits, group "
                    + group);
        }
        if (levels.length != (long) rows * cols || scales.length != (long) rows / group * cols) {
            throw new IllegalArgumentException("a " + rows + "x" + cols + " matrix in groups of " + group + " needs "
                    + (long) rows * cols + " levels and " + (long) rows / group * cols + " scales, got "
                    + levels.length + " and " + scales.length);
        }
        if (offsets != null && offsets.length != scales.length) {
            throw new IllegalArgumentException(scales.length + " scales but " + offsets.length + " offsets");
        }
        boolean nibbles = bits <= 4 && group % 2 == 0;
        byte[] data;
        if (nibbles) {
            data = new byte[rows / 2 * cols];
            for (int r = 0; r < rows; r += 2) {
                int base = r / 2 * cols;
                for (int c = 0; c < cols; c++) {
                    data[base + c] = (byte) ((levels[r * cols + c] & 0xF) | (levels[(r + 1) * cols + c] << 4));
                }
            }
        } else {
            data = levels.clone();
        }
        return new PackedMatrix(rows, cols, bits, group, nibbles, data, scales.clone(),
                offsets == null ? null : offsets.clone());
    }

    /**
     * Joins matrices side by side: the columns of each in turn, as one product computes
     * several projections of the same input. All must have the same rows, bits, group and
     * level layout, and all or none offsets.
     *
     * @param parts the matrices, at least one
     * @return the joined matrix
     */
    public static PackedMatrix concatColumns(PackedMatrix... parts) {
        PackedMatrix f = parts[0];
        int cols = 0;
        for (PackedMatrix p : parts) {
            if (p.rows != f.rows || p.bits != f.bits || p.group != f.group || p.nibbles != f.nibbles
                    || (p.offsets == null) != (f.offsets == null)) {
                throw new IllegalArgumentException("cannot join " + p + " with " + f);
            }
            cols += p.cols;
        }
        int dataRows = f.nibbles ? f.rows / 2 : f.rows;
        int scaleRows = f.rows / f.group;
        byte[] data = new byte[dataRows * cols];
        float[] scales = new float[scaleRows * cols];
        float[] offsets = f.offsets == null ? null : new float[scales.length];
        int at = 0;
        for (PackedMatrix p : parts) {
            for (int r = 0; r < dataRows; r++) {
                System.arraycopy(p.data, r * p.cols, data, r * cols + at, p.cols);
            }
            for (int r = 0; r < scaleRows; r++) {
                System.arraycopy(p.scales, r * p.cols, scales, r * cols + at, p.cols);
                if (offsets != null) {
                    System.arraycopy(p.offsets, r * p.cols, offsets, r * cols + at, p.cols);
                }
            }
            at += p.cols;
        }
        return new PackedMatrix(f.rows, cols, f.bits, f.group, f.nibbles, data, scales, offsets);
    }

    /**
     * Quantizes values: each column's scale for a group of rows is its largest magnitude
     * there over {@code 2^(bits-1) - 1}, and each level is the value over its scale,
     * rounded half to even. A group of zeros gets the smallest normal float as its scale.
     *
     * @param values {@code rows * cols} values, row-major
     * @param rows the rows
     * @param cols the columns
     * @param bits the bits per level, 2 to 8
     * @param group the rows per scale; must divide {@code rows}
     * @return the packed matrix
     */
    public static PackedMatrix quantize(float[] values, int rows, int cols, int bits, int group) {
        if (rows < 1 || cols < 1 || bits < 2 || bits > 8 || group < 1 || rows % group != 0
                || values.length != (long) rows * cols) {
            throw new IllegalArgumentException("cannot quantize " + values.length + " values as " + rows + "x" + cols
                    + ", " + bits + " bits, groups of " + group);
        }
        int limit = (1 << (bits - 1)) - 1;
        float[] scales = new float[rows / group * cols];
        for (int r = 0; r < rows; r++) {
            int s = r / group * cols;
            for (int c = 0; c < cols; c++) {
                scales[s + c] = Math.max(scales[s + c], Math.abs(values[r * cols + c]));
            }
        }
        for (int i = 0; i < scales.length; i++) {
            scales[i] = Math.max(scales[i], Float.MIN_NORMAL) / limit;
        }
        byte[] levels = new byte[values.length];
        for (int r = 0; r < rows; r++) {
            int s = r / group * cols;
            for (int c = 0; c < cols; c++) {
                int q = (int) Math.rint(values[r * cols + c] / scales[s + c]);
                levels[r * cols + c] = (byte) Math.max(-limit, Math.min(limit, q));
            }
        }
        return of(levels, scales, rows, cols, bits, group);
    }

    @Override
    public int rows() {
        return rows;
    }

    @Override
    public int cols() {
        return cols;
    }

    /** @return the bits per level */
    public int bits() {
        return bits;
    }

    /** @return the rows per scale */
    public int group() {
        return group;
    }

    /** @return whether levels take half a byte each (see the class description) */
    public boolean nibbles() {
        return nibbles;
    }

    /** @return the packed levels; not a copy, do not change */
    public byte[] data() {
        return data;
    }

    /** @return the scales, {@code [rows / group, cols]}; not a copy, do not change */
    public float[] scales() {
        return scales;
    }

    /** @return the offsets, like {@link #scales()}, or null for none; not a copy, do not change */
    public float[] offsets() {
        return offsets;
    }

    /**
     * Returns one level.
     *
     * @param r the row
     * @param c the column
     * @return the level
     */
    public int level(int r, int c) {
        if (!nibbles) {
            return data[r * cols + c];
        }
        byte b = data[r / 2 * cols + c];
        return (r & 1) == 0 ? (byte) (b << 4) >> 4 : b >> 4;
    }

    /**
     * Computes the values, row-major.
     *
     * @param out {@code rows * cols} floats to fill
     */
    public void dequantize(float[] out) {
        for (int r = 0; r < rows; r++) {
            int s = r / group * cols;
            for (int c = 0; c < cols; c++) {
                out[r * cols + c] = offsets == null ? level(r, c) * scales[s + c]
                        : level(r, c) * scales[s + c] + offsets[s + c];
            }
        }
    }

    @Override
    public long size() {
        return (long) rows * cols;
    }

    @Override
    public long bytes() {
        return data.length + 4L * scales.length * (offsets == null ? 1 : 2);
    }

    @Override
    public String toString() {
        return "PackedMatrix[" + rows + "x" + cols + ", " + bits + " bits, groups of " + group + "]";
    }
}
