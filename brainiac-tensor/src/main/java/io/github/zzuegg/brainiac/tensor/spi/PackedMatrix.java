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
 *
 * <p>Scales and offsets may come {@linkplain Factors factored}, as GGUF's k-quants store
 * them: a small whole-number code per group times a value per super-group of rows. Backends
 * can keep that form on the device, at a quarter of the memory; {@link #scales()} and
 * {@link #offsets()} still give the products.
 */
public final class PackedMatrix implements PackedStorage {

    private final int rows;
    private final int cols;
    private final int bits;
    private final int group;
    private final boolean nibbles;
    private final byte[] data;
    private float[] scales;
    private float[] offsets;
    private final Factors factors;

    /**
     * Scales and offsets as codes times super-group values: the scale of group {@code g} in
     * column {@code c} is {@code scaleCodes[g * cols + c] * superScales[(g / perSuper) * cols + c]}
     * in float, {@code perSuper} being {@code superGroup / group}; the offset is
     * {@code offsetCodes[g * cols + c] * superOffsets[(g / perSuper) * cols + c] + scale * shift},
     * the product in float and the sum in double rounded to float. Null codes are all 1.
     *
     * @param scaleCodes {@code [rows / group, cols]} codes, or null for 1
     * @param superScales {@code [rows / superGroup, cols]}
     * @param offsetCodes like the scale codes, or null for 1
     * @param superOffsets like the super scales, or null for no offsets
     * @param superGroup the rows per super value: a multiple of the group that divides the rows
     * @param shift the multiple of the scale each offset adds
     */
    public record Factors(byte[] scaleCodes, float[] superScales, byte[] offsetCodes, float[] superOffsets,
            int superGroup, int shift) {

        /** @return the bytes these take */
        long bytes() {
            return (scaleCodes == null ? 0 : scaleCodes.length) + 4L * superScales.length
                    + (offsetCodes == null ? 0 : offsetCodes.length)
                    + (superOffsets == null ? 0 : 4L * superOffsets.length);
        }
    }

    private PackedMatrix(int rows, int cols, int bits, int group, boolean nibbles, byte[] data, float[] scales,
            float[] offsets) {
        this(rows, cols, bits, group, nibbles, data, scales, offsets, null);
    }

    private PackedMatrix(int rows, int cols, int bits, int group, boolean nibbles, byte[] data, float[] scales,
            float[] offsets, Factors factors) {
        this.factors = factors;
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
     * Packs levels with factored scales and offsets (see {@link Factors}).
     *
     * @param levels one level per value, row-major, each within {@code bits} bits signed
     * @param factors the scales and offsets
     * @param rows the rows
     * @param cols the columns
     * @param bits the bits per level, 2 to 8
     * @param group the rows per scale; must divide {@code rows}
     * @return the packed matrix
     */
    public static PackedMatrix of(byte[] levels, Factors factors, int rows, int cols, int bits, int group) {
        Objects.requireNonNull(factors, "factors");
        int sg = factors.superGroup();
        if (group < 1 || sg < group || sg % group != 0 || rows % sg != 0) {
            throw new IllegalArgumentException("super-groups of " + sg + " rows for groups of " + group + " in "
                    + rows + " rows");
        }
        long groups = (long) rows / group * cols;
        long supers = (long) rows / sg * cols;
        if (factors.superScales().length != supers || (factors.scaleCodes() != null && factors.scaleCodes().length != groups)
                || (factors.superOffsets() != null && factors.superOffsets().length != supers)
                || (factors.offsetCodes() != null && (factors.offsetCodes().length != groups || factors.superOffsets() == null))) {
            throw new IllegalArgumentException("factors do not fit a " + rows + "x" + cols + " matrix in groups of "
                    + group + " and super-groups of " + sg);
        }
        PackedMatrix plain = of(levels, new float[(int) groups], null, rows, cols, bits, group);
        return new PackedMatrix(rows, cols, bits, group, plain.nibbles, plain.data, null, null, factors);
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
        boolean factored = f.factors != null;
        for (PackedMatrix p : parts) {
            if (p.rows != f.rows || p.bits != f.bits || p.group != f.group || p.nibbles != f.nibbles
                    || p.hasOffsets() != f.hasOffsets()) {
                throw new IllegalArgumentException("cannot join " + p + " with " + f);
            }
            factored = factored && p.factors != null && p.factors.superGroup() == f.factors.superGroup()
                    && p.factors.shift() == f.factors.shift()
                    && (p.factors.scaleCodes() == null) == (f.factors.scaleCodes() == null)
                    && (p.factors.offsetCodes() == null) == (f.factors.offsetCodes() == null);
            cols += p.cols;
        }
        int dataRows = f.nibbles ? f.rows / 2 : f.rows;
        byte[] data = new byte[dataRows * cols];
        int at = 0;
        for (PackedMatrix p : parts) {
            for (int r = 0; r < dataRows; r++) {
                System.arraycopy(p.data, r * p.cols, data, r * cols + at, p.cols);
            }
            at += p.cols;
        }
        int groups = f.rows / f.group;
        if (factored) {
            int supers = f.rows / f.factors.superGroup();
            Factors x = f.factors;
            Factors joined = new Factors(
                    x.scaleCodes() == null ? null : join(parts, groups, cols, p -> p.factors.scaleCodes()),
                    join(parts, supers, cols, p -> p.factors.superScales()),
                    x.offsetCodes() == null ? null : join(parts, groups, cols, p -> p.factors.offsetCodes()),
                    x.superOffsets() == null ? null : join(parts, supers, cols, p -> p.factors.superOffsets()),
                    x.superGroup(), x.shift());
            return new PackedMatrix(f.rows, cols, f.bits, f.group, f.nibbles, data, null, null, joined);
        }
        float[] scales = join(parts, groups, cols, PackedMatrix::scales);
        float[] offsets = f.hasOffsets() ? join(parts, groups, cols, PackedMatrix::offsets) : null;
        return new PackedMatrix(f.rows, cols, f.bits, f.group, f.nibbles, data, scales, offsets);
    }

    /** Joins row-major arrays of {@code rows} rows side by side, one array per part. */
    @SuppressWarnings("unchecked")
    private static <A> A join(PackedMatrix[] parts, int rows, int cols, java.util.function.Function<PackedMatrix, A> of) {
        A first = of.apply(parts[0]);
        A out = (A) java.lang.reflect.Array.newInstance(first.getClass().getComponentType(), rows * cols);
        int at = 0;
        for (PackedMatrix p : parts) {
            A a = of.apply(p);
            for (int r = 0; r < rows; r++) {
                System.arraycopy(a, r * p.cols, out, r * cols + at, p.cols);
            }
            at += p.cols;
        }
        return out;
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
        if (scales == null) {
            expand();
        }
        return scales;
    }

    /** @return the offsets, like {@link #scales()}, or null for none; not a copy, do not change */
    public float[] offsets() {
        if (scales == null) {
            expand();
        }
        return offsets;
    }

    /** @return whether values have offsets */
    public boolean hasOffsets() {
        return factors != null ? factors.superOffsets() != null : offsets != null;
    }

    /** @return the factored scales and offsets these were packed with, or null */
    public Factors factors() {
        return factors;
    }

    /** Computes the scales and offsets from their factors, once. */
    private synchronized void expand() {
        if (scales != null) {
            return;
        }
        Factors f = factors;
        int perSuper = f.superGroup() / group;
        int count = rows / group * cols;
        float[] sc = new float[count];
        float[] of = f.superOffsets() == null ? null : new float[count];
        for (int i = 0; i < count; i++) {
            int g = i / cols;
            int at = g / perSuper * cols + i % cols;
            float s = (f.scaleCodes() == null ? 1 : f.scaleCodes()[i]) * f.superScales()[at];
            sc[i] = s;
            if (of != null) {
                float o = (f.offsetCodes() == null ? 1 : f.offsetCodes()[i]) * f.superOffsets()[at];
                of[i] = (float) (o + (double) s * f.shift());
            }
        }
        offsets = of;
        scales = sc;
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
                out[r * cols + c] = offsets() == null ? level(r, c) * scales()[s + c]
                        : level(r, c) * scales()[s + c] + offsets()[s + c];
            }
        }
    }

    @Override
    public long size() {
        return (long) rows * cols;
    }

    @Override
    public long bytes() {
        return factors != null ? data.length + factors.bytes()
                : data.length + 4L * scales.length * (offsets == null ? 1 : 2);
    }

    @Override
    public String toString() {
        return "PackedMatrix[" + rows + "x" + cols + ", " + bits + " bits, groups of " + group + "]";
    }
}
