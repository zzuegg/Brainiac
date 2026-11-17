package io.github.zzuegg.brainiac.tensor.spi;

import io.github.zzuegg.brainiac.tensor.DType;

/**
 * Rows of 8-bit levels with one float scale per row, in Java memory: the form of
 * {@link Backend#allocateQuantizedRows} that backends without their own use, such as a
 * key/value cache at half of float16's memory.
 *
 * <p>A row's scale is its largest magnitude over 127 and each level its value over the
 * scale, rounded half to even, so the values are {@code level * scale}. Quantizing values
 * read back from here gives the same levels and scales again.
 */
public final class QuantizedRows implements RowQuantizedStorage {

    private final long rows;
    private final int cols;
    private final DType dtype;
    private final byte[] levels;
    private final float[] scales;

    /**
     * Creates rows of zeros.
     *
     * @param rows the rows
     * @param cols the values per row
     * @param dtype the type the values are read as
     */
    public QuantizedRows(long rows, int cols, DType dtype) {
        if (rows < 0 || cols < 1) {
            throw new IllegalArgumentException("bad quantized rows: " + rows + " x " + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.dtype = dtype;
        this.levels = new byte[Math.toIntExact(rows * cols)];
        this.scales = new float[Math.toIntExact(rows)];
    }

    @Override
    public int rows() {
        return Math.toIntExact(rows);
    }

    @Override
    public int cols() {
        return cols;
    }

    @Override
    public long size() {
        return rows * cols;
    }

    @Override
    public DType dtype() {
        return dtype;
    }

    @Override
    public long bytes() {
        return levels.length + 4L * scales.length;
    }

    /** @return the levels, row-major; not a copy */
    public byte[] levels() {
        return levels;
    }

    /** @return one scale per row; not a copy */
    public float[] scales() {
        return scales;
    }

    /**
     * Quantizes rows into place: for each of {@code outer} blocks of {@code blockRows} rows
     * here, source rows {@code [o * sourceRows, (o + 1) * sourceRows)} go to rows
     * {@code start ..} of block {@code o}.
     *
     * @param values the source rows, row-major
     * @param outer the blocks
     * @param blockRows rows per block here
     * @param sourceRows rows per block in {@code values}
     * @param start the first row written in each block
     */
    public void write(float[] values, long outer, int blockRows, int sourceRows, int start) {
        for (long o = 0; o < outer; o++) {
            for (int t = 0; t < sourceRows; t++) {
                int from = Math.toIntExact((o * sourceRows + t) * cols);
                int row = Math.toIntExact(o * blockRows + start + t);
                quantize(values, from, cols, levels, row * cols, scales, row);
            }
        }
    }

    /**
     * Quantizes one row.
     *
     * @param values the source
     * @param from where the row starts in it
     * @param cols the row's length
     * @param levels where the levels go
     * @param to where the row's levels start
     * @param scales where the scale goes
     * @param row the scale's index
     */
    public static void quantize(float[] values, int from, int cols, byte[] levels, int to, float[] scales, int row) {
        float largest = 0;
        for (int c = 0; c < cols; c++) {
            largest = Math.max(largest, Math.abs(values[from + c]));
        }
        float scale = largest / 127;
        scales[row] = scale;
        for (int c = 0; c < cols; c++) {
            levels[to + c] = scale == 0 ? 0 : (byte) Math.clamp((long) Math.rint(values[from + c] / scale), -127, 127);
        }
    }

    /**
     * Computes the values, row-major.
     *
     * @param out {@code rows * cols} floats to fill
     */
    public void dequantize(float[] out) {
        for (int r = 0; r < scales.length; r++) {
            float s = scales[r];
            for (int c = 0; c < cols; c++) {
                out[r * cols + c] = levels[r * cols + c] * s;
            }
        }
    }

    @Override
    public String toString() {
        return "QuantizedRows[" + rows + "x" + cols + ", " + dtype + "]";
    }
}
