package io.github.zzuegg.brainiac.backend.hip;

import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;

import java.util.stream.IntStream;

/**
 * A packed matrix's arrays as they go to the device, in the layout of the Vulkan backend's
 * packed matrices (see {@code PackedLayout} in kernels.hip): levels split into a nibble plane
 * and a high-bit plane, or into 4-bit indices into a table, and scales and offsets kept
 * factored and as float16 where they are exactly that. Made without the backend's lock.
 *
 * @param flags {@link #NIBBLES} and the rest
 * @param perSuper the groups per super value
 * @param shift the multiple of the scale each offset adds
 * @param levels the levels, nibbles or indices
 * @param hi the high-bit plane, the table of 16 levels, or null
 * @param scaleCodes a signed byte per group and column, or null for 1
 * @param offsetCodes likewise for the offsets
 * @param scaleHalves the super scales as float16, or null for {@code scales}
 * @param offsetHalves the super offsets as float16, or null
 * @param scales the scales as float32 when not halves
 * @param offsets the offsets as float32 when not halves, or null
 */
record PackedLayout(PackedMatrix matrix, int flags, int perSuper, float shift, byte[] levels, byte[] hi,
        byte[] scaleCodes, byte[] offsetCodes, short[] scaleHalves, short[] offsetHalves, float[] scales,
        float[] offsets) {

    static final int NIBBLES = 1;
    static final int OFFSETS = 2;
    static final int SCALE_CODES = 4;
    static final int OFFSET_CODES = 8;
    static final int HALF = 16;
    static final int CODEBOOK = 32;

    /** Whether 5- and 6-bit levels go as nibbles plus high-bit planes ({@code -Dbrainiac.hip.packedPlanes=false}: a byte each). */
    private static final boolean PLANES = !"false".equals(System.getProperty("brainiac.hip.packedPlanes"));
    /** Whether levels of at most 16 distinct values go as 4-bit indices into a table ({@code -Dbrainiac.hip.packedCodebook=false}: never). */
    private static final boolean CODEBOOK_ON = !"false".equals(System.getProperty("brainiac.hip.packedCodebook"));
    /** Whether factored scales stay factored ({@code -Dbrainiac.hip.packedFactors=false}: float32 per group). */
    private static final boolean FACTORS = !"false".equals(System.getProperty("brainiac.hip.packedFactors"));

    /** @return the flags the kernels read: the above, the plane bits from bit 8 and the level bits from bit 12 */
    int layout() {
        int planeBits = hi == null || (flags & CODEBOOK) != 0 ? 0 : matrix.bits() - 4;
        return flags | planeBits << 8 | matrix.bits() << 12;
    }

    /** Prepares a packed matrix's arrays for the device. */
    static PackedLayout of(PackedMatrix matrix) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        int bits = matrix.bits();
        int flags = matrix.nibbles() ? NIBBLES : 0;
        byte[] levels;
        byte[] hi = null;
        byte[] table;
        // the 4-bit layouts read two rows from a byte, so groups (and the pieces of k cut at
        // them) must have an even number of rows
        boolean even = matrix.group() % 2 == 0;
        if (PLANES && even && !matrix.nibbles() && (bits == 5 || bits == 6) && rows % 8 == 0) {
            int planeBits = bits - 4;
            int perByte = 8 / planeBits;
            byte[] data = matrix.data();
            byte[] low = new byte[rows / 2 * cols];
            byte[] high = new byte[rows / perByte * cols];
            int half = 1 << (bits - 1);
            // in parallel over runs of 8 rows, which write their own bytes of both planes
            IntStream.range(0, rows / 8).parallel().forEach(run -> {
                for (int r = run * 8; r < run * 8 + 8; r++) {
                    int lowAt = r / 2 * cols;
                    int highAt = r / perByte * cols;
                    int lowShift = (r & 1) * 4;
                    int highShift = (r % perByte) * planeBits;
                    for (int c = 0, at = r * cols; c < cols; c++, at++) {
                        int u = data[at] + half;
                        low[lowAt + c] |= (byte) ((u & 15) << lowShift);
                        high[highAt + c] |= (byte) ((u >> 4) << highShift);
                    }
                }
            });
            levels = low;
            hi = high;
        } else if (CODEBOOK_ON && even && !matrix.nibbles() && rows % 2 == 0
                && (table = codebook(matrix.data())) != null) {
            // at most 16 distinct levels (IQ4_NL, IQ4_XS, MXFP4): a 4-bit index each into a table
            int[] index = new int[256];
            for (int i = 0; i < table.length; i++) {
                index[table[i] & 0xFF] = i;
            }
            byte[] data = matrix.data();
            byte[] packed = new byte[rows / 2 * cols];
            IntStream.range(0, rows / 2).parallel().forEach(pair -> {
                int evenAt = 2 * pair * cols;
                int odd = evenAt + cols;
                int at = pair * cols;
                for (int c = 0; c < cols; c++) {
                    packed[at + c] = (byte) (index[data[evenAt + c] & 0xFF] | index[data[odd + c] & 0xFF] << 4);
                }
            });
            levels = packed;
            hi = java.util.Arrays.copyOf(table, 16);
            flags |= CODEBOOK;
        } else {
            levels = matrix.data();
        }
        PackedMatrix.Factors f = matrix.factors();
        if (FACTORS && f != null && halves(f.superScales()) && (f.superOffsets() == null || halves(f.superOffsets()))) {
            flags |= HALF | (f.scaleCodes() != null ? SCALE_CODES : 0) | (f.superOffsets() != null ? OFFSETS : 0)
                    | (f.offsetCodes() != null ? OFFSET_CODES : 0);
            return new PackedLayout(matrix, flags, f.superGroup() / matrix.group(), f.shift(), levels, hi,
                    f.scaleCodes(), f.offsetCodes(), toHalves(f.superScales()),
                    f.superOffsets() == null ? null : toHalves(f.superOffsets()), null, null);
        }
        flags |= matrix.offsets() != null ? OFFSETS : 0;
        if (FACTORS && halves(matrix.scales()) && (matrix.offsets() == null || halves(matrix.offsets()))) {
            // float16 values per group, as IQ4_NL and MXFP4 store them
            return new PackedLayout(matrix, flags | HALF, 1, 0, levels, hi, null, null, toHalves(matrix.scales()),
                    matrix.offsets() == null ? null : toHalves(matrix.offsets()), null, null);
        }
        return new PackedLayout(matrix, flags, 1, 0, levels, hi, null, null, null, null, matrix.scales(),
                matrix.offsets());
    }

    /** @return the distinct levels in increasing order if there are at most 16, else null */
    private static byte[] codebook(byte[] levels) {
        // the levels seen, as a 256-bit set, gathered in parallel over chunks; a chunk with more
        // than 16 on its own (an 8-bit format) stops the others early
        int chunk = 1 << 20;
        java.util.concurrent.atomic.AtomicBoolean tooMany = new java.util.concurrent.atomic.AtomicBoolean();
        long[] seen = IntStream.range(0, (int) Math.ceilDiv((long) levels.length, chunk)).parallel()
                .mapToObj(part -> {
                    long[] bits = new long[4];
                    int count = 0;
                    for (int i = part * chunk, end = Math.min(levels.length, i + chunk); i < end; i++) {
                        int v = levels[i] & 0xFF;
                        long bit = 1L << v;
                        if ((bits[v >> 6] & bit) == 0) {
                            bits[v >> 6] |= bit;
                            if (++count > 16 || tooMany.get()) {
                                tooMany.set(true);
                                break;
                            }
                        }
                    }
                    return bits;
                })
                .reduce(new long[4], (a, b) -> new long[] {a[0] | b[0], a[1] | b[1], a[2] | b[2], a[3] | b[3]});
        int count = Long.bitCount(seen[0]) + Long.bitCount(seen[1]) + Long.bitCount(seen[2]) + Long.bitCount(seen[3]);
        if (tooMany.get() || count > 16) {
            return null;
        }
        byte[] table = new byte[count];
        int at = 0;
        for (int v = -128; v < 128; v++) {
            if ((seen[(v & 0xFF) >> 6] & 1L << v) != 0) {
                table[at++] = (byte) v;
            }
        }
        return table;
    }

    /** Whether every value is exactly a float16. */
    private static boolean halves(float[] values) {
        for (float v : values) {
            if (Float.float16ToFloat(Float.floatToFloat16(v)) != v) {
                return false;
            }
        }
        return true;
    }

    private static short[] toHalves(float[] values) {
        short[] h = new short[values.length];
        for (int i = 0; i < h.length; i++) {
            h[i] = Float.floatToFloat16(values[i]);
        }
        return h;
    }
}
