package io.github.zzuegg.brainiac.tensor;

/**
 * The number format of a tensor's values.
 *
 * <p>Operations on two tensors of different types promote to the wider type
 * ({@code FLOAT16} and {@code BFLOAT16} meet at {@code FLOAT32}), except that a
 * single-value operand, such as a constant, takes the type of the other side.
 */
public enum DType {
    /** IEEE 754 single precision: the default. */
    FLOAT32(4, 2),
    /** IEEE 754 double precision. */
    FLOAT64(8, 3),
    /** IEEE 754 half precision: 11 significant bits, range up to 65504. */
    FLOAT16(2, 1),
    /** bfloat16: the range of float32 with 8 significant bits. */
    BFLOAT16(2, 1);

    private final int bytes;
    private final int rank;

    DType(int bytes, int rank) {
        this.bytes = bytes;
        this.rank = rank;
    }

    /** @return the size of one value in bytes */
    public int bytes() {
        return bytes;
    }

    /**
     * Returns the type two types combine to.
     *
     * @param a one type
     * @param b the other type
     * @return the wider type
     */
    public static DType promote(DType a, DType b) {
        if (a == b) {
            return a;
        }
        if (a.rank == b.rank) {
            return FLOAT32;
        }
        return a.rank > b.rank ? a : b;
    }

    /**
     * Rounds a value to this type's precision, the way storing it would.
     *
     * @param v the value
     * @return the nearest value this type can hold
     */
    public double round(double v) {
        return switch (this) {
            case FLOAT64 -> v;
            case FLOAT32 -> (float) v;
            case FLOAT16 -> Float.float16ToFloat(Float.floatToFloat16((float) v));
            case BFLOAT16 -> bfloat16ToFloat(floatToBfloat16((float) v));
        };
    }

    /**
     * Converts a float to bfloat16 bits, rounding to nearest even.
     *
     * @param f the value
     * @return the 16 bits
     */
    public static short floatToBfloat16(float f) {
        int bits = Float.floatToRawIntBits(f);
        if (Float.isNaN(f)) {
            return (short) ((bits >>> 16) | 0x40);
        }
        int rounding = 0x7FFF + ((bits >>> 16) & 1);
        return (short) ((bits + rounding) >>> 16);
    }

    /**
     * Converts bfloat16 bits to a float.
     *
     * @param b the 16 bits
     * @return the value
     */
    public static float bfloat16ToFloat(short b) {
        return Float.intBitsToFloat((b & 0xFFFF) << 16);
    }
}
