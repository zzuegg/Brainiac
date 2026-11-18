package io.github.zzuegg.brainiac.tensor;

/**
 * The number format of a tensor's values.
 *
 * <p>Operations on two tensors of different types promote to the wider type
 * ({@code FLOAT16} and {@code BFLOAT16} meet at {@code FLOAT32}), except that a
 * single-value operand, such as a constant, takes the type of the other side.
 *
 * <p>The integer and boolean types hold token ids, indices and masks exactly. The CPU
 * backend stores and computes with them; other backends hold floating point values
 * only, so integers move there by converting with {@link Tensor#to(DType)}. Mixed with a
 * floating point type they promote to it; arithmetic on two integer tensors stays
 * integer (results are truncated toward zero).
 */
public enum DType {
    /** IEEE 754 single precision: the default. */
    FLOAT32(4, 2),
    /** IEEE 754 double precision. */
    FLOAT64(8, 3),
    /** IEEE 754 half precision: 11 significant bits, range up to 65504. */
    FLOAT16(2, 1),
    /** bfloat16: the range of float32 with 8 significant bits. */
    BFLOAT16(2, 1),
    /** 32-bit signed integers. */
    INT32(4, 0),
    /** 64-bit signed integers; computed through doubles, so exact up to 2^53. */
    INT64(8, 0),
    /** 8-bit unsigned integers, 0 to 255. */
    UINT8(1, 0),
    /** {@code true} or {@code false}, stored as one byte and read as 1 or 0. */
    BOOL(1, 0);

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
        if (!a.floating() && !b.floating()) {
            // integers widen to the larger one; a bool with anything takes the other side
            if (a == BOOL || b == BOOL) {
                return a == BOOL ? b : a;
            }
            return a == INT64 || b == INT64 ? INT64 : INT32;
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
            case INT32 -> Double.isNaN(v) ? 0 : (int) v;
            case INT64 -> Double.isNaN(v) ? 0 : (double) (long) v;
            case UINT8 -> ((int) Math.max(0, Math.min(255, Double.isNaN(v) ? 0 : v)));
            case BOOL -> v != 0 && !Double.isNaN(v) ? 1 : 0;
        };
    }

    /** @return whether the type holds floating point numbers, rather than integers or booleans */
    public boolean floating() {
        return rank > 0;
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
