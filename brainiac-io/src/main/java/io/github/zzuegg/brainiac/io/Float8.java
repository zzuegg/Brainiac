package io.github.zzuegg.brainiac.io;

/**
 * Decodes the two 8-bit float formats models are stored in: E4M3 (the "fn" variant: no
 * infinities, one NaN pattern) and E5M2 (IEEE-like).
 */
final class Float8 {

    private Float8() {
    }

    /**
     * Decodes an E4M3FN value: 1 sign, 4 exponent (bias 7), 3 mantissa bits.
     *
     * @param b the bits
     * @return the value
     */
    static float e4m3(byte b) {
        int bits = b & 0xFF;
        int sign = bits >>> 7;
        int exp = (bits >>> 3) & 0xF;
        int man = bits & 0x7;
        float v;
        if (exp == 0xF && man == 0x7) {
            return Float.NaN;
        } else if (exp == 0) {
            v = man / 8f * 0x1p-6f;
        } else {
            v = (1 + man / 8f) * Math.scalb(1f, exp - 7);
        }
        return sign == 0 ? v : -v;
    }

    /**
     * Decodes an E5M2 value: 1 sign, 5 exponent (bias 15), 2 mantissa bits; the top half
     * of a float16.
     *
     * @param b the bits
     * @return the value
     */
    static float e5m2(byte b) {
        return Float.float16ToFloat((short) ((b & 0xFF) << 8));
    }
}
