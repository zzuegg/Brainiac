package io.github.zzuegg.brainiac.tensor.spi;

import io.github.zzuegg.brainiac.tensor.DType;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Reads values stored the way a {@link DType} stores them (little-endian) from memory, for
 * backends that receive weights from a memory-mapped file.
 */
public final class RawValues {

    /** Little-endian float32, unaligned. */
    public static final ValueLayout.OfFloat FLOAT =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    /** Little-endian float64, unaligned. */
    public static final ValueLayout.OfDouble DOUBLE =
            ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    /** Little-endian 16-bit values, unaligned. */
    public static final ValueLayout.OfShort SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private RawValues() {
    }

    /**
     * Whether this machine stores numbers little-endian, so raw values can be copied as
     * they are.
     *
     * @return {@code true} on little-endian machines
     */
    public static boolean nativeLittleEndian() {
        return ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN;
    }

    /**
     * Decodes {@code count} values of a type into doubles.
     *
     * @param data the raw values
     * @param dtype their type
     * @param count how many
     * @return the values
     */
    public static double[] decode(MemorySegment data, DType dtype, long count) {
        double[] out = new double[Math.toIntExact(count)];
        switch (dtype) {
            case FLOAT64 -> MemorySegment.copy(data, DOUBLE, 0, out, 0, out.length);
            case FLOAT32 -> {
                for (int i = 0; i < out.length; i++) {
                    out[i] = data.getAtIndex(FLOAT, i);
                }
            }
            case FLOAT16 -> {
                for (int i = 0; i < out.length; i++) {
                    out[i] = Float.float16ToFloat(data.getAtIndex(SHORT, i));
                }
            }
            case BFLOAT16 -> {
                for (int i = 0; i < out.length; i++) {
                    out[i] = DType.bfloat16ToFloat(data.getAtIndex(SHORT, i));
                }
            }
        }
        return out;
    }
}
