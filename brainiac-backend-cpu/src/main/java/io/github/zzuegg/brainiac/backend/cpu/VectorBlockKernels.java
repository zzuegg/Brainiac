package io.github.zzuegg.brainiac.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD kernels for {@link BlockKernels}. Only loaded when the {@code jdk.incubator.vector}
 * module is present.
 *
 * <p>The formats of most large models (MXFP4, Q8_0, Q4_K, Q6_K, IQ4_XS) are multiplied straight
 * from their blocks, without decoding a row into floats first: codes are loaded 8 bytes at
 * a time and widened to 8 floats (the non-linear formats: 16 bytes, looked up in a
 * 16-entry table with one byte shuffle), multiplied by the inputs, and each block's (or
 * sub-block's) sum is scaled once. Q4_K's minimums are folded in through each sub-block's
 * sum of inputs. Needs 8 float lanes (256-bit vectors).
 */
final class VectorBlockKernels {

    private static final VectorSpecies<Float> FLOATS = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = FLOATS.length();
    private static final VectorSpecies<Float> F8 = FloatVector.SPECIES_256;
    private static final VectorSpecies<Byte> B8 = ByteVector.SPECIES_64;
    private static final VectorSpecies<Byte> B16 = ByteVector.SPECIES_128;
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

    /** MXFP4's E2M1 values, doubled to whole numbers (the scale is halved to match). */
    private static final byte[] MXFP4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    /** IQ4_NL's and IQ4_XS's levels (llama.cpp's kvalues_iq4nl). */
    private static final byte[] IQ4 = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    private VectorBlockKernels() {
    }

    /** @return whether the fused kernels can run: 8 float lanes or more */
    static boolean fused() {
        return LANES >= 8;
    }

    /** @return whether {@link #dot(String, MemorySegment, long, float[], int, int)} reads this format directly */
    static boolean reads(String format) {
        return fused() && switch (format) {
            case "MXFP4", "Q8_0", "Q4_K", "Q6_K", "IQ4_XS" -> true;
            default -> false;
        };
    }

    /**
     * The dot product of the stored row at {@code row} in {@code m}, of {@code k} values in
     * {@code format}, with {@code x[at, at + k)}.
     */
    static float dot(String format, MemorySegment m, long row, float[] x, int at, int k) {
        return switch (format) {
            case "MXFP4" -> mxfp4(m, row, x, at, k);
            case "Q8_0" -> q80(m, row, x, at, k);
            case "Q4_K" -> q4k(m, row, x, at, k);
            case "Q6_K" -> q6k(m, row, x, at, k);
            case "IQ4_XS" -> iq4xs(m, row, x, at, k);
            default -> throw new IllegalArgumentException(format);
        };
    }

    /**
     * Shifts each byte right by {@code n}, the top bits filled from the next byte: callers
     * mask what they keep. Shifted as ints, which every SIMD instruction set has; a byte
     * shift is not, and was not compiled to vector code.
     */
    private static ByteVector shr(ByteVector v, int n) {
        return v.reinterpretAsInts().lanewise(VectorOperators.LSHR, n).reinterpretAsBytes();
    }

    private static FloatVector lane(ByteVector v, int part) {
        return (FloatVector) v.convertShape(VectorOperators.B2F, F8, part);
    }

    /**
     * Eight bytes as eight floats. Converted between shapes of the same lane count: measured,
     * taking a part of a wider vector ({@link #lane}) compiled to slow fallback code wherever
     * the method around it grew large, and a whole kernel ran 4 to 5 times slower.
     */
    private static FloatVector f8(ByteVector v) {
        return (FloatVector) v.convertShape(VectorOperators.B2F, F8, 0);
    }

    private static FloatVector xs(float[] x, int at) {
        return FloatVector.fromArray(F8, x, at);
    }

    /** Sum over 32 inputs of table[low code] (first 16) and table[high code] (last 16) times x. */
    private static FloatVector lookup32(ByteVector table, ByteVector q, float[] x, int at) {
        ByteVector lo = q.and((byte) 0xF).selectFrom(table);
        ByteVector hi = shr(q, 4).and((byte) 0xF).selectFrom(table);
        FloatVector s = lane(lo, 0).mul(xs(x, at));
        s = lane(lo, 1).fma(xs(x, at + 8), s);
        s = lane(hi, 0).fma(xs(x, at + 16), s);
        return lane(hi, 1).fma(xs(x, at + 24), s);
    }

    private static float f16(MemorySegment m, long at) {
        return Float.float16ToFloat(m.get(SHORT, at));
    }

    /** Byte {@code i} (0 to 11) of the 12 bytes held as {@code lo} (0 to 7) and {@code hi}, unsigned. */
    private static int sb(long lo, int hi, int i) {
        return i < 8 ? (int) (lo >>> (8 * i)) & 0xFF : (hi >>> (8 * (i - 8))) & 0xFF;
    }

    private static int u8(MemorySegment m, long at) {
        return Byte.toUnsignedInt(m.get(ValueLayout.JAVA_BYTE, at));
    }

    private static float mxfp4(MemorySegment m, long row, float[] x, int at, int k) {
        ByteVector table = ByteVector.fromArray(B16, MXFP4, 0);
        FloatVector acc = FloatVector.zero(F8);
        for (int b = 0; b < k / 32; b++) {
            long o = row + 17L * b;
            float d = Math.scalb(1f, u8(m, o) - 128);
            FloatVector s = lookup32(table, ByteVector.fromMemorySegment(B16, m, o + 1, LE), x, at + 32 * b);
            acc = s.fma(FloatVector.broadcast(F8, d), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float q80(MemorySegment m, long row, float[] x, int at, int k) {
        FloatVector acc = FloatVector.zero(F8);
        for (int b = 0; b < k / 32; b++) {
            long o = row + 34L * b;
            float d = f16(m, o);
            int xo = at + 32 * b;
            FloatVector s = f8(ByteVector.fromMemorySegment(B8, m, o + 2, LE)).mul(xs(x, xo));
            s = f8(ByteVector.fromMemorySegment(B8, m, o + 10, LE)).fma(xs(x, xo + 8), s);
            s = f8(ByteVector.fromMemorySegment(B8, m, o + 18, LE)).fma(xs(x, xo + 16), s);
            s = f8(ByteVector.fromMemorySegment(B8, m, o + 26, LE)).fma(xs(x, xo + 24), s);
            acc = s.fma(FloatVector.broadcast(F8, d), acc);
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float iq4xs(MemorySegment m, long row, float[] x, int at, int k) {
        ByteVector table = ByteVector.fromArray(B16, IQ4, 0);
        FloatVector acc = FloatVector.zero(F8);
        for (int b = 0; b < k / 256; b++) {
            long o = row + 136L * b;
            float d = f16(m, o);
            int high = Short.toUnsignedInt(m.get(SHORT, o + 2));
            for (int ib = 0; ib < 8; ib++) {
                int ls = ((u8(m, o + 4 + ib / 2) >>> (4 * (ib % 2))) & 0xF) | (((high >>> (2 * ib)) & 3) << 4);
                FloatVector s = lookup32(table, ByteVector.fromMemorySegment(B16, m, o + 8 + 16L * ib, LE), x,
                        at + 256 * b + 32 * ib);
                acc = s.fma(FloatVector.broadcast(F8, d * (ls - 32)), acc);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float q4k(MemorySegment m, long row, float[] x, int at, int k) {
        FloatVector acc = FloatVector.zero(F8);
        FloatVector macc = FloatVector.zero(F8);
        for (int b = 0; b < k / 256; b++) {
            long o = row + 144L * b;
            float d = f16(m, o);
            float min = f16(m, o + 2);
            long lo = m.get(LONG, o + 4);
            int hi = m.get(INT, o + 12);
            for (int ib = 0; ib < 8; ib++) {
                int sm = q4kScale(lo, hi, ib);
                long qo = o + 16 + 32L * (ib >>> 1);
                int shift = 4 * (ib & 1);
                int xo = at + 256 * b + 32 * ib;
                FloatVector x0 = xs(x, xo);
                FloatVector x1 = xs(x, xo + 8);
                FloatVector x2 = xs(x, xo + 16);
                FloatVector x3 = xs(x, xo + 24);
                FloatVector v = f8(nib8(m, qo, shift)).mul(x0);
                v = f8(nib8(m, qo + 8, shift)).fma(x1, v);
                v = f8(nib8(m, qo + 16, shift)).fma(x2, v);
                v = f8(nib8(m, qo + 24, shift)).fma(x3, v);
                acc = v.fma(FloatVector.broadcast(F8, d * (sm & 63)), acc);
                macc = x0.add(x1).add(x2.add(x3)).fma(FloatVector.broadcast(F8, min * (sm >>> 8)), macc);
            }
        }
        return acc.sub(macc).reduceLanes(VectorOperators.ADD);
    }

    /** The 4-bit codes at bit {@code shift} of 8 bytes. */
    private static ByteVector nib8(MemorySegment m, long at, int shift) {
        return ByteVector.fromMemorySegment(B8, m, at, LE).reinterpretAsInts().lanewise(VectorOperators.LSHR, shift)
                .reinterpretAsBytes().and((byte) 0xF);
    }

    /** Q4_K sub-block {@code ib}'s 6-bit scale and minimum, from the block's 12 bytes of them: {@code s | m << 8}. */
    private static int q4kScale(long lo, int hi, int ib) {
        if (ib < 4) {
            return (sb(lo, hi, ib) & 63) | (sb(lo, hi, ib + 4) & 63) << 8;
        }
        return ((sb(lo, hi, ib + 4) & 0xF) | ((sb(lo, hi, ib - 4) >>> 6) << 4))
                | ((sb(lo, hi, ib + 4) >>> 4) | ((sb(lo, hi, ib) >>> 6) << 4)) << 8;
    }

    private static float q6k(MemorySegment m, long row, float[] x, int at, int k) {
        FloatVector acc = FloatVector.zero(F8);
        for (int b = 0; b < k / 256; b++) {
            long o = row + 210L * b;
            float d = f16(m, o + 208);
            for (int i = 0; i < 16; i++) {
                // half (128 values), c (16 positions l), g: the values at l + 32 g
                int half = i >>> 3;
                int c = (i >>> 2) & 1;
                int g = i & 3;
                long ql = o + 64L * half + 16 * c + 32 * (g & 1);
                long qh = o + 128 + 32L * half + 16 * c;
                int xo = at + 256 * b + 128 * half + 16 * c + 32 * g;
                float scale = d * m.get(ValueLayout.JAVA_BYTE, o + 192 + 8L * half + 2 * g + c);
                FloatVector v = f8(q6k8(m, ql, 4 * (g >> 1), qh, 2 * g)).mul(xs(x, xo));
                v = f8(q6k8(m, ql + 8, 4 * (g >> 1), qh + 8, 2 * g)).fma(xs(x, xo + 8), v);
                acc = v.fma(FloatVector.broadcast(F8, scale), acc);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    /** 8 of Q6_K's values, {@code -32} to {@code 31}: 4 low bits at {@code lowShift}, 2 high at {@code highShift}. */
    private static ByteVector q6k8(MemorySegment m, long ql, int lowShift, long qh, int highShift) {
        IntVector low = ByteVector.fromMemorySegment(B8, m, ql, LE).reinterpretAsInts()
                .lanewise(VectorOperators.LSHR, lowShift).and(0x0F0F0F0F);
        IntVector high = ByteVector.fromMemorySegment(B8, m, qh, LE).reinterpretAsInts()
                .lanewise(VectorOperators.LSHR, highShift).and(0x03030303).lanewise(VectorOperators.LSHL, 4);
        return low.or(high).reinterpretAsBytes().sub((byte) 32);
    }

    /** As {@link BlockKernels#dot}, two vectors of running sums at a time. */
    static float dot(float[] x, int at, float[] y, int k) {
        FloatVector a = FloatVector.zero(FLOATS);
        FloatVector b = FloatVector.zero(FLOATS);
        int p = 0;
        for (; p + 2 * LANES <= k; p += 2 * LANES) {
            a = FloatVector.fromArray(FLOATS, x, at + p).fma(FloatVector.fromArray(FLOATS, y, p), a);
            b = FloatVector.fromArray(FLOATS, x, at + p + LANES).fma(FloatVector.fromArray(FLOATS, y, p + LANES), b);
        }
        float s = a.add(b).reduceLanes(VectorOperators.ADD);
        for (; p < k; p++) {
            s = Math.fma(x[at + p], y[p], s);
        }
        return s;
    }
}
