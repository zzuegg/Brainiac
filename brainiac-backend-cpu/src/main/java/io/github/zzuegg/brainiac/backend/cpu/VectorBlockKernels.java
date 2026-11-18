package io.github.zzuegg.brainiac.backend.cpu;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
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
     * Eight bytes as eight floats. Converted between shapes of the same lane count: measured on
     * JDK 27 (x86, AVX2), taking a part of a wider vector ({@code convertShape(.., part)}) and
     * reinterpreting 8 bytes as ints both compile to slow fallback code (boxed vectors), and
     * whole kernels ran 4 to 7 times slower. Bytes are shifted as bytes
     * ({@code lanewise(LSHR, n)} on 8 or 16 lanes), which does compile to vector code.
     */
    private static FloatVector f8(ByteVector v) {
        return (FloatVector) v.convertShape(VectorOperators.B2F, F8, 0);
    }

    private static FloatVector xs(float[] x, int at) {
        return FloatVector.fromArray(F8, x, at);
    }

    /**
     * Sum over 32 inputs of table[low code] (first 16) and table[high code] (last 16) times x.
     * The two 16-byte lookups go through {@code scratch} (32 bytes) and come back 8 bytes at a
     * time: measured faster than widening halves of the 16-byte vectors, which falls back.
     */
    private static FloatVector lookup32(ByteVector table, ByteVector q, float[] x, int at, byte[] scratch) {
        q.and((byte) 0xF).selectFrom(table).intoArray(scratch, 0);
        q.lanewise(VectorOperators.LSHR, 4).and((byte) 0xF).selectFrom(table).intoArray(scratch, 16);
        FloatVector s = f8(ByteVector.fromArray(B8, scratch, 0)).mul(xs(x, at));
        s = f8(ByteVector.fromArray(B8, scratch, 8)).fma(xs(x, at + 8), s);
        s = f8(ByteVector.fromArray(B8, scratch, 16)).fma(xs(x, at + 16), s);
        return f8(ByteVector.fromArray(B8, scratch, 24)).fma(xs(x, at + 24), s);
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
        byte[] scratch = new byte[32];
        FloatVector acc = FloatVector.zero(F8);
        for (int b = 0; b < k / 32; b++) {
            long o = row + 17L * b;
            float d = Math.scalb(1f, u8(m, o) - 128);
            FloatVector s = lookup32(table, ByteVector.fromMemorySegment(B16, m, o + 1, LE), x, at + 32 * b, scratch);
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
        byte[] scratch = new byte[32];
        FloatVector acc = FloatVector.zero(F8);
        for (int b = 0; b < k / 256; b++) {
            long o = row + 136L * b;
            float d = f16(m, o);
            int high = Short.toUnsignedInt(m.get(SHORT, o + 2));
            for (int ib = 0; ib < 8; ib++) {
                int ls = ((u8(m, o + 4 + ib / 2) >>> (4 * (ib % 2))) & 0xF) | (((high >>> (2 * ib)) & 3) << 4);
                FloatVector s = lookup32(table, ByteVector.fromMemorySegment(B16, m, o + 8 + 16L * ib, LE), x,
                        at + 256 * b + 32 * ib, scratch);
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
            // pairs of sub-blocks share 32 bytes: low codes for the first, high codes for the next
            for (int j = 0; j < 4; j++) {
                int s0 = q4kScale(lo, hi, 2 * j);
                int s1 = q4kScale(lo, hi, 2 * j + 1);
                long qo = o + 16 + 32L * j;
                int xo = at + 256 * b + 64 * j;
                FloatVector v0 = FloatVector.zero(F8);
                FloatVector v1 = FloatVector.zero(F8);
                FloatVector in0 = FloatVector.zero(F8);
                FloatVector in1 = FloatVector.zero(F8);
                for (int i = 0; i < 4; i++) {
                    ByteVector q = ByteVector.fromMemorySegment(B8, m, qo + 8L * i, LE);
                    FloatVector xl = xs(x, xo + 8 * i);
                    FloatVector xh = xs(x, xo + 32 + 8 * i);
                    v0 = f8(q.and((byte) 0xF)).fma(xl, v0);
                    v1 = f8(q.lanewise(VectorOperators.LSHR, 4).and((byte) 0xF)).fma(xh, v1);
                    in0 = in0.add(xl);
                    in1 = in1.add(xh);
                }
                acc = v0.fma(FloatVector.broadcast(F8, d * (s0 & 63)), acc);
                acc = v1.fma(FloatVector.broadcast(F8, d * (s1 & 63)), acc);
                macc = in0.fma(FloatVector.broadcast(F8, min * (s0 >>> 8)), macc);
                macc = in1.fma(FloatVector.broadcast(F8, min * (s1 >>> 8)), macc);
            }
        }
        return acc.sub(macc).reduceLanes(VectorOperators.ADD);
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
        ByteVector low = ByteVector.fromMemorySegment(B8, m, ql, LE).lanewise(VectorOperators.LSHR, lowShift)
                .and((byte) 0x0F);
        ByteVector high = ByteVector.fromMemorySegment(B8, m, qh, LE).lanewise(VectorOperators.LSHR, highShift)
                .and((byte) 0x03).lanewise(VectorOperators.LSHL, 4);
        return low.or(high).sub((byte) 32);
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
