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
 * <p>The formats of most large models (MXFP4, Q8_0, Q4_K, IQ4_XS) are multiplied straight
 * from their blocks, without decoding a row into floats first: levels are loaded 16 bytes
 * at a time, 4-bit codes split into their two halves and, for the non-linear formats,
 * looked up in a 16-entry table with one byte shuffle, widened to floats and multiplied
 * by the inputs; each block's (or sub-block's) sum is scaled once. Q4_K's minimums are folded in
 * through each sub-block's sum of inputs. Needs 8 float lanes (256-bit vectors).
 */
final class VectorBlockKernels {

    private static final VectorSpecies<Float> FLOATS = FloatVector.SPECIES_PREFERRED;
    private static final int LANES = FLOATS.length();
    private static final VectorSpecies<Float> F8 = FloatVector.SPECIES_256;
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
            case "MXFP4", "Q8_0", "Q4_K", "IQ4_XS" -> true;
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
            case "IQ4_XS" -> iq4xs(m, row, x, at, k);
            default -> throw new IllegalArgumentException(format);
        };
    }

    private static FloatVector lane(ByteVector v, int part) {
        return (FloatVector) v.convertShape(VectorOperators.B2F, F8, part);
    }

    private static FloatVector xs(float[] x, int at) {
        return FloatVector.fromArray(F8, x, at);
    }

    /** Sum over 32 inputs of table[low code] (first 16) and table[high code] (last 16) times x. */
    private static FloatVector lookup32(ByteVector table, ByteVector q, float[] x, int at) {
        ByteVector lo = q.and((byte) 0xF).selectFrom(table);
        ByteVector hi = q.lanewise(VectorOperators.LSHR, 4).and((byte) 0xF).selectFrom(table);
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
            ByteVector q0 = ByteVector.fromMemorySegment(B16, m, o + 2, LE);
            ByteVector q1 = ByteVector.fromMemorySegment(B16, m, o + 18, LE);
            int xo = at + 32 * b;
            FloatVector s = lane(q0, 0).mul(xs(x, xo));
            s = lane(q0, 1).fma(xs(x, xo + 8), s);
            s = lane(q1, 0).fma(xs(x, xo + 16), s);
            s = lane(q1, 1).fma(xs(x, xo + 24), s);
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
        float sum = 0;
        for (int b = 0; b < k / 256; b++) {
            long o = row + 144L * b;
            float d = f16(m, o);
            float min = f16(m, o + 2);
            // the 12 bytes of 6-bit scales and minimums, read at once
            long lo = m.get(LONG, o + 4);
            int hi = m.get(INT, o + 12);
            for (int j = 0; j < 4; j++) {
                int is = 2 * j;
                int s0;
                int m0;
                int s1;
                int m1;
                if (is < 4) {
                    s0 = sb(lo, hi, is) & 63;
                    m0 = sb(lo, hi, is + 4) & 63;
                    s1 = sb(lo, hi, is + 1) & 63;
                    m1 = sb(lo, hi, is + 5) & 63;
                } else {
                    s0 = (sb(lo, hi, is + 4) & 0xF) | ((sb(lo, hi, is - 4) >>> 6) << 4);
                    m0 = (sb(lo, hi, is + 4) >>> 4) | ((sb(lo, hi, is) >>> 6) << 4);
                    s1 = (sb(lo, hi, is + 5) & 0xF) | ((sb(lo, hi, is - 3) >>> 6) << 4);
                    m1 = (sb(lo, hi, is + 5) >>> 4) | ((sb(lo, hi, is + 1) >>> 6) << 4);
                }
                sum += q4kChunk(m, o + 16 + 32L * j, x, at + 256 * b + 64 * j, d * s0, -(min * m0), d * s1,
                        -(min * m1));
            }
        }
        return sum;
    }

    /**
     * One 64-input piece of a Q4_K block: 32 bytes whose low codes weigh the first 32 inputs
     * (scale {@code d0}, offset {@code o0}) and high codes the next 32. Returns a float, not a
     * vector: the JIT compiles this on its own (too large to inline), and a vector passed
     * across a call that is not inlined would be boxed.
     */
    private static float q4kChunk(MemorySegment m, long qo, float[] x, int xo, float d0, float o0, float d1,
            float o1) {
        ByteVector q0 = ByteVector.fromMemorySegment(B16, m, qo, LE);
        ByteVector q1 = ByteVector.fromMemorySegment(B16, m, qo + 16, LE);
        ByteVector a = q0.and((byte) 0xF);
        ByteVector b = q1.and((byte) 0xF);
        FloatVector x0 = xs(x, xo);
        FloatVector x1 = xs(x, xo + 8);
        FloatVector x2 = xs(x, xo + 16);
        FloatVector x3 = xs(x, xo + 24);
        FloatVector lo = lane(a, 0).mul(x0);
        lo = lane(a, 1).fma(x1, lo);
        lo = lane(b, 0).fma(x2, lo);
        lo = lane(b, 1).fma(x3, lo);
        FloatVector loInputs = x0.add(x1).add(x2.add(x3));
        a = q0.lanewise(VectorOperators.LSHR, 4).and((byte) 0xF);
        b = q1.lanewise(VectorOperators.LSHR, 4).and((byte) 0xF);
        x0 = xs(x, xo + 32);
        x1 = xs(x, xo + 40);
        x2 = xs(x, xo + 48);
        x3 = xs(x, xo + 56);
        FloatVector hi = lane(a, 0).mul(x0);
        hi = lane(a, 1).fma(x1, hi);
        hi = lane(b, 0).fma(x2, hi);
        hi = lane(b, 1).fma(x3, hi);
        FloatVector hiInputs = x0.add(x1).add(x2.add(x3));
        FloatVector sum = lo.mul(d0).add(hi.mul(d1)).add(loInputs.mul(o0)).add(hiInputs.mul(o1));
        return sum.reduceLanes(VectorOperators.ADD);
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
