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

    /** Each thread's copy of the Q8_0 row being multiplied (see {@link #q80}). */
    private static final ThreadLocal<byte[]> ROW = ThreadLocal.withInitial(() -> new byte[0]);

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
        // the row copied to an array first, then read from there: read from the segment, the
        // 2-byte scale and the 8-byte loads of a 34-byte block stayed slow out-of-line accesses
        // on JDK 27 (0.1 GMAC/s; 1.2 with the block in its own method)
        int bytes = 34 * (k / 32);
        byte[] q = ROW.get();
        if (q.length < bytes) {
            q = new byte[bytes];
            ROW.set(q);
        }
        MemorySegment.copy(m, ValueLayout.JAVA_BYTE, row, q, 0, bytes);
        float sum = 0;
        for (int b = 0; b < k / 32; b++) {
            int o = 34 * b;
            short bits = (short) ((q[o] & 0xFF) | q[o + 1] << 8);
            sum += Float.float16ToFloat(bits) * q80Block(q, o + 2, x, at + 32 * b);
        }
        return sum;
    }

    /** One Q8_0 block's 32 codes, at {@code o} in {@code q}, times x, unscaled. */
    private static float q80Block(byte[] q, int o, float[] x, int xo) {
        FloatVector s = ((FloatVector) ByteVector.fromArray(B8, q, o).convertShape(VectorOperators.B2F, F8, 0))
                .mul(FloatVector.fromArray(F8, x, xo));
        s = ((FloatVector) ByteVector.fromArray(B8, q, o + 8).convertShape(VectorOperators.B2F, F8, 0))
                .fma(FloatVector.fromArray(F8, x, xo + 8), s);
        s = ((FloatVector) ByteVector.fromArray(B8, q, o + 16).convertShape(VectorOperators.B2F, F8, 0))
                .fma(FloatVector.fromArray(F8, x, xo + 16), s);
        s = ((FloatVector) ByteVector.fromArray(B8, q, o + 24).convertShape(VectorOperators.B2F, F8, 0))
                .fma(FloatVector.fromArray(F8, x, xo + 24), s);
        return s.reduceLanes(VectorOperators.ADD);
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

    /**
     * Q6_K: per half (128 values) and 16 positions {@code l}, the values at {@code l + 32 g}
     * (g 0 to 3) come from the same bytes: low 4 bits from {@code ql[l]} (g 0, 2) and
     * {@code ql[l + 32]} (g 1, 3), the top 2 from bits {@code 2 g} of {@code qh[l]}. Each 8 of
     * those bytes is loaded once and gives 32 values, with constant byte shifts and masks.
     */
    private static float q6k(MemorySegment m, long row, float[] x, int at, int k) {
        float sum = 0;
        for (int b = 0; b < k / 256; b++) {
            long o = row + 210L * b;
            float d = f16(m, o + 208);
            for (int i = 0; i < 4; i++) {
                int half = i >>> 1;
                int c = i & 1;
                long sc = o + 192 + 8L * half + c;
                sum += q6kGroup(m, o + 64L * half + 16 * c, o + 128 + 32L * half + 16 * c, x,
                        at + 256 * b + 128 * half + 16 * c, d * m.get(ValueLayout.JAVA_BYTE, sc),
                        d * m.get(ValueLayout.JAVA_BYTE, sc + 2), d * m.get(ValueLayout.JAVA_BYTE, sc + 4),
                        d * m.get(ValueLayout.JAVA_BYTE, sc + 6));
            }
        }
        return sum;
    }

    /**
     * 64 of a Q6_K block's values (16 positions, 4 groups 32 apart) times x, the groups scaled
     * by {@code s0} to {@code s3}. Kept apart from {@link #q6k} and returning a float, with no
     * helper calls taking or returning vectors: measured on JDK 27, either made the JIT box the
     * vectors, and the kernel ran 2 times slower.
     */
    private static float q6kGroup(MemorySegment m, long ql, long qh, float[] x, int xo, float s0, float s1, float s2,
            float s3) {
        FloatVector v = FloatVector.zero(F8);
        for (int j = 0; j < 16; j += 8) {
            ByteVector a = ByteVector.fromMemorySegment(B8, m, ql + j, LE);
            ByteVector bb = ByteVector.fromMemorySegment(B8, m, ql + 32 + j, LE);
            ByteVector h = ByteVector.fromMemorySegment(B8, m, qh + j, LE);
            // written out, not through helpers taking vectors: a small helper the JIT compiled on
            // its own was not inlined again here, and its vectors were boxed (measured, 2x slower)
            ByteVector q0 = a.and((byte) 0x0F).or(h.lanewise(VectorOperators.LSHL, 4).and((byte) 0x30)).sub((byte) 32);
            ByteVector q1 = bb.and((byte) 0x0F).or(h.lanewise(VectorOperators.LSHL, 2).and((byte) 0x30)).sub((byte) 32);
            ByteVector q2 = a.lanewise(VectorOperators.LSHR, 4).or(h.and((byte) 0x30)).sub((byte) 32);
            ByteVector q3 = bb.lanewise(VectorOperators.LSHR, 4)
                    .or(h.lanewise(VectorOperators.LSHR, 2).and((byte) 0x30)).sub((byte) 32);
            int xj = xo + j;
            v = ((FloatVector) q0.convertShape(VectorOperators.B2F, F8, 0)).mul(FloatVector.fromArray(F8, x, xj))
                    .fma(FloatVector.broadcast(F8, s0), v);
            v = ((FloatVector) q1.convertShape(VectorOperators.B2F, F8, 0)).mul(FloatVector.fromArray(F8, x, xj + 32))
                    .fma(FloatVector.broadcast(F8, s1), v);
            v = ((FloatVector) q2.convertShape(VectorOperators.B2F, F8, 0)).mul(FloatVector.fromArray(F8, x, xj + 64))
                    .fma(FloatVector.broadcast(F8, s2), v);
            v = ((FloatVector) q3.convertShape(VectorOperators.B2F, F8, 0)).mul(FloatVector.fromArray(F8, x, xj + 96))
                    .fma(FloatVector.broadcast(F8, s3), v);
        }
        return v.reduceLanes(VectorOperators.ADD);
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
