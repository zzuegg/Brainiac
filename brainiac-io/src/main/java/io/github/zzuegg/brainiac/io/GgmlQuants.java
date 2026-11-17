package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.spi.RawValues;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/** Decodes GGML's block-quantized formats into values, block by block, as llama.cpp does. */
final class GgmlQuants {

    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private GgmlQuants() {
    }

    /** Receives value {@code i} as {@code q * scale + offset}, q a whole number. */
    interface Sink {
        void put(int i, int q, float scale, float offset);

        /**
         * Says, before the values of a group starting at {@code i}, how its scale and offset
         * factor: {@code scale = scaleCode * d} and {@code offset = offsetCode * m}, products in float.
         */
        default void factors(int i, int scaleCode, int offsetCode, float d, float m) {
        }
    }

    /** A sink that keeps the levels of a row and the scale and offset of each of its groups. */
    record Row(byte[] levels, float[] scales, float[] offsets, int group, byte[] scaleCodes, byte[] offsetCodes,
            float[] d, float[] m) implements Sink {

        Row(int length, int group) {
            this(new byte[length], new float[length / group], new float[length / group], group,
                    new byte[length / group], new byte[length / group], new float[length / group],
                    new float[length / group]);
        }

        @Override
        public void factors(int i, int scaleCode, int offsetCode, float d, float m) {
            int g = i / group;
            scaleCodes[g] = (byte) scaleCode;
            offsetCodes[g] = (byte) offsetCode;
            this.d[g] = d;
            this.m[g] = m;
        }

        @Override
        public void put(int i, int q, float scale, float offset) {
            levels[i] = (byte) q;
            if (i % group == 0) {
                scales[i / group] = scale;
                offsets[i / group] = offset;
            }
        }
    }

    /** A sink that only computes the values, rounding as llama.cpp does. */
    private record Values(double[] y, int base) implements Sink {
        @Override
        public void put(int i, int q, float scale, float offset) {
            y[base + i] = offset == 0 ? q * scale : q * scale + offset;
        }
    }

    static double[] decode(Gguf.Format format, MemorySegment m, long count) {
        double[] y = new double[Math.toIntExact(count)];
        switch (format) {
            case I8, I16, I32, I64 -> {
                for (int i = 0; i < y.length; i++) {
                    long at = (long) i * format.bytes;
                    y[i] = switch (format) {
                        case I8 -> i8(m, at);
                        case I16 -> m.get(RawValues.SHORT, at);
                        case I32 -> m.get(INT, at);
                        default -> m.get(LONG, at);
                    };
                }
                return y;
            }
            default -> {
            }
        }
        // in pieces of whole blocks, in parallel
        int blocks = Math.toIntExact(count / format.block);
        int perTask = Math.max(1, 65536 / format.block);
        java.util.stream.IntStream.range(0, Math.ceilDiv(blocks, perTask)).parallel().forEach(task -> {
            int first = task * perTask;
            int n = Math.min(blocks, first + perTask) - first;
            levels(format, m.asSlice((long) first * format.bytes, (long) n * format.bytes), (long) n * format.block,
                    new Values(y, first * format.block));
        });
        return y;
    }

    /**
     * Rows per scale of the formats that {@link #levels} reads, or 0: 32 for the simple
     * formats, Q4_K and Q5_K; 16 for Q2_K, Q3_K and Q6_K.
     */
    static int group(Gguf.Format format) {
        return switch (format) {
            case Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, IQ4_NL, IQ4_XS, MXFP4, IQ2_XXS, IQ3_XXS, IQ3_S, IQ1_S -> 32;
            case Q2_K, Q3_K, Q6_K, IQ2_XS, IQ2_S, IQ1_M -> 16;
            default -> 0;
        };
    }

    /** Bits per level, as {@link #levels} gives them (unsigned where the format has an offset). */
    static int bits(Gguf.Format format) {
        return switch (format) {
            case Q2_K -> 2;
            case Q3_K -> 3;
            case Q4_0, Q4_1, Q4_K -> 4;
            case Q5_0, Q5_1, Q5_K -> 5;
            case Q6_K -> 6;
            default -> 8;
        };
    }

    /** Whether the format adds an offset per group (its levels are then unsigned). */
    static boolean offsets(Gguf.Format format) {
        return format == Gguf.Format.Q4_1 || format == Gguf.Format.Q5_1 || format == Gguf.Format.Q2_K
                || format == Gguf.Format.Q4_K || format == Gguf.Format.Q5_K;
    }

    /** Reads every value as a level, a scale and an offset. */
    static void levels(Gguf.Format format, MemorySegment m, long count, Sink y) {
        int blocks = (int) (count / format.block);
        for (int b = 0; b < blocks; b++) {
            long at = (long) b * format.bytes;
            int o = b * format.block;
            switch (format) {
                case Q4_0 -> q4(m, at, y, o, false);
                case Q4_1 -> q4(m, at, y, o, true);
                case Q5_0 -> q5(m, at, y, o, false);
                case Q5_1 -> q5(m, at, y, o, true);
                case Q8_0 -> {
                    float d = f16(m, at);
                    y.factors(o, 1, 0, d, 0);
                    for (int j = 0; j < 32; j++) {
                        y.put(o + j, i8(m, at + 2 + j), d, 0);
                    }
                }
                case Q8_1 -> {
                    float d = f16(m, at);
                    for (int j = 0; j < 32; j++) {
                        y.put(o + j, i8(m, at + 4 + j), d, 0);
                    }
                }
                case Q2_K -> q2k(m, at, y, o);
                case Q3_K -> q3k(m, at, y, o);
                case Q4_K -> q4k(m, at, y, o);
                case Q5_K -> q5k(m, at, y, o);
                case Q6_K -> q6k(m, at, y, o);
                case IQ4_NL -> nonLinear(m, at + 2, y, o, IQ4, f16(m, at));
                case IQ4_XS -> iq4xs(m, at, y, o);
                case IQ2_XXS -> iq2xxs(m, at, y, o);
                case IQ2_XS -> iq2xs(m, at, y, o);
                case IQ2_S -> iq2s(m, at, y, o);
                case IQ3_XXS -> iq3xxs(m, at, y, o);
                case IQ3_S -> iq3s(m, at, y, o);
                case IQ1_S -> iq1s(m, at, y, o);
                case IQ1_M -> iq1m(m, at, y, o);
                case MXFP4 -> nonLinear(m, at + 1, y, o, MXFP4_VALUES, Math.scalb(1f, u8(m, at) - 128));
                case Q8_K -> {
                    float d = m.get(RawValues.FLOAT, at);
                    for (int j = 0; j < 256; j++) {
                        y.put(o + j, i8(m, at + 4 + j), d, 0);
                    }
                }
                default -> throw new IllegalArgumentException("cannot decode " + format);
            }
        }
    }

    /** IQ4_NL's and IQ4_XS's sixteen levels (llama.cpp's kvalues_iq4nl). */
    private static final int[] IQ4 = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    /** MXFP4's sixteen E2M1 values, doubled to whole numbers (the scale is halved to match). */
    private static final int[] MXFP4_VALUES = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    /** 32 values from 16 bytes of 4-bit indices into a table: low nibbles first, then high. */
    private static void nonLinear(MemorySegment m, long qs, Sink y, int o, int[] table, float d) {
        for (int j = 0; j < 16; j++) {
            int q = u8(m, qs + j);
            y.put(o + j, table[q & 0xF], d, 0);
            y.put(o + j + 16, table[q >>> 4], d, 0);
        }
    }

    private static void iq4xs(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        int high = Short.toUnsignedInt(m.get(RawValues.SHORT, at + 2));
        long low = at + 4;
        long qs = at + 8;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((u8(m, low + ib / 2) >>> (4 * (ib % 2))) & 0xF) | (((high >>> (2 * ib)) & 3) << 4);
            nonLinear(m, qs + 16L * ib, y, o + 32 * ib, IQ4, d * (ls - 32));
        }
    }

    // ---------------------------------------------------------------- i-quants
    // Grids of whole levels from llama.cpp's ggml-common.h, as gguf-py unpacks them (iq-grids.bin):
    // iq2xxs 256x8, iq2xs 512x8, iq2s 1024x8, iq3xxs 256x4, iq3s 512x4, iq1s 2048x8.

    private static final byte[] GRIDS;
    private static final int IQ2XXS = 0;
    private static final int IQ2XS = IQ2XXS + 256 * 8;
    private static final int IQ2S = IQ2XS + 512 * 8;
    private static final int IQ3XXS = IQ2S + 1024 * 8;
    private static final int IQ3S = IQ3XXS + 256 * 4;
    private static final int IQ1S = IQ3S + 512 * 4;

    static {
        try (java.io.InputStream in = GgmlQuants.class.getResourceAsStream("iq-grids.bin")) {
            GRIDS = java.util.Objects.requireNonNull(in, "iq-grids.bin").readAllBytes();
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
        if (GRIDS.length != IQ1S + 2048 * 8) {
            throw new IllegalStateException("iq-grids.bin has " + GRIDS.length + " bytes");
        }
    }

    /** Seven sign bits and an eighth that makes their count even (llama.cpp's ksigns_iq2xs). */
    private static int signs(int index) {
        return index | ((Integer.bitCount(index) & 1) << 7);
    }

    private static int sign(int bits, int j) {
        return (bits >>> j & 1) != 0 ? -1 : 1;
    }

    private static long u32(MemorySegment m, long at) {
        return Integer.toUnsignedLong(m.get(INT, at));
    }

    private static int u16(MemorySegment m, long at) {
        return Short.toUnsignedInt(m.get(RawValues.SHORT, at));
    }

    private static void iq2xxs(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        for (int ib = 0; ib < 8; ib++) {
            long a0 = u32(m, at + 2 + 8L * ib);
            long a1 = u32(m, at + 6 + 8L * ib);
            float db = d * (0.5f + (a1 >>> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                int grid = IQ2XXS + (int) (a0 >>> (8 * l) & 0xFF) * 8;
                int bits = signs((int) (a1 >>> (7 * l) & 0x7F));
                for (int j = 0; j < 8; j++) {
                    y.put(o + 32 * ib + 8 * l + j, GRIDS[grid + j] * sign(bits, j), db, 0);
                }
            }
        }
    }

    private static void iq2xs(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        long scales = at + 66;
        for (int i = 0; i < 32; i++) {   // 32 groups of 8 values, two per scale
            int q = u16(m, at + 2 + 2L * i);
            int s = u8(m, scales + i / 4) >>> (4 * (i / 2 % 2)) & 0xF;
            float db = d * (0.5f + s) * 0.25f;
            int grid = IQ2XS + (q & 511) * 8;
            int bits = signs(q >>> 9);
            for (int j = 0; j < 8; j++) {
                y.put(o + 8 * i + j, GRIDS[grid + j] * sign(bits, j), db, 0);
            }
        }
    }

    private static void iq2s(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        long qs = at + 2;
        long sg = at + 34;
        long qh = at + 66;
        long scales = at + 74;
        for (int i = 0; i < 32; i++) {
            int index = u8(m, qs + i) | (u8(m, qh + i / 4) >>> (2 * (i % 4)) & 3) << 8;
            int s = u8(m, scales + i / 4) >>> (4 * (i / 2 % 2)) & 0xF;
            float db = d * (0.5f + s) * 0.25f;
            int bits = u8(m, sg + i);
            int grid = IQ2S + index * 8;
            for (int j = 0; j < 8; j++) {
                y.put(o + 8 * i + j, GRIDS[grid + j] * sign(bits, j), db, 0);
            }
        }
    }

    private static void iq3xxs(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        long qs = at + 2;
        for (int ib = 0; ib < 8; ib++) {
            long a = u32(m, at + 66 + 4L * ib);
            float db = d * (0.5f + (a >>> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                int bits = signs((int) (a >>> (7 * l) & 0x7F));
                for (int j = 0; j < 8; j++) {
                    int grid = IQ3XXS + u8(m, qs + 8L * ib + 2 * l + j / 4) * 4;
                    y.put(o + 32 * ib + 8 * l + j, GRIDS[grid + j % 4] * sign(bits, j), db, 0);
                }
            }
        }
    }

    private static void iq3s(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        long qs = at + 2;
        long qh = at + 66;
        long sg = at + 74;
        long scales = at + 106;
        for (int i = 0; i < 64; i++) {   // 64 groups of 4 values
            int index = u8(m, qs + i) | (u8(m, qh + i / 8) >>> (i % 8) & 1) << 8;
            int ib = i / 8;
            int s = u8(m, scales + ib / 2) >>> (4 * (ib % 2)) & 0xF;
            float db = d * (1 + 2 * s);
            int grid = IQ3S + index * 4;
            for (int j = 0; j < 4; j++) {
                int v = 4 * i + j;
                y.put(o + v, GRIDS[grid + j] * sign(u8(m, sg + v / 8), v % 8), db, 0);
            }
        }
    }

    /** IQ1_S: levels -1, 0, 1 shifted by 1/8 of the step, kept as 8 times the level plus or minus 1. */
    private static void iq1s(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        long qs = at + 2;
        for (int ib = 0; ib < 8; ib++) {
            int h = u16(m, at + 34 + 2L * ib);
            float dl = d * (2 * (h >>> 12 & 7) + 1) / 8;
            int delta = (h & 0x8000) == 0 ? 1 : -1;
            for (int l = 0; l < 4; l++) {
                int grid = IQ1S + (u8(m, qs + 4L * ib + l) | (h >>> (3 * l) & 7) << 8) * 8;
                for (int j = 0; j < 8; j++) {
                    y.put(o + 32 * ib + 8 * l + j, 8 * GRIDS[grid + j] + delta, dl, 0);
                }
            }
        }
    }

    private static void iq1m(MemorySegment m, long at, Sink y, int o) {
        long qs = at;
        long qh = at + 32;
        int[] sc = new int[4];
        for (int k = 0; k < 4; k++) {
            sc[k] = u16(m, at + 48 + 2L * k);
        }
        short bits = (short) ((sc[0] & 0xF000) >>> 12 | (sc[1] & 0xF000) >>> 8 | (sc[2] & 0xF000) >>> 4 | sc[3] & 0xF000);
        float d = Float.float16ToFloat(bits);
        for (int i = 0; i < 32; i++) {   // 32 groups of 8 values, two per scale
            int nibble = u8(m, qh + i / 2) >>> (4 * (i % 2)) & 0xF;
            int index = u8(m, qs + i) | (nibble & 7) << 8;
            int delta = (nibble & 8) == 0 ? 1 : -1;
            int k = i / 2;   // sixteen 3-bit scales, four per 16-bit word
            float dl = d * (2 * (sc[k / 4] >>> (3 * (k % 4)) & 7) + 1) / 8;
            int grid = IQ1S + index * 8;
            for (int j = 0; j < 8; j++) {
                y.put(o + 8 * i + j, 8 * GRIDS[grid + j] + delta, dl, 0);
            }
        }
    }

    private static float f16(MemorySegment m, long at) {
        return Float.float16ToFloat(m.get(RawValues.SHORT, at));
    }

    private static int u8(MemorySegment m, long at) {
        return Byte.toUnsignedInt(m.get(ValueLayout.JAVA_BYTE, at));
    }

    private static int i8(MemorySegment m, long at) {
        return m.get(ValueLayout.JAVA_BYTE, at);
    }

    private static void q4(MemorySegment m, long at, Sink y, int o, boolean min) {
        float d = f16(m, at);
        float mn = min ? f16(m, at + 2) : 0;
        y.factors(o, 1, min ? 1 : 0, d, mn);
        long qs = at + (min ? 4 : 2);
        for (int j = 0; j < 16; j++) {
            int q = u8(m, qs + j);
            if (min) {
                y.put(o + j, q & 0xF, d, mn);
                y.put(o + j + 16, q >>> 4, d, mn);
            } else {
                y.put(o + j, (q & 0xF) - 8, d, 0);
                y.put(o + j + 16, (q >>> 4) - 8, d, 0);
            }
        }
    }

    private static void q5(MemorySegment m, long at, Sink y, int o, boolean min) {
        float d = f16(m, at);
        float mn = min ? f16(m, at + 2) : 0;
        y.factors(o, 1, min ? 1 : 0, d, mn);
        long hAt = at + (min ? 4 : 2);
        int qh = m.get(INT, hAt);
        long qs = hAt + 4;
        for (int j = 0; j < 16; j++) {
            int q = u8(m, qs + j);
            int x0 = (q & 0xF) | (((qh >>> j) & 1) << 4);
            int x1 = (q >>> 4) | (((qh >>> (j + 16)) & 1) << 4);
            if (min) {
                y.put(o + j, x0, d, mn);
                y.put(o + j + 16, x1, d, mn);
            } else {
                y.put(o + j, x0 - 16, d, 0);
                y.put(o + j + 16, x1 - 16, d, 0);
            }
        }
    }

    private static void q2k(MemorySegment m, long at, Sink y, int o) {
        long scales = at;
        long qs = at + 16;
        float d = f16(m, at + 80);
        float min = f16(m, at + 82);
        int is = 0;
        int out = o;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int sc = u8(m, scales + is++);
                float dl = d * (sc & 0xF);
                float ml = min * (sc >>> 4);
                y.factors(out, sc & 0xF, sc >>> 4, d, -min);
                for (int l = 0; l < 16; l++) {
                    y.put(out++, (u8(m, qs + l) >>> shift) & 3, dl, -ml);
                }
                sc = u8(m, scales + is++);
                dl = d * (sc & 0xF);
                ml = min * (sc >>> 4);
                y.factors(out, sc & 0xF, sc >>> 4, d, -min);
                for (int l = 0; l < 16; l++) {
                    y.put(out++, (u8(m, qs + l + 16) >>> shift) & 3, dl, -ml);
                }
                shift += 2;
            }
            qs += 32;
        }
    }

    private static void q3k(MemorySegment m, long at, Sink y, int o) {
        long hm = at;
        long qs = at + 32;
        long sAt = at + 96;
        float d = f16(m, at + 108);
        int[] aux = {m.get(INT, sAt), m.get(INT, sAt + 4), m.get(INT, sAt + 8), 0};
        int kmask1 = 0x03030303;
        int kmask2 = 0x0f0f0f0f;
        int tmp = aux[2];
        aux[2] = ((aux[0] >>> 4) & kmask2) | (((tmp >>> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >>> 4) & kmask2) | (((tmp >>> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >>> 2) & kmask1) << 4);
        byte[] scales = new byte[16];
        for (int i = 0; i < 16; i++) {
            scales[i] = (byte) (aux[i / 4] >>> (8 * (i % 4)));
        }
        int bit = 1;
        int is = 0;
        int out = o;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                y.factors(out, scales[is] - 32, 0, d, 0);
                float dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = (u8(m, qs + l) >>> shift) & 3;
                    y.put(out++, q - ((u8(m, hm + l) & bit) != 0 ? 0 : 4), dl, 0);
                }
                y.factors(out, scales[is] - 32, 0, d, 0);
                dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = (u8(m, qs + l + 16) >>> shift) & 3;
                    y.put(out++, q - ((u8(m, hm + l + 16) & bit) != 0 ? 0 : 4), dl, 0);
                }
                shift += 2;
                bit <<= 1;
            }
            qs += 32;
        }
    }

    /** The 6-bit scale and min number {@code j} of a K-quant super-block. */
    private static int[] scaleMin(MemorySegment m, long q, int j) {
        if (j < 4) {
            return new int[] {u8(m, q + j) & 63, u8(m, q + j + 4) & 63};
        }
        return new int[] {(u8(m, q + j + 4) & 0xF) | ((u8(m, q + j - 4) >>> 6) << 4),
                (u8(m, q + j + 4) >>> 4) | ((u8(m, q + j) >>> 6) << 4)};
    }

    private static void q4k(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        float min = f16(m, at + 2);
        long scales = at + 4;
        long q = at + 16;
        int is = 0;
        int out = o;
        for (int j = 0; j < 256; j += 64) {
            int[] a = scaleMin(m, scales, is);
            int[] b = scaleMin(m, scales, is + 1);
            float d1 = d * a[0];
            float m1 = min * a[1];
            float d2 = d * b[0];
            float m2 = min * b[1];
            y.factors(out, a[0], a[1], d, -min);
            for (int l = 0; l < 32; l++) {
                y.put(out++, u8(m, q + l) & 0xF, d1, -m1);
            }
            y.factors(out, b[0], b[1], d, -min);
            for (int l = 0; l < 32; l++) {
                y.put(out++, u8(m, q + l) >>> 4, d2, -m2);
            }
            q += 32;
            is += 2;
        }
    }

    private static void q5k(MemorySegment m, long at, Sink y, int o) {
        float d = f16(m, at);
        float min = f16(m, at + 2);
        long scales = at + 4;
        long qh = at + 16;
        long ql = at + 48;
        int is = 0;
        int u1 = 1;
        int u2 = 2;
        int out = o;
        for (int j = 0; j < 256; j += 64) {
            int[] a = scaleMin(m, scales, is);
            int[] b = scaleMin(m, scales, is + 1);
            float d1 = d * a[0];
            float m1 = min * a[1];
            float d2 = d * b[0];
            float m2 = min * b[1];
            y.factors(out, a[0], a[1], d, -min);
            for (int l = 0; l < 32; l++) {
                y.put(out++, (u8(m, ql + l) & 0xF) + ((u8(m, qh + l) & u1) != 0 ? 16 : 0), d1, -m1);
            }
            y.factors(out, b[0], b[1], d, -min);
            for (int l = 0; l < 32; l++) {
                y.put(out++, (u8(m, ql + l) >>> 4) + ((u8(m, qh + l) & u2) != 0 ? 16 : 0), d2, -m2);
            }
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    private static void q6k(MemorySegment m, long at, Sink y, int o) {
        long ql = at;
        long qh = at + 128;
        long sc = at + 192;
        float d = f16(m, at + 208);
        int out = o;
        for (int n = 0; n < 256; n += 128) {
            for (int g = 0; g < 8; g++) {
                y.factors(out + g * 16, i8(m, sc + (g % 2) + (g / 2) * 2), 0, d, 0);
            }
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int h = u8(m, qh + l);
                int q1 = ((u8(m, ql + l) & 0xF) | ((h & 3) << 4)) - 32;
                int q2 = ((u8(m, ql + l + 32) & 0xF) | (((h >>> 2) & 3) << 4)) - 32;
                int q3 = ((u8(m, ql + l) >>> 4) | (((h >>> 4) & 3) << 4)) - 32;
                int q4 = ((u8(m, ql + l + 32) >>> 4) | (((h >>> 6) & 3) << 4)) - 32;
                y.put(out + l, q1, d * i8(m, sc + is), 0);
                y.put(out + l + 32, q2, d * i8(m, sc + is + 2), 0);
                y.put(out + l + 64, q3, d * i8(m, sc + is + 4), 0);
                y.put(out + l + 96, q4, d * i8(m, sc + is + 6), 0);
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
