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
    }

    /** A sink that keeps the levels of a row and the scale and offset of each of its groups. */
    record Row(byte[] levels, float[] scales, float[] offsets, int group) implements Sink {

        Row(int length, int group) {
            this(new byte[length], new float[length / group], new float[length / group], group);
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
            case Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K -> 32;
            case Q2_K, Q3_K, Q6_K -> 16;
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
                for (int l = 0; l < 16; l++) {
                    y.put(out++, (u8(m, qs + l) >>> shift) & 3, dl, -ml);
                }
                sc = u8(m, scales + is++);
                dl = d * (sc & 0xF);
                ml = min * (sc >>> 4);
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
                float dl = d * (scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int q = (u8(m, qs + l) >>> shift) & 3;
                    y.put(out++, q - ((u8(m, hm + l) & bit) != 0 ? 0 : 4), dl, 0);
                }
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
            for (int l = 0; l < 32; l++) {
                y.put(out++, u8(m, q + l) & 0xF, d1, -m1);
            }
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
            for (int l = 0; l < 32; l++) {
                y.put(out++, (u8(m, ql + l) & 0xF) + ((u8(m, qh + l) & u1) != 0 ? 16 : 0), d1, -m1);
            }
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
