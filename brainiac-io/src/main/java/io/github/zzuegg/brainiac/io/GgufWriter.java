package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Writes GGUF files (version 3), as llama.cpp reads them: metadata, then tensors in float32
 * or float16 or quantized ({@link Type}), aligned to 32 bytes.
 *
 * <p>Metadata values map to GGUF types by their Java type: {@code String} to string,
 * {@code Boolean} to bool, {@code Integer} to int32, {@code Long} to uint32 (uint64 when it
 * does not fit), {@code Float} and {@code Double} to float32, and a {@code List} to an array
 * of its first element's type. {@link Typed} sets another type.
 */
public final class GgufWriter {

    private static final int MAGIC = 0x46554747;
    private static final int ALIGNMENT = 32;

    /** GGUF value types. */
    public static final int UINT8 = 0;
    /** GGUF value type int8. */
    public static final int INT8 = 1;
    /** GGUF value type uint16. */
    public static final int UINT16 = 2;
    /** GGUF value type int16. */
    public static final int INT16 = 3;
    /** GGUF value type uint32. */
    public static final int UINT32 = 4;
    /** GGUF value type int32. */
    public static final int INT32 = 5;
    /** GGUF value type float32. */
    public static final int FLOAT32 = 6;
    /** GGUF value type bool. */
    public static final int BOOL = 7;
    /** GGUF value type string. */
    public static final int STRING = 8;
    /** GGUF value type array. */
    public static final int ARRAY = 9;
    /** GGUF value type uint64. */
    public static final int UINT64 = 10;
    /** GGUF value type int64. */
    public static final int INT64 = 11;
    /** GGUF value type float64. */
    public static final int FLOAT64 = 12;

    /**
     * A metadata value with an explicit GGUF type; for an array, the type of its items.
     *
     * @param type one of the type constants
     * @param value a number, string, boolean or list
     */
    public record Typed(int type, Object value) {
    }

    private GgufWriter() {
    }

    /**
     * How a tensor is stored: as floats, or quantized in one of llama.cpp's block formats,
     * written as {@code ggml-quants.c}'s reference quantizers lay them out. Rows whose length
     * is not a multiple of the block go in Q8_0 (for a k-quant, when they fit its blocks of 32)
     * or float16 instead.
     */
    public enum Type {
        /** 32-bit floats. */
        F32(0, 1, 4),
        /** 16-bit floats. */
        F16(1, 1, 2),
        /** 4.5 bits: blocks of 32 with a float16 scale. */
        Q4_0(2, 32, 18),
        /** 8.5 bits: blocks of 32 with a float16 scale; nearly lossless. */
        Q8_0(8, 32, 34),
        /** 4.5 bits: super-blocks of 256, eight 6-bit scales and minimums; llama.cpp's Q4_K_M base. */
        Q4_K(12, 256, 144),
        /** 6.56 bits: super-blocks of 256 with sixteen 8-bit scales. */
        Q6_K(14, 256, 210);

        final int id;
        final int block;
        final int bytes;

        Type(int id, int block, int bytes) {
            this.id = id;
            this.block = block;
            this.bytes = bytes;
        }

        /** @return the GGML type id written to the file */
        public int id() {
            return id;
        }

        /**
         * @param rowLength the values per row
         * @return whether rows of that length fit this format's blocks
         */
        public boolean fits(int rowLength) {
            return rowLength % block == 0;
        }
    }

    /**
     * Writes a file.
     *
     * @param file where to write
     * @param metadata the key-value pairs, in order
     * @param tensors the tensors by GGUF name, in order, any shape
     * @param half whether to store matrices (rank 2 and more) as float16; vectors stay float32
     */
    public static void write(Path file, Map<String, Object> metadata, Map<String, Tensor> tensors, boolean half) {
        write(file, metadata, tensors, name -> half ? Type.F16 : Type.F32);
    }

    /**
     * Writes a file with a storage type per tensor. Vectors and scalars always stay float32,
     * as llama.cpp keeps norms and biases; a matrix whose rows do not fit the type's blocks
     * falls back as {@link Type} says.
     *
     * @param file where to write
     * @param metadata the key-value pairs, in order
     * @param tensors the tensors by GGUF name, in order, any shape
     * @param type the type of each matrix, by name
     */
    public static void write(Path file, Map<String, Object> metadata, Map<String, Tensor> tensors,
            java.util.function.Function<String, Type> type) {
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(file), 1 << 20)) {
            Out o = new Out(out);
            o.u32(MAGIC);
            o.u32(3);
            o.u64(tensors.size());
            o.u64(metadata.size());
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                o.string(e.getKey());
                Object v = e.getValue();
                int vt = typeOf(v);
                o.u32(vt);
                o.value(vt, v instanceof Typed t ? t.value() : v, v instanceof Typed t ? t.type() : -1);
            }
            long offset = 0;
            Type[] types = new Type[tensors.size()];
            int i = 0;
            for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
                int[] dims = e.getValue().shape().toArray();
                Type t = dims.length >= 2 ? type.apply(e.getKey()) : Type.F32;
                if (!t.fits(dims[dims.length - 1])) {
                    // as llama-quantize falls back: k-quants to a 32-block format, else float16
                    t = Type.Q8_0.fits(dims[dims.length - 1]) && t.block > 32 ? Type.Q8_0 : Type.F16;
                }
                types[i] = t;
                o.string(e.getKey());
                o.u32(Math.max(1, dims.length));
                if (dims.length == 0) {
                    o.u64(1);
                }
                for (int d = dims.length - 1; d >= 0; d--) {
                    o.u64(dims[d]);   // fastest first
                }
                o.u32(t.id);
                o.u64(offset);
                offset = align(offset + e.getValue().size() / t.block * t.bytes);
                i++;
            }
            o.pad();
            i = 0;
            for (Tensor t : tensors.values()) {
                o.bytes(encode(t.to(DType.FLOAT32).toArray(), types[i]));
                o.pad();
                i++;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Encodes values in a storage type, as {@code ggml-quants.c}'s reference quantizers do
     * (Q4_K and Q6_K with plain min/max scales rather than its iterative search).
     *
     * @param values the values, a whole number of blocks
     * @param type the format
     * @return the bytes
     */
    public static byte[] encode(float[] values, Type type) {
        if (values.length % type.block != 0) {
            throw new IllegalArgumentException(values.length + " values are not whole " + type + " blocks");
        }
        ByteBuffer b = ByteBuffer.allocate(values.length / type.block * type.bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int at = 0; at < values.length; at += type.block) {
            switch (type) {
                case F32 -> b.putFloat(values[at]);
                case F16 -> b.putShort(Float.floatToFloat16(values[at]));
                case Q8_0 -> q8_0(values, at, b);
                case Q4_0 -> q4_0(values, at, b);
                case Q4_K -> q4_k(values, at, b);
                case Q6_K -> q6_k(values, at, b);
            }
        }
        return b.array();
    }

    private static void q8_0(float[] x, int at, ByteBuffer b) {
        float amax = 0;
        for (int i = 0; i < 32; i++) {
            amax = Math.max(amax, Math.abs(x[at + i]));
        }
        float d = amax / 127;
        float id = d != 0 ? 1 / d : 0;
        b.putShort(Float.floatToFloat16(d));
        for (int i = 0; i < 32; i++) {
            b.put((byte) Math.round(x[at + i] * id));
        }
    }

    private static void q4_0(float[] x, int at, ByteBuffer b) {
        float amax = 0;
        float max = 0;
        for (int i = 0; i < 32; i++) {
            if (Math.abs(x[at + i]) > amax) {
                amax = Math.abs(x[at + i]);
                max = x[at + i];
            }
        }
        float d = max / -8;
        float id = d != 0 ? 1 / d : 0;
        b.putShort(Float.floatToFloat16(d));
        for (int i = 0; i < 16; i++) {
            int lo = Math.min(15, (int) (x[at + i] * id + 8.5f));
            int hi = Math.min(15, (int) (x[at + 16 + i] * id + 8.5f));
            b.put((byte) (lo | hi << 4));
        }
    }

    private static void q4_k(float[] x, int at, ByteBuffer b) {
        float[] scales = new float[8];
        float[] mins = new float[8];
        float maxScale = 0;
        float maxMin = 0;
        for (int j = 0; j < 8; j++) {
            float lo = 0;   // the minimum is at most 0, as in ggml
            float hi = -Float.MAX_VALUE;
            for (int i = 0; i < 32; i++) {
                lo = Math.min(lo, x[at + 32 * j + i]);
                hi = Math.max(hi, x[at + 32 * j + i]);
            }
            scales[j] = Math.max(0, (hi - lo) / 15);
            mins[j] = -lo;
            maxScale = Math.max(maxScale, scales[j]);
            maxMin = Math.max(maxMin, mins[j]);
        }
        float invScale = maxScale > 0 ? 63 / maxScale : 0;
        float invMin = maxMin > 0 ? 63 / maxMin : 0;
        int[] ls = new int[8];
        int[] lm = new int[8];
        byte[] packed = new byte[12];
        for (int j = 0; j < 8; j++) {
            ls[j] = Math.min(63, Math.round(invScale * scales[j]));
            lm[j] = Math.min(63, Math.round(invMin * mins[j]));
            if (j < 4) {
                packed[j] = (byte) ls[j];
                packed[j + 4] = (byte) lm[j];
            } else {
                packed[j + 4] = (byte) ((ls[j] & 0xF) | (lm[j] & 0xF) << 4);
                packed[j - 4] |= (byte) ((ls[j] >> 4) << 6);
                packed[j] |= (byte) ((lm[j] >> 4) << 6);
            }
        }
        short dh = Float.floatToFloat16(maxScale / 63);
        short mh = Float.floatToFloat16(maxMin / 63);
        float d = Float.float16ToFloat(dh);
        float dmin = Float.float16ToFloat(mh);
        int[] q = new int[256];
        for (int j = 0; j < 8; j++) {
            float sd = d * ls[j];
            float dm = dmin * lm[j];
            for (int i = 0; i < 32; i++) {
                int l = sd == 0 ? 0 : Math.round((x[at + 32 * j + i] + dm) / sd);
                q[32 * j + i] = Math.clamp(l, 0, 15);
            }
        }
        b.putShort(dh);
        b.putShort(mh);
        b.put(packed);
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; l++) {
                b.put((byte) (q[j + l] | q[j + l + 32] << 4));
            }
        }
    }

    private static void q6_k(float[] x, int at, ByteBuffer b) {
        float[] scales = new float[16];
        float maxScale = 0;
        float maxAbs = 0;
        for (int j = 0; j < 16; j++) {
            float amax = 0;
            float max = 0;
            for (int i = 0; i < 16; i++) {
                float v = x[at + 16 * j + i];
                if (Math.abs(v) > amax) {
                    amax = Math.abs(v);
                    max = v;
                }
            }
            scales[j] = max / -32;
            if (Math.abs(scales[j]) > maxAbs) {
                maxAbs = Math.abs(scales[j]);
                maxScale = scales[j];
            }
        }
        float iscale = maxScale != 0 ? -128 / maxScale : 0;
        byte[] sc = new byte[16];
        for (int j = 0; j < 16; j++) {
            sc[j] = (byte) Math.min(127, Math.round(iscale * scales[j]));
        }
        short dh = Float.floatToFloat16(iscale != 0 ? 1 / iscale : 0);
        float d = Float.float16ToFloat(dh);
        int[] q = new int[256];
        for (int j = 0; j < 16; j++) {
            float sd = d * sc[j];
            for (int i = 0; i < 16; i++) {
                int l = sd == 0 ? 0 : Math.round(x[at + 16 * j + i] / sd);
                q[16 * j + i] = Math.clamp(l, -32, 31) + 32;
            }
        }
        byte[] ql = new byte[128];
        byte[] qh = new byte[64];
        for (int n = 0; n < 2; n++) {
            for (int l = 0; l < 32; l++) {
                int q1 = q[n * 128 + l];
                int q2 = q[n * 128 + l + 32];
                int q3 = q[n * 128 + l + 64];
                int q4 = q[n * 128 + l + 96];
                ql[n * 64 + l] = (byte) ((q1 & 0xF) | (q3 & 0xF) << 4);
                ql[n * 64 + l + 32] = (byte) ((q2 & 0xF) | (q4 & 0xF) << 4);
                qh[n * 32 + l] = (byte) ((q1 >> 4) | (q2 >> 4) << 2 | (q3 >> 4) << 4 | (q4 >> 4) << 6);
            }
        }
        b.put(ql);
        b.put(qh);
        b.put(sc);
        b.putShort(dh);
    }

    private static long align(long n) {
        return (n + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }

    private static int typeOf(Object v) {
        return switch (v) {
            case Typed t -> t.value() instanceof List<?> ? ARRAY : t.type();
            case String s -> STRING;
            case Boolean b -> BOOL;
            case Integer n -> INT32;
            case Long n -> n >= 0 && n <= 0xFFFFFFFFL ? UINT32 : n < 0 ? INT64 : UINT64;
            case Float f -> FLOAT32;
            case Double d -> FLOAT32;
            case List<?> l -> ARRAY;
            default -> throw new IllegalArgumentException("no GGUF type for " + v.getClass().getName());
        };
    }

    private static final class Out {
        private final OutputStream out;
        private long written;
        private final ByteBuffer scratch = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);

        Out(OutputStream out) {
            this.out = out;
        }

        void bytes(byte[] b) throws IOException {
            out.write(b);
            written += b.length;
        }

        void u32(int v) throws IOException {
            scratch.clear();
            scratch.putInt(v);
            bytes(java.util.Arrays.copyOf(scratch.array(), 4));
        }

        void u64(long v) throws IOException {
            scratch.clear();
            scratch.putLong(v);
            bytes(scratch.array().clone());
        }

        void string(String s) throws IOException {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            u64(b.length);
            bytes(b);
        }

        void pad() throws IOException {
            int n = (int) (align(written) - written);
            if (n > 0) {
                bytes(new byte[n]);
            }
        }

        /** Writes a value of {@code type}; {@code itemType} is an array's items' type, or -1 to infer it. */
        void value(int type, Object v, int itemType) throws IOException {
            switch (type) {
                case UINT8, INT8 -> bytes(new byte[] {((Number) v).byteValue()});
                case UINT16, INT16 -> {
                    short s = ((Number) v).shortValue();
                    bytes(new byte[] {(byte) s, (byte) (s >> 8)});
                }
                case UINT32, INT32 -> u32(((Number) v).intValue());
                case UINT64, INT64 -> u64(((Number) v).longValue());
                case FLOAT32 -> u32(Float.floatToRawIntBits(((Number) v).floatValue()));
                case FLOAT64 -> u64(Double.doubleToRawLongBits(((Number) v).doubleValue()));
                case BOOL -> bytes(new byte[] {(byte) (Boolean.TRUE.equals(v) ? 1 : 0)});
                case STRING -> string((String) v);
                case ARRAY -> {
                    List<?> list = (List<?>) v;
                    int item = itemType >= 0 ? itemType : list.isEmpty() ? INT32 : typeOf(list.getFirst());
                    u32(item);
                    u64(list.size());
                    for (Object x : list) {
                        value(item, x, -1);
                    }
                }
                default -> throw new IllegalArgumentException("GGUF type " + type);
            }
        }
    }
}
