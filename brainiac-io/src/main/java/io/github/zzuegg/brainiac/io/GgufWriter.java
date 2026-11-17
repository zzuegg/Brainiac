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
 * or float16, aligned to 32 bytes.
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
     * Writes a file.
     *
     * @param file where to write
     * @param metadata the key-value pairs, in order
     * @param tensors the tensors by GGUF name, in order, any shape
     * @param half whether to store matrices (rank 2 and more) as float16; vectors stay float32
     */
    public static void write(Path file, Map<String, Object> metadata, Map<String, Tensor> tensors, boolean half) {
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(file), 1 << 20)) {
            Out o = new Out(out);
            o.u32(MAGIC);
            o.u32(3);
            o.u64(tensors.size());
            o.u64(metadata.size());
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                o.string(e.getKey());
                Object v = e.getValue();
                int type = typeOf(v);
                o.u32(type);
                o.value(type, v instanceof Typed t ? t.value() : v, v instanceof Typed t ? t.type() : -1);
            }
            long offset = 0;
            boolean[] halves = new boolean[tensors.size()];
            int i = 0;
            for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
                int[] dims = e.getValue().shape().toArray();
                halves[i] = half && dims.length >= 2;
                o.string(e.getKey());
                o.u32(Math.max(1, dims.length));
                if (dims.length == 0) {
                    o.u64(1);
                }
                for (int d = dims.length - 1; d >= 0; d--) {
                    o.u64(dims[d]);   // fastest first
                }
                o.u32(halves[i] ? 1 : 0);
                o.u64(offset);
                offset = align(offset + e.getValue().size() * (halves[i] ? 2 : 4));
                i++;
            }
            o.pad();
            i = 0;
            for (Tensor t : tensors.values()) {
                float[] values = t.to(DType.FLOAT32).toArray();
                ByteBuffer b = ByteBuffer.allocate(values.length * (halves[i] ? 2 : 4)).order(ByteOrder.LITTLE_ENDIAN);
                for (float f : values) {
                    if (halves[i]) {
                        b.putShort(Float.floatToFloat16(f));
                    } else {
                        b.putFloat(f);
                    }
                }
                o.bytes(b.array());
                o.pad();
                i++;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
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
