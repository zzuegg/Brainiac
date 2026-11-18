package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.RawValues;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A <a href="https://github.com/huggingface/safetensors">safetensors</a> file, memory-mapped:
 * the format most open models are published in.
 *
 * <p>The file is mapped, not read, so opening it costs only the size of its JSON header.
 * A tensor's bytes go from the mapping straight to its device ({@link
 * Tensor#of(Shape, MemorySegment, DType, Device)}); on the GPU backends they never pass
 * through the Java heap, and a type conversion such as bfloat16 to float32 runs on the
 * device.
 *
 * <p>Stored types: {@code F64, F32, F16, BF16} load directly; {@code F8_E4M3, F8_E5M2},
 * the integer types and {@code BOOL} are decoded on the host into float32.
 *
 * <pre>{@code
 * try (SafeTensors file = SafeTensors.open(Path.of("model.safetensors"))) {
 *     for (String name : file.names()) {
 *         System.out.println(name + " " + file.format(name) + " " + file.shape(name));
 *     }
 *     Tensor w = file.tensor("lm_head.weight", Device.gpu());
 * }
 * SafeTensors.write(Path.of("out.safetensors"), Map.of("w", w));
 * }</pre>
 */
public final class SafeTensors implements WeightSource {

    /** Where one tensor sits in the file. */
    private record Entry(String format, Shape shape, long begin, long end) {
    }

    private final Path path;
    private final Arena arena;
    private final MemorySegment data;
    private final Map<String, Entry> entries;
    private final Map<String, String> metadata;

    private SafeTensors(Path path, Arena arena, MemorySegment data, Map<String, Entry> entries,
            Map<String, String> metadata) {
        this.path = path;
        this.arena = arena;
        this.data = data;
        this.entries = entries;
        this.metadata = metadata;
    }

    /**
     * Opens and maps a file.
     *
     * @param file the {@code .safetensors} file
     * @return the opened file; close it to unmap
     * @throws UncheckedIOException if it cannot be read
     * @throws IllegalArgumentException if it is not a valid safetensors file
     */
    public static SafeTensors open(Path file) {
        Objects.requireNonNull(file, "file");
        Arena arena = Arena.ofShared();
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long size = ch.size();
            if (size < 8) {
                throw new IllegalArgumentException(file + " is too short for a safetensors file");
            }
            MemorySegment all = ch.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
            long headerSize = all.get(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 0);
            if (headerSize < 2 || headerSize > size - 8 || headerSize > 100_000_000) {
                throw new IllegalArgumentException(file + " has a bad safetensors header size " + headerSize);
            }
            byte[] header = all.asSlice(8, headerSize).toArray(ValueLayout.JAVA_BYTE);
            MemorySegment data = all.asSlice(8 + headerSize);
            Map<String, Object> json = Json.object(Json.parse(new String(header, StandardCharsets.UTF_8).strip()));
            Map<String, Entry> entries = new LinkedHashMap<>();
            Map<String, String> metadata = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : json.entrySet()) {
                if (e.getKey().equals("__metadata__")) {
                    for (Map.Entry<String, Object> m : Json.object(e.getValue()).entrySet()) {
                        metadata.put(m.getKey(), String.valueOf(m.getValue()));
                    }
                    continue;
                }
                Map<String, Object> info = Json.object(e.getValue());
                String format = Json.string(info, "dtype", null);
                List<Object> dims = Json.array(info.get("shape"));
                int[] shape = new int[dims.size()];
                for (int i = 0; i < shape.length; i++) {
                    shape[i] = Math.toIntExact(((Number) dims.get(i)).longValue());
                }
                List<Object> offsets = Json.array(info.get("data_offsets"));
                long begin = ((Number) offsets.get(0)).longValue();
                long end = ((Number) offsets.get(1)).longValue();
                Shape s = Shape.of(shape);
                if (begin < 0 || end < begin || end > data.byteSize()) {
                    throw new IllegalArgumentException(file + ": tensor " + e.getKey() + " lies outside the file");
                }
                if (end - begin != s.size() * bytesPer(format)) {
                    throw new IllegalArgumentException(file + ": tensor " + e.getKey() + " of " + format + " " + s
                            + " needs " + s.size() * bytesPer(format) + " bytes, has " + (end - begin));
                }
                entries.put(e.getKey(), new Entry(format, s, begin, end));
            }
            return new SafeTensors(file, arena, data, entries, Collections.unmodifiableMap(metadata));
        } catch (IOException e) {
            arena.close();
            throw new UncheckedIOException(e);
        } catch (RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /** Bytes per value of a stored format. */
    private static long bytesPer(String format) {
        return switch (format) {
            case "F64", "I64", "U64" -> 8;
            case "F32", "I32", "U32" -> 4;
            case "F16", "BF16", "I16", "U16" -> 2;
            case "F8_E4M3", "F8_E5M2", "I8", "U8", "BOOL" -> 1;
            default -> throw new IllegalArgumentException("unknown safetensors dtype " + format);
        };
    }

    /**
     * Returns the file's free-form metadata ({@code __metadata__}).
     *
     * @return the metadata, possibly empty
     */
    public Map<String, String> metadata() {
        return metadata;
    }

    /** @return the file this was opened from */
    public Path path() {
        return path;
    }

    @Override
    public List<String> names() {
        return List.copyOf(entries.keySet());
    }

    @Override
    public boolean contains(String name) {
        return entries.containsKey(name);
    }

    private Entry entry(String name) {
        Entry e = entries.get(name);
        if (e == null) {
            throw new IllegalArgumentException("no tensor named " + name + " in " + path);
        }
        return e;
    }

    @Override
    public Shape shape(String name) {
        return entry(name).shape;
    }

    @Override
    public String format(String name) {
        return entry(name).format;
    }

    @Override
    public DType naturalType(String name) {
        DType t = direct(entry(name).format);
        return t == null ? DType.FLOAT32 : t;
    }

    /** The tensor type a stored format is identical to, or null. */
    static DType direct(String format) {
        return switch (format) {
            case "F64" -> DType.FLOAT64;
            case "F32" -> DType.FLOAT32;
            case "F16" -> DType.FLOAT16;
            case "BF16" -> DType.BFLOAT16;
            default -> null;
        };
    }

    /**
     * Returns a tensor's raw bytes, little-endian as stored, without copying. Valid until
     * this file is closed.
     *
     * @param name the name
     * @return its bytes
     */
    public MemorySegment raw(String name) {
        Entry e = entry(name);
        return data.asSlice(e.begin, e.end - e.begin);
    }

    @Override
    public Tensor tensor(String name, Device device, DType dtype) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(dtype, "dtype");
        Entry e = entry(name);
        return read(raw(name), e.format, e.shape, device, dtype);
    }

    /** Makes a tensor of row-major values stored in a safetensors format. */
    static Tensor read(MemorySegment raw, String format, Shape shape, Device device, DType dtype) {
        DType stored = direct(format);
        if (stored != null) {
            if (stored == dtype || !device.supports(stored)) {
                if (stored == dtype) {
                    return Tensor.of(shape, raw, stored, device);
                }
                // the device cannot hold the stored type: convert on the host
                return Tensor.of(shape, RawValues.decode(raw, stored, shape.size()), dtype, device);
            }
            return Tensor.of(shape, raw, stored, device).to(dtype);
        }
        return Tensor.of(shape, decode(format, raw, shape.size()), dtype, device);
    }

    /** Decodes a format with no tensor type of its own into doubles. */
    private static double[] decode(String format, MemorySegment src, long count) {
        double[] out = new double[Math.toIntExact(count)];
        ByteOrder le = ByteOrder.LITTLE_ENDIAN;
        for (int i = 0; i < out.length; i++) {
            out[i] = switch (format) {
                case "I64" -> src.getAtIndex(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(le), i);
                case "U64" -> {
                    long v = src.getAtIndex(ValueLayout.JAVA_LONG_UNALIGNED.withOrder(le), i);
                    yield v >= 0 ? v : (double) (v >>> 1) * 2 + (v & 1);
                }
                case "I32" -> src.getAtIndex(ValueLayout.JAVA_INT_UNALIGNED.withOrder(le), i);
                case "U32" -> Integer.toUnsignedLong(src.getAtIndex(ValueLayout.JAVA_INT_UNALIGNED.withOrder(le), i));
                case "I16" -> src.getAtIndex(RawValues.SHORT, i);
                case "U16" -> Short.toUnsignedInt(src.getAtIndex(RawValues.SHORT, i));
                case "I8" -> src.get(ValueLayout.JAVA_BYTE, i);
                case "U8", "BOOL" -> Byte.toUnsignedInt(src.get(ValueLayout.JAVA_BYTE, i));
                case "F8_E4M3" -> Float8.e4m3(src.get(ValueLayout.JAVA_BYTE, i));
                case "F8_E5M2" -> Float8.e5m2(src.get(ValueLayout.JAVA_BYTE, i));
                default -> throw new IllegalArgumentException("unknown safetensors dtype " + format);
            };
        }
        return out;
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }

    @Override
    public String toString() {
        return "SafeTensors[" + path + ", " + entries.size() + " tensors]";
    }

    // ---------------------------------------------------------------- writing

    /**
     * Writes tensors to a safetensors file, each in its own type.
     *
     * @param file the file to create or replace
     * @param tensors the tensors by name, written in iteration order
     * @throws UncheckedIOException if the file cannot be written
     */
    public static void write(Path file, Map<String, Tensor> tensors) {
        write(file, tensors, Map.of());
    }

    /**
     * Writes tensors and metadata to a safetensors file, each tensor in its own type.
     *
     * @param file the file to create or replace
     * @param tensors the tensors by name, written in iteration order
     * @param metadata free-form strings stored in the header
     * @throws UncheckedIOException if the file cannot be written
     */
    public static void write(Path file, Map<String, Tensor> tensors, Map<String, String> metadata) {
        Map<String, Object> header = new LinkedHashMap<>();
        if (!metadata.isEmpty()) {
            header.put("__metadata__", new LinkedHashMap<>(metadata));
        }
        long offset = 0;
        List<Tensor> order = new ArrayList<>();
        for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
            Tensor t = e.getValue().unpacked();
            order.add(t);
            long bytes = t.size() * t.dtype().bytes();
            Map<String, Object> info = new LinkedHashMap<>();
            info.put("dtype", switch (t.dtype()) {
                case FLOAT64 -> "F64";
                case FLOAT32 -> "F32";
                case FLOAT16 -> "F16";
                case BFLOAT16 -> "BF16";
            });
            info.put("shape", t.shape().toArray());
            info.put("data_offsets", List.of(offset, offset + bytes));
            header.put(e.getKey(), info);
            offset += bytes;
        }
        byte[] json = Json.write(header).getBytes(StandardCharsets.UTF_8);
        int padded = (json.length + 7) & ~7;
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer head = ByteBuffer.allocate(8 + padded).order(ByteOrder.LITTLE_ENDIAN);
            head.putLong(padded).put(json);
            while (head.hasRemaining()) {
                head.put((byte) ' ');
            }
            writeFully(ch, head.flip());
            for (Tensor t : order) {
                writeFully(ch, bytes(t));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void writeFully(FileChannel ch, ByteBuffer b) throws IOException {
        while (b.hasRemaining()) {
            ch.write(b);
        }
    }

    /** A tensor's values as its type stores them, little-endian. */
    private static ByteBuffer bytes(Tensor t) {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(t.size() * t.dtype().bytes())).order(ByteOrder.LITTLE_ENDIAN);
        switch (t.dtype()) {
            case FLOAT32 -> b.asFloatBuffer().put(t.toArray());
            case FLOAT64 -> b.asDoubleBuffer().put(t.toDoubleArray());
            case FLOAT16 -> {
                var s = b.asShortBuffer();
                for (double v : t.toDoubleArray()) {
                    s.put(Float.floatToFloat16((float) v));
                }
            }
            case BFLOAT16 -> {
                var s = b.asShortBuffer();
                for (double v : t.toDoubleArray()) {
                    s.put(DType.floatToBfloat16((float) v));
                }
            }
        }
        return b;
    }
}
