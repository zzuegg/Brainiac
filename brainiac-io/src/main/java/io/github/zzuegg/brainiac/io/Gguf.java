package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.RawValues;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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
 * A <a href="https://github.com/ggml-org/ggml/blob/master/docs/gguf.md">GGUF</a> file, the
 * format llama.cpp runs quantized models from, memory-mapped.
 *
 * <p>The metadata (architecture, hyperparameters, the tokenizer's vocabulary) is read into
 * {@link #metadata()}. Tensors are read on demand:
 *
 * <ul>
 *   <li>{@code F32, F16, BF16, F64} go straight from the mapping to the device;
 *   <li>the quantized formats {@code Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
 *       Q6_K, Q8_K} and the integer types are decoded on the host by {@link
 *       #tensor(String, Device, DType)};
 *   <li>matrices in any of those formats but {@code Q8_K} also map onto Brainiac's own
 *       packed weights with {@link #packedTransposed(String, Device)}, so they stay at 2 to 8
 *       bits on the device and multiply without being expanded.
 * </ul>
 *
 * <p>Shapes are reported row-major, as everywhere in Brainiac: GGUF lists dimensions
 * fastest first, so a weight GGUF calls {@code [4096, 11008]} is {@code [11008, 4096]} here
 * (output rows, input columns).
 */
public final class Gguf implements WeightSource {

    private static final int MAGIC = 0x46554747;   // "GGUF" little-endian
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** The GGML tensor types this reader knows: id, values per block, bytes per block. */
    enum Format {
        F32(0, 1, 4), F16(1, 1, 2), Q4_0(2, 32, 18), Q4_1(3, 32, 20), Q5_0(6, 32, 22), Q5_1(7, 32, 24),
        Q8_0(8, 32, 34), Q8_1(9, 32, 36), Q2_K(10, 256, 84), Q3_K(11, 256, 110), Q4_K(12, 256, 144),
        Q5_K(13, 256, 176), Q6_K(14, 256, 210), Q8_K(15, 256, 292), I8(24, 1, 1), I16(25, 1, 2), I32(26, 1, 4),
        I64(27, 1, 8), F64(28, 1, 8), BF16(30, 1, 2);

        final int id;
        final int block;
        final int bytes;

        Format(int id, int block, int bytes) {
            this.id = id;
            this.block = block;
            this.bytes = bytes;
        }

        static Format of(int id) {
            for (Format f : values()) {
                if (f.id == id) {
                    return f;
                }
            }
            throw new IllegalArgumentException("GGML tensor type " + id + " is not supported");
        }
    }

    private record Entry(Format format, Shape shape, long offset, long bytes) {
    }

    private final Path path;
    private final Arena arena;
    private final int version;
    private final Map<String, Object> metadata;
    private final Map<String, Entry> entries;
    private final MemorySegment data;

    private Gguf(Path path, Arena arena, int version, Map<String, Object> metadata, Map<String, Entry> entries,
            MemorySegment data) {
        this.path = path;
        this.arena = arena;
        this.version = version;
        this.metadata = metadata;
        this.entries = entries;
        this.data = data;
    }

    /** Reads the header fields in order. */
    private static final class Reader {
        final MemorySegment m;
        long at;

        Reader(MemorySegment m) {
            this.m = m;
        }

        int u32() {
            int v = m.get(INT, at);
            at += 4;
            return v;
        }

        long u64() {
            long v = m.get(LONG, at);
            at += 8;
            return v;
        }

        String string() {
            long n = u64();
            if (n < 0 || n > m.byteSize() - at) {
                throw new IllegalArgumentException("bad GGUF string length " + n);
            }
            String s = new String(m.asSlice(at, n).toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
            at += n;
            return s;
        }

        Object value(int type) {
            return switch (type) {
                case 0 -> (long) Byte.toUnsignedInt(m.get(ValueLayout.JAVA_BYTE, at++));
                case 1 -> (long) m.get(ValueLayout.JAVA_BYTE, at++);
                case 2 -> {
                    long v = Short.toUnsignedInt(m.get(RawValues.SHORT, at));
                    at += 2;
                    yield v;
                }
                case 3 -> {
                    long v = m.get(RawValues.SHORT, at);
                    at += 2;
                    yield v;
                }
                case 4 -> Integer.toUnsignedLong(u32());
                case 5 -> (long) u32();
                case 6 -> {
                    double v = m.get(RawValues.FLOAT, at);
                    at += 4;
                    yield v;
                }
                case 7 -> m.get(ValueLayout.JAVA_BYTE, at++) != 0;
                case 8 -> string();
                case 9 -> {
                    int itemType = u32();
                    long n = u64();
                    if (n < 0 || n > m.byteSize()) {
                        throw new IllegalArgumentException("bad GGUF array length " + n);
                    }
                    List<Object> list = new ArrayList<>((int) Math.min(n, 1 << 20));
                    for (long i = 0; i < n; i++) {
                        list.add(value(itemType));
                    }
                    yield Collections.unmodifiableList(list);
                }
                case 10, 11 -> u64();
                case 12 -> {
                    double v = m.get(RawValues.DOUBLE, at);
                    at += 8;
                    yield v;
                }
                default -> throw new IllegalArgumentException("unknown GGUF value type " + type);
            };
        }
    }

    /**
     * Opens and maps a GGUF file (versions 2 and 3).
     *
     * @param file the {@code .gguf} file
     * @return the opened file; close it to unmap
     * @throws UncheckedIOException if it cannot be read
     * @throws IllegalArgumentException if it is not a valid GGUF file
     */
    public static Gguf open(Path file) {
        Objects.requireNonNull(file, "file");
        Arena arena = Arena.ofShared();
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            MemorySegment all = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            Reader r = new Reader(all);
            if (all.byteSize() < 24 || r.u32() != MAGIC) {
                throw new IllegalArgumentException(file + " is not a GGUF file");
            }
            int version = r.u32();
            if (version < 2 || version > 3) {
                throw new IllegalArgumentException(file + " is GGUF version " + version + "; 2 and 3 are supported");
            }
            long tensors = r.u64();
            long kvs = r.u64();
            Map<String, Object> metadata = new LinkedHashMap<>();
            for (long i = 0; i < kvs; i++) {
                String key = r.string();
                metadata.put(key, r.value(r.u32()));
            }
            record Info(String name, Format format, Shape shape, long offset) {
            }
            List<Info> infos = new ArrayList<>();
            for (long i = 0; i < tensors; i++) {
                String name = r.string();
                int dims = r.u32();
                int[] shape = new int[dims];
                for (int d = 0; d < dims; d++) {
                    shape[dims - 1 - d] = Math.toIntExact(r.u64());   // fastest first in the file
                }
                Format format = Format.of(r.u32());
                infos.add(new Info(name, format, Shape.of(shape), r.u64()));
            }
            long alignment = metadata.get("general.alignment") instanceof Long a ? a : 32;
            long start = (r.at + alignment - 1) / alignment * alignment;
            MemorySegment data = all.asSlice(Math.min(start, all.byteSize()));
            Map<String, Entry> entries = new LinkedHashMap<>();
            for (Info info : infos) {
                long n = info.shape.size();
                int row = info.shape.rank() == 0 ? 1 : info.shape.dim(info.shape.rank() - 1);
                if (row % info.format.block != 0) {
                    throw new IllegalArgumentException(info.name + ": rows of " + row + " values do not fill "
                            + info.format + " blocks of " + info.format.block);
                }
                long bytes = n / info.format.block * info.format.bytes;
                if (info.offset < 0 || info.offset + bytes > data.byteSize()) {
                    throw new IllegalArgumentException(file + ": tensor " + info.name + " lies outside the file");
                }
                entries.put(info.name, new Entry(info.format, info.shape, info.offset, bytes));
            }
            return new Gguf(file, arena, version, Collections.unmodifiableMap(metadata), entries, data);
        } catch (IOException e) {
            arena.close();
            throw new UncheckedIOException(e);
        } catch (RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /** @return the file this was opened from */
    public Path path() {
        return path;
    }

    /** @return the GGUF version, 2 or 3 */
    public int version() {
        return version;
    }

    /**
     * Returns the metadata: whole numbers as {@code Long}, floats as {@code Double}, and
     * strings, booleans and lists of those.
     *
     * @return the key-value pairs in file order
     */
    public Map<String, Object> metadata() {
        return metadata;
    }

    /** @return the model architecture ({@code general.architecture}), such as {@code "llama"} */
    public String architecture() {
        return String.valueOf(metadata.get("general.architecture"));
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
        return entry(name).format.name();
    }

    @Override
    public DType naturalType(String name) {
        return switch (entry(name).format) {
            case F64 -> DType.FLOAT64;
            case F16 -> DType.FLOAT16;
            case BF16 -> DType.BFLOAT16;
            default -> DType.FLOAT32;
        };
    }

    /**
     * Returns a tensor's raw bytes as stored, without copying. Valid until this file is
     * closed.
     *
     * @param name the name
     * @return its bytes
     */
    public MemorySegment raw(String name) {
        Entry e = entry(name);
        return data.asSlice(e.offset, e.bytes);
    }

    @Override
    public Tensor tensor(String name, Device device, DType dtype) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(dtype, "dtype");
        Entry e = entry(name);
        DType stored = switch (e.format) {
            case F32 -> DType.FLOAT32;
            case F64 -> DType.FLOAT64;
            case F16 -> DType.FLOAT16;
            case BF16 -> DType.BFLOAT16;
            default -> null;
        };
        if (stored != null && device.supports(stored)) {
            Tensor t = Tensor.of(e.shape, raw(name), stored, device);
            return t.to(dtype);
        }
        double[] values = stored != null ? RawValues.decode(raw(name), stored, e.shape.size())
                : GgmlQuants.decode(e.format, raw(name), e.shape.size());
        return Tensor.of(e.shape, values, dtype, device);
    }

    /**
     * Returns whether {@link #packedTransposed} can keep a tensor packed: a matrix in {@code
     * Q8_0}, {@code Q4_0}, {@code Q4_1}, {@code Q5_0}, {@code Q5_1}, {@code Q2_K}, {@code
     * Q3_K}, {@code Q4_K}, {@code Q5_K} or {@code Q6_K}.
     *
     * @param name the name
     * @return {@code true} if it can
     */
    public boolean packable(String name) {
        Entry e = entry(name);
        return e.shape.rank() == 2 && GgmlQuants.group(e.format) > 0 && e.shape.dim(1) % e.format.block == 0;
    }

    /**
     * Reads a quantized weight matrix as a {@linkplain Tensor#packed packed} tensor,
     * transposed: GGUF stores a layer's weights as {@code [out, in]} with blocks along the
     * input, while Brainiac multiplies {@code x.matmul(w)} with {@code w} as {@code [in, out]}.
     * The levels are kept exactly, with a scale for every 32 inputs of an output (16 for
     * {@code Q2_K}, {@code Q3_K} and {@code Q6_K}) and, for the formats with a minimum
     * ({@code Q4_1}, {@code Q5_1}, {@code Q2_K}, {@code Q4_K}, {@code Q5_K}), an offset. Their
     * unsigned levels are shifted to signed ones, the shift moving into the offset, so values
     * can differ from the file's in the last bit of a float.
     *
     * @param name the name of a {@linkplain #packable packable} matrix
     * @param device where to put it
     * @return the packed {@code [in, out]} matrix
     * @throws IllegalArgumentException if the tensor is not packable
     */
    public Tensor packedTransposed(String name, Device device) {
        return packedTransposed(name, device, null);
    }

    /**
     * As {@link #packedTransposed(String, Device)}, with the outputs reordered: output
     * {@code c} of the result is row {@code order[c]} of the stored matrix. Used to undo the
     * row permutation llama.cpp's converter applies to Llama's query and key weights.
     *
     * @param name the name of a {@linkplain #packable packable} matrix
     * @param device where to put it
     * @param order the stored row for each output, or {@code null} to keep the order
     * @return the packed {@code [in, out]} matrix
     */
    public Tensor packedTransposed(String name, Device device, int[] order) {
        if (!packable(name)) {
            throw new IllegalArgumentException(name + " is " + format(name) + " " + shape(name)
                    + "; it cannot stay packed");
        }
        Entry e = entry(name);
        int out = e.shape.dim(0);
        int in = e.shape.dim(1);
        int group = GgmlQuants.group(e.format);
        int bits = GgmlQuants.bits(e.format);
        boolean shifted = GgmlQuants.offsets(e.format);
        int shift = shifted ? 1 << (bits - 1) : 0;
        if (order != null && order.length != out) {
            throw new IllegalArgumentException("order has " + order.length + " entries for " + out + " outputs");
        }
        int[] column = new int[out];   // the output each stored row becomes
        for (int c = 0; c < out; c++) {
            column[order == null ? c : order[c]] = c;
        }
        byte[] levels = new byte[in * out];
        float[] scales = new float[in / group * out];
        float[] offsets = shifted ? new float[scales.length] : null;
        GgmlQuants.levels(e.format, raw(name), (long) in * out, (i, q, scale, offset) -> {
            int c = column[i / in];
            int r = i % in;
            levels[r * out + c] = (byte) (q - shift);
            if (r % group == 0) {
                int g = r / group * out + c;
                scales[g] = scale;
                if (shifted) {
                    offsets[g] = (float) (offset + (double) scale * shift);
                }
            }
        });
        return Tensor.packed(PackedMatrix.of(levels, scales, offsets, in, out, bits, group), device);
    }

    @Override
    public void close() {
        if (arena.scope().isAlive()) {
            arena.close();
        }
    }

    @Override
    public String toString() {
        return "Gguf[" + path + ", " + architecture() + ", " + entries.size() + " tensors]";
    }
}
