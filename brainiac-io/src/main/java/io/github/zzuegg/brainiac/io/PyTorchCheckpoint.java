package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A PyTorch checkpoint written by {@code torch.save}, in its zip format (PyTorch 1.6 and
 * later) or the older legacy format, such as a Hugging Face {@code pytorch_model.bin}: a
 * pickled state dict whose tensors point into the file's storages. The pickle is interpreted, never executed: only
 * dictionaries, tensors and parameters are understood, and anything else in it is skipped.
 * Storages are memory-mapped, so opening a checkpoint reads only its pickle.
 */
final class PyTorchCheckpoint implements WeightSource {

    private static final ValueLayout.OfShort U16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt U32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    /** A pickle of the magic number 0x1950a86a20f9469cfc6c that starts legacy checkpoints. */
    private static final int[] LEGACY_MAGIC = {0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8,
        0x50, 0x19, 0x2e};
    private static final ValueLayout.OfLong U64 = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** A tensor: a view of {@code storage} with an element offset and strides. */
    private record Entry(String storage, String format, Shape shape, long offset, long[] stride) {
    }

    private final Path path;
    private final Arena arena;
    private final MemorySegment file;
    /** Each storage's bytes in the file, by key. */
    private final Map<String, MemorySegment> storages = new HashMap<>();
    private final Map<String, Entry> entries = new LinkedHashMap<>();

    private PyTorchCheckpoint(Path path) throws IOException {
        this.path = path;
        arena = Arena.ofShared();
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            file = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
        }
        try {
            if (file.byteSize() >= LEGACY_MAGIC.length && (file.get(ValueLayout.JAVA_BYTE, 0) & 0xFF) == 0x80) {
                legacy();
                return;
            }
            Map<String, MemorySegment> zip = entries(file);
            String pickle = null;
            for (String n : zip.keySet()) {
                if (n.endsWith("/data.pkl") && n.indexOf('/') == n.length() - "/data.pkl".length()) {
                    pickle = n;
                }
            }
            if (pickle == null) {
                throw new IllegalArgumentException(path + " is not a PyTorch zip checkpoint"
                        + " (no data.pkl)");
            }
            String root = pickle.substring(0, pickle.length() - "data.pkl".length());
            for (Map.Entry<String, MemorySegment> e : zip.entrySet()) {
                if (e.getKey().startsWith(root + "data/")) {
                    storages.put(e.getKey().substring(root.length() + "data/".length()), e.getValue());
                }
            }
            Object state = new Unpickler(zip.get(pickle), 0).load();
            collect(state);
        } catch (RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /** Takes the tensors out of a loaded state dict, or a checkpoint holding one. */
    private void collect(Object state) {
        if (state instanceof Map<?, ?> m && m.get("state_dict") instanceof Map<?, ?> inner
                && !(m.values().stream().anyMatch(Entry.class::isInstance))) {
            state = inner;   // a training checkpoint holding the weights under "state_dict"
        }
        if (!(state instanceof Map<?, ?> dict)) {
            throw new IllegalArgumentException(path + " does not hold a state dict");
        }
        for (Map.Entry<?, ?> e : dict.entrySet()) {
            if (e.getKey() instanceof String name && e.getValue() instanceof Entry t) {
                if (!storages.containsKey(t.storage)) {
                    throw new IllegalArgumentException(path + ": storage " + t.storage + " is missing");
                }
                entries.put(name, t);
            }
        }
    }

    /**
     * The format before PyTorch 1.6: pickles of a magic number, the protocol version and
     * system facts, then the state dict, then the storage keys in the order their data
     * follows, each as a little-endian element count and the raw elements.
     */
    private void legacy() {
        for (int i = 0; i < LEGACY_MAGIC.length; i++) {
            if ((file.get(ValueLayout.JAVA_BYTE, i) & 0xFF) != LEGACY_MAGIC[i]) {
                throw new IllegalArgumentException(path + " is neither a zip nor a legacy PyTorch checkpoint");
            }
        }
        // the magic number is an 80-bit integer, past what the unpickler reads, so it is matched as bytes
        Unpickler u = new Unpickler(file, LEGACY_MAGIC.length);
        u.load();   // protocol version
        u = new Unpickler(file, u.position());
        u.load();   // system information
        Unpickler main = new Unpickler(file, u.position());
        Object state = main.load();
        Unpickler keys = new Unpickler(file, main.position());
        Object order = keys.load();
        if (!(order instanceof List<?> list)) {
            throw new IllegalArgumentException(path + ": no storage list in a legacy checkpoint");
        }
        Map<String, Integer> sizes = new HashMap<>();
        collectSizes(state, sizes);
        long at = keys.position();
        for (Object k : list) {
            String key = String.valueOf(k);
            long count = file.get(U64, at);
            at += 8;
            int element = sizes.getOrDefault(key, 4);
            long length = count * element;
            if (at + length > file.byteSize()) {
                throw new IllegalArgumentException(path + ": storage " + key + " reaches past the end of the file");
            }
            storages.put(key, file.asSlice(at, length));
            at += length;
        }
        collect(state);
    }

    private static void collectSizes(Object v, Map<String, Integer> sizes) {
        if (v instanceof Entry e) {
            sizes.put(e.storage, bytes(e.format));
        } else if (v instanceof Map<?, ?> m) {
            m.values().forEach(x -> collectSizes(x, sizes));
        } else if (v instanceof List<?> l) {
            l.forEach(x -> collectSizes(x, sizes));
        }
    }

    static PyTorchCheckpoint open(Path path) {
        Objects.requireNonNull(path, "path");
        try {
            return new PyTorchCheckpoint(path);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
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
            throw new IllegalArgumentException("no tensor " + name + " in " + path);
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
        DType t = SafeTensors.direct(entry(name).format);
        return t == null ? DType.FLOAT32 : t;
    }

    @Override
    public Tensor tensor(String name, Device device, DType dtype) {
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(dtype, "dtype");
        Entry e = entry(name);
        return SafeTensors.read(contiguous(e), e.format, e.shape, device, dtype);
    }

    /** The tensor's values in row-major order: a slice of its storage, or a gathered copy. */
    private MemorySegment contiguous(Entry e) {
        int size = bytes(e.format);
        MemorySegment storage = storages.get(e.storage);
        long count = e.shape.size();
        int rank = e.stride.length;
        long[] dims = new long[rank];
        boolean dense = true;
        long expect = 1;
        for (int d = rank - 1; d >= 0; d--) {
            dims[d] = e.shape.dim(d);
            if (dims[d] != 1 && e.stride[d] != expect) {
                dense = false;
            }
            expect *= dims[d];
        }
        long last = e.offset;
        for (int d = 0; d < rank; d++) {
            last += (dims[d] - 1) * e.stride[d];
        }
        if (count > 0 && (e.offset < 0 || (last + 1) * size > storage.byteSize())) {
            throw new IllegalArgumentException(path + ": a tensor reaches past its storage");
        }
        if (dense) {
            return storage.asSlice(e.offset * size, count * size);
        }
        MemorySegment out = Arena.ofAuto().allocate(count * size);
        long[] index = new long[rank];
        for (long i = 0; i < count; i++) {
            long at = e.offset;
            for (int d = 0; d < rank; d++) {
                at += index[d] * e.stride[d];
            }
            MemorySegment.copy(storage, at * size, out, i * size, size);
            for (int d = rank - 1; d >= 0 && ++index[d] == dims[d]; d--) {
                index[d] = 0;
            }
        }
        return out;
    }

    private static int bytes(String format) {
        return switch (format) {
            case "F64", "I64" -> 8;
            case "F32", "I32" -> 4;
            case "F16", "BF16", "I16" -> 2;
            default -> 1;
        };
    }

    @Override
    public void close() {
        arena.close();
    }

    @Override
    public String toString() {
        return "PyTorch checkpoint " + path;
    }

    // ------------------------------------------------------------------ zip

    /** The archive's stored (uncompressed) entries, by name. */
    private static Map<String, MemorySegment> entries(MemorySegment zip) {
        long size = zip.byteSize();
        long eocd = -1;
        for (long at = size - 22; at >= Math.max(0, size - 22 - 65535); at--) {
            if (zip.get(U32, at) == 0x06054b50) {
                eocd = at;
                break;
            }
        }
        if (eocd < 0) {
            throw new IllegalArgumentException("not a zip archive");
        }
        long count = Short.toUnsignedLong(zip.get(U16, eocd + 10));
        long offset = Integer.toUnsignedLong(zip.get(U32, eocd + 16));
        if ((count == 0xFFFF || offset == 0xFFFFFFFFL) && eocd >= 20 && zip.get(U32, eocd - 20) == 0x07064b50) {
            long z = zip.get(U64, eocd - 20 + 8);
            if (zip.get(U32, z) != 0x06064b50) {
                throw new IllegalArgumentException("broken zip64 directory");
            }
            count = zip.get(U64, z + 32);
            offset = zip.get(U64, z + 48);
        }
        Map<String, MemorySegment> out = new LinkedHashMap<>();
        long at = offset;
        for (long i = 0; i < count; i++) {
            if (zip.get(U32, at) != 0x02014b50) {
                throw new IllegalArgumentException("broken zip directory");
            }
            int method = Short.toUnsignedInt(zip.get(U16, at + 10));
            long compressed = Integer.toUnsignedLong(zip.get(U32, at + 20));
            long length = Integer.toUnsignedLong(zip.get(U32, at + 24));
            int nameLength = Short.toUnsignedInt(zip.get(U16, at + 28));
            int extraLength = Short.toUnsignedInt(zip.get(U16, at + 30));
            int commentLength = Short.toUnsignedInt(zip.get(U16, at + 32));
            long local = Integer.toUnsignedLong(zip.get(U32, at + 42));
            String name = new String(zip.asSlice(at + 46, nameLength).toArray(ValueLayout.JAVA_BYTE),
                    StandardCharsets.UTF_8);
            // zip64: the fields that overflowed follow in this order
            long extra = at + 46 + nameLength;
            for (long x = extra; x + 4 <= extra + extraLength; ) {
                int id = Short.toUnsignedInt(zip.get(U16, x));
                int n = Short.toUnsignedInt(zip.get(U16, x + 2));
                if (id == 1) {
                    long f = x + 4;
                    if (length == 0xFFFFFFFFL) {
                        length = zip.get(U64, f);
                        f += 8;
                    }
                    if (compressed == 0xFFFFFFFFL) {
                        compressed = zip.get(U64, f);
                        f += 8;
                    }
                    if (local == 0xFFFFFFFFL) {
                        local = zip.get(U64, f);
                    }
                }
                x += 4 + n;
            }
            at = extra + extraLength + commentLength;
            if (method != 0) {
                continue;   // torch.save stores everything uncompressed
            }
            if (zip.get(U32, local) != 0x04034b50) {
                throw new IllegalArgumentException("broken zip entry " + name);
            }
            long data = local + 30 + Short.toUnsignedInt(zip.get(U16, local + 26))
                    + Short.toUnsignedInt(zip.get(U16, local + 28));
            out.put(name, zip.asSlice(data, length));
        }
        return out;
    }

    // ------------------------------------------------------------------ pickle

    /** A {@code module.name} the pickle refers to; never looked up, only recognized. */
    private record Global(String module, String name) {
    }

    /** A call the unpickler does not understand, kept as a value nobody uses. */
    private record Opaque(Object function) {
    }

    /** Interprets the subset of the pickle protocol (0 to 5) that state dicts use. */
    private static final class Unpickler {

        private static final Object MARK = new Object();

        private final MemorySegment in;
        private long at;
        private final List<Object> stack = new ArrayList<>();
        private final Map<Long, Object> memo = new HashMap<>();

        Unpickler(MemorySegment in, long start) {
            this.in = in;
            this.at = start;
        }

        /** @return where the next pickle starts, once {@link #load()} returned */
        long position() {
            return at;
        }

        Object load() {
            while (true) {
                int op = u8();
                switch (op) {
                    case 0x80 -> u8();                                    // PROTO
                    case 0x95 -> at += 8;                                 // FRAME
                    case '.' -> {                                         // STOP
                        return pop();
                    }
                    case '(' -> stack.add(MARK);
                    case '}' -> stack.add(new LinkedHashMap<>());         // EMPTY_DICT
                    case ']' -> stack.add(new ArrayList<>());             // EMPTY_LIST
                    case ')' -> stack.add(List.of());                     // EMPTY_TUPLE
                    case 0x8f -> stack.add(new ArrayList<>());            // EMPTY_SET
                    case 'N' -> stack.add(null);
                    case 0x88 -> stack.add(Boolean.TRUE);
                    case 0x89 -> stack.add(Boolean.FALSE);
                    case 'K' -> stack.add((long) u8());                   // BININT1
                    case 'M' -> stack.add((long) (u8() | u8() << 8));     // BININT2
                    case 'J' -> stack.add((long) i32());                  // BININT
                    case 0x8a -> stack.add(longOf(u8()));                 // LONG1
                    case 0x8b -> stack.add(longOf(i32()));                // LONG4
                    case 'G' -> {                                         // BINFLOAT, big-endian
                        long bits = 0;
                        for (int i = 0; i < 8; i++) {
                            bits = bits << 8 | u8();
                        }
                        stack.add(Double.longBitsToDouble(bits));
                    }
                    case 'X' -> stack.add(text(i32()));                   // BINUNICODE
                    case 0x8c -> stack.add(text(u8()));                   // SHORT_BINUNICODE
                    case 0x8d -> stack.add(text(Math.toIntExact(i64()))); // BINUNICODE8
                    case 'C' -> stack.add(bytes(u8()));                   // SHORT_BINBYTES
                    case 'B' -> stack.add(bytes(i32()));                  // BINBYTES
                    case 0x8e -> stack.add(bytes(Math.toIntExact(i64()))); // BINBYTES8
                    case 'U' -> stack.add(text(u8()));                    // SHORT_BINSTRING
                    case 'T' -> stack.add(text(i32()));                   // BINSTRING
                    case 'c' -> stack.add(new Global(line(), line()));    // GLOBAL
                    case 0x93 -> {                                        // STACK_GLOBAL
                        String name = (String) pop();
                        stack.add(new Global((String) pop(), name));
                    }
                    case 'q' -> memo.put((long) u8(), peek());            // BINPUT
                    case 'r' -> memo.put(Integer.toUnsignedLong(i32()), peek()); // LONG_BINPUT
                    case 0x94 -> memo.put((long) memo.size(), peek());    // MEMOIZE
                    case 'h' -> stack.add(memo.get((long) u8()));         // BINGET
                    case 'j' -> stack.add(memo.get(Integer.toUnsignedLong(i32()))); // LONG_BINGET
                    case 't' -> stack.add(nullSafe(popMark())); // TUPLE
                    case 0x85 -> stack.add(tuple(1));
                    case 0x86 -> stack.add(tuple(2));
                    case 0x87 -> stack.add(tuple(3));
                    case 'l' -> stack.add(new ArrayList<>(popMark()));    // LIST
                    case 'd' -> {                                         // DICT
                        List<Object> items = popMark();
                        Map<Object, Object> m = new LinkedHashMap<>();
                        for (int i = 0; i + 1 < items.size(); i += 2) {
                            m.put(items.get(i), items.get(i + 1));
                        }
                        stack.add(m);
                    }
                    case 'a' -> {                                         // APPEND
                        Object v = pop();
                        list(peek()).add(v);
                    }
                    case 'e', 0x90 -> {                                   // APPENDS, ADDITEMS
                        List<Object> items = popMark();
                        list(peek()).addAll(items);
                    }
                    case 's' -> {                                         // SETITEM
                        Object v = pop();
                        Object k = pop();
                        map(peek()).put(k, v);
                    }
                    case 'u' -> {                                         // SETITEMS
                        List<Object> items = popMark();
                        Map<Object, Object> m = map(peek());
                        for (int i = 0; i + 1 < items.size(); i += 2) {
                            m.put(items.get(i), items.get(i + 1));
                        }
                    }
                    case 'b' -> pop();                                    // BUILD: object state, unused
                    case 'R' -> {                                         // REDUCE
                        List<?> args = (List<?>) pop();
                        stack.add(call(pop(), args));
                    }
                    case 0x81 -> {                                        // NEWOBJ
                        List<?> args = (List<?>) pop();
                        stack.add(call(pop(), args));
                    }
                    case 'Q' -> stack.add(persistent(pop()));             // BINPERSID
                    case '0' -> pop();                                    // POP
                    case '2' -> stack.add(peek());                        // DUP
                    case '1' -> popMark();                                // POP_MARK
                    default -> throw new IllegalArgumentException(
                            String.format("unsupported pickle opcode 0x%02x at %d", op, at - 1));
                }
            }
        }

        /** The storages behind tensors: {@code ('storage', type, key, location, numel)}. */
        private static Object persistent(Object id) {
            if (id instanceof List<?> t && t.size() >= 3 && "storage".equals(t.get(0))
                    && t.get(1) instanceof Global type) {
                return new Storage(format(type), String.valueOf(t.get(2)));
            }
            throw new IllegalArgumentException("unknown persistent id " + id);
        }

        private record Storage(String format, String key) {
        }

        private static String format(Global type) {
            return switch (type.name) {
                case "DoubleStorage" -> "F64";
                case "FloatStorage" -> "F32";
                case "HalfStorage" -> "F16";
                case "BFloat16Storage" -> "BF16";
                case "LongStorage" -> "I64";
                case "IntStorage" -> "I32";
                case "ShortStorage" -> "I16";
                case "CharStorage" -> "I8";
                case "ByteStorage" -> "U8";
                case "BoolStorage" -> "BOOL";
                case "Float8_e4m3fnStorage" -> "F8_E4M3";
                case "Float8_e5m2Storage" -> "F8_E5M2";
                default -> throw new IllegalArgumentException("unsupported storage type " + type.module + "." + type.name);
            };
        }

        private static Object call(Object function, List<?> args) {
            if (function instanceof Global g) {
                String name = g.module + "." + g.name;
                switch (name) {
                    case "collections.OrderedDict", "builtins.dict", "__builtin__.dict" -> {
                        return new LinkedHashMap<>();
                    }
                    case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor" -> {
                        Storage s = (Storage) args.get(0);
                        List<?> size = (List<?>) args.get(2);
                        List<?> stride = (List<?>) args.get(3);
                        int[] dims = new int[size.size()];
                        long[] strides = new long[stride.size()];
                        for (int i = 0; i < dims.length; i++) {
                            dims[i] = Math.toIntExact((Long) size.get(i));
                            strides[i] = (Long) stride.get(i);
                        }
                        return new Entry(s.key, s.format, Shape.of(dims), (Long) args.get(1), strides);
                    }
                    case "torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state" -> {
                        return args.get(0);
                    }
                    default -> {
                        return new Opaque(g);
                    }
                }
            }
            return new Opaque(function);
        }

        private static Long longOf(byte[] b) {
            if (b.length > 8) {
                throw new IllegalArgumentException("integer too large in pickle");
            }
            long v = 0;
            for (int i = b.length - 1; i >= 0; i--) {
                v = v << 8 | (b[i] & 0xFF);
            }
            if (b.length > 0 && b.length < 8 && b[b.length - 1] < 0) {
                v -= 1L << 8 * b.length;   // sign-extend
            }
            return v;
        }

        private Long longOf(int n) {
            return longOf(rawBytes(n));
        }

        private List<Object> tuple(int n) {
            List<Object> items = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                items.addFirst(pop());
            }
            return nullSafe(items);
        }

        /** Tuples may hold None, which {@link List#of} refuses. */
        private static List<Object> nullSafe(List<Object> items) {
            return java.util.Collections.unmodifiableList(new ArrayList<>(items));
        }

        @SuppressWarnings("unchecked")
        private static List<Object> list(Object o) {
            if (o instanceof List<?> l) {
                return (List<Object>) l;
            }
            throw new IllegalArgumentException("pickle appends to a " + o);
        }

        @SuppressWarnings("unchecked")
        private static Map<Object, Object> map(Object o) {
            if (o instanceof Map<?, ?> m) {
                return (Map<Object, Object>) m;
            }
            throw new IllegalArgumentException("pickle sets an item on a " + o);
        }

        private List<Object> popMark() {
            int m = stack.lastIndexOf(MARK);
            if (m < 0) {
                throw new IllegalArgumentException("pickle mark missing");
            }
            List<Object> items = new ArrayList<>(stack.subList(m + 1, stack.size()));
            stack.subList(m, stack.size()).clear();
            return items;
        }

        private Object pop() {
            if (stack.isEmpty()) {
                throw new IllegalArgumentException("pickle stack underflow");
            }
            return stack.removeLast();
        }

        private Object peek() {
            if (stack.isEmpty()) {
                throw new IllegalArgumentException("pickle stack underflow");
            }
            return stack.getLast();
        }

        private int u8() {
            if (at >= in.byteSize()) {
                throw new IllegalArgumentException("pickle ends early");
            }
            return in.get(ValueLayout.JAVA_BYTE, at++) & 0xFF;
        }

        private int i32() {
            return u8() | u8() << 8 | u8() << 16 | u8() << 24;
        }

        private long i64() {
            return Integer.toUnsignedLong(i32()) | (long) i32() << 32;
        }

        private byte[] rawBytes(int n) {
            if (n < 0 || at + n > in.byteSize()) {
                throw new IllegalArgumentException("pickle ends early");
            }
            byte[] b = in.asSlice(at, n).toArray(ValueLayout.JAVA_BYTE);
            at += n;
            return b;
        }

        private byte[] bytes(int n) {
            return rawBytes(n);
        }

        private String text(int n) {
            return new String(rawBytes(n), StandardCharsets.UTF_8);
        }

        private String line() {
            long end = at;
            while (end < in.byteSize() && in.get(ValueLayout.JAVA_BYTE, end) != '\n') {
                end++;
            }
            String s = new String(in.asSlice(at, end - at).toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
            at = end + 1;
            return s;
        }
    }
}
