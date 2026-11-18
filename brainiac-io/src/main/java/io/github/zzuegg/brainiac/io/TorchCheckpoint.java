package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/**
 * PyTorch checkpoints ({@code .pt}, {@code .pth}, {@code .bin} written by {@code torch.save}
 * since PyTorch 1.6): a zip of a pickled state dict and one file per storage. The pickle is
 * read by a small, safe interpreter that only builds dicts, lists, tuples, numbers, strings
 * and tensors; any other class is refused, so a checkpoint cannot run code.
 *
 * <p>Nested dicts are flattened with dots ({@code {"bert": {"x.weight": t}}} becomes
 * {@code bert.x.weight}); a checkpoint that is a single tensor (as Kokoro's voice files) has
 * the name {@code ""}.
 *
 * <pre>{@code
 * try (WeightSource w = TorchCheckpoint.open(Path.of("pytorch_model.bin"))) {
 *     Tensor t = w.tensor("decoder.conv_pre.weight", Device.cpu(), DType.FLOAT32);
 * }
 * }</pre>
 */
public final class TorchCheckpoint implements WeightSource {

    /** Where a tensor's values are: its storage, element type, offset, shape and strides. */
    private record Entry(String storage, String type, long offset, int[] shape, long[] stride) {
    }

    private final ZipFile zip;
    private final String prefix;
    private final Map<String, Entry> entries;

    private TorchCheckpoint(ZipFile zip, String prefix, Map<String, Entry> entries) {
        this.zip = zip;
        this.prefix = prefix;
        this.entries = entries;
    }

    /**
     * Opens a checkpoint.
     *
     * @param file a zip-format {@code torch.save} file
     * @return its tensors
     * @throws UncheckedIOException if it cannot be read
     * @throws IllegalArgumentException if it is not a zip checkpoint or holds classes other than tensors
     */
    public static TorchCheckpoint open(Path file) {
        ZipFile zip;
        try {
            zip = new ZipFile(file.toFile());
        } catch (IOException e) {
            throw new IllegalArgumentException(file + " is not a zip-format PyTorch checkpoint (saved by PyTorch 1.6"
                    + " or later); re-save it with torch.save or convert it to safetensors", e);
        }
        try {
            ZipEntry pkl = null;
            for (var en = zip.entries(); en.hasMoreElements();) {
                ZipEntry e = en.nextElement();
                if (e.getName().endsWith("data.pkl") && e.getName().indexOf('/') == e.getName().lastIndexOf('/')) {
                    pkl = e;
                    break;
                }
            }
            if (pkl == null) {
                throw new IllegalArgumentException(file + " has no data.pkl; not a PyTorch checkpoint");
            }
            String prefix = pkl.getName().substring(0, pkl.getName().length() - "data.pkl".length());
            byte[] bytes;
            try (InputStream in = zip.getInputStream(pkl)) {
                bytes = in.readAllBytes();
            }
            Object root = new Unpickler(bytes).load();
            Map<String, Entry> entries = new LinkedHashMap<>();
            flatten("", root, entries);
            return new TorchCheckpoint(zip, prefix, entries);
        } catch (IOException e) {
            closeQuietly(zip);
            throw new UncheckedIOException("cannot read " + file + ": " + e.getMessage(), e);
        } catch (RuntimeException e) {
            closeQuietly(zip);
            throw e;
        }
    }

    private static void closeQuietly(ZipFile zip) {
        try {
            zip.close();
        } catch (IOException ignored) {
            // nothing to do
        }
    }

    private static void flatten(String name, Object v, Map<String, Entry> out) {
        if (v instanceof Entry e) {
            out.put(name, e);
        } else if (v instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                String k = String.valueOf(e.getKey());
                flatten(name.isEmpty() ? k : name + "." + k, e.getValue(), out);
            }
        } else if (v instanceof List<?> l) {
            for (int i = 0; i < l.size(); i++) {
                flatten(name.isEmpty() ? String.valueOf(i) : name + "." + i, l.get(i), out);
            }
        }
        // numbers, strings and other values are not tensors
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
            throw new IllegalArgumentException("no tensor named '" + name + "' in this checkpoint");
        }
        return e;
    }

    @Override
    public Shape shape(String name) {
        return Shape.of(entry(name).shape);
    }

    @Override
    public String format(String name) {
        return entry(name).type;
    }

    @Override
    public DType naturalType(String name) {
        return switch (entry(name).type) {
            case "DoubleStorage" -> DType.FLOAT64;
            case "HalfStorage" -> DType.FLOAT16;
            case "BFloat16Storage" -> DType.BFLOAT16;
            default -> DType.FLOAT32;
        };
    }

    @Override
    public Tensor tensor(String name, Device device, DType dtype) {
        Entry e = entry(name);
        ZipEntry z = zip.getEntry(prefix + "data/" + e.storage);
        if (z == null) {
            throw new IllegalArgumentException("the checkpoint lacks the storage of " + name);
        }
        byte[] raw;
        try (InputStream in = zip.getInputStream(z)) {
            raw = in.readAllBytes();
        } catch (IOException ex) {
            throw new UncheckedIOException(ex);
        }
        ByteBuffer b = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        int n = 1;
        for (int d : e.shape) {
            n = Math.multiplyExact(n, d);
        }
        float[] out = new float[n];
        int rank = e.shape.length;
        int[] idx = new int[rank];
        for (int i = 0; i < n; i++) {
            long at = e.offset;
            for (int d = 0; d < rank; d++) {
                at += idx[d] * e.stride[d];
            }
            out[i] = read(b, e.type, at);
            for (int d = rank - 1; d >= 0; d--) {
                if (++idx[d] < e.shape[d]) {
                    break;
                }
                idx[d] = 0;
            }
        }
        return Tensor.of(Shape.of(e.shape), out, device).to(dtype);
    }

    private static float read(ByteBuffer b, String type, long element) {
        int i = Math.toIntExact(element);
        return switch (type) {
            case "FloatStorage" -> b.getFloat(i * 4);
            case "DoubleStorage" -> (float) b.getDouble(i * 8);
            case "HalfStorage" -> Float.float16ToFloat(b.getShort(i * 2));
            case "BFloat16Storage" -> Float.intBitsToFloat((b.getShort(i * 2) & 0xFFFF) << 16);
            case "LongStorage" -> b.getLong(i * 8);
            case "IntStorage" -> b.getInt(i * 4);
            case "ShortStorage" -> b.getShort(i * 2);
            case "CharStorage" -> b.get(i);
            case "ByteStorage", "BoolStorage" -> b.get(i) & 0xFF;
            default -> throw new IllegalArgumentException("unsupported tensor storage " + type);
        };
    }

    @Override
    public void close() {
        closeQuietly(zip);
    }

    @Override
    public String toString() {
        return "TorchCheckpoint[" + entries.size() + " tensors]";
    }

    // ---------------------------------------------------------------- pickle

    /** A class reference from GLOBAL, resolved only for the few allowed classes. */
    private record Global(String module, String name) {
        String full() {
            return module + "." + name;
        }
    }

    /** A storage reference from persistent_load: its type and key. */
    private record Storage(String type, String key) {
    }

    private static final Object MARK = new Object();

    /** The pickle opcodes torch.save writes (protocol 2), building only plain values and tensors. */
    private static final class Unpickler {
        private final byte[] data;
        private int pos;
        private final List<Object> stack = new ArrayList<>();
        private final Map<Integer, Object> memo = new HashMap<>();

        Unpickler(byte[] data) {
            this.data = data;
        }

        private int u8() {
            return data[pos++] & 0xFF;
        }

        private int i32() {
            int v = ByteBuffer.wrap(data, pos, 4).order(ByteOrder.LITTLE_ENDIAN).getInt();
            pos += 4;
            return v;
        }

        private String line() {
            int start = pos;
            while (data[pos] != '\n') {
                pos++;
            }
            String s = new String(data, start, pos - start, StandardCharsets.UTF_8);
            pos++;
            return s;
        }

        private String str(int n) {
            String s = new String(data, pos, n, StandardCharsets.UTF_8);
            pos += n;
            return s;
        }

        private Object pop() {
            return stack.removeLast();
        }

        private List<Object> popMark() {
            int m = stack.size() - 1;
            while (stack.get(m) != MARK) {
                m--;
            }
            List<Object> items = new ArrayList<>(stack.subList(m + 1, stack.size()));
            while (stack.size() > m) {
                stack.removeLast();
            }
            return items;
        }

        @SuppressWarnings("unchecked")
        Object load() {
            while (true) {
                int op = u8();
                switch (op) {
                    case 0x80 -> u8();   // PROTO
                    case 0x95 -> pos += 8;   // FRAME
                    case '.' -> {
                        return pop();
                    }
                    case '(' -> stack.add(MARK);
                    case '}' -> stack.add(new LinkedHashMap<>());
                    case ']' -> stack.add(new ArrayList<>());
                    case ')' -> stack.add(List.of());
                    case 'N' -> stack.add(null);
                    case 0x88 -> stack.add(Boolean.TRUE);
                    case 0x89 -> stack.add(Boolean.FALSE);
                    case 'K' -> stack.add((long) u8());
                    case 'M' -> {
                        int v = u8() | u8() << 8;
                        stack.add((long) v);
                    }
                    case 'J' -> stack.add((long) i32());
                    case 0x8a -> {   // LONG1
                        int n = u8();
                        long v = 0;
                        for (int i = 0; i < n; i++) {
                            v |= (long) (data[pos + i] & 0xFF) << (8 * i);
                        }
                        if (n > 0 && n < 8 && (data[pos + n - 1] & 0x80) != 0) {
                            v -= 1L << (8 * n);
                        }
                        pos += n;
                        stack.add(v);
                    }
                    case 'G' -> {
                        double v = ByteBuffer.wrap(data, pos, 8).order(ByteOrder.BIG_ENDIAN).getDouble();
                        pos += 8;
                        stack.add(v);
                    }
                    case 'X' -> stack.add(str(i32()));
                    case 0x8c -> stack.add(str(u8()));   // SHORT_BINUNICODE
                    case 'U' -> stack.add(str(u8()));    // SHORT_BINSTRING
                    case 'T' -> stack.add(str(i32()));   // BINSTRING
                    case 'C' -> {   // SHORT_BINBYTES
                        int n = u8();
                        pos += n;
                        stack.add(new byte[0]);
                    }
                    case 'B' -> {   // BINBYTES
                        int n = i32();
                        pos += n;
                        stack.add(new byte[0]);
                    }
                    case 'c' -> stack.add(new Global(line(), line()));
                    case 0x93 -> {   // STACK_GLOBAL
                        String name = (String) pop();
                        String module = (String) pop();
                        stack.add(new Global(module, name));
                    }
                    case 'q' -> memo.put(u8(), stack.getLast());
                    case 'r' -> memo.put(i32(), stack.getLast());
                    case 0x94 -> memo.put(memo.size(), stack.getLast());   // MEMOIZE
                    case 'h' -> stack.add(memo.get(u8()));
                    case 'j' -> stack.add(memo.get(i32()));
                    case 't' -> stack.add(popMark());
                    case 0x85 -> stack.add(tuple(pop()));
                    case 0x86 -> {
                        Object b = pop();
                        Object a = pop();
                        stack.add(tuple(a, b));
                    }
                    case 0x87 -> {
                        Object c = pop();
                        Object b = pop();
                        Object a = pop();
                        stack.add(tuple(a, b, c));
                    }
                    case 'a' -> {
                        Object v = pop();
                        ((List<Object>) stack.getLast()).add(v);
                    }
                    case 'e' -> {
                        List<Object> items = popMark();
                        ((List<Object>) stack.getLast()).addAll(items);
                    }
                    case 's' -> {
                        Object v = pop();
                        Object k = pop();
                        ((Map<Object, Object>) stack.getLast()).put(k, v);
                    }
                    case 'u' -> {
                        List<Object> items = popMark();
                        Map<Object, Object> m = (Map<Object, Object>) stack.getLast();
                        for (int i = 0; i < items.size(); i += 2) {
                            m.put(items.get(i), items.get(i + 1));
                        }
                    }
                    case 'Q' -> {   // BINPERSID: ('storage', type, key, location, numel)
                        List<?> pid = (List<?>) pop();
                        Object type = pid.get(1);
                        String t = type instanceof Global g ? g.name() : String.valueOf(type);
                        stack.add(new Storage(t, String.valueOf(pid.get(2))));
                    }
                    case 'R' -> {   // REDUCE
                        List<?> args = (List<?>) pop();
                        Global f = (Global) pop();
                        stack.add(reduce(f, args));
                    }
                    case 0x81 -> {   // NEWOBJ
                        List<?> args = (List<?>) pop();
                        Global cls = (Global) pop();
                        stack.add(reduce(cls, args));
                    }
                    case 'b' -> {   // BUILD: set state; the plain values here ignore it
                        Object state = pop();
                        Object obj = stack.getLast();
                        if (obj instanceof Map<?, ?> m && state instanceof Map<?, ?> s && m.isEmpty()) {
                            ((Map<Object, Object>) obj).putAll(s);
                        }
                    }
                    default -> throw new IllegalArgumentException(String.format(
                            "unsupported pickle opcode 0x%02x at %d; not a plain PyTorch state dict", op, pos - 1));
                }
            }
        }

        private static List<Object> tuple(Object... items) {
            return java.util.Arrays.asList(items);
        }

        private Object reduce(Global f, List<?> a) {
            switch (f.full()) {
                case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor" -> {
                    Storage s = (Storage) a.get(0);
                    long offset = ((Number) a.get(1)).longValue();
                    List<?> size = (List<?>) a.get(2);
                    List<?> stride = (List<?>) a.get(3);
                    int[] shape = new int[size.size()];
                    long[] st = new long[stride.size()];
                    for (int i = 0; i < shape.length; i++) {
                        shape[i] = ((Number) size.get(i)).intValue();
                        st[i] = ((Number) stride.get(i)).longValue();
                    }
                    return new Entry(s.key(), s.type(), offset, shape, st);
                }
                case "torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state" -> {
                    return a.get(0);
                }
                case "collections.OrderedDict", "builtins.dict", "__builtin__.dict" -> {
                    Map<Object, Object> m = new LinkedHashMap<>();
                    if (!a.isEmpty() && a.get(0) instanceof List<?> pairs) {
                        for (Object p : pairs) {
                            List<?> kv = (List<?>) p;
                            m.put(kv.get(0), kv.get(1));
                        }
                    }
                    return m;
                }
                case "torch.device", "builtins.set", "builtins.slice", "argparse.Namespace", "pathlib.PosixPath",
                        "pathlib.WindowsPath", "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar",
                        "numpy.dtype" -> {
                    return a;   // metadata that sometimes rides along; never a tensor
                }
                default -> throw new IllegalArgumentException("the checkpoint holds a " + f.full()
                        + ", which Brainiac does not load (only state dicts of tensors)");
            }
        }
    }
}
