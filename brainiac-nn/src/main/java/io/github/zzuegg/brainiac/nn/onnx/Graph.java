package io.github.zzuegg.brainiac.nn.onnx;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * An ONNX model file read into plain Java objects: the nodes in order, the initializers
 * and the declared inputs and outputs. Tensor data is copied out of the file, so nothing
 * stays mapped once reading is done.
 */
final class Graph {

    static final int FLOAT = 1;
    static final int UINT8 = 2;
    static final int INT8 = 3;
    static final int UINT16 = 4;
    static final int INT16 = 5;
    static final int INT32 = 6;
    static final int INT64 = 7;
    static final int STRING = 8;
    static final int BOOL = 9;
    static final int FLOAT16 = 10;
    static final int DOUBLE = 11;
    static final int UINT32 = 12;
    static final int UINT64 = 13;
    static final int BFLOAT16 = 16;

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfShort SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** Tensor contents: floating point types in {@code f}, the others in {@code l}. */
    record Raw(int type, int[] dims, float[] f, long[] l) {
        boolean floating() {
            return isFloat(type);
        }
    }

    /** A node attribute; only the fields of its kind are set. */
    static final class Attr {
        float f;
        long i;
        String s;
        Raw t;
        float[] floats;
        long[] ints;
        List<String> strings = new ArrayList<>();
        boolean graph;
    }

    record Node(String op, String domain, String name, String[] in, String[] out, Map<String, Attr> attrs) {
        Attr attr(String key) {
            return attrs.get(key);
        }

        long i(String key, long otherwise) {
            Attr a = attrs.get(key);
            return a == null ? otherwise : a.i;
        }

        float f(String key, float otherwise) {
            Attr a = attrs.get(key);
            return a == null ? otherwise : a.f;
        }

        String s(String key, String otherwise) {
            Attr a = attrs.get(key);
            return a == null || a.s == null ? otherwise : a.s;
        }

        long[] ints(String key) {
            Attr a = attrs.get(key);
            return a == null ? null : a.ints;
        }

        /** The input at {@code k}, or null when absent or left empty. */
        String input(int k) {
            return k < in.length && !in[k].isEmpty() ? in[k] : null;
        }

        @Override
        public String toString() {
            return op + (name.isEmpty() ? "" : " '" + name + "'");
        }
    }

    final List<Node> nodes = new ArrayList<>();
    final Map<String, Raw> initializers = new LinkedHashMap<>();
    final Map<String, Integer> inputs = new LinkedHashMap<>();
    final Map<String, Integer> outputs = new LinkedHashMap<>();
    int opset = 1;
    String producer = "";

    static boolean isFloat(int type) {
        return type == FLOAT || type == FLOAT16 || type == DOUBLE || type == BFLOAT16;
    }

    static Graph read(Path file) {
        try (Arena arena = Arena.ofConfined(); FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            MemorySegment m = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
            Graph g = new Graph();
            g.model(new Protobuf(m), file.toAbsolutePath().getParent());
            return g;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void model(Protobuf p, Path dir) {
        Protobuf graph = null;
        while (p.next()) {
            switch (p.field()) {
                case 2 -> producer = p.string();
                case 7 -> graph = p.message();
                case 8 -> {
                    Protobuf o = p.message();
                    String domain = "";
                    long version = 1;
                    while (o.next()) {
                        if (o.field() == 1) {
                            domain = o.string();
                        } else if (o.field() == 2) {
                            version = o.varint();
                        } else {
                            o.skip();
                        }
                    }
                    if (domain.isEmpty() || domain.equals("ai.onnx")) {
                        opset = (int) version;
                    }
                }
                default -> p.skip();
            }
        }
        if (graph == null) {
            throw new IllegalArgumentException("not an ONNX model: no graph");
        }
        graph(graph, dir);
    }

    private void graph(Protobuf p, Path dir) {
        Map<String, Integer> declared = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> nodes.add(node(p.message(), dir));
                case 5 -> {
                    Map<String, Object> t = new HashMap<>();
                    Raw r = tensor(p.message(), dir, t);
                    initializers.put((String) t.get("name"), r);
                }
                case 11 -> valueInfo(p.message(), declared);
                case 12 -> valueInfo(p.message(), outputs);
                default -> p.skip();
            }
        }
        // older exporters list the initializers among the inputs too
        declared.forEach((name, type) -> {
            if (!initializers.containsKey(name)) {
                inputs.put(name, type);
            }
        });
    }

    private static void valueInfo(Protobuf p, Map<String, Integer> into) {
        String name = "";
        int type = FLOAT;
        while (p.next()) {
            if (p.field() == 1) {
                name = p.string();
            } else if (p.field() == 2) {
                Protobuf t = p.message();
                while (t.next()) {
                    if (t.field() == 1) {
                        Protobuf tt = t.message();
                        while (tt.next()) {
                            if (tt.field() == 1) {
                                type = (int) tt.varint();
                            } else {
                                tt.skip();
                            }
                        }
                    } else {
                        t.skip();
                    }
                }
            } else {
                p.skip();
            }
        }
        into.put(name, type);
    }

    private static Node node(Protobuf p, Path dir) {
        List<String> in = new ArrayList<>();
        List<String> out = new ArrayList<>();
        String name = "";
        String op = "";
        String domain = "";
        Map<String, Attr> attrs = new LinkedHashMap<>();
        while (p.next()) {
            switch (p.field()) {
                case 1 -> in.add(p.string());
                case 2 -> out.add(p.string());
                case 3 -> name = p.string();
                case 4 -> op = p.string();
                case 5 -> {
                    Map<String, Object> holder = new HashMap<>();
                    Attr a = attribute(p.message(), dir, holder);
                    attrs.put((String) holder.get("name"), a);
                }
                case 7 -> domain = p.string();
                default -> p.skip();
            }
        }
        return new Node(op, domain, name, in.toArray(String[]::new), out.toArray(String[]::new), attrs);
    }

    private static Attr attribute(Protobuf p, Path dir, Map<String, Object> holder) {
        Attr a = new Attr();
        List<Float> floats = new ArrayList<>();
        List<Long> ints = new ArrayList<>();
        holder.put("name", "");
        while (p.next()) {
            switch (p.field()) {
                case 1 -> holder.put("name", p.string());
                case 2 -> a.f = p.float32();
                case 3 -> a.i = p.varint();
                case 4 -> a.s = p.string();
                case 5 -> a.t = tensor(p.message(), dir, new HashMap<>());
                case 6 -> {
                    a.graph = true;
                    p.skip();
                }
                case 7 -> p.floats(floats);
                case 8 -> p.longs(ints);
                case 9 -> a.strings.add(p.string());
                default -> p.skip();
            }
        }
        a.floats = new float[floats.size()];
        for (int k = 0; k < a.floats.length; k++) {
            a.floats[k] = floats.get(k);
        }
        a.ints = ints.stream().mapToLong(Long::longValue).toArray();
        return a;
    }

    /** Reads a TensorProto; its name goes into {@code meta}. */
    static Raw tensor(Protobuf p, Path dir, Map<String, Object> meta) {
        List<Long> dims = new ArrayList<>();
        int type = FLOAT;
        List<Float> floatData = new ArrayList<>();
        List<Long> intData = new ArrayList<>();
        List<Double> doubleData = new ArrayList<>();
        MemorySegment raw = null;
        Map<String, String> external = new HashMap<>();
        boolean isExternal = false;
        meta.put("name", "");
        while (p.next()) {
            switch (p.field()) {
                case 1 -> p.longs(dims);
                case 2 -> type = (int) p.varint();
                case 4 -> p.floats(floatData);
                case 5, 7, 11 -> p.longs(intData);
                case 8 -> meta.put("name", p.string());
                case 9 -> raw = p.bytes();
                case 10 -> p.doubles(doubleData);
                case 13 -> {
                    Protobuf e = p.message();
                    String k = "";
                    String v = "";
                    while (e.next()) {
                        if (e.field() == 1) {
                            k = e.string();
                        } else if (e.field() == 2) {
                            v = e.string();
                        } else {
                            e.skip();
                        }
                    }
                    external.put(k, v);
                }
                case 14 -> isExternal = p.varint() == 1;
                default -> p.skip();
            }
        }
        int[] shape = dims.stream().mapToInt(Long::intValue).toArray();
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor '" + meta.get("name") + "' is too large: " + n);
        }
        int count = (int) n;
        if (type == STRING) {
            return new Raw(type, shape, null, new long[0]);
        }
        if (isExternal) {
            return external(type, shape, count, external, dir, (String) meta.get("name"));
        }
        if (raw != null) {
            return fromBytes(type, shape, count, raw);
        }
        if (isFloat(type)) {
            float[] f = new float[count];
            if (!floatData.isEmpty()) {
                for (int k = 0; k < count; k++) {
                    f[k] = floatData.get(k);
                }
            } else if (!doubleData.isEmpty()) {
                for (int k = 0; k < count; k++) {
                    f[k] = doubleData.get(k).floatValue();
                }
            } else if (!intData.isEmpty()) { // half precision bits in int32_data
                for (int k = 0; k < count; k++) {
                    int bits = intData.get(k).intValue() & 0xFFFF;
                    f[k] = type == BFLOAT16 ? Float.intBitsToFloat(bits << 16) : Float.float16ToFloat((short) bits);
                }
            }
            return new Raw(type, shape, f, null);
        }
        long[] l = new long[count];
        for (int k = 0; k < count && k < intData.size(); k++) {
            l[k] = intData.get(k);
        }
        return new Raw(type, shape, null, l);
    }

    private static Raw external(int type, int[] shape, int count, Map<String, String> info, Path dir, String name) {
        String location = info.get("location");
        if (location == null || dir == null) {
            throw new IllegalArgumentException("tensor '" + name + "' has external data without a location");
        }
        Path file = dir.resolve(location).normalize();
        if (!file.startsWith(dir.normalize())) {
            throw new IllegalArgumentException("external data of '" + name + "' points outside the model folder");
        }
        long offset = info.containsKey("offset") ? Long.parseLong(info.get("offset")) : 0;
        try (Arena arena = Arena.ofConfined(); FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            long length = info.containsKey("length") ? Long.parseLong(info.get("length")) : ch.size() - offset;
            MemorySegment m = ch.map(FileChannel.MapMode.READ_ONLY, offset, length, arena);
            return fromBytes(type, shape, count, m);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Raw fromBytes(int type, int[] shape, int count, MemorySegment s) {
        if (isFloat(type)) {
            float[] f = new float[count];
            for (int k = 0; k < count; k++) {
                f[k] = switch (type) {
                    case FLOAT -> Float.intBitsToFloat(s.get(INT, 4L * k));
                    case DOUBLE -> (float) Double.longBitsToDouble(s.get(LONG, 8L * k));
                    case FLOAT16 -> Float.float16ToFloat(s.get(SHORT, 2L * k));
                    default -> Float.intBitsToFloat((s.get(SHORT, 2L * k) & 0xFFFF) << 16);
                };
            }
            return new Raw(type, shape, f, null);
        }
        long[] l = new long[count];
        for (int k = 0; k < count; k++) {
            l[k] = switch (type) {
                case INT64, UINT64 -> s.get(LONG, 8L * k);
                case INT32 -> s.get(INT, 4L * k);
                case UINT32 -> s.get(INT, 4L * k) & 0xFFFFFFFFL;
                case INT16 -> s.get(SHORT, 2L * k);
                case UINT16 -> s.get(SHORT, 2L * k) & 0xFFFF;
                case INT8 -> s.get(ValueLayout.JAVA_BYTE, k);
                case UINT8, BOOL -> s.get(ValueLayout.JAVA_BYTE, k) & 0xFF;
                default -> throw new IllegalArgumentException("unsupported tensor type " + type);
            };
        }
        return new Raw(type, shape, null, l);
    }
}
