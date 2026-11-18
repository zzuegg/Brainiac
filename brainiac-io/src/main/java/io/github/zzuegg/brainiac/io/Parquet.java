package io.github.zzuegg.brainiac.io;

import java.io.ByteArrayInputStream;
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
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.function.Consumer;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.zip.GZIPInputStream;

/**
 * Reads Apache Parquet files, the format most Hugging Face datasets are published in, into
 * rows shaped like {@link Jsonl}'s: a map per row from column name to value.
 *
 * <pre>{@code
 * for (Map<String, Object> row : Parquet.read(Path.of("train-00000-of-00001.parquet"))) {
 *     List<?> messages = (List<?>) row.get("messages");   // [{role=user, content=...}, ...]
 * }
 * }</pre>
 *
 * <p>Values are {@link Long} for integers, {@link Double} for floating point,
 * {@link Boolean}, {@link String} for text columns, {@code byte[]} for other binary columns
 * (an image's bytes), {@link List} for lists, {@link Map} for structs and maps, and
 * {@code null} for missing values. Nested columns (lists of structs, maps, lists of lists)
 * are assembled from their repetition and definition levels.
 *
 * <p>Supported: data pages v1 and v2; the PLAIN, dictionary, RLE, DELTA_BINARY_PACKED,
 * DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY and BYTE_STREAM_SPLIT encodings; uncompressed,
 * Snappy, gzip, LZ4 (raw) and Zstandard compression. The file is memory-mapped and read one
 * row group at a time.
 */
public final class Parquet implements AutoCloseable {

    private static final byte[] MAGIC = "PAR1".getBytes(StandardCharsets.US_ASCII);

    /** A schema node: a group (with children) or a leaf column. */
    private record Node(String name, int repetition, int type, int typeLength, int converted, int logical,
            List<Node> children) {

        boolean leaf() {
            return children.isEmpty();
        }

        boolean text() {
            return converted == 0 || converted == 4 || converted == 19 || logical == 1 || logical == 4
                    || logical == 12;
        }
    }

    /** A leaf column with the nodes from the root to it. */
    private record Column(List<Node> path, int maxDef, int maxRep) {
        Node leaf() {
            return path.getLast();
        }
    }

    private final Path path;
    private final Arena arena;
    private final MemorySegment file;
    private final Node root;
    private final List<Column> columns = new ArrayList<>();
    private final List<Map<Integer, Object>> rowGroups;
    private final long rows;

    private Parquet(Path path) throws IOException {
        this.path = path;
        arena = Arena.ofShared();
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            file = ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size(), arena);
        }
        try {
            long size = file.byteSize();
            if (size < 12 || !Arrays.equals(bytes(0, 4), MAGIC) || !Arrays.equals(bytes(size - 4, 4), MAGIC)) {
                throw new IllegalArgumentException(path + " is not a Parquet file");
            }
            int footer = file.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), size - 8);
            if (footer <= 0 || footer > size - 12) {
                throw new IllegalArgumentException(path + ": broken Parquet footer");
            }
            Map<Integer, Object> meta = new Thrift(file, size - 8 - footer).struct();
            List<?> schema = list(meta.get(2));
            int[] next = {0};
            root = node(schema, next);
            leaves(root, new ArrayList<>(), 0, 0);
            rows = num(meta.get(3));
            rowGroups = new ArrayList<>();
            for (Object g : list(meta.get(4))) {
                rowGroups.add(struct(g));
            }
        } catch (RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Opens a Parquet file.
     *
     * @param file the file
     * @return the reader; close it when done
     * @throws UncheckedIOException if it cannot be read
     */
    public static Parquet open(Path file) {
        Objects.requireNonNull(file, "file");
        try {
            return new Parquet(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Reads every row of a file.
     *
     * @param file the file
     * @return the rows, a map from column name to value each
     */
    public static List<Map<String, Object>> read(Path file) {
        try (Parquet p = open(file)) {
            List<Map<String, Object>> out = new ArrayList<>();
            for (int g = 0; g < p.rowGroups(); g++) {
                out.addAll(p.rowGroup(g));
            }
            return out;
        }
    }

    /**
     * Reads a file's rows lazily, one row group at a time; close the stream to close the file.
     *
     * @param file the file
     * @return the rows
     */
    public static Stream<Map<String, Object>> stream(Path file) {
        Parquet p = open(file);
        Spliterator<Map<String, Object>> s = new Spliterators.AbstractSpliterator<>(p.rows, Spliterator.ORDERED) {
            private int group;
            private List<Map<String, Object>> current = List.of();
            private int at;

            @Override
            public boolean tryAdvance(Consumer<? super Map<String, Object>> action) {
                while (at == current.size()) {
                    if (group == p.rowGroups()) {
                        return false;
                    }
                    current = p.rowGroup(group++);
                    at = 0;
                }
                action.accept(current.get(at++));
                return true;
            }
        };
        return StreamSupport.stream(s, false).onClose(p::close);
    }

    /**
     * Reads one text column of a file, skipping missing values.
     *
     * @param file the file
     * @param column the column's name
     * @return its values
     */
    public static List<String> texts(Path file, String column) {
        try (Stream<Map<String, Object>> rows = stream(file)) {
            return rows.map(r -> r.get(column)).filter(Objects::nonNull).map(String::valueOf).toList();
        }
    }

    /** @return the top-level column names, in schema order */
    public List<String> columns() {
        return root.children.stream().map(Node::name).toList();
    }

    /** @return the number of rows */
    public long rows() {
        return rows;
    }

    /** @return the number of row groups */
    public int rowGroups() {
        return rowGroups.size();
    }

    /**
     * Reads one row group.
     *
     * @param index the row group, from 0
     * @return its rows
     */
    public List<Map<String, Object>> rowGroup(int index) {
        Map<Integer, Object> group = rowGroups.get(index);
        int n = Math.toIntExact(num(group.get(3)));
        List<Map<String, Object>> raw = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            raw.add(new LinkedHashMap<>());
        }
        List<?> chunks = list(group.get(1));
        if (chunks.size() != columns.size()) {
            throw new IllegalArgumentException(path + ": row group has " + chunks.size() + " columns, schema "
                    + columns.size());
        }
        for (int c = 0; c < columns.size(); c++) {
            Map<Integer, Object> chunk = struct(chunks.get(c));
            if (chunk.get(1) != null) {
                throw new IllegalArgumentException(path + ": columns in other files are not supported");
            }
            assemble(columns.get(c), decode(columns.get(c), struct(chunk.get(3))), raw);
        }
        List<Map<String, Object>> out = new ArrayList<>(n);
        for (Map<String, Object> r : raw) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (Node child : root.children) {
                row.put(child.name, field(r, child));
            }
            out.add(row);
        }
        return out;
    }

    @Override
    public void close() {
        arena.close();
    }

    @Override
    public String toString() {
        return "Parquet " + path + " (" + rows + " rows, columns " + columns() + ")";
    }

    // ------------------------------------------------------------------ schema

    private static Node node(List<?> schema, int[] next) {
        Map<Integer, Object> e = struct(schema.get(next[0]++));
        int children = e.get(5) == null ? 0 : (int) num(e.get(5));
        List<Node> kids = new ArrayList<>();
        for (int i = 0; i < children; i++) {
            kids.add(node(schema, next));
        }
        int logical = -1;
        if (e.get(10) instanceof Map<?, ?> l && !l.isEmpty()) {
            logical = (Integer) l.keySet().iterator().next();
        }
        return new Node(new String((byte[]) e.get(4), StandardCharsets.UTF_8),
                e.get(3) == null ? 0 : (int) num(e.get(3)), e.get(1) == null ? -1 : (int) num(e.get(1)),
                e.get(2) == null ? 0 : (int) num(e.get(2)), e.get(6) == null ? -1 : (int) num(e.get(6)), logical,
                List.copyOf(kids));
    }

    private void leaves(Node n, List<Node> path, int def, int rep) {
        for (Node c : n.children) {
            List<Node> p = new ArrayList<>(path);
            p.add(c);
            int d = def + (c.repetition != 0 ? 1 : 0);
            int r = rep + (c.repetition == 2 ? 1 : 0);
            if (c.leaf()) {
                columns.add(new Column(List.copyOf(p), d, r));
            } else {
                leaves(c, p, d, r);
            }
        }
    }

    // ------------------------------------------------------------------ assembly

    /** A column chunk's levels and its non-null values, in order. */
    private record Levels(int[] rep, int[] def, Object[] values, int count) {
    }

    /** Puts a column's entries into the rows' nested maps and lists. */
    @SuppressWarnings("unchecked")
    private static void assemble(Column col, Levels lv, List<Map<String, Object>> rows) {
        int k = col.path.size();
        int[] defAt = new int[k];
        int[] repAt = new int[k];
        int d = 0;
        int r = 0;
        for (int i = 0; i < k; i++) {
            Node n = col.path.get(i);
            d += n.repetition != 0 ? 1 : 0;
            r += n.repetition == 2 ? 1 : 0;
            defAt[i] = d;
            repAt[i] = r;
        }
        int[] index = new int[col.maxRep + 1];
        int row = -1;
        int value = 0;
        for (int e = 0; e < lv.count; e++) {
            int rep = lv.rep == null ? 0 : lv.rep[e];
            int def = lv.def == null ? col.maxDef : lv.def[e];
            if (rep == 0) {
                row++;
                if (row >= rows.size()) {
                    throw new IllegalArgumentException("column " + col.leaf().name + " has more rows than its row group");
                }
            }
            Object container = rows.get(row);
            for (int i = 0; i < k; i++) {
                Node n = col.path.get(i);
                Map<String, Object> parent = (Map<String, Object>) container;
                boolean last = i == k - 1;
                if (defAt[i] > def) {
                    // not defined from here on: an empty list or a null
                    if (n.repetition == 2) {
                        parent.putIfAbsent(n.name, new ArrayList<>());
                    } else {
                        parent.putIfAbsent(n.name, null);
                    }
                    break;
                }
                if (n.repetition == 2) {
                    List<Object> list = (List<Object>) parent.computeIfAbsent(n.name, x -> new ArrayList<>());
                    int level = repAt[i];
                    if (rep < level) {
                        index[level] = 0;
                    } else if (rep == level) {
                        index[level]++;
                    }
                    int at = index[level];
                    if (last) {
                        set(list, at, lv.values[value++]);
                    } else {
                        while (list.size() <= at) {
                            list.add(new LinkedHashMap<String, Object>());
                        }
                        container = list.get(at);
                    }
                } else if (last) {
                    parent.put(n.name, lv.values[value++]);
                } else {
                    Object child = parent.get(n.name);
                    if (child == null) {
                        child = new LinkedHashMap<String, Object>();
                        parent.put(n.name, child);
                    }
                    container = child;
                }
            }
        }
    }

    private static void set(List<Object> list, int at, Object v) {
        while (list.size() <= at) {
            list.add(null);
        }
        list.set(at, v);
    }

    /** A field of an assembled group, turned into its final form. */
    private static Object field(Map<?, ?> group, Node n) {
        Object v = group == null ? null : group.get(n.name);
        if (n.repetition == 2) {
            List<Object> out = new ArrayList<>();
            if (v instanceof List<?> l) {
                for (Object item : l) {
                    out.add(value(item, n));
                }
            }
            return out;
        }
        return value(v, n);
    }

    /** One value of a node, ignoring its repetition: LIST and MAP groups become lists and maps. */
    private static Object value(Object v, Node n) {
        if (v == null) {
            return null;
        }
        if (n.leaf()) {
            return v instanceof byte[] b && n.text() ? new String(b, StandardCharsets.UTF_8) : v;
        }
        Map<?, ?> m = (Map<?, ?>) v;
        boolean list = n.converted == 3 || n.logical == 3;
        boolean map = n.converted == 1 || n.converted == 2 || n.logical == 2;
        if ((list || map) && n.children.size() == 1 && n.children.getFirst().repetition == 2) {
            Node rep = n.children.getFirst();
            Object items = m.get(rep.name);
            List<?> entries = items instanceof List<?> l ? l : List.of();
            if (map && !rep.leaf() && rep.children.size() == 2) {
                Map<Object, Object> out = new LinkedHashMap<>();
                for (Object e : entries) {
                    Map<?, ?> kv = (Map<?, ?>) e;
                    out.put(field(kv, rep.children.get(0)), field(kv, rep.children.get(1)));
                }
                return out;
            }
            List<Object> out = new ArrayList<>();
            // three levels (list { element }) unless the repeated node is itself the element
            boolean threeLevel = !rep.leaf() && rep.children.size() == 1 && !rep.name.equals("array")
                    && !rep.name.equals(n.name + "_tuple");
            for (Object e : entries) {
                out.add(threeLevel ? field((Map<?, ?>) e, rep.children.getFirst()) : value(e, rep));
            }
            return out;
        }
        Map<String, Object> out = new LinkedHashMap<>();
        for (Node c : n.children) {
            out.put(c.name, field(m, c));
        }
        return out;
    }

    // ------------------------------------------------------------------ pages

    private Levels decode(Column col, Map<Integer, Object> meta) {
        int codec = (int) num(meta.get(4));
        long total = num(meta.get(5));
        long start = num(meta.get(9));
        if (meta.get(11) != null && num(meta.get(11)) > 0 && num(meta.get(11)) < start) {
            start = num(meta.get(11));
        }
        Node leaf = col.leaf();
        int n = Math.toIntExact(total);
        int[] rep = col.maxRep > 0 ? new int[n] : null;
        int[] def = col.maxDef > 0 ? new int[n] : null;
        Object[] values = new Object[n];
        int count = 0;
        int stored = 0;
        Object[] dictionary = null;
        long at = start;
        while (count < n) {
            Thrift t = new Thrift(file, at);
            Map<Integer, Object> header = t.struct();
            at = t.at;
            int type = (int) num(header.get(1));
            int raw = (int) num(header.get(2));
            int compressed = (int) num(header.get(3));
            byte[] page = bytes(at, compressed);
            at += compressed;
            if (type == 2) {
                Map<Integer, Object> dh = struct(header.get(7));
                byte[] data = decompress(codec, page, raw);
                Object[] dict = new Object[(int) num(dh.get(1))];
                new Values(data, 0, data.length, leaf).plain(dict, 0, dict.length);
                dictionary = dict;
            } else if (type == 0) {
                Map<Integer, Object> dh = struct(header.get(5));
                int m = (int) num(dh.get(1));
                byte[] data = decompress(codec, page, raw);
                int p = 0;
                if (rep != null) {
                    int len = le32(data, p);
                    hybrid(data, p + 4, p + 4 + len, bitWidth(col.maxRep), rep, count, m);
                    p += 4 + len;
                }
                int nonNull = m;
                if (def != null) {
                    int len = le32(data, p);
                    hybrid(data, p + 4, p + 4 + len, bitWidth(col.maxDef), def, count, m);
                    p += 4 + len;
                    nonNull = nonNull(def, count, m, col.maxDef);
                }
                values(data, p, data.length, (int) num(dh.get(2)), leaf, dictionary, values, stored, nonNull);
                stored += nonNull;
                count += m;
            } else if (type == 3) {
                Map<Integer, Object> dh = struct(header.get(8));
                int m = (int) num(dh.get(1));
                int nulls = (int) num(dh.get(2));
                int defLen = (int) num(dh.get(5));
                int repLen = (int) num(dh.get(6));
                boolean isCompressed = !(dh.get(7) instanceof Boolean b) || b;
                if (rep != null) {
                    hybrid(page, 0, repLen, bitWidth(col.maxRep), rep, count, m);
                }
                if (def != null) {
                    hybrid(page, repLen, repLen + defLen, bitWidth(col.maxDef), def, count, m);
                }
                byte[] rest = Arrays.copyOfRange(page, repLen + defLen, page.length);
                byte[] data = isCompressed ? decompress(codec, rest, raw - repLen - defLen) : rest;
                int nonNull = def != null ? nonNull(def, count, m, col.maxDef) : m - nulls;
                values(data, 0, data.length, (int) num(dh.get(4)), leaf, dictionary, values, stored, nonNull);
                stored += nonNull;
                count += m;
            }
            // index pages (1) are skipped
        }
        return new Levels(rep, def, values, n);
    }

    private static int nonNull(int[] def, int from, int count, int max) {
        int n = 0;
        for (int i = from; i < from + count; i++) {
            if (def[i] == max) {
                n++;
            }
        }
        return n;
    }

    private static int bitWidth(int max) {
        return 32 - Integer.numberOfLeadingZeros(max);
    }

    /** Decodes {@code count} values of a data page into {@code out} from {@code at}. */
    private static void values(byte[] data, int from, int to, int encoding, Node leaf, Object[] dictionary,
            Object[] out, int at, int count) {
        switch (encoding) {
            case 0 -> new Values(data, from, to, leaf).plain(out, at, count);
            case 2, 8 -> {
                if (dictionary == null) {
                    throw new IllegalArgumentException("dictionary page missing for column " + leaf.name);
                }
                int[] ids = new int[count];
                if (count > 0) {
                    hybrid(data, from + 1, to, data[from] & 0xFF, ids, 0, count);
                }
                for (int i = 0; i < count; i++) {
                    out[at + i] = dictionary[ids[i]];
                }
            }
            case 3 -> {   // RLE: booleans, with a length prefix
                int[] bits = new int[count];
                hybrid(data, from + 4, from + 4 + le32(data, from), 1, bits, 0, count);
                for (int i = 0; i < count; i++) {
                    out[at + i] = bits[i] != 0;
                }
            }
            case 5 -> {
                long[] v = new long[count];
                deltas(data, from, v, count);
                for (int i = 0; i < count; i++) {
                    out[at + i] = leaf.type == 1 ? (long) (int) v[i] : v[i];
                }
            }
            case 6 -> {
                long[] lengths = new long[count];
                int p = deltas(data, from, lengths, count);
                for (int i = 0; i < count; i++) {
                    int len = (int) lengths[i];
                    out[at + i] = Arrays.copyOfRange(data, p, p + len);
                    p += len;
                }
            }
            case 7 -> {
                long[] prefix = new long[count];
                int p = deltas(data, from, prefix, count);
                long[] suffix = new long[count];
                p = deltas(data, p, suffix, count);
                byte[] previous = new byte[0];
                for (int i = 0; i < count; i++) {
                    int pre = (int) prefix[i];
                    int suf = (int) suffix[i];
                    byte[] v = new byte[pre + suf];
                    System.arraycopy(previous, 0, v, 0, pre);
                    System.arraycopy(data, p, v, pre, suf);
                    p += suf;
                    out[at + i] = v;
                    previous = v;
                }
            }
            case 9 -> {   // BYTE_STREAM_SPLIT: byte k of every value, then byte k + 1, ...
                int width = switch (leaf.type) {
                    case 1, 4 -> 4;
                    case 2, 5 -> 8;
                    case 7 -> leaf.typeLength;
                    default -> throw new IllegalArgumentException("BYTE_STREAM_SPLIT on type " + leaf.type);
                };
                byte[] joined = new byte[count * width];
                for (int i = 0; i < count; i++) {
                    for (int b = 0; b < width; b++) {
                        joined[i * width + b] = data[from + b * count + i];
                    }
                }
                new Values(joined, 0, joined.length, leaf).plain(out, at, count);
            }
            default -> throw new IllegalArgumentException("Parquet encoding " + encoding + " is not supported");
        }
    }

    /** PLAIN values of a leaf's physical type. */
    private static final class Values {
        private final ByteBuffer in;
        private final Node leaf;

        Values(byte[] data, int from, int to, Node leaf) {
            in = ByteBuffer.wrap(data, from, to - from).order(ByteOrder.LITTLE_ENDIAN);
            this.leaf = leaf;
        }

        void plain(Object[] out, int at, int count) {
            switch (leaf.type) {
                case 0 -> {
                    int base = in.position();
                    for (int i = 0; i < count; i++) {
                        out[at + i] = (in.get(base + i / 8) >> (i % 8) & 1) != 0;
                    }
                }
                case 1 -> {
                    for (int i = 0; i < count; i++) {
                        out[at + i] = (long) in.getInt();
                    }
                }
                case 2 -> {
                    for (int i = 0; i < count; i++) {
                        out[at + i] = in.getLong();
                    }
                }
                case 3 -> fixed(out, at, count, 12);
                case 4 -> {
                    for (int i = 0; i < count; i++) {
                        out[at + i] = (double) in.getFloat();
                    }
                }
                case 5 -> {
                    for (int i = 0; i < count; i++) {
                        out[at + i] = in.getDouble();
                    }
                }
                case 6 -> {
                    for (int i = 0; i < count; i++) {
                        byte[] b = new byte[in.getInt()];
                        in.get(b);
                        out[at + i] = b;
                    }
                }
                case 7 -> fixed(out, at, count, leaf.typeLength);
                default -> throw new IllegalArgumentException("Parquet type " + leaf.type + " is not supported");
            }
        }

        private void fixed(Object[] out, int at, int count, int width) {
            for (int i = 0; i < count; i++) {
                byte[] b = new byte[width];
                in.get(b);
                out[at + i] = b;
            }
        }
    }

    /** The RLE / bit-packed hybrid encoding: {@code count} values into {@code out} from {@code at}. */
    private static void hybrid(byte[] data, int from, int to, int width, int[] out, int at, int count) {
        int p = from;
        int n = 0;
        int bytes = (width + 7) / 8;
        while (n < count) {
            if (p >= to) {
                throw new IllegalArgumentException("Parquet levels end early");
            }
            long header = 0;
            int shift = 0;
            int b;
            do {
                b = data[p++] & 0xFF;
                header |= (long) (b & 0x7F) << shift;
                shift += 7;
            } while ((b & 0x80) != 0);
            if ((header & 1) == 0) {
                int run = (int) (header >>> 1);
                int v = 0;
                for (int i = 0; i < bytes; i++) {
                    v |= (data[p++] & 0xFF) << 8 * i;
                }
                for (int i = 0; i < run && n < count; i++) {
                    out[at + n++] = v;
                }
            } else {
                int values = (int) (header >>> 1) * 8;
                long bitPos = (long) p * 8;
                for (int i = 0; i < values; i++) {
                    int v = 0;
                    for (int k = 0; k < width; k++) {
                        long bit = bitPos + (long) i * width + k;
                        v |= (data[(int) (bit >>> 3)] >> (bit & 7) & 1) << k;
                    }
                    if (n < count) {
                        out[at + n++] = v;
                    }
                }
                p += values / 8 * width;
            }
        }
    }

    /** DELTA_BINARY_PACKED; returns where the encoded values end. */
    private static int deltas(byte[] data, int from, long[] out, int count) {
        int[] p = {from};
        int block = (int) varint(data, p);
        int miniblocks = (int) varint(data, p);
        int total = (int) varint(data, p);
        long value = zigzag(varint(data, p));
        int perMini = block / miniblocks;
        int n = 0;
        if (total > 0 && n < count) {
            out[n++] = value;
        }
        int done = 1;
        while (done < total) {
            long minDelta = zigzag(varint(data, p));
            int[] widths = new int[miniblocks];
            for (int i = 0; i < miniblocks; i++) {
                widths[i] = data[p[0]++] & 0xFF;
            }
            for (int m = 0; m < miniblocks && done < total; m++) {
                long bitPos = (long) p[0] * 8;
                for (int i = 0; i < perMini; i++) {
                    long d = 0;
                    for (int k = 0; k < widths[m]; k++) {
                        long bit = bitPos + (long) i * widths[m] + k;
                        d |= (long) (data[(int) (bit >>> 3)] >> (bit & 7) & 1) << k;
                    }
                    if (done < total) {
                        value += minDelta + d;
                        if (n < count) {
                            out[n++] = value;
                        }
                        done++;
                    }
                }
                p[0] += perMini * widths[m] / 8;
            }
        }
        return p[0];
    }

    private static long varint(byte[] data, int[] p) {
        long v = 0;
        int shift = 0;
        int b;
        do {
            b = data[p[0]++] & 0xFF;
            v |= (long) (b & 0x7F) << shift;
            shift += 7;
        } while ((b & 0x80) != 0);
        return v;
    }

    private static long zigzag(long v) {
        return v >>> 1 ^ -(v & 1);
    }

    private static int le32(byte[] b, int at) {
        return b[at] & 0xFF | (b[at + 1] & 0xFF) << 8 | (b[at + 2] & 0xFF) << 16 | (b[at + 3] & 0xFF) << 24;
    }

    // ------------------------------------------------------------------ compression

    private static byte[] decompress(int codec, byte[] in, int size) {
        return switch (codec) {
            case 0 -> in;
            case 1 -> Snappy.decompress(in, size);
            case 2 -> {
                try (GZIPInputStream z = new GZIPInputStream(new ByteArrayInputStream(in))) {
                    yield z.readNBytes(size);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
            case 6 -> Zstd.decompress(in, size);
            case 7 -> lz4(in, size);
            default -> throw new IllegalArgumentException("Parquet compression codec " + codec + " is not supported");
        };
    }

    /** An LZ4 block (LZ4_RAW). */
    static byte[] lz4(byte[] in, int size) {
        byte[] out = new byte[size];
        int p = 0;
        int o = 0;
        while (p < in.length) {
            int token = in[p++] & 0xFF;
            int literals = token >>> 4;
            if (literals == 15) {
                int b;
                do {
                    b = in[p++] & 0xFF;
                    literals += b;
                } while (b == 255);
            }
            System.arraycopy(in, p, out, o, literals);
            p += literals;
            o += literals;
            if (p >= in.length) {
                break;
            }
            int offset = in[p] & 0xFF | (in[p + 1] & 0xFF) << 8;
            p += 2;
            int match = token & 15;
            if (match == 15) {
                int b;
                do {
                    b = in[p++] & 0xFF;
                    match += b;
                } while (b == 255);
            }
            match += 4;
            for (int i = 0; i < match; i++) {
                out[o] = out[o - offset];
                o++;
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ thrift

    /** Thrift's compact protocol, read into maps from field id to value. */
    private static final class Thrift {
        private final MemorySegment in;
        long at;

        Thrift(MemorySegment in, long at) {
            this.in = in;
            this.at = at;
        }

        private int next() {
            if (at >= in.byteSize()) {
                throw new IllegalArgumentException("broken Parquet metadata (ends early)");
            }
            return in.get(ValueLayout.JAVA_BYTE, at++) & 0xFF;
        }

        Map<Integer, Object> struct() {
            Map<Integer, Object> out = new HashMap<>();
            int id = 0;
            while (true) {
                int b = next();
                if (b == 0) {
                    return out;
                }
                int type = b & 0x0F;
                int delta = b >>> 4;
                id = delta == 0 ? (int) zigzag(varint()) : id + delta;
                out.put(id, value(type));
            }
        }

        private Object value(int type) {
            return switch (type) {
                case 1 -> Boolean.TRUE;
                case 2 -> Boolean.FALSE;
                case 3 -> (long) (byte) next();
                case 4, 5, 6 -> zigzag(varint());
                case 7 -> {
                    long bits = 0;
                    for (int i = 0; i < 8; i++) {
                        bits |= (long) next() << 8 * i;
                    }
                    yield Double.longBitsToDouble(bits);
                }
                case 8 -> {
                    int n = (int) varint();
                    if (n < 0 || at + n > in.byteSize()) {
                        throw new IllegalArgumentException("broken Parquet metadata (ends early)");
                    }
                    byte[] b = in.asSlice(at, n).toArray(ValueLayout.JAVA_BYTE);
                    at += n;
                    yield b;
                }
                case 9, 10 -> {
                    int h = next();
                    int n = h >>> 4;
                    if (n == 15) {
                        n = (int) varint();
                    }
                    int elem = h & 0x0F;
                    List<Object> out = new ArrayList<>(n);
                    for (int i = 0; i < n; i++) {
                        out.add(elem == 1 || elem == 2 ? next() == 1 : value(elem));
                    }
                    yield out;
                }
                case 11 -> {
                    int n = (int) varint();
                    Map<Object, Object> out = new HashMap<>();
                    if (n > 0) {
                        int kv = next();
                        for (int i = 0; i < n; i++) {
                            out.put(value(kv >>> 4), value(kv & 0x0F));
                        }
                    }
                    yield out;
                }
                case 12 -> struct();
                default -> throw new IllegalArgumentException("broken Parquet metadata (thrift type " + type + ")");
            };
        }

        private long varint() {
            long v = 0;
            int shift = 0;
            int b;
            do {
                b = next();
                v |= (long) (b & 0x7F) << shift;
                shift += 7;
            } while ((b & 0x80) != 0);
            return v;
        }
    }

    // ------------------------------------------------------------------ helpers

    private byte[] bytes(long at, int n) {
        if (at < 0 || at + n > file.byteSize()) {
            throw new IllegalArgumentException(path + ": a page reaches past the end of the file");
        }
        return file.asSlice(at, n).toArray(ValueLayout.JAVA_BYTE);
    }

    private static long num(Object o) {
        return o == null ? 0 : ((Number) o).longValue();
    }

    private static List<?> list(Object o) {
        return o instanceof List<?> l ? l : List.of();
    }

    @SuppressWarnings("unchecked")
    private static Map<Integer, Object> struct(Object o) {
        return o instanceof Map<?, ?> m ? (Map<Integer, Object>) m : Map.of();
    }
}
