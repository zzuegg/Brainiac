package io.github.zzuegg.brainiac.experimental.nn;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * An exact nearest-neighbour index over float vectors, each with an attached payload vector:
 * the shared primitive behind kNN memory attention, kNN-LM, episodic control and archival
 * memory. Search is a brute-force scan (exact, deterministic, and fast enough for the tens of
 * thousands of rows these experiments use); the index grows by appending and can be pruned,
 * saved and loaded, so a memory persists across sessions.
 *
 * <p>Three similarity measures: the dot product (as attention scores keys), the cosine and
 * the negative squared Euclidean distance (as kNN-LM and episodic control use).
 *
 * <pre>{@code
 * KnnIndex index = KnnIndex.create(64, 16, KnnIndex.Metric.DOT);
 * index.add(key, value);
 * KnnIndex.Hit[] hits = index.search(query, 8);
 * }</pre>
 *
 * <p>Not thread safe.
 */
public final class KnnIndex {

    /** How rows are compared with a query; larger is closer for every metric. */
    public enum Metric {
        /** The dot product. */
        DOT,
        /** The cosine of the angle. */
        COSINE,
        /** The negative squared Euclidean distance. */
        L2
    }

    /**
     * One search result.
     *
     * @param row the row's index in the order rows were added (after pruning, the new order)
     * @param score the similarity, larger is closer
     */
    public record Hit(int row, float score) {
    }

    private final int keySize;
    private final int valueSize;
    private final Metric metric;
    private float[] keys;
    private float[] values;
    private float[] norms;
    private long[] stamps;
    private int size;
    private long clock;

    private KnnIndex(int keySize, int valueSize, Metric metric) {
        if (keySize < 1 || valueSize < 0) {
            throw new IllegalArgumentException("bad sizes " + keySize + ", " + valueSize);
        }
        this.keySize = keySize;
        this.valueSize = valueSize;
        this.metric = metric;
        keys = new float[16 * keySize];
        values = new float[16 * Math.max(1, valueSize)];
        norms = new float[16];
        stamps = new long[16];
    }

    /**
     * Creates an empty index.
     *
     * @param keySize the length of the searched vectors
     * @param valueSize the length of the payload stored with each, 0 for none
     * @param metric how keys are compared
     * @return the index
     */
    public static KnnIndex create(int keySize, int valueSize, Metric metric) {
        return new KnnIndex(keySize, valueSize, metric);
    }

    /** @return the number of rows */
    public int size() {
        return size;
    }

    /** @return the key length */
    public int keySize() {
        return keySize;
    }

    /** @return the payload length */
    public int valueSize() {
        return valueSize;
    }

    /** @return the metric */
    public Metric metric() {
        return metric;
    }

    /**
     * Appends a row.
     *
     * @param key the key, {@link #keySize()} long
     * @param value the payload, {@link #valueSize()} long
     * @return the new row's index
     */
    public int add(float[] key, float[] value) {
        if (key.length != keySize || (value == null ? 0 : value.length) != valueSize) {
            throw new IllegalArgumentException("expected key " + keySize + " and value " + valueSize);
        }
        if (size == norms.length) {
            int cap = norms.length * 2;
            keys = Arrays.copyOf(keys, cap * keySize);
            values = Arrays.copyOf(values, cap * Math.max(1, valueSize));
            norms = Arrays.copyOf(norms, cap);
            stamps = Arrays.copyOf(stamps, cap);
        }
        System.arraycopy(key, 0, keys, size * keySize, keySize);
        if (valueSize > 0) {
            System.arraycopy(value, 0, values, size * valueSize, valueSize);
        }
        norms[size] = (float) Math.sqrt(dot(key, 0, key, 0, keySize));
        stamps[size] = clock++;
        return size++;
    }

    /**
     * Copies a row's key.
     *
     * @param row the row
     * @return the key
     */
    public float[] key(int row) {
        check(row);
        return Arrays.copyOfRange(keys, row * keySize, (row + 1) * keySize);
    }

    /**
     * Copies a row's payload.
     *
     * @param row the row
     * @return the payload
     */
    public float[] value(int row) {
        check(row);
        return Arrays.copyOfRange(values, row * valueSize, (row + 1) * valueSize);
    }

    /**
     * Overwrites a row's payload, for memories whose stored outcome is refined (episodic control).
     *
     * @param row the row
     * @param value the new payload
     */
    public void setValue(int row, float[] value) {
        check(row);
        System.arraycopy(value, 0, values, row * valueSize, valueSize);
    }

    /**
     * Marks a row as used now, for least-recently-used pruning.
     *
     * @param row the row
     */
    public void touch(int row) {
        check(row);
        stamps[row] = clock++;
    }

    /**
     * Scores one row against a query.
     *
     * @param query the query
     * @param row the row
     * @return the similarity
     */
    public float score(float[] query, int row) {
        check(row);
        return score(query, (float) Math.sqrt(dot(query, 0, query, 0, keySize)), row);
    }

    private float score(float[] q, float qNorm, int row) {
        int off = row * keySize;
        return switch (metric) {
            case DOT -> (float) dot(q, 0, keys, off, keySize);
            case COSINE -> (float) (dot(q, 0, keys, off, keySize) / (qNorm * norms[row] + 1e-12));
            case L2 -> {
                double s = 0;
                for (int i = 0; i < keySize; i++) {
                    double d = q[i] - keys[off + i];
                    s += d * d;
                }
                yield (float) -s;
            }
        };
    }

    /**
     * Finds the {@code k} closest rows, closest first.
     *
     * @param query the query, {@link #keySize()} long
     * @param k how many to return at most
     * @return the hits, at most {@code min(k, size())}
     */
    public Hit[] search(float[] query, int k) {
        if (query.length != keySize) {
            throw new IllegalArgumentException("query has " + query.length + " values, keys " + keySize);
        }
        int n = Math.min(k, size);
        if (n <= 0) {
            return new Hit[0];
        }
        float qNorm = (float) Math.sqrt(dot(query, 0, query, 0, keySize));
        // a min-heap of the best n
        int[] heapRow = new int[n];
        float[] heapScore = new float[n];
        int filled = 0;
        for (int r = 0; r < size; r++) {
            float s = score(query, qNorm, r);
            if (filled < n) {
                heapRow[filled] = r;
                heapScore[filled] = s;
                siftUp(heapRow, heapScore, filled++);
            } else if (s > heapScore[0]) {
                heapRow[0] = r;
                heapScore[0] = s;
                siftDown(heapRow, heapScore, n);
            }
        }
        Hit[] hits = new Hit[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new Hit(heapRow[i], heapScore[i]);
        }
        Arrays.sort(hits, (a, b) -> a.score != b.score ? Float.compare(b.score, a.score) : Integer.compare(a.row, b.row));
        return hits;
    }

    private static void siftUp(int[] rows, float[] scores, int i) {
        while (i > 0) {
            int p = (i - 1) / 2;
            if (scores[p] <= scores[i]) {
                return;
            }
            swap(rows, scores, i, p);
            i = p;
        }
    }

    private static void siftDown(int[] rows, float[] scores, int n) {
        int i = 0;
        while (true) {
            int l = 2 * i + 1;
            int r = l + 1;
            int m = i;
            if (l < n && scores[l] < scores[m]) {
                m = l;
            }
            if (r < n && scores[r] < scores[m]) {
                m = r;
            }
            if (m == i) {
                return;
            }
            swap(rows, scores, i, m);
            i = m;
        }
    }

    private static void swap(int[] rows, float[] scores, int a, int b) {
        int t = rows[a];
        rows[a] = rows[b];
        rows[b] = t;
        float s = scores[a];
        scores[a] = scores[b];
        scores[b] = s;
    }

    /**
     * Keeps only some rows, in their current order.
     *
     * @param keep for each row, whether it stays
     * @return how many rows were removed
     */
    public int retain(boolean[] keep) {
        if (keep.length != size) {
            throw new IllegalArgumentException("need one flag per row (" + size + ")");
        }
        int n = 0;
        for (int r = 0; r < size; r++) {
            if (keep[r]) {
                if (n != r) {
                    System.arraycopy(keys, r * keySize, keys, n * keySize, keySize);
                    if (valueSize > 0) {
                        System.arraycopy(values, r * valueSize, values, n * valueSize, valueSize);
                    }
                    norms[n] = norms[r];
                    stamps[n] = stamps[r];
                }
                n++;
            }
        }
        int removed = size - n;
        size = n;
        return removed;
    }

    /**
     * Removes the least recently added or {@linkplain #touch touched} rows until at most
     * {@code capacity} remain.
     *
     * @param capacity the rows to keep
     * @return how many rows were removed
     */
    public int pruneLeastRecentlyUsed(int capacity) {
        if (size <= capacity) {
            return 0;
        }
        long[] sorted = Arrays.copyOf(stamps, size);
        Arrays.sort(sorted);
        long cut = sorted[size - capacity];
        boolean[] keep = new boolean[size];
        for (int r = 0; r < size; r++) {
            keep[r] = stamps[r] >= cut;
        }
        return retain(keep);
    }

    /** Removes every row. */
    public void clear() {
        size = 0;
    }

    /**
     * Writes the index.
     *
     * @param out the stream; left open
     * @throws IOException if writing fails
     */
    public void write(OutputStream out) throws IOException {
        DataOutputStream d = new DataOutputStream(out);
        d.writeInt(0x564958_31);   // "VIX1"
        d.writeInt(keySize);
        d.writeInt(valueSize);
        d.writeInt(metric.ordinal());
        d.writeInt(size);
        for (int i = 0; i < size * keySize; i++) {
            d.writeFloat(keys[i]);
        }
        for (int i = 0; i < size * valueSize; i++) {
            d.writeFloat(values[i]);
        }
        d.flush();
    }

    /**
     * Reads an index that {@link #write} wrote.
     *
     * @param in the stream; left open
     * @return the index
     * @throws IOException if reading fails or the data is not an index
     */
    public static KnnIndex read(InputStream in) throws IOException {
        DataInputStream d = new DataInputStream(in);
        if (d.readInt() != 0x564958_31) {
            throw new IOException("not a vector index");
        }
        int ks = d.readInt();
        int vs = d.readInt();
        Metric m = Metric.values()[d.readInt()];
        int n = d.readInt();
        KnnIndex index = new KnnIndex(ks, vs, m);
        float[] k = new float[ks];
        float[] v = new float[vs];
        float[][] allK = new float[n][];
        for (int r = 0; r < n; r++) {
            for (int i = 0; i < ks; i++) {
                k[i] = d.readFloat();
            }
            allK[r] = k.clone();
        }
        for (int r = 0; r < n; r++) {
            for (int i = 0; i < vs; i++) {
                v[i] = d.readFloat();
            }
            index.add(allK[r], v);
        }
        return index;
    }

    /**
     * Saves the index to a file.
     *
     * @param path the file
     * @throws IOException if writing fails
     */
    public void save(Path path) throws IOException {
        try (OutputStream out = new java.io.BufferedOutputStream(Files.newOutputStream(path))) {
            write(out);
        }
    }

    /**
     * Loads an index from a file.
     *
     * @param path the file
     * @return the index
     * @throws IOException if reading fails
     */
    public static KnnIndex load(Path path) throws IOException {
        try (InputStream in = new java.io.BufferedInputStream(Files.newInputStream(path))) {
            return read(in);
        }
    }

    private void check(int row) {
        if (row < 0 || row >= size) {
            throw new IndexOutOfBoundsException("row " + row + " of " + size);
        }
    }

    static double dot(float[] a, int ao, float[] b, int bo, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += (double) a[ao + i] * b[bo + i];
        }
        return s;
    }

    @Override
    public String toString() {
        return "KnnIndex(" + size + " rows, key " + keySize + ", value " + valueSize + ", " + metric + ")";
    }
}
