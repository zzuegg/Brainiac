package io.github.zzuegg.brainiac.data;

import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * The hashing trick: maps named features (words, n-grams, categorical values) into a fixed
 * number of columns without a vocabulary, as scikit-learn's {@code FeatureHasher}. Each name
 * is hashed with 32-bit MurmurHash3 (seed 0) of its UTF-8 bytes; the column is
 * {@code |hash| mod columns} and, with alternating signs, the value is negated for negative
 * hashes so that collisions cancel in expectation.
 */
public final class FeatureHasher {

    private final int columns;
    private final boolean alternateSign;

    /**
     * @param columns the number of output columns
     * @param alternateSign whether to negate the values of names with a negative hash
     */
    public FeatureHasher(int columns, boolean alternateSign) {
        if (columns < 1) {
            throw new IllegalArgumentException("columns must be at least 1, got " + columns);
        }
        this.columns = columns;
        this.alternateSign = alternateSign;
    }

    /** @return the number of output columns */
    public int columns() {
        return columns;
    }

    /**
     * @param name a feature name
     * @return its column
     */
    public int column(String name) {
        int h = murmur3(name.getBytes(StandardCharsets.UTF_8), 0);
        // |Integer.MIN_VALUE| as a long, as scikit-learn's abs on the signed hash
        return (int) (Math.abs((long) h) % columns);
    }

    /**
     * @param name a feature name
     * @return the factor its value is multiplied by: 1, or -1 for a negative hash with
     *     alternating signs
     */
    public float sign(String name) {
        return alternateSign && murmur3(name.getBytes(StandardCharsets.UTF_8), 0) < 0 ? -1f : 1f;
    }

    /**
     * Hashes named values into one row; values of names that share a column add up.
     *
     * @param features feature names and their values
     * @return the row, {@link #columns()} wide
     */
    public float[] transform(Map<String, ? extends Number> features) {
        float[] row = new float[columns];
        for (Map.Entry<String, ? extends Number> e : features.entrySet()) {
            row[column(e.getKey())] += sign(e.getKey()) * e.getValue().floatValue();
        }
        return row;
    }

    /**
     * Hashes tokens into one row, each occurrence counting 1.
     *
     * @param tokens the tokens, repeats allowed
     * @return the row, {@link #columns()} wide
     */
    public float[] transform(Iterable<String> tokens) {
        float[] row = new float[columns];
        for (String t : tokens) {
            row[column(t)] += sign(t);
        }
        return row;
    }

    /**
     * 32-bit MurmurHash3 (x86 variant) of bytes.
     *
     * @param data the bytes
     * @param seed the seed
     * @return the hash, as a signed int
     */
    public static int murmur3(byte[] data, int seed) {
        final int c1 = 0xcc9e2d51;
        final int c2 = 0x1b873593;
        int h = seed;
        int blocks = data.length / 4;
        for (int i = 0; i < blocks; i++) {
            int k = data[4 * i] & 0xff | (data[4 * i + 1] & 0xff) << 8 | (data[4 * i + 2] & 0xff) << 16
                    | (data[4 * i + 3] & 0xff) << 24;
            k *= c1;
            k = Integer.rotateLeft(k, 15);
            k *= c2;
            h ^= k;
            h = Integer.rotateLeft(h, 13);
            h = h * 5 + 0xe6546b64;
        }
        int tail = blocks * 4;
        int rest = data.length & 3;
        if (rest > 0) {
            int k = 0;
            for (int b = rest - 1; b >= 0; b--) {
                k = k << 8 | data[tail + b] & 0xff;
            }
            k *= c1;
            k = Integer.rotateLeft(k, 15);
            k *= c2;
            h ^= k;
        }
        h ^= data.length;
        h ^= h >>> 16;
        h *= 0x85ebca6b;
        h ^= h >>> 13;
        h *= 0xc2b2ae35;
        h ^= h >>> 16;
        return h;
    }
}
