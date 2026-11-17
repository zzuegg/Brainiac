package io.github.zzuegg.brainiac.experimental.data;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * A small byte-level language model: interpolated n-gram counts with additive smoothing,
 * trained in one pass over a corpus. It gives the next-byte distribution and its entropy for
 * any context, which is what {@link EntropyPatcher} needs to decide where patches begin; it
 * is cheap enough to run ahead of a large model on every byte.
 *
 * <p>The distribution mixes the orders {@code 0..order} with weights that favour longer
 * contexts that have been seen more often (Witten-Bell style: weight {@code c / (c + u)}
 * for a context seen {@code c} times with {@code u} distinct followers).
 */
public final class ByteNgramModel {

    private final int order;
    private final Map<Long, int[]> counts = new HashMap<>();

    private ByteNgramModel(int order) {
        this.order = order;
    }

    /**
     * Trains a model on text, as UTF-8 bytes.
     *
     * @param text the corpus
     * @param order the longest context, for example 3
     * @return the model
     */
    public static ByteNgramModel train(String text, int order) {
        return train(text.getBytes(StandardCharsets.UTF_8), order);
    }

    /**
     * Trains a model on bytes.
     *
     * @param bytes the corpus
     * @param order the longest context, at most 7
     * @return the model
     */
    public static ByteNgramModel train(byte[] bytes, int order) {
        if (order < 0 || order > 7) {
            throw new IllegalArgumentException("order must be in [0, 7], got " + order);
        }
        ByteNgramModel m = new ByteNgramModel(order);
        for (int i = 0; i < bytes.length; i++) {
            for (int k = 0; k <= Math.min(order, i); k++) {
                m.counts.computeIfAbsent(key(bytes, i - k, k), x -> new int[257])[bytes[i] & 0xff]++;
            }
        }
        for (int[] c : m.counts.values()) {
            for (int i = 0; i < 256; i++) {
                c[256] += c[i];
            }
        }
        return m;
    }

    /** Packs a context of length {@code k} into a key that includes its length. */
    private static long key(byte[] b, int from, int k) {
        long key = k;
        for (int i = 0; i < k; i++) {
            key = key * 257 + (b[from + i] & 0xff) + 1;
        }
        return key;
    }

    /** @return the longest context length */
    public int order() {
        return order;
    }

    /**
     * Returns the next-byte distribution after a position.
     *
     * @param bytes the text
     * @param position the index of the byte to predict; the context is the bytes before it
     * @return 256 probabilities
     */
    public double[] distribution(byte[] bytes, int position) {
        double[] p = new double[256];
        java.util.Arrays.fill(p, 1.0 / 256);
        for (int k = 0; k <= Math.min(order, position); k++) {
            int[] c = counts.get(key(bytes, position - k, k));
            if (c == null || c[256] == 0) {
                break;
            }
            int distinct = 0;
            for (int i = 0; i < 256; i++) {
                distinct += c[i] > 0 ? 1 : 0;
            }
            double lambda = c[256] / (double) (c[256] + distinct);
            for (int i = 0; i < 256; i++) {
                p[i] = lambda * c[i] / c[256] + (1 - lambda) * p[i];
            }
        }
        return p;
    }

    /**
     * Returns the entropy of the next-byte distribution at every position.
     *
     * @param bytes the text
     * @return entropies in bits, one per byte
     */
    public double[] entropies(byte[] bytes) {
        double[] h = new double[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            double e = 0;
            for (double q : distribution(bytes, i)) {
                if (q > 0) {
                    e -= q * Math.log(q) / Math.log(2);
                }
            }
            h[i] = e;
        }
        return h;
    }
}
