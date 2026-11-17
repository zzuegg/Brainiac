package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Near-duplicate detection with MinHash and locality-sensitive hashing (Broder 1997; as used
 * to deduplicate GPT-3's, The Pile's and FineWeb's training data, Lee et al. 2021,
 * <a href="https://arxiv.org/abs/2107.06499">arXiv:2107.06499</a>). Each document becomes
 * its set of character {@code k}-shingles; {@code numHashes} random hash functions keep the
 * minimum hash of the set, and the share of equal minima estimates the Jaccard similarity.
 * The signature is cut into {@code bands} of {@code rows}; documents that agree on a whole
 * band become candidate pairs, so pairs above about {@code (1/bands)^(1/rows)} similarity
 * are found without comparing every pair, and candidates are confirmed by their estimate.
 */
public final class MinHashDeduplicator {

    private static final long PRIME = (1L << 61) - 1;

    private final int shingle;
    private final int bands;
    private final int rows;
    private final long[] a;
    private final long[] b;

    /**
     * @param shingle the characters per shingle, for example 5
     * @param bands the LSH bands, for example 16
     * @param rows the hashes per band, for example 8
     * @param seed the seed of the hash functions
     */
    public MinHashDeduplicator(int shingle, int bands, int rows, long seed) {
        this.shingle = shingle;
        this.bands = bands;
        this.rows = rows;
        SplittableRandom r = new SplittableRandom(seed);
        int n = bands * rows;
        a = new long[n];
        b = new long[n];
        for (int i = 0; i < n; i++) {
            a[i] = 1 + r.nextLong(PRIME - 1);
            b[i] = r.nextLong(PRIME);
        }
    }

    /**
     * @param text a document
     * @return its MinHash signature
     */
    public long[] signature(String text) {
        long[] sig = new long[a.length];
        java.util.Arrays.fill(sig, Long.MAX_VALUE);
        String t = text.length() < shingle ? text + " ".repeat(shingle - text.length()) : text;
        for (int i = 0; i + shingle <= t.length(); i++) {
            long h = mix(t.substring(i, i + shingle).hashCode() * 0x9E3779B97F4A7C15L) & PRIME;
            for (int k = 0; k < sig.length; k++) {
                long v = mulMod(a[k], h) + b[k];
                v = v >= PRIME ? v - PRIME : v;
                if (v < sig[k]) {
                    sig[k] = v;
                }
            }
        }
        return sig;
    }

    /**
     * @param x a signature
     * @param y another
     * @return the estimated Jaccard similarity of their shingle sets
     */
    public static double similarity(long[] x, long[] y) {
        int same = 0;
        for (int i = 0; i < x.length; i++) {
            same += x[i] == y[i] ? 1 : 0;
        }
        return (double) same / x.length;
    }

    /**
     * @param x a text
     * @param y another
     * @param k the shingle size
     * @return the exact Jaccard similarity of their {@code k}-shingle sets
     */
    public static double jaccard(String x, String y, int k) {
        Set<String> sx = shingles(x, k);
        Set<String> sy = shingles(y, k);
        Set<String> inter = new HashSet<>(sx);
        inter.retainAll(sy);
        Set<String> union = new HashSet<>(sx);
        union.addAll(sy);
        return union.isEmpty() ? 1 : (double) inter.size() / union.size();
    }

    private static Set<String> shingles(String t, int k) {
        Set<String> s = new HashSet<>();
        for (int i = 0; i + k <= t.length(); i++) {
            s.add(t.substring(i, i + k));
        }
        return s;
    }

    /**
     * Finds near-duplicates.
     *
     * @param documents the documents
     * @param threshold the estimated similarity above which two are duplicates
     * @return for each document the index of the first document of its cluster (itself if
     *     it is kept)
     */
    public int[] clusters(List<String> documents, double threshold) {
        int n = documents.size();
        List<long[]> sigs = new ArrayList<>();
        for (String d : documents) {
            sigs.add(signature(d));
        }
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        for (int band = 0; band < bands; band++) {
            Map<List<Long>, List<Integer>> buckets = new HashMap<>();
            for (int i = 0; i < n; i++) {
                List<Long> key = new ArrayList<>(rows);
                for (int r = 0; r < rows; r++) {
                    key.add(sigs.get(i)[band * rows + r]);
                }
                buckets.computeIfAbsent(key, x -> new ArrayList<>()).add(i);
            }
            for (List<Integer> bucket : buckets.values()) {
                for (int x = 0; x < bucket.size(); x++) {
                    for (int y = x + 1; y < bucket.size(); y++) {
                        int i = bucket.get(x);
                        int j = bucket.get(y);
                        if (find(parent, i) != find(parent, j) && similarity(sigs.get(i), sigs.get(j)) >= threshold) {
                            union(parent, i, j);
                        }
                    }
                }
            }
        }
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = find(parent, i);
        }
        return out;
    }

    /**
     * @param documents the documents
     * @param threshold the duplicate threshold
     * @return the documents with near-duplicates removed (the first of each cluster kept)
     */
    public List<String> deduplicate(List<String> documents, double threshold) {
        int[] c = clusters(documents, threshold);
        List<String> out = new ArrayList<>();
        for (int i = 0; i < c.length; i++) {
            if (c[i] == i) {
                out.add(documents.get(i));
            }
        }
        return out;
    }

    private static int find(int[] p, int i) {
        while (p[i] != i) {
            p[i] = p[p[i]];
            i = p[i];
        }
        return i;
    }

    private static void union(int[] p, int i, int j) {
        int a = find(p, i);
        int b = find(p, j);
        if (a < b) {
            p[b] = a;
        } else {
            p[a] = b;
        }
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private static long mulMod(long x, long y) {
        long hi = Math.multiplyHigh(x, y);
        long lo = x * y;
        // (hi * 2^64 + lo) mod (2^61 - 1)
        long r = (lo & PRIME) + (lo >>> 61) + (hi << 3);
        r = (r & PRIME) + (r >>> 61);
        return r >= PRIME ? r - PRIME : r;
    }
}
