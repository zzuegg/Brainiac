package io.github.zzuegg.brainiac.experimental.compress;

import java.util.Arrays;
import java.util.PriorityQueue;

/**
 * A canonical Huffman code over the symbols {@code 0 .. alphabet-1}, the last stage of Deep
 * Compression (Han et al. 2016, <a href="https://arxiv.org/abs/1510.00149">arXiv:1510.00149</a>):
 * after pruning and weight sharing the cluster indices and the sparse index gaps are far from
 * uniform, so a variable-length code stores them in fewer bits than their fixed width. The code is
 * canonical, so storing it takes only one code length per symbol ({@link #tableBits}).
 */
public final class HuffmanCoder {

    private final int[] lengths;
    private final long[] codes;

    private HuffmanCoder(int[] lengths) {
        this.lengths = lengths;
        this.codes = canonical(lengths);
    }

    /**
     * Builds the code for a symbol stream.
     *
     * @param symbols the stream
     * @param alphabet the number of possible symbols
     * @return the code
     */
    public static HuffmanCoder fit(int[] symbols, int alphabet) {
        long[] freq = new long[alphabet];
        for (int s : symbols) {
            freq[s]++;
        }
        return fromFrequencies(freq);
    }

    /**
     * @param freq the frequency of every symbol
     * @return the code
     */
    public static HuffmanCoder fromFrequencies(long[] freq) {
        int n = freq.length;
        int[] lengths = new int[n];
        int used = 0;
        for (long f : freq) {
            used += f > 0 ? 1 : 0;
        }
        if (used == 1) {
            for (int i = 0; i < n; i++) {
                lengths[i] = freq[i] > 0 ? 1 : 0;
            }
            return new HuffmanCoder(lengths);
        }
        // nodes: leaves 0..n-1, internal n..; parent links give the depths
        int[] parent = new int[2 * n];
        Arrays.fill(parent, -1);
        long[] weight = new long[2 * n];
        PriorityQueue<long[]> q = new PriorityQueue<>((a, b) -> a[0] != b[0] ? Long.compare(a[0], b[0])
                : Long.compare(a[1], b[1]));
        for (int i = 0; i < n; i++) {
            if (freq[i] > 0) {
                weight[i] = freq[i];
                q.add(new long[] {freq[i], i});
            }
        }
        int next = n;
        while (q.size() > 1) {
            long[] a = q.poll();
            long[] b = q.poll();
            weight[next] = a[0] + b[0];
            parent[(int) a[1]] = next;
            parent[(int) b[1]] = next;
            q.add(new long[] {weight[next], next});
            next++;
        }
        for (int i = 0; i < n; i++) {
            if (freq[i] > 0) {
                int d = 0;
                for (int p = parent[i]; p >= 0; p = parent[p]) {
                    d++;
                }
                lengths[i] = d;
            }
        }
        return new HuffmanCoder(lengths);
    }

    /**
     * @param symbol a symbol
     * @return its code length in bits, 0 if it never occurs
     */
    public int length(int symbol) {
        return lengths[symbol];
    }

    /**
     * @param symbols a stream
     * @return its encoded size in bits
     */
    public long encodedBits(int[] symbols) {
        long b = 0;
        for (int s : symbols) {
            if (lengths[s] == 0) {
                throw new IllegalArgumentException("symbol " + s + " has no code");
            }
            b += lengths[s];
        }
        return b;
    }

    /** @return the bits to store the canonical code: one 5-bit length per symbol */
    public long tableBits() {
        return 5L * lengths.length;
    }

    /**
     * @param symbols a stream
     * @return the packed bits, most significant first
     */
    public long[] encode(int[] symbols) {
        long[] out = new long[(int) ((encodedBits(symbols) + 63) / 64) + 1];
        long pos = 0;
        for (int s : symbols) {
            long c = codes[s];
            for (int b = lengths[s] - 1; b >= 0; b--) {
                if (((c >>> b) & 1) != 0) {
                    out[(int) (pos >>> 6)] |= 1L << (63 - (pos & 63));
                }
                pos++;
            }
        }
        return out;
    }

    /**
     * @param bits packed bits from {@link #encode}
     * @param count how many symbols to read
     * @return the symbols
     */
    public int[] decode(long[] bits, int count) {
        int[] out = new int[count];
        long pos = 0;
        for (int i = 0; i < count; i++) {
            long c = 0;
            int len = 0;
            int found = -1;
            while (found < 0) {
                int bit = (int) ((bits[(int) (pos >>> 6)] >>> (63 - (pos & 63))) & 1);
                pos++;
                c = (c << 1) | bit;
                len++;
                for (int s = 0; s < lengths.length; s++) {
                    if (lengths[s] == len && codes[s] == c) {
                        found = s;
                        break;
                    }
                }
                if (len > 63) {
                    throw new IllegalStateException("corrupt stream");
                }
            }
            out[i] = found;
        }
        return out;
    }

    private static long[] canonical(int[] lengths) {
        int n = lengths.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> lengths[a] != lengths[b] ? Integer.compare(lengths[a], lengths[b])
                : Integer.compare(a, b));
        long[] codes = new long[n];
        long code = 0;
        int prev = 0;
        boolean first = true;
        for (int i : order) {
            if (lengths[i] == 0) {
                continue;
            }
            if (first) {
                code = 0;
                first = false;
            } else {
                code = (code + 1) << (lengths[i] - prev);
            }
            prev = lengths[i];
            codes[i] = code;
        }
        return codes;
    }
}
