package io.github.zzuegg.brainiac.experimental.codec;

import java.util.Arrays;

/**
 * A probability distribution quantised to integer frequencies that sum to {@code 2^bits}, every
 * symbol at least 1 so that anything can be coded. Both {@link ArithmeticCoder} and {@link Rans}
 * code with such tables: a symbol costs {@code bits - log2(freq)} bits, so the quantisation loss
 * against the real distribution is the only overhead besides the coder's few bits of flush.
 */
public final class FrequencyTable {

    private final int[] freq;
    private final int[] cum;
    private final int bits;

    private FrequencyTable(int[] freq, int bits) {
        this.freq = freq;
        this.bits = bits;
        this.cum = new int[freq.length + 1];
        for (int i = 0; i < freq.length; i++) {
            cum[i + 1] = cum[i] + freq[i];
        }
    }

    /**
     * Quantises probabilities.
     *
     * @param p the probabilities (need not be normalised; non-finite or negative values count as 0)
     * @param bits the total {@code 2^bits}, at most 24, and {@code 2^bits >= p.length}
     * @return the table
     */
    public static FrequencyTable of(double[] p, int bits) {
        int n = p.length;
        int total = 1 << bits;
        if (n > total) {
            throw new IllegalArgumentException("alphabet larger than the total");
        }
        double sum = 0;
        for (double v : p) {
            sum += v > 0 && Double.isFinite(v) ? v : 0;
        }
        int[] f = new int[n];
        int spare = total - n;
        int used = 0;
        for (int i = 0; i < n; i++) {
            // a uniform table only when nothing is positive; otherwise non-positive entries get
            // just the minimum frequency 1
            double v = sum > 0 ? (p[i] > 0 && Double.isFinite(p[i]) ? p[i] / sum : 0) : 1.0 / n;
            f[i] = 1 + (int) Math.floor(v * spare);
            used += f[i];
        }
        // hand the rounding remainder to the largest entries
        int left = total - used;
        if (left > 0) {
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Integer.compare(f[b], f[a]));
            for (int k = 0; left > 0; k = (k + 1) % n) {
                f[order[k]]++;
                left--;
            }
        }
        return new FrequencyTable(f, bits);
    }

    /**
     * A table from counts already summing to {@code 2^bits}.
     *
     * @param freq positive frequencies
     * @param bits the precision
     * @return the table
     */
    public static FrequencyTable ofFrequencies(int[] freq, int bits) {
        long s = 0;
        for (int f : freq) {
            if (f <= 0) {
                throw new IllegalArgumentException("frequencies must be positive");
            }
            s += f;
        }
        if (s != 1L << bits) {
            throw new IllegalArgumentException("frequencies must sum to 2^bits");
        }
        return new FrequencyTable(freq.clone(), bits);
    }

    /** @return the number of symbols */
    public int size() {
        return freq.length;
    }

    /** @return the precision: frequencies sum to {@code 2^bits} */
    public int bits() {
        return bits;
    }

    /** @return {@code 2^bits} */
    public int total() {
        return 1 << bits;
    }

    /**
     * @param symbol a symbol
     * @return its frequency
     */
    public int frequency(int symbol) {
        return freq[symbol];
    }

    /**
     * @param symbol a symbol
     * @return the summed frequency of the symbols before it
     */
    public int start(int symbol) {
        return cum[symbol];
    }

    /**
     * The symbol whose interval holds a cumulative value.
     *
     * @param value in {@code [0, total)}
     * @return the symbol
     */
    public int symbol(int value) {
        int lo = 0;
        int hi = freq.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (cum[mid] <= value) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /**
     * @param symbol a symbol
     * @return its ideal code length under the table, {@code bits - log2(freq)}
     */
    public double cost(int symbol) {
        return bits - Math.log(freq[symbol]) / Math.log(2);
    }
}
