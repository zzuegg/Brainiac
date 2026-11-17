package io.github.zzuegg.brainiac.experimental.tensor;

import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A binary hypervector for hyperdimensional computing / vector-symbolic architectures
 * (Kanerva; survey by Kleyko et al., <a href="https://arxiv.org/abs/2111.06077">2111.06077</a>):
 * thousands of random bits packed into {@code long}s, so every operation is a few
 * XORs and popcounts per 64 dimensions.
 *
 * <ul>
 * <li>{@link #bind} (XOR) makes a vector unlike both inputs, and is its own inverse:
 *     {@code a.bind(b).bind(b) == a}. Use it for key-value pairs.</li>
 * <li>{@link #bundle} (bitwise majority) makes a vector like every input: a set or a
 *     prototype. Ties are broken by a fixed random vector.</li>
 * <li>{@link #permute} (cyclic rotation) marks position or order, and is invertible.</li>
 * <li>{@link #similarity} is {@code 1 - 2 * hamming / dims}: about 0 for unrelated random
 *     vectors (they are nearly orthogonal in high dimensions), 1 for equal ones.</li>
 * </ul>
 *
 * <p>Instances are immutable.
 */
public final class Hypervector {
    private final int dims;
    private final long[] bits;

    private Hypervector(int dims, long[] bits) {
        this.dims = dims;
        this.bits = bits;
    }

    /**
     * A uniformly random vector.
     *
     * @param dims the number of bits, a multiple of 64
     * @param random the random source
     * @return the vector
     */
    public static Hypervector random(int dims, RandomGenerator random) {
        check(dims);
        long[] b = new long[dims / 64];
        for (int i = 0; i < b.length; i++) {
            b[i] = random.nextLong();
        }
        return new Hypervector(dims, b);
    }

    /**
     * The all-zero vector, the identity of {@link #bind}.
     *
     * @param dims the number of bits, a multiple of 64
     * @return the vector
     */
    public static Hypervector zeros(int dims) {
        check(dims);
        return new Hypervector(dims, new long[dims / 64]);
    }

    /**
     * Wraps raw bits.
     *
     * @param dims the number of bits
     * @param bits {@code dims / 64} longs, copied
     * @return the vector
     */
    public static Hypervector of(int dims, long[] bits) {
        check(dims);
        if (bits.length != dims / 64) {
            throw new IllegalArgumentException("need " + dims / 64 + " longs");
        }
        return new Hypervector(dims, bits.clone());
    }

    private static void check(int dims) {
        if (dims < 64 || dims % 64 != 0) {
            throw new IllegalArgumentException("dims must be a positive multiple of 64, got " + dims);
        }
    }

    /** @return the number of bits */
    public int dims() {
        return dims;
    }

    /** @return a copy of the packed bits */
    public long[] bits() {
        return bits.clone();
    }

    /**
     * Reads one bit.
     *
     * @param i the index
     * @return the bit
     */
    public boolean get(int i) {
        return (bits[i >>> 6] >>> (i & 63) & 1L) != 0;
    }

    /**
     * Binds with XOR.
     *
     * @param other another vector of the same size
     * @return the bound vector
     */
    public Hypervector bind(Hypervector other) {
        same(other);
        long[] b = new long[bits.length];
        for (int i = 0; i < b.length; i++) {
            b[i] = bits[i] ^ other.bits[i];
        }
        return new Hypervector(dims, b);
    }

    /**
     * Rotates cyclically by {@code shift} positions; negative shifts undo positive ones.
     *
     * @param shift the rotation
     * @return the rotated vector
     */
    public Hypervector permute(int shift) {
        int s = Math.floorMod(shift, dims);
        if (s == 0) {
            return this;
        }
        long[] b = new long[bits.length];
        for (int i = 0; i < dims; i++) {
            if (get(i)) {
                int j = (i + s) % dims;
                b[j >>> 6] |= 1L << (j & 63);
            }
        }
        return new Hypervector(dims, b);
    }

    /**
     * Counts differing bits.
     *
     * @param other another vector
     * @return the Hamming distance
     */
    public int hamming(Hypervector other) {
        same(other);
        int d = 0;
        for (int i = 0; i < bits.length; i++) {
            d += Long.bitCount(bits[i] ^ other.bits[i]);
        }
        return d;
    }

    /**
     * Normalized similarity: 1 equal, about 0 unrelated, -1 complementary.
     *
     * @param other another vector
     * @return {@code 1 - 2 hamming / dims}
     */
    public double similarity(Hypervector other) {
        return 1 - 2.0 * hamming(other) / dims;
    }

    /**
     * Bitwise majority of several vectors. With an even count, ties go to the bit of a
     * fixed pseudo-random tie-breaker vector.
     *
     * @param vectors the vectors, at least one
     * @return the bundle
     */
    public static Hypervector bundle(List<Hypervector> vectors) {
        Accumulator acc = new Accumulator(vectors.get(0).dims);
        for (Hypervector v : vectors) {
            acc.add(v, 1);
        }
        return acc.toVector();
    }

    private void same(Hypervector other) {
        if (other.dims != dims) {
            throw new IllegalArgumentException("dimension mismatch " + dims + " vs " + other.dims);
        }
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Hypervector h && h.dims == dims && Arrays.equals(h.bits, bits);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(bits);
    }

    /**
     * Integer counters per dimension for incremental, weighted bundling: add many vectors,
     * then threshold once. This is how prototypes are learned in one pass.
     */
    public static final class Accumulator {
        private final int dims;
        private final int[] counts;
        private int total;

        /**
         * Creates empty counters.
         *
         * @param dims the vector size
         */
        public Accumulator(int dims) {
            check(dims);
            this.dims = dims;
            this.counts = new int[dims];
        }

        /**
         * Adds a vector with a weight (negative to subtract, as in retraining).
         *
         * @param v the vector
         * @param weight its weight
         */
        public void add(Hypervector v, int weight) {
            for (int w = 0; w < v.bits.length; w++) {
                long word = v.bits[w];
                for (int k = 0; k < 64; k++) {
                    counts[w * 64 + k] += ((word >>> k & 1L) != 0) ? weight : -weight;
                }
            }
            total += Math.abs(weight);
        }

        /** @return the number of (weighted) vectors added */
        public int total() {
            return total;
        }

        /**
         * Thresholds the counters at zero; ties take the bit of a fixed tie-breaker.
         *
         * @return the majority vector
         */
        public Hypervector toVector() {
            long[] b = new long[dims / 64];
            long tie = 0x9E3779B97F4A7C15L;
            for (int i = 0; i < dims; i++) {
                boolean bit;
                if (counts[i] == 0) {
                    tie = tie * 6364136223846793005L + 1442695040888963407L;
                    bit = tie < 0;
                } else {
                    bit = counts[i] > 0;
                }
                if (bit) {
                    b[i >>> 6] |= 1L << (i & 63);
                }
            }
            return new Hypervector(dims, b);
        }
    }
}
