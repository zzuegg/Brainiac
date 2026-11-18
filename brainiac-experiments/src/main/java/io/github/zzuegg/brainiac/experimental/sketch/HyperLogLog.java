package io.github.zzuegg.brainiac.experimental.sketch;

/**
 * HyperLogLog distinct counting: {@code m = 2^p} registers, each holding the longest run of
 * leading zeros (plus one) among the hashes routed to it; the estimate is the bias-corrected
 * harmonic mean {@code alpha_m m^2 / sum 2^-M[j]}, with linear counting when many registers
 * are still empty. The relative standard error is about {@code 1.04 / sqrt(m)}.
 *
 * <p>Paper: Flajolet, Fusy, Gandouet and Meunier, "HyperLogLog: the analysis of a
 * near-optimal cardinality estimation algorithm" (AofA 2007).
 */
public final class HyperLogLog {

    private final int p;
    private final byte[] registers;
    private final long seed;

    /**
     * Creates an empty sketch.
     *
     * @param p register bits (4-18): {@code 2^p} registers
     * @param seed the hash seed
     */
    public HyperLogLog(int p, long seed) {
        if (p < 4 || p > 18) {
            throw new IllegalArgumentException("p must be 4-18, got " + p);
        }
        this.p = p;
        this.registers = new byte[1 << p];
        this.seed = seed;
    }

    /**
     * Adds an item.
     *
     * @param item the item
     */
    public void add(long item) {
        long h = FrequencySketches.hash(item, seed);
        int j = (int) (h >>> (64 - p));
        long rest = h << p;
        int rank = rest == 0 ? 64 - p + 1 : Long.numberOfLeadingZeros(rest) + 1;
        if (rank > registers[j]) {
            registers[j] = (byte) rank;
        }
    }

    /** @return the estimated number of distinct items added */
    public double estimate() {
        int m = registers.length;
        double alpha = m == 16 ? 0.673 : m == 32 ? 0.697 : m == 64 ? 0.709 : 0.7213 / (1 + 1.079 / m);
        double sum = 0;
        int zeros = 0;
        for (byte r : registers) {
            sum += Math.pow(2, -r);
            zeros += r == 0 ? 1 : 0;
        }
        double e = alpha * m * (double) m / sum;
        if (e <= 2.5 * m && zeros > 0) {
            return m * Math.log(m / (double) zeros);
        }
        return e;
    }

    /** @return the register count */
    public int registers() {
        return registers.length;
    }

    /** @return the theoretical relative standard error {@code 1.04 / sqrt(m)} */
    public double standardError() {
        return 1.04 / Math.sqrt(registers.length);
    }
}
