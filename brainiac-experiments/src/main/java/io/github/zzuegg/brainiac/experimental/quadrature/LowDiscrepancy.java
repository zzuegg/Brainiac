package io.github.zzuegg.brainiac.experimental.quadrature;

import io.github.zzuegg.brainiac.tensor.Special;

import java.util.SplittableRandom;

/**
 * Low-discrepancy point sets on {@code [0, 1)^d} (Dick, Kuo and Sloan, Acta Numerica 2013):
 * <ul>
 *   <li>{@link #halton}: radical inverses in the first {@code d} primes, optionally scrambled by
 *       independent random digit permutations per base and digit position (Owen-type
 *       permutation scrambling);</li>
 *   <li>{@link #sobol}: base-2 digital sequence with Joe and Kuo's direction numbers (the
 *       {@code new-joe-kuo-6.21201} primitive polynomials and initial values, up to 32
 *       dimensions), Gray-code ordered, with {@link Scramble#LINEAR} (Matousek's random
 *       lower-triangular linear scrambling plus a digital shift) or {@link Scramble#OWEN}
 *       (Owen's nested uniform scrambling, each bit flipped by a hashed random tree over the
 *       higher bits);</li>
 *   <li>{@link #lattice}: a rank-1 lattice {@code {i z / n + shift}} with a Korobov generating
 *       vector {@code z = (1, a, a^2, ...) mod n} chosen to minimise the worst-case error
 *       {@code P_2} in a weighted Korobov space (weights {@code 1/j^2}), and a uniform random
 *       shift (Cranley-Patterson).</li>
 * </ul>
 * {@link #toGaussian} maps points to standard normals by the inverse CDF.
 */
public final class LowDiscrepancy {

    private LowDiscrepancy() {
    }

    /** Sobol randomisation. */
    public enum Scramble {
        /** The raw (deterministic) sequence. */
        NONE,
        /** Random linear matrix scrambling plus a digital shift. */
        LINEAR,
        /** Owen's nested uniform scrambling. */
        OWEN
    }

    private static final int[] PRIMES = primes(64);

    private static int[] primes(int count) {
        int[] p = new int[count];
        int found = 0;
        for (int c = 2; found < count; c++) {
            boolean prime = true;
            for (int i = 0; i < found && p[i] * p[i] <= c; i++) {
                if (c % p[i] == 0) {
                    prime = false;
                    break;
                }
            }
            if (prime) {
                p[found++] = c;
            }
        }
        return p;
    }

    /**
     * Halton points.
     *
     * @param n the number of points
     * @param d the dimension (at most 64)
     * @param scrambled whether to apply random digit permutations
     * @param seed the scrambling seed
     * @return points {@code [n][d]}
     */
    public static double[][] halton(int n, int d, boolean scrambled, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int digits = 40;
        double[][] out = new double[n][d];
        for (int j = 0; j < d; j++) {
            int b = PRIMES[j];
            int[][] perm = new int[digits][];
            for (int k = 0; k < digits; k++) {
                perm[k] = new int[b];
                for (int v = 0; v < b; v++) {
                    perm[k][v] = v;
                }
                if (scrambled) {
                    for (int v = b - 1; v > 0; v--) {
                        int w = r.nextInt(v + 1);
                        int t = perm[k][v];
                        perm[k][v] = perm[k][w];
                        perm[k][w] = t;
                    }
                }
            }
            int depth = (int) Math.ceil(52 / (Math.log(b) / Math.log(2)));
            for (int i = 0; i < n; i++) {
                long idx = i + (scrambled ? 0 : 1);
                double f = 1.0 / b;
                double x = 0;
                for (int k = 0; k < Math.min(depth, digits); k++) {
                    int digit = (int) (idx % b);
                    idx /= b;
                    x += perm[k][digit] * f;
                    f /= b;
                    if (idx == 0 && !scrambled) {
                        break;
                    }
                }
                out[i][j] = x;
            }
        }
        return out;
    }

    // Joe-Kuo new-joe-kuo-6.21201, dimensions 2..32: {degree s, coefficients a, m_1..m_s}
    private static final int[][] JOE_KUO = {
        {1, 0, 1}, {2, 1, 1, 3}, {3, 1, 1, 3, 1}, {3, 2, 1, 1, 1}, {4, 1, 1, 1, 3, 3}, {4, 4, 1, 3, 5, 13},
        {5, 2, 1, 1, 5, 5, 17}, {5, 4, 1, 1, 5, 5, 5}, {5, 7, 1, 1, 7, 11, 19}, {5, 11, 1, 1, 5, 1, 1},
        {5, 13, 1, 1, 1, 3, 11}, {5, 14, 1, 3, 5, 5, 31}, {6, 1, 1, 3, 3, 9, 7, 49}, {6, 13, 1, 1, 1, 15, 21, 21},
        {6, 16, 1, 3, 1, 13, 27, 49}, {6, 19, 1, 1, 1, 15, 7, 5}, {6, 22, 1, 3, 1, 15, 13, 25},
        {6, 25, 1, 1, 5, 5, 19, 61}, {7, 1, 1, 3, 7, 11, 23, 15, 103}, {7, 4, 1, 3, 7, 13, 13, 15, 69},
        {7, 7, 1, 1, 3, 13, 7, 35, 63}, {7, 8, 1, 3, 5, 9, 1, 25, 53}, {7, 14, 1, 3, 1, 13, 9, 35, 107},
        {7, 19, 1, 3, 1, 5, 27, 61, 31}, {7, 21, 1, 1, 5, 11, 19, 41, 61}, {7, 28, 1, 3, 5, 3, 3, 13, 69},
        {7, 31, 1, 1, 7, 13, 1, 19, 1}, {7, 32, 1, 3, 7, 5, 13, 19, 59}, {7, 37, 1, 1, 3, 9, 25, 29, 41},
        {7, 41, 1, 3, 5, 13, 23, 1, 55}, {7, 42, 1, 3, 7, 3, 13, 59, 17}};

    private static final int BITS = 32;

    /** The largest Sobol dimension. */
    public static final int SOBOL_MAX_DIMENSION = JOE_KUO.length + 1;

    private static long[][] directions(int d) {
        long[][] v = new long[d][BITS];
        for (int i = 0; i < BITS; i++) {
            v[0][i] = 1L << (BITS - 1 - i);
        }
        for (int j = 1; j < d; j++) {
            int[] row = JOE_KUO[j - 1];
            int s = row[0];
            int a = row[1];
            for (int i = 0; i < Math.min(s, BITS); i++) {
                v[j][i] = (long) row[2 + i] << (BITS - 1 - i);
            }
            for (int i = s; i < BITS; i++) {
                long x = v[j][i - s] ^ (v[j][i - s] >>> s);
                for (int k = 1; k < s; k++) {
                    if (((a >>> (s - 1 - k)) & 1) != 0) {
                        x ^= v[j][i - k];
                    }
                }
                v[j][i] = x;
            }
        }
        return v;
    }

    private static long mix(long z) {
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }

    /**
     * Sobol points (the first {@code n} of the sequence, starting at the origin).
     *
     * @param n the number of points
     * @param d the dimension, at most {@link #SOBOL_MAX_DIMENSION}
     * @param scramble the randomisation
     * @param seed the scrambling seed
     * @return points {@code [n][d]}
     */
    public static double[][] sobol(int n, int d, Scramble scramble, long seed) {
        if (d > SOBOL_MAX_DIMENSION) {
            throw new IllegalArgumentException("Sobol supports up to " + SOBOL_MAX_DIMENSION + " dimensions");
        }
        long[][] v = directions(d);
        SplittableRandom r = new SplittableRandom(seed);
        long[][] lms = new long[d][BITS];
        long[] shift = new long[d];
        long[] owenSeed = new long[d];
        for (int j = 0; j < d; j++) {
            // lower-triangular with unit diagonal: output bit k (from the top) mixes input bits 0..k
            for (int k = 0; k < BITS; k++) {
                long rowBits = r.nextLong() & ((1L << k) - 1);
                lms[j][k] = ((rowBits << (BITS - k)) | (1L << (BITS - 1 - k))) & 0xFFFFFFFFL;
            }
            shift[j] = r.nextLong() & 0xFFFFFFFFL;
            owenSeed[j] = r.nextLong();
        }
        double[][] out = new double[n][d];
        long[] x = new long[d];
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                int c = Long.numberOfTrailingZeros(i);
                for (int j = 0; j < d; j++) {
                    x[j] ^= v[j][c];
                }
            }
            for (int j = 0; j < d; j++) {
                long y = x[j];
                if (scramble == Scramble.LINEAR) {
                    long z = 0;
                    for (int k = 0; k < BITS; k++) {
                        // output bit k = parity(row_k AND y) over the top k+1 bits
                        long bit = Long.bitCount(lms[j][k] & y) & 1L;
                        z |= bit << (BITS - 1 - k);
                    }
                    y = z ^ shift[j];
                } else if (scramble == Scramble.OWEN) {
                    long z = y;
                    for (int k = 0; k < BITS; k++) {
                        long prefix = k == 0 ? 0 : y >>> (BITS - k);
                        long flip = mix(owenSeed[j] ^ mix(((long) k << 40) ^ prefix)) & 1L;
                        z ^= flip << (BITS - 1 - k);
                    }
                    y = z;
                }
                double u = (y + 0.5) / 4294967296.0;
                out[i][j] = scramble == Scramble.NONE ? y / 4294967296.0 : u;
            }
        }
        return out;
    }

    /**
     * A Korobov generating vector for {@code n} points in {@code d} dimensions, by exhaustive
     * search over the multiplier.
     *
     * @param n the number of points
     * @param d the dimension
     * @return {@code z}
     */
    public static long[] korobov(int n, int d) {
        long bestA = 1;
        double best = Double.POSITIVE_INFINITY;
        int step = Math.max(1, n / 512);
        for (long a = 1; a < n; a += step) {
            if (gcd(a, n) != 1) {
                continue;
            }
            long[] z = powers(a, n, d);
            double e = p2(z, n);
            if (e < best) {
                best = e;
                bestA = a;
            }
        }
        return powers(bestA, n, d);
    }

    private static long gcd(long a, long b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    private static long[] powers(long a, int n, int d) {
        long[] z = new long[d];
        z[0] = 1;
        for (int j = 1; j < d; j++) {
            z[j] = z[j - 1] * a % n;
        }
        return z;
    }

    /** Squared worst-case error in the weighted Korobov space with alpha = 2, weights 1/j^2. */
    private static double p2(long[] z, int n) {
        double s = 0;
        for (int i = 0; i < n; i++) {
            double prod = 1;
            for (int j = 0; j < z.length; j++) {
                double x = (double) (i * z[j] % n) / n;
                double b2 = x * x - x + 1.0 / 6;
                prod *= 1 + 2 * Math.PI * Math.PI * b2 / ((j + 1.0) * (j + 1.0));
            }
            s += prod;
        }
        return s / n - 1;
    }

    /**
     * A randomly shifted rank-1 lattice.
     *
     * @param n the number of points
     * @param z the generating vector (see {@link #korobov})
     * @param seed the shift seed (a negative seed gives no shift)
     * @return points {@code [n][d]}
     */
    public static double[][] lattice(int n, long[] z, long seed) {
        int d = z.length;
        double[] shift = new double[d];
        if (seed >= 0) {
            SplittableRandom r = new SplittableRandom(seed);
            for (int j = 0; j < d; j++) {
                shift[j] = r.nextDouble();
            }
        }
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double x = (double) ((long) i * z[j] % n) / n + shift[j];
                out[i][j] = x - Math.floor(x);
            }
        }
        return out;
    }

    /**
     * Maps unit-cube points to standard normals by the inverse CDF (clamped away from 0 and 1).
     *
     * @param u points {@code [n][d]}
     * @return normals {@code [n][d]}
     */
    public static double[][] toGaussian(double[][] u) {
        double[][] out = new double[u.length][u[0].length];
        for (int i = 0; i < u.length; i++) {
            for (int j = 0; j < u[0].length; j++) {
                out[i][j] = Special.normalQuantile(Math.min(1 - 1e-12, Math.max(1e-12, u[i][j])));
            }
        }
        return out;
    }
}
