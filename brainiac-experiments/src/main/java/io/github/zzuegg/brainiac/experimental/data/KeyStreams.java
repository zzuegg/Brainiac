package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Synthetic key streams for sketches, learned indexes and caches:
 *
 * <ul>
 *   <li>{@link #zipf}: i.i.d. items from a Zipf law {@code P(rank r) ~ r^-s} over a universe,
 *       with ranks mapped to item ids by a fixed random permutation (so ids carry no rank
 *       information unless a model learns it from features);</li>
 *   <li>{@link #driftingZipf}: the same with the rank-to-id map partly reshuffled at each of
 *       several phases;</li>
 *   <li>{@link #sortedKeys}: sorted distinct keys from a uniform, lognormal or piecewise
 *       (a mixture of dense and sparse segments) distribution;</li>
 *   <li>{@link #pageTrace}: page-request traces with Zipf popularity, loops longer than a
 *       cache, scans mixed into a hot set, and phase changes.</li>
 * </ul>
 */
public final class KeyStreams {

    private KeyStreams() {
    }

    /** Sorted-key distributions. */
    public enum KeyDistribution {
        /** uniform over a range */
        UNIFORM,
        /** lognormal (heavy right tail) */
        LOGNORMAL,
        /** dense and sparse segments of very different density */
        PIECEWISE
    }

    /** Page-trace patterns. */
    public enum TraceKind {
        /** Zipf popularity, i.i.d. */
        ZIPF,
        /** a loop over more pages than the cache, with a little noise */
        LOOP,
        /** a Zipf hot set interleaved with one-off scans of cold pages */
        SCAN,
        /** Zipf with the hot set moving at phase boundaries */
        PHASES
    }

    /**
     * The rank-to-id permutation of {@link #zipf} for a universe and seed.
     *
     * @param universe the item count
     * @param seed the seed
     * @return {@code ids[rank]}
     */
    public static int[] rankToId(int universe, long seed) {
        SplittableRandom r = new SplittableRandom(seed ^ 0x5DEECE66DL);
        int[] p = new int[universe];
        for (int i = 0; i < universe; i++) {
            p[i] = i;
        }
        for (int i = universe - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /**
     * The Zipf CDF over ranks {@code 0..universe-1}.
     *
     * @param universe the item count
     * @param s the exponent
     * @return the cumulative probabilities
     */
    public static double[] zipfCdf(int universe, double s) {
        double[] cdf = new double[universe];
        double z = 0;
        for (int r = 0; r < universe; r++) {
            z += Math.pow(r + 1, -s);
            cdf[r] = z;
        }
        for (int r = 0; r < universe; r++) {
            cdf[r] /= z;
        }
        return cdf;
    }

    private static int sample(double[] cdf, SplittableRandom r) {
        int i = Arrays.binarySearch(cdf, r.nextDouble());
        return Math.min(cdf.length - 1, i >= 0 ? i : -i - 1);
    }

    /**
     * An i.i.d. Zipf stream.
     *
     * @param length the stream length
     * @param universe the item count
     * @param s the exponent (0.8-1.5)
     * @param seed the seed (also fixes the rank-to-id map)
     * @return item ids
     */
    public static int[] zipf(int length, int universe, double s, long seed) {
        double[] cdf = zipfCdf(universe, s);
        int[] ids = rankToId(universe, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int[] out = new int[length];
        for (int i = 0; i < length; i++) {
            out[i] = ids[sample(cdf, r)];
        }
        return out;
    }

    /**
     * A drifting Zipf stream: at each phase boundary a share of the ranks swap ids.
     *
     * @param length the stream length
     * @param universe the item count
     * @param s the exponent
     * @param phases the number of phases
     * @param drift the share of ranks reshuffled per phase
     * @param seed the seed
     * @return item ids
     */
    public static int[] driftingZipf(int length, int universe, double s, int phases, double drift, long seed) {
        double[] cdf = zipfCdf(universe, s);
        int[] ids = rankToId(universe, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int[] out = new int[length];
        int per = Math.max(1, length / phases);
        for (int i = 0; i < length; i++) {
            if (i > 0 && i % per == 0) {
                int swaps = (int) (drift * universe / 2);
                for (int k = 0; k < swaps; k++) {
                    int a = r.nextInt(Math.max(1, universe / 10));
                    int b = r.nextInt(universe);
                    int t = ids[a];
                    ids[a] = ids[b];
                    ids[b] = t;
                }
            }
            out[i] = ids[sample(cdf, r)];
        }
        return out;
    }

    /**
     * Sorted distinct non-negative keys.
     *
     * @param n the key count
     * @param distribution the distribution
     * @param seed the seed
     * @return sorted distinct keys
     */
    public static long[] sortedKeys(int n, KeyDistribution distribution, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        long[] k = new long[n];
        int filled = 0;
        long scale = 1L << 40;
        int segments = 16;
        double[] segStart = new double[segments + 1];
        double[] segWeight = new double[segments];
        if (distribution == KeyDistribution.PIECEWISE) {
            for (int s = 1; s <= segments; s++) {
                segStart[s] = segStart[s - 1] + Math.exp(3 * r.nextDouble());
            }
            for (int s = 0; s < segments; s++) {
                segWeight[s] = Math.exp(4 * r.nextDouble());
            }
        }
        double total = 0;
        for (double w : segWeight) {
            total += w;
        }
        while (filled < n) {
            int need = n - filled;
            for (int i = 0; i < need; i++) {
                double u = switch (distribution) {
                    case UNIFORM -> r.nextDouble();
                    case LOGNORMAL -> Math.min(1, Math.exp(gaussian(r)) / 60);
                    case PIECEWISE -> {
                        double pick = r.nextDouble() * total;
                        int s = 0;
                        while (s < segments - 1 && pick > segWeight[s]) {
                            pick -= segWeight[s];
                            s++;
                        }
                        double lo = segStart[s] / segStart[segments];
                        double hi = segStart[s + 1] / segStart[segments];
                        yield lo + (hi - lo) * r.nextDouble();
                    }
                };
                k[filled++] = (long) (u * scale);
            }
            Arrays.sort(k, 0, filled);
            int w = 0;
            for (int i = 0; i < filled; i++) {
                if (w == 0 || k[i] != k[w - 1]) {
                    k[w++] = k[i];
                }
            }
            filled = w;
        }
        return k;
    }

    /**
     * A page-request trace.
     *
     * @param kind the pattern
     * @param length requests
     * @param pages the page universe
     * @param loopLength the loop length (for {@link TraceKind#LOOP}) or scan length (for
     *     {@link TraceKind#SCAN})
     * @param seed the seed
     * @return page ids
     */
    public static int[] pageTrace(TraceKind kind, int length, int pages, int loopLength, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] out = new int[length];
        double[] cdf = zipfCdf(pages, 1.0);
        int[] ids = rankToId(pages, seed);
        int scanNext = pages / 2;
        int i = 0;
        while (i < length) {
            switch (kind) {
                case ZIPF -> out[i++] = ids[sample(cdf, r)];
                case LOOP -> {
                    int p = i % loopLength;
                    out[i++] = r.nextDouble() < 0.05 ? r.nextInt(pages) : p;
                }
                case SCAN -> {
                    if (r.nextDouble() < 0.02) {
                        for (int s = 0; s < loopLength && i < length; s++) {
                            out[i++] = pages + (scanNext++ % (4 * pages));
                        }
                    } else {
                        out[i++] = ids[sample(cdf, r)];
                    }
                }
                case PHASES -> {
                    int phase = 4 * i / length;
                    out[i++] = (ids[sample(cdf, r)] + phase * pages / 4) % pages;
                }
            }
        }
        return out;
    }

    private static double gaussian(SplittableRandom r) {
        double u = Math.max(1e-12, r.nextDouble());
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble());
    }
}
