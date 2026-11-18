package io.github.zzuegg.brainiac.experimental.sketch;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Streaming frequency sketches over integer items, sized in counters ("words"):
 *
 * <ul>
 *   <li>{@link Kind#COUNT_MIN}: {@code d} rows of {@code w} counters; an item adds one to
 *       its hashed counter in every row and is estimated by the minimum (never an
 *       underestimate; overestimate at most {@code e N / w} with probability
 *       {@code 1 - e^-d}) (Cormode and Muthukrishnan, 2005);</li>
 *   <li>{@link Kind#CONSERVATIVE_UPDATE}: Count-Min that only raises the counters that equal
 *       the current minimum (Estan and Varghese, 2002);</li>
 *   <li>{@link Kind#COUNT_SKETCH}: signed counters and the median over rows, an unbiased
 *       estimate (Charikar, Chen and Farach-Colton, 2002);</li>
 *   <li>{@link Kind#MISRA_GRIES}: {@code k} candidate counters, all decremented when a new
 *       item finds none free (underestimate by at most {@code N / (k + 1)});</li>
 *   <li>{@link Kind#SPACE_SAVING}: {@code k} counters, a new item replaces the smallest and
 *       inherits its count (overestimate at most {@code N / k}) (Metwally, Agrawal and
 *       El Abbadi, 2005).</li>
 * </ul>
 *
 * <p>Counter-based summaries store an id and a count per entry, so at {@code words}
 * counters they keep {@code words / 2} entries.
 */
public final class FrequencySketches {

    private FrequencySketches() {
    }

    /** The sketches. */
    public enum Kind {
        /** Count-Min */
        COUNT_MIN,
        /** Count-Min with conservative update */
        CONSERVATIVE_UPDATE,
        /** Count-Sketch */
        COUNT_SKETCH,
        /** Misra-Gries */
        MISRA_GRIES,
        /** Space-Saving */
        SPACE_SAVING
    }

    /** A frequency summary of a stream. */
    public interface FrequencySketch {
        /**
         * Adds one occurrence.
         *
         * @param item the item
         */
        void add(int item);

        /**
         * @param item an item
         * @return its estimated count
         */
        long estimate(int item);

        /** @return memory in counters (an id or a count is one counter) */
        int words();
    }

    /**
     * Creates a sketch.
     *
     * @param kind the sketch
     * @param words its memory in counters
     * @param depth rows of the hashed sketches (ignored by the counter-based ones)
     * @param seed the hash seed
     * @return the sketch
     */
    public static FrequencySketch of(Kind kind, int words, int depth, long seed) {
        return switch (kind) {
            case COUNT_MIN -> new CountMin(Math.max(1, words / depth), depth, false, seed);
            case CONSERVATIVE_UPDATE -> new CountMin(Math.max(1, words / depth), depth, true, seed);
            case COUNT_SKETCH -> new CountSketch(Math.max(1, words / depth), depth, seed);
            case MISRA_GRIES -> new MisraGries(Math.max(1, words / 2));
            case SPACE_SAVING -> new SpaceSaving(Math.max(1, words / 2));
        };
    }

    /**
     * A seeded 64-bit mix of an item (SplitMix64's finaliser).
     *
     * @param item the item
     * @param seed the seed
     * @return the hash
     */
    static long hash(long item, long seed) {
        long z = item * 0x9E3779B97F4A7C15L + seed * 0xBF58476D1CE4E5B9L + 0x94D049BB133111EBL;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Count-Min, optionally with conservative update. */
    static final class CountMin implements FrequencySketch {
        private final long[][] table;
        private final long[] seeds;
        private final boolean conservative;

        CountMin(int width, int depth, boolean conservative, long seed) {
            this.table = new long[depth][width];
            this.seeds = new SplittableRandom(seed).longs(depth).toArray();
            this.conservative = conservative;
        }

        private int bucket(int row, int item) {
            return (int) Math.floorMod(hash(item, seeds[row]), (long) table[row].length);
        }

        @Override
        public void add(int item) {
            if (!conservative) {
                for (int r = 0; r < table.length; r++) {
                    table[r][bucket(r, item)]++;
                }
                return;
            }
            long min = estimate(item);
            for (int r = 0; r < table.length; r++) {
                int b = bucket(r, item);
                table[r][b] = Math.max(table[r][b], min + 1);
            }
        }

        @Override
        public long estimate(int item) {
            long min = Long.MAX_VALUE;
            for (int r = 0; r < table.length; r++) {
                min = Math.min(min, table[r][bucket(r, item)]);
            }
            return min;
        }

        @Override
        public int words() {
            return table.length * table[0].length;
        }
    }

    /** Count-Sketch. */
    static final class CountSketch implements FrequencySketch {
        private final long[][] table;
        private final long[] seeds;

        CountSketch(int width, int depth, long seed) {
            this.table = new long[depth][width];
            this.seeds = new SplittableRandom(seed).longs(depth).toArray();
        }

        @Override
        public void add(int item) {
            for (int r = 0; r < table.length; r++) {
                long h = hash(item, seeds[r]);
                table[r][(int) Math.floorMod(h >>> 1, (long) table[r].length)] += (h & 1) == 0 ? 1 : -1;
            }
        }

        @Override
        public long estimate(int item) {
            long[] v = new long[table.length];
            for (int r = 0; r < table.length; r++) {
                long h = hash(item, seeds[r]);
                v[r] = table[r][(int) Math.floorMod(h >>> 1, (long) table[r].length)] * ((h & 1) == 0 ? 1 : -1);
            }
            Arrays.sort(v);
            long med = v.length % 2 == 1 ? v[v.length / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2;
            return Math.max(0, med);
        }

        @Override
        public int words() {
            return table.length * table[0].length;
        }
    }

    /** Misra-Gries with {@code k} counters (decrements are a shared offset, so each is O(log k)). */
    public static final class MisraGries implements FrequencySketch {
        private final int k;
        private final Map<Integer, Long> stored = new HashMap<>();
        private final java.util.TreeMap<Long, java.util.Set<Integer>> byCount = new java.util.TreeMap<>();
        private long offset;

        /**
         * Creates the summary.
         *
         * @param k counters (entries)
         */
        public MisraGries(int k) {
            this.k = k;
        }

        private void put(int item, long value) {
            Long old = stored.put(item, value);
            if (old != null) {
                java.util.Set<Integer> s = byCount.get(old);
                s.remove(item);
                if (s.isEmpty()) {
                    byCount.remove(old);
                }
            }
            byCount.computeIfAbsent(value, x -> new java.util.HashSet<>()).add(item);
        }

        @Override
        public void add(int item) {
            Long c = stored.get(item);
            if (c != null) {
                put(item, c + 1);
            } else if (stored.size() < k) {
                put(item, offset + 1);
            } else {
                offset++;
                while (!byCount.isEmpty() && byCount.firstKey() <= offset) {
                    for (int victim : byCount.pollFirstEntry().getValue()) {
                        stored.remove(victim);
                    }
                }
            }
        }

        @Override
        public long estimate(int item) {
            Long c = stored.get(item);
            return c == null ? 0 : c - offset;
        }

        /** @return the items currently held */
        public java.util.Set<Integer> items() {
            return java.util.Collections.unmodifiableSet(stored.keySet());
        }

        @Override
        public int words() {
            return 2 * k;
        }
    }

    /** Space-Saving with {@code k} counters. */
    static final class SpaceSaving implements FrequencySketch {
        private final int k;
        private final Map<Integer, Long> counts = new HashMap<>();
        private final java.util.TreeMap<Long, java.util.Set<Integer>> byCount = new java.util.TreeMap<>();

        SpaceSaving(int k) {
            this.k = k;
        }

        private void move(int item, long from, long to) {
            if (from > 0) {
                java.util.Set<Integer> s = byCount.get(from);
                s.remove(item);
                if (s.isEmpty()) {
                    byCount.remove(from);
                }
            }
            byCount.computeIfAbsent(to, x -> new java.util.HashSet<>()).add(item);
            counts.put(item, to);
        }

        @Override
        public void add(int item) {
            Long c = counts.get(item);
            if (c != null) {
                move(item, c, c + 1);
            } else if (counts.size() < k) {
                move(item, 0, 1);
            } else {
                Map.Entry<Long, java.util.Set<Integer>> min = byCount.firstEntry();
                int victim = min.getValue().iterator().next();
                long m = min.getKey();
                min.getValue().remove(victim);
                if (min.getValue().isEmpty()) {
                    byCount.remove(m);
                }
                counts.remove(victim);
                move(item, 0, m + 1);
            }
        }

        @Override
        public long estimate(int item) {
            return counts.getOrDefault(item, 0L);
        }

        @Override
        public int words() {
            return 2 * k;
        }
    }
}
