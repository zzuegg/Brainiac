package io.github.zzuegg.brainiac.experimental.sketch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Classical page-cache policies, the references for caching with predictions: LRU, LFU, FIFO,
 * random eviction, Belady's offline optimum (evict the page requested furthest in the
 * future) and the heavy-hitter retention of the core {@code KvCache} ({@code h2o:N}: always
 * keep the {@code N} most recent entries and, among the rest, the ones accessed most so far,
 * the page analogue of H2O's accumulated attention).
 *
 * <p>A {@link PageCache} answers one request at a time; {@link #simulate} replays a trace.
 * Belady needs the whole trace up front, so it is built with {@link #belady}.
 *
 * <p>References: Belady, "A study of replacement algorithms for a virtual-storage computer"
 * (IBM Systems Journal 1966); Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient
 * Generative Inference of Large Language Models" (NeurIPS 2023),
 * <a href="https://arxiv.org/abs/2306.14048">arXiv:2306.14048</a>.
 */
public final class ClassicCaches {

    private ClassicCaches() {
    }

    /** A cache of pages that answers requests in order. */
    public interface PageCache {
        /**
         * Requests a page at time {@code t} (consecutive from 0), loading it on a miss and
         * evicting if full.
         *
         * @param page the page
         * @param t the request index
         * @return whether it was a hit
         */
        boolean request(int page, int t);

        /** @return the pages held */
        int size();
    }

    /**
     * A replay's outcome.
     *
     * @param hits hits
     * @param misses misses
     */
    public record Outcome(long hits, long misses) {
        /** @return hits over requests */
        public double hitRate() {
            return hits / (double) Math.max(1, hits + misses);
        }
    }

    /**
     * Replays a trace.
     *
     * @param trace page ids
     * @param cache the cache
     * @return the outcome
     */
    public static Outcome simulate(int[] trace, PageCache cache) {
        long hits = 0;
        for (int t = 0; t < trace.length; t++) {
            hits += cache.request(trace[t], t) ? 1 : 0;
        }
        return new Outcome(hits, trace.length - hits);
    }

    /**
     * For each request, the index of the next request of the same page
     * ({@code Integer.MAX_VALUE} if none).
     *
     * @param trace page ids
     * @return next-use indices
     */
    public static int[] nextUse(int[] trace) {
        int[] next = new int[trace.length];
        Map<Integer, Integer> last = new HashMap<>();
        for (int t = trace.length - 1; t >= 0; t--) {
            next[t] = last.getOrDefault(trace[t], Integer.MAX_VALUE);
            last.put(trace[t], t);
        }
        return next;
    }

    /**
     * Least recently used.
     *
     * @param capacity pages held
     * @return the cache
     */
    public static PageCache lru(int capacity) {
        return new PageCache() {
            private final LinkedHashMap<Integer, Boolean> map = new LinkedHashMap<>(16, 0.75f, true);

            @Override
            public boolean request(int page, int t) {
                if (map.get(page) != null) {
                    return true;
                }
                if (map.size() >= capacity) {
                    Integer eldest = map.keySet().iterator().next();
                    map.remove(eldest);
                }
                map.put(page, Boolean.TRUE);
                return false;
            }

            @Override
            public int size() {
                return map.size();
            }
        };
    }

    /**
     * First in, first out.
     *
     * @param capacity pages held
     * @return the cache
     */
    public static PageCache fifo(int capacity) {
        return new PageCache() {
            private final LinkedHashMap<Integer, Boolean> map = new LinkedHashMap<>();

            @Override
            public boolean request(int page, int t) {
                if (map.containsKey(page)) {
                    return true;
                }
                if (map.size() >= capacity) {
                    map.remove(map.keySet().iterator().next());
                }
                map.put(page, Boolean.TRUE);
                return false;
            }

            @Override
            public int size() {
                return map.size();
            }
        };
    }

    /**
     * Random eviction.
     *
     * @param capacity pages held
     * @param seed the seed
     * @return the cache
     */
    public static PageCache random(int capacity, long seed) {
        return new ScoredCache(capacity) {
            private final SplittableRandom r = new SplittableRandom(seed);

            @Override
            int victim(int t) {
                return pages.get(r.nextInt(pages.size()));
            }
        };
    }

    /**
     * Least frequently used, counting accesses while cached; ties go to the least recent.
     *
     * @param capacity pages held
     * @return the cache
     */
    public static PageCache lfu(int capacity) {
        return new ScoredCache(capacity) {
            private final Map<Integer, Integer> count = new HashMap<>();

            @Override
            void touched(int page, int t, boolean hit) {
                count.merge(page, 1, Integer::sum);
            }

            @Override
            void evicted(int page) {
                count.remove(page);
            }

            @Override
            int victim(int t) {
                int best = pages.get(0);
                for (int p : pages) {
                    int c = count.get(p);
                    int cb = count.get(best);
                    if (c < cb || (c == cb && lastUse.get(p) < lastUse.get(best))) {
                        best = p;
                    }
                }
                return best;
            }
        };
    }

    /**
     * The heavy-hitter retention of the core {@code KvCache} ({@code h2o:recent}) on pages:
     * the {@code recent} most recently used pages are always kept; the victim is the other
     * page with the fewest accesses since the start of the trace.
     *
     * @param capacity pages held
     * @param recent the most recent pages always kept
     * @return the cache
     */
    public static PageCache heavyHitters(int capacity, int recent) {
        return new ScoredCache(capacity) {
            private final Map<Integer, Integer> count = new HashMap<>();

            @Override
            void touched(int page, int t, boolean hit) {
                count.merge(page, 1, Integer::sum);
            }

            @Override
            int victim(int t) {
                List<Integer> byRecency = new ArrayList<>(pages);
                byRecency.sort((a, b) -> Integer.compare(lastUse.get(b), lastUse.get(a)));
                int best = -1;
                for (int k = Math.min(recent, byRecency.size() - 1); k < byRecency.size(); k++) {
                    int p = byRecency.get(k);
                    if (best < 0 || count.get(p) < count.get(best)) {
                        best = p;
                    }
                }
                return best;
            }
        };
    }

    /**
     * Belady's offline optimum for a known trace.
     *
     * @param capacity pages held
     * @param trace the whole trace that will be replayed
     * @return the cache
     */
    public static PageCache belady(int capacity, int[] trace) {
        int[] next = nextUse(trace);
        return new ScoredCache(capacity) {
            private final Map<Integer, Integer> nextOf = new HashMap<>();

            @Override
            void touched(int page, int t, boolean hit) {
                nextOf.put(page, next[t]);
            }

            @Override
            int victim(int t) {
                int best = pages.get(0);
                for (int p : pages) {
                    if (nextOf.get(p) > nextOf.get(best)) {
                        best = p;
                    }
                }
                return best;
            }
        };
    }

    /**
     * A cache that keeps a list of its pages and the last use of each, for policies that
     * scan the pages to choose a victim.
     */
    abstract static class ScoredCache implements PageCache {
        final int capacity;
        final List<Integer> pages = new ArrayList<>();
        final Map<Integer, Integer> lastUse = new HashMap<>();

        ScoredCache(int capacity) {
            this.capacity = capacity;
        }

        abstract int victim(int t);

        void touched(int page, int t, boolean hit) {
        }

        void evicted(int page) {
        }

        @Override
        public boolean request(int page, int t) {
            boolean hit = lastUse.containsKey(page);
            if (!hit && pages.size() >= capacity) {
                int v = victim(t);
                pages.remove(Integer.valueOf(v));
                lastUse.remove(v);
                evicted(v);
            }
            if (!hit) {
                pages.add(page);
            }
            lastUse.put(page, t);
            touched(page, t, hit);
            return hit;
        }

        @Override
        public int size() {
            return pages.size();
        }
    }
}
