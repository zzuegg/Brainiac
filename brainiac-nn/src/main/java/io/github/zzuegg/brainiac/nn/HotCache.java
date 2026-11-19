package io.github.zzuegg.brainiac.nn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.ToLongFunction;

/**
 * Copies of the most used of many items, within a byte budget: the experts of a mixture whose
 * weights live in host memory, a few of them kept on the GPU too. Callers report each use with
 * {@link #use}, which returns the copy when one is held. After {@code warmup} uses the cache
 * fills with the items used most, and every {@code every} uses after that it ranks them again
 * and swaps in what became hotter, at most {@code maxSwapBytes} per refresh so that a refresh
 * never stalls a stream for long. Counts halve at each refresh, so the ranking follows what the
 * stream does now; a held item counts a quarter more than its uses, so two items of about the
 * same heat do not trade places on every refresh.
 *
 * @param <K> the item's key, with value equality
 * @param <V> the copy
 */
public final class HotCache<K, V> {
    private static final double HELD_BONUS = 1.25;

    private final long budget;
    private final ToLongFunction<K> size;
    private final Function<K, V> load;
    private final Consumer<V> unload;
    private final long warmup;
    private final long every;
    private final long maxSwapBytes;
    private final Map<K, double[]> counts = new HashMap<>();
    private final Map<K, V> held = new LinkedHashMap<>();
    private long bytes;
    private long untilRefresh;
    private boolean filled;
    private long hits;
    private long misses;

    /**
     * @param budget the bytes the copies may take
     * @param size an item's bytes as a copy
     * @param load makes an item's copy
     * @param unload frees a copy that leaves the cache
     * @param warmup uses before the first fill
     * @param every uses between refreshes
     * @param maxSwapBytes the bytes a refresh may load, after the first fill
     */
    public HotCache(long budget, ToLongFunction<K> size, Function<K, V> load, Consumer<V> unload, long warmup,
            long every, long maxSwapBytes) {
        if (budget < 0 || warmup < 0 || every <= 0 || maxSwapBytes < 0) {
            throw new IllegalArgumentException("budget " + budget + ", warmup " + warmup + ", every " + every
                    + ", maxSwapBytes " + maxSwapBytes);
        }
        this.budget = budget;
        this.size = Objects.requireNonNull(size, "size");
        this.load = Objects.requireNonNull(load, "load");
        this.unload = Objects.requireNonNull(unload, "unload");
        this.warmup = warmup;
        this.every = every;
        this.maxSwapBytes = maxSwapBytes;
        this.untilRefresh = warmup;
    }

    /**
     * Counts {@code times} uses of {@code key} and returns its copy, or null when none is held.
     * May fill or refresh the cache first, loading and unloading copies.
     */
    public synchronized V use(K key, int times) {
        if (times <= 0) {
            throw new IllegalArgumentException("times is " + times);
        }
        counts.computeIfAbsent(key, k -> new double[1])[0] += times;
        untilRefresh -= times;
        if (untilRefresh <= 0) {
            refresh();
        }
        V v = held.get(key);
        if (filled && v != null) {
            hits += times;
        } else if (filled) {
            misses += times;
        }
        return v;
    }

    /** Ranks the items again and swaps copies; the first time, fills the whole budget. */
    public synchronized void refresh() {
        List<Map.Entry<K, double[]>> ranked = new ArrayList<>(counts.entrySet());
        ranked.sort((a, b) -> Double.compare(heat(b), heat(a)));
        Map<K, Long> want = new LinkedHashMap<>();
        long total = 0;
        for (Map.Entry<K, double[]> e : ranked) {
            long s = size.applyAsLong(e.getKey());
            if (total + s <= budget) {
                want.put(e.getKey(), s);
                total += s;
            }
        }
        // free what fell out first, then load the hottest newcomers within the swap limit
        for (var it = held.entrySet().iterator(); it.hasNext();) {
            Map.Entry<K, V> e = it.next();
            if (!want.containsKey(e.getKey())) {
                unload.accept(e.getValue());
                bytes -= size.applyAsLong(e.getKey());
                it.remove();
            }
        }
        long loaded = 0;
        for (Map.Entry<K, Long> e : want.entrySet()) {
            if (held.containsKey(e.getKey())) {
                continue;
            }
            if (filled && loaded + e.getValue() > maxSwapBytes) {
                break;
            }
            held.put(e.getKey(), load.apply(e.getKey()));
            bytes += e.getValue();
            loaded += e.getValue();
        }
        filled = true;
        for (double[] c : counts.values()) {
            c[0] /= 2;
        }
        untilRefresh = every;
    }

    private double heat(Map.Entry<K, double[]> e) {
        return held.containsKey(e.getKey()) ? e.getValue()[0] * HELD_BONUS : e.getValue()[0];
    }

    /** Whether {@code key}'s copy is held. */
    public synchronized boolean holds(K key) {
        return held.containsKey(key);
    }

    /** The number of copies held. */
    public synchronized int size() {
        return held.size();
    }

    /** The bytes the held copies take. */
    public synchronized long bytes() {
        return bytes;
    }

    /** The share of uses since the first fill that found a copy, 0 before it. */
    public synchronized double hitRate() {
        return hits + misses == 0 ? 0 : (double) hits / (hits + misses);
    }

    /** Frees every copy; the cache fills again after another warm-up. */
    public synchronized void clear() {
        for (V v : held.values()) {
            unload.accept(v);
        }
        held.clear();
        counts.clear();
        bytes = 0;
        filled = false;
        untilRefresh = warmup;
        hits = 0;
        misses = 0;
    }

    @Override
    public synchronized String toString() {
        return String.format("HotCache[%d held, %.1f of %.1f MB, hit rate %.2f]", held.size(), bytes / 1e6,
                budget / 1e6, hitRate());
    }
}
