package io.github.zzuegg.brainiac.nn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
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
 * <p>Copies can be made on another thread ({@code loader}): a refresh then only starts them,
 * and each joins the cache at the first use after it is ready, so making gigabytes of copies
 * never holds up the stream that uses them.
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
    private final Executor loader;
    /** Copies being made, and those no longer wanted, to free once they are made. */
    private final Map<K, CompletableFuture<V>> pending = new LinkedHashMap<>();
    private final List<CompletableFuture<V>> dropped = new ArrayList<>();
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
        this(budget, size, load, unload, warmup, every, maxSwapBytes, Runnable::run);
    }

    /**
     * As the other constructor, with copies made on {@code loader}; {@code load} must then be
     * safe to call from it, several at once if it runs several.
     *
     * @param loader runs {@code load}; {@code Runnable::run} makes copies during the use that asks
     */
    public HotCache(long budget, ToLongFunction<K> size, Function<K, V> load, Consumer<V> unload, long warmup,
            long every, long maxSwapBytes, Executor loader) {
        this.loader = Objects.requireNonNull(loader, "loader");
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
        settle();
        V v = held.get(key);
        if (filled && v != null) {
            hits += times;
        } else if (filled) {
            misses += times;
        }
        return v;
    }

    /**
     * Fills the cache with {@code keys}, in order, as far as the budget goes, and waits for the
     * copies: a start for a stream with no uses yet, made while nothing else runs (a model
     * loading). The refreshes then move the cache toward what the stream uses, within the swap
     * limit, as after a first fill.
     *
     * @param keys the items to hold first, the most likely first
     */
    public void preload(Iterable<K> keys) {
        List<CompletableFuture<V>> started = new ArrayList<>();
        synchronized (this) {
            long total = bytes;
            for (K key : keys) {
                long s = size.applyAsLong(key);
                if (total + s > budget) {
                    break;
                }
                if (held.containsKey(key) || pending.containsKey(key)) {
                    continue;
                }
                CompletableFuture<V> f = start(key);
                pending.put(key, f);
                started.add(f);
                total += s;
            }
            filled = true;
        }
        for (CompletableFuture<V> f : started) {
            try {
                f.join();
            } catch (RuntimeException e) {
                // left out; a refresh may try it again
            }
        }
        synchronized (this) {
            settle();
        }
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
        // the hottest newcomers within the swap limit, then room for them: what fell out of the
        // ranking leaves, coldest first, only as far as the newcomers need, so that a refresh
        // that may load little does not empty the cache
        Map<K, Long> newcomers = new LinkedHashMap<>();
        long loading = 0;
        for (Map.Entry<K, Long> e : want.entrySet()) {
            if (held.containsKey(e.getKey()) || pending.containsKey(e.getKey())) {
                continue;
            }
            if (filled && loading + e.getValue() > maxSwapBytes) {
                break;
            }
            newcomers.put(e.getKey(), e.getValue());
            loading += e.getValue();
        }
        long kept = 0;
        List<K> leaving = new ArrayList<>();
        for (K k : held.keySet()) {
            kept += size.applyAsLong(k);
            if (!want.containsKey(k)) {
                leaving.add(k);
            }
        }
        for (K k : pending.keySet()) {
            kept += size.applyAsLong(k);
            if (!want.containsKey(k)) {
                leaving.add(k);
            }
        }
        leaving.sort((x, y) -> Double.compare(heat(x), heat(y)));
        for (K k : leaving) {
            if (kept + loading <= budget) {
                break;
            }
            kept -= size.applyAsLong(k);
            V v = held.remove(k);
            if (v != null) {
                unload.accept(v);
                bytes -= size.applyAsLong(k);
            } else {
                dropped.add(pending.remove(k));
            }
        }
        for (K key : newcomers.keySet()) {
            pending.put(key, start(key));
        }
        filled = true;
        for (double[] c : counts.values()) {
            c[0] /= 2;
        }
        untilRefresh = every;
    }

    /**
     * Starts making a copy. The copy's future completes whatever happens: a load that fails
     * with an error (out of memory, say) and a loader that refuses the task leave it failed,
     * where a future that never completes held up {@link #preload} for good (measured: a
     * heap that ran out while copies were made).
     */
    private CompletableFuture<V> start(K key) {
        CompletableFuture<V> f = new CompletableFuture<>();
        try {
            loader.execute(() -> {
                try {
                    f.complete(load.apply(key));
                } catch (Throwable t) {
                    f.completeExceptionally(t);
                } finally {
                    if (!f.isDone()) {
                        f.completeExceptionally(LOAD_FAILED);   // completing above failed too
                    }
                }
            });
        } catch (RuntimeException e) {
            f.completeExceptionally(e);
        }
        return f;
    }

    /** Made beforehand: when a load fails because memory ran out, there may be none for another. */
    private static final RuntimeException LOAD_FAILED = new RuntimeException("a copy failed", null, false,
            false) {
        private static final long serialVersionUID = 1L;
    };

    /** Moves copies that are ready into the cache, and frees those made after they stopped being wanted. */
    private void settle() {
        for (var it = pending.entrySet().iterator(); it.hasNext();) {
            Map.Entry<K, CompletableFuture<V>> e = it.next();
            if (e.getValue().isDone()) {
                it.remove();
                if (!e.getValue().isCompletedExceptionally()) {
                    // a copy that failed (out of memory, say) is tried again at a later refresh
                    held.put(e.getKey(), e.getValue().join());
                    bytes += size.applyAsLong(e.getKey());
                }
            }
        }
        for (var it = dropped.iterator(); it.hasNext();) {
            CompletableFuture<V> f = it.next();
            if (f.isDone()) {
                it.remove();
                if (!f.isCompletedExceptionally()) {
                    unload.accept(f.join());
                }
            }
        }
    }

    /** The number of copies being made. */
    public synchronized int loading() {
        return pending.size();
    }

    private double heat(Map.Entry<K, double[]> e) {
        return heat(e.getKey());
    }

    private double heat(K key) {
        double[] c = counts.get(key);
        double n = c == null ? 0 : c[0];
        return held.containsKey(key) || pending.containsKey(key) ? n * HELD_BONUS : n;
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
        dropped.addAll(pending.values());
        pending.clear();
        settle();
        counts.clear();
        bytes = 0;
        filled = false;
        untilRefresh = warmup;
        hits = 0;
        misses = 0;
    }

    @Override
    public synchronized String toString() {
        return String.format(java.util.Locale.ROOT, "HotCache[%d held, %d loading, %.1f of %.1f MB, hit rate %.2f]",
                held.size(), pending.size(), bytes / 1e6, budget / 1e6, hitRate());
    }
}
