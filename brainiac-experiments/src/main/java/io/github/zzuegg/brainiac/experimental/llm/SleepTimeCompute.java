package io.github.zzuegg.brainiac.experimental.llm;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Sleep-time compute (Lin et al. 2025, <a href="https://arxiv.org/abs/2504.13187">arXiv:2504.13187</a>):
 * while the user is idle, a model thinks about the context ahead of time. Here: whenever the
 * context changes, the queries it is likely to be asked are predicted and answered in the
 * background; a later query that was anticipated is served from that work (or waits for it
 * if it is still running) instead of costing a full model call at test time. A new context
 * makes earlier work stale: queued tasks for it are skipped and its results never served.
 *
 * @param <C> the context, such as a document or conversation
 * @param <A> the answer
 */
public final class SleepTimeCompute<C, A> {

    /**
     * Counts so far.
     *
     * @param sleepCalls model calls made in the background
     * @param testCalls model calls made while a query waited
     * @param hits queries served from background work
     * @param misses queries that needed a model call
     * @param skipped background tasks skipped because the context changed first
     */
    public record Stats(int sleepCalls, int testCalls, int hits, int misses, int skipped) { }

    private record Snapshot<C, A>(long version, C context, Map<String, CompletableFuture<A>> work) { }

    private final Function<C, List<String>> anticipate;
    private final BiFunction<C, String, A> answer;
    private final Executor executor;
    private volatile Snapshot<C, A> current;
    private final AtomicInteger sleepCalls = new AtomicInteger();
    private final AtomicInteger testCalls = new AtomicInteger();
    private final AtomicInteger hits = new AtomicInteger();
    private final AtomicInteger misses = new AtomicInteger();
    private final AtomicInteger skipped = new AtomicInteger();

    private SleepTimeCompute(Function<C, List<String>> anticipate, BiFunction<C, String, A> answer,
            Executor executor) {
        this.anticipate = anticipate;
        this.answer = answer;
        this.executor = executor;
    }

    /**
     * Creates the scheduler.
     *
     * @param <C> the context type
     * @param <A> the answer type
     * @param anticipate predicts likely queries for a context (itself a model call, in practice)
     * @param answer answers a query about a context: the expensive model call
     * @param executor runs the background work
     * @return the scheduler
     */
    public static <C, A> SleepTimeCompute<C, A> of(Function<C, List<String>> anticipate,
            BiFunction<C, String, A> answer, Executor executor) {
        return new SleepTimeCompute<>(anticipate, answer, executor);
    }

    private static String key(String query) {
        return query.trim().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    }

    /**
     * Sets a new context and starts thinking about it in the background.
     *
     * @param context the context
     */
    public synchronized void contextChanged(C context) {
        long version = current == null ? 0 : current.version() + 1;
        Snapshot<C, A> snap = new Snapshot<>(version, context, new ConcurrentHashMap<>());
        current = snap;
        for (String q : anticipate.apply(context)) {
            snap.work().computeIfAbsent(key(q), k -> CompletableFuture.supplyAsync(() -> {
                if (current != snap) {
                    skipped.incrementAndGet();
                    return null;
                }
                sleepCalls.incrementAndGet();
                return answer.apply(context, q);
            }, executor));
        }
    }

    /**
     * Answers a query about the current context.
     *
     * @param query the query
     * @return the answer
     */
    public A ask(String query) {
        Snapshot<C, A> snap = current;
        if (snap == null) {
            throw new IllegalStateException("no context yet");
        }
        CompletableFuture<A> f = snap.work().get(key(query));
        if (f != null) {
            A a = f.join();
            if (a != null && current == snap) {
                hits.incrementAndGet();
                return a;
            }
        }
        misses.incrementAndGet();
        testCalls.incrementAndGet();
        return answer.apply(snap.context(), query);
    }

    /** Waits until the background work for the current context is done. */
    public void awaitIdle() {
        Snapshot<C, A> snap = current;
        if (snap != null) {
            CompletableFuture.allOf(snap.work().values().toArray(CompletableFuture[]::new)).join();
        }
    }

    /** @return the counts so far */
    public Stats stats() {
        return new Stats(sleepCalls.get(), testCalls.get(), hits.get(), misses.get(), skipped.get());
    }

    @Override
    public String toString() {
        return "SleepTimeCompute(" + stats() + ")";
    }
}
