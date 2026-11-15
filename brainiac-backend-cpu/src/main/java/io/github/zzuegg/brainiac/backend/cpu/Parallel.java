package io.github.zzuegg.brainiac.backend.cpu;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

/**
 * Splits loops over a shared fork-join pool. Work is only split when it is large enough
 * to pay for the hand-off, and every split computes each output value in the same order
 * as a single thread would, so results do not depend on the number of threads.
 *
 * <p>The number of threads is read once from the system property
 * {@code brainiac.cpu.threads} and defaults to the number of available processors.
 */
final class Parallel {

    /**
     * Rough number of floating-point operations below which splitting is not worth it.
     * Tunable with {@code -Dbrainiac.cpu.minWork}; the default was picked with the
     * benchmark harness.
     */
    private static final long MIN_WORK_PER_TASK = Long.getLong("brainiac.cpu.minWork", 1L << 20);

    private static final int THREADS = threadCount();
    private static final ForkJoinPool POOL = THREADS > 1 ? new ForkJoinPool(THREADS) : null;

    /** A loop body over the index range {@code [from, to)}. */
    @FunctionalInterface
    interface Range {
        void run(int from, int to);
    }

    private Parallel() {
    }

    static int threads() {
        return THREADS;
    }

    /**
     * Runs {@code body} over {@code [0, n)}, split into parallel chunks if the total work
     * is large enough.
     *
     * @param n the number of items
     * @param workPerItem rough cost of one item, in floating-point operations
     * @param body the loop body
     */
    static void forRange(int n, long workPerItem, Range body) {
        if (n <= 0) {
            return;
        }
        long total = n * Math.max(1, workPerItem);
        int tasks = (int) Math.min(Math.min(n, (long) THREADS * 4), total / MIN_WORK_PER_TASK);
        if (POOL == null || tasks < 2) {
            body.run(0, n);
            return;
        }
        int grain = Math.ceilDiv(n, tasks);
        Split task = new Split(body, 0, n, grain);
        if (ForkJoinTask.getPool() == POOL) {
            task.invoke();
        } else {
            POOL.invoke(task);
        }
    }

    private static int threadCount() {
        String configured = System.getProperty("brainiac.cpu.threads");
        if (configured != null) {
            try {
                return Math.max(1, Integer.parseInt(configured.trim()));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("brainiac.cpu.threads must be a number, got " + configured, e);
            }
        }
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    private static final class Split extends RecursiveAction {

        private static final long serialVersionUID = 1L;

        private final transient Range body;
        private final int from;
        private final int to;
        private final int grain;

        Split(Range body, int from, int to, int grain) {
            this.body = body;
            this.from = from;
            this.to = to;
            this.grain = grain;
        }

        @Override
        protected void compute() {
            if (to - from <= grain) {
                body.run(from, to);
                return;
            }
            int mid = from + (to - from) / 2;
            invokeAll(new Split(body, from, mid, grain), new Split(body, mid, to, grain));
        }
    }
}
