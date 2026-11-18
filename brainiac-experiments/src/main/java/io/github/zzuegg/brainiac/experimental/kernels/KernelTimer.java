package io.github.zzuegg.brainiac.experimental.kernels;

import java.util.Arrays;

/**
 * The timing harness shared by the kernel experiments, in the spirit of JMH without its
 * machinery: untimed warm-up runs let the JIT compile the kernel, then the median of several
 * timed runs is reported (the median resists the occasional garbage-collection or scheduler
 * hiccup better than the mean). It also gives the correctness check every kernel is held to,
 * the largest error against a reference result relative to the reference's largest magnitude.
 */
public final class KernelTimer {

    private KernelTimer() {
    }

    /**
     * A timing.
     *
     * @param medianMillis the median of the timed runs
     * @param minMillis the fastest timed run
     */
    public record Timing(double medianMillis, double minMillis) {

        /**
         * @param flops the floating-point operations of one run
         * @return the throughput at the median, in GFLOP/s
         */
        public double gflops(double flops) {
            return flops / (medianMillis * 1e6);
        }

        /**
         * @param flops the floating-point operations of one run
         * @return the throughput of the fastest run, in GFLOP/s (robust to interference)
         */
        public double peakGflops(double flops) {
            return flops / (minMillis * 1e6);
        }
    }

    /**
     * Times a task.
     *
     * @param task the work
     * @param warmup untimed runs first
     * @param repeats timed runs, at least 1
     * @return the median and fastest run
     */
    public static Timing time(Runnable task, int warmup, int repeats) {
        for (int i = 0; i < warmup; i++) {
            task.run();
        }
        double[] t = new double[Math.max(1, repeats)];
        for (int i = 0; i < t.length; i++) {
            long t0 = System.nanoTime();
            task.run();
            t[i] = (System.nanoTime() - t0) / 1e6;
        }
        Arrays.sort(t);
        return new Timing(t[t.length / 2], t[0]);
    }

    /**
     * @param task the work
     * @param warmup untimed runs
     * @param repeats timed runs
     * @return the median time in milliseconds
     */
    public static double millis(Runnable task, int warmup, int repeats) {
        return time(task, warmup, repeats).medianMillis();
    }

    /**
     * @param result a kernel's output
     * @param reference the reference output, same length
     * @return {@code max |result - reference| / max |reference|}
     */
    public static double relativeError(float[] result, float[] reference) {
        if (result.length != reference.length) {
            throw new IllegalArgumentException("lengths differ: " + result.length + " vs " + reference.length);
        }
        double err = 0;
        double scale = 1e-30;
        for (int i = 0; i < result.length; i++) {
            err = Math.max(err, Math.abs(result[i] - reference[i]));
            scale = Math.max(scale, Math.abs(reference[i]));
        }
        return err / scale;
    }
}
