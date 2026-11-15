package io.github.zzuegg.brainiac.rl;

/** A value that changes with the number of steps taken, such as the exploration rate. */
@FunctionalInterface
public interface Schedule {

    /**
     * @param step the steps taken so far
     * @return the value at that step
     */
    double at(long step);

    /**
     * @param value the value
     * @return a schedule that never changes
     */
    static Schedule constant(double value) {
        return step -> value;
    }

    /**
     * Moves in a straight line from {@code start} to {@code end} over {@code steps}, then
     * stays at {@code end}.
     *
     * @param start the first value
     * @param end the final value
     * @param steps how long the change takes
     * @return the schedule
     */
    static Schedule linear(double start, double end, long steps) {
        return step -> step >= steps ? end : start + (end - start) * step / steps;
    }

    /**
     * Multiplies by {@code factor} every step, never passing {@code end}.
     *
     * @param start the first value
     * @param end the limit
     * @param factor the change per step, for example {@code 0.999}
     * @return the schedule
     */
    static Schedule exponential(double start, double end, double factor) {
        return step -> {
            double v = start * Math.pow(factor, step);
            return start > end ? Math.max(end, v) : Math.min(end, v);
        };
    }
}
