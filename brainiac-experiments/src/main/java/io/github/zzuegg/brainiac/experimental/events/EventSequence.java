package io.github.zzuegg.brainiac.experimental.events;

import java.util.Arrays;

/**
 * One marked event sequence observed on {@code [0, horizon]}: strictly increasing event times
 * and one type (mark) per event.
 *
 * @param times event times, increasing, inside {@code (0, horizon]}
 * @param types the event types, in {@code [0, types)}
 * @param horizon the end of the observation window
 */
public record EventSequence(double[] times, int[] types, double horizon) {

    /**
     * Checks the arrays.
     *
     * @param times event times
     * @param types event types
     * @param horizon window end
     */
    public EventSequence {
        if (times.length != types.length) {
            throw new IllegalArgumentException(times.length + " times, " + types.length + " types");
        }
        times = times.clone();
        types = types.clone();
    }

    /** @return the number of events */
    public int size() {
        return times.length;
    }

    /**
     * @param n how many events to keep
     * @param newHorizon the new window end
     * @return the first {@code n} events observed on {@code [0, newHorizon]}
     */
    public EventSequence prefix(int n, double newHorizon) {
        return new EventSequence(Arrays.copyOf(times, n), Arrays.copyOf(types, n), newHorizon);
    }
}
