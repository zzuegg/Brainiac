package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Synthetic series with labelled anomalies and change points, for anomaly and change-point
 * detectors.
 * <ul>
 *   <li>{@link #anomalies}: a seasonal series with planted anomalies of the four kinds of the
 *       usual taxonomy: <b>point</b> (a single spike far outside the range), <b>contextual</b>
 *       (a few values inside the global range but wrong for their phase of the season),
 *       <b>collective</b> (a stretch whose shape is wrong although every value is ordinary)
 *       and <b>seasonal shift</b> (a stretch played at a different period);</li>
 *   <li>{@link #changePoints}: piecewise-stationary segments whose mean, variance or seasonal
 *       amplitude changes at labelled change points.</li>
 * </ul>
 */
public final class AnomalySeries {

    private AnomalySeries() {
    }

    /** The planted anomaly kinds. */
    public enum Kind {
        /** A single extreme value. */
        POINT,
        /** Ordinary values at the wrong phase. */
        CONTEXTUAL,
        /** A stretch of the wrong shape. */
        COLLECTIVE,
        /** A stretch at a different period. */
        SEASONAL_SHIFT
    }

    /**
     * One planted anomaly.
     *
     * @param kind its kind
     * @param start the first anomalous step
     * @param end one past the last anomalous step
     */
    public record Event(Kind kind, int start, int end) {
    }

    /**
     * A series with labels.
     *
     * @param values the series
     * @param labels true at anomalous steps
     * @param events the planted anomalies
     * @param period the season length
     */
    public record Labelled(float[] values, boolean[] labels, List<Event> events, int period) {
    }

    /**
     * A seasonal series with {@code count} anomalies of each kind in the given order, spread
     * over the second half (the first {@code clean} steps stay normal for training).
     *
     * @param length steps
     * @param period the season length
     * @param clean the anomaly-free prefix
     * @param count anomalies per kind
     * @param kinds the kinds to plant
     * @param seed the seed
     * @return the labelled series
     */
    public static Labelled anomalies(int length, int period, int clean, int count, List<Kind> kinds, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] y = new float[length];
        double phase = r.nextDouble(0, 2 * Math.PI);
        for (int t = 0; t < length; t++) {
            y[t] = (float) seasonal(t, period, phase) + (float) (0.08 * r.nextGaussian());
        }
        boolean[] labels = new boolean[length];
        List<Event> events = new ArrayList<>();
        int slots = count * kinds.size();
        int slotLength = (length - clean) / Math.max(1, slots);
        int slot = 0;
        for (int k = 0; k < count; k++) {
            for (Kind kind : kinds) {
                int base = clean + slot * slotLength + period / 2;
                int start = base + r.nextInt(Math.max(1, slotLength - 2 * period));
                int end = switch (kind) {
                    case POINT -> start + 1;
                    case CONTEXTUAL -> start + 3;
                    case COLLECTIVE, SEASONAL_SHIFT -> start + period;
                };
                end = Math.min(end, length);
                for (int t = start; t < end; t++) {
                    switch (kind) {
                        case POINT -> y[t] += (float) ((r.nextBoolean() ? 1 : -1) * (2.5 + r.nextDouble()));
                        // the value half a season away: inside the global range, wrong for this phase
                        case CONTEXTUAL -> y[t] = (float) seasonal(t + period / 2, period, phase)
                                + (float) (0.08 * r.nextGaussian());
                        // the shape is flattened to a slow ramp between the endpoints
                        case COLLECTIVE -> {
                            double a = seasonal(start, period, phase);
                            double b = seasonal(end, period, phase);
                            y[t] = (float) (a + (b - a) * (t - start) / (end - start))
                                    + (float) (0.08 * r.nextGaussian());
                        }
                        case SEASONAL_SHIFT -> y[t] = (float) seasonal(start + 2 * (t - start), period, phase)
                                + (float) (0.08 * r.nextGaussian());
                    }
                    labels[t] = true;
                }
                events.add(new Event(kind, start, end));
                slot++;
            }
        }
        return new Labelled(y, labels, events, period);
    }

    private static double seasonal(int t, int period, double phase) {
        double w = 2 * Math.PI * t / period + phase;
        return Math.sin(w) + 0.5 * Math.sin(2 * w + 0.7);
    }

    /** What changes at a change point. */
    public enum Change {
        /** The mean jumps. */
        MEAN,
        /** The noise standard deviation changes. */
        VARIANCE,
        /** The seasonal amplitude changes. */
        SEASONALITY
    }

    /**
     * A piecewise-stationary series.
     *
     * @param values the series
     * @param changePoints the first step of each new segment
     * @param changes what changed at each change point
     */
    public record Segmented(float[] values, int[] changePoints, List<Change> changes) {
    }

    /**
     * Segments of 60-140 steps; at each boundary the mean, the noise level or the seasonal
     * amplitude changes (the kinds cycle through {@code kinds}).
     *
     * @param length steps
     * @param period the season length (for seasonality changes)
     * @param kinds the change kinds to cycle through
     * @param seed the seed
     * @return the series and its change points
     */
    public static Segmented changePoints(int length, int period, List<Change> kinds, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] y = new float[length];
        List<Integer> cps = new ArrayList<>();
        List<Change> changes = new ArrayList<>();
        double mean = 0;
        double sigma = 0.3;
        double amp = 0;
        int next = 60 + r.nextInt(80);
        int k = 0;
        for (int t = 0; t < length; t++) {
            if (t == next && length - t > 40) {
                Change c = kinds.get(k++ % kinds.size());
                switch (c) {
                    case MEAN -> mean += (r.nextBoolean() ? 1 : -1) * (1.0 + r.nextDouble());
                    case VARIANCE -> sigma = sigma < 0.5 ? 0.9 + 0.4 * r.nextDouble() : 0.2 + 0.1 * r.nextDouble();
                    case SEASONALITY -> amp = amp > 0.5 ? 0 : 1.2 + 0.5 * r.nextDouble();
                }
                cps.add(t);
                changes.add(c);
                next = t + 60 + r.nextInt(80);
            }
            y[t] = (float) (mean + amp * Math.sin(2 * Math.PI * t / period) + sigma * r.nextGaussian());
        }
        int[] points = new int[cps.size()];
        for (int i = 0; i < points.length; i++) {
            points[i] = cps.get(i);
        }
        return new Segmented(y, points, changes);
    }
}
