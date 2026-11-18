package io.github.zzuegg.brainiac.experimental.stream;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Concept-drift detectors behind one interface: they watch a stream of values (usually a model's
 * 0/1 errors) and signal a warning or a drift when its distribution changes.
 * <ul>
 *   <li>{@link Adwin}: an adaptive window of recent values kept in an exponential histogram of
 *       buckets; whenever two sub-windows (an older and a newer part) have means differing by more
 *       than a Hoeffding-style bound, the older part is dropped (Bifet and Gavaldà, 2007);</li>
 *   <li>{@link Ddm}: tracks the error rate {@code p} and its deviation {@code s}, and warns or
 *       alarms when {@code p + s} exceeds its recorded minimum by 2 or 3 deviations (Gama et al.,
 *       2004);</li>
 *   <li>{@link Eddm}: the same on the distance between consecutive errors, which reacts to gradual
 *       drift (Baena-García et al., 2006);</li>
 *   <li>{@link PageHinkley}: a cumulative sum of deviations from the running mean, alarming when it
 *       rises a threshold above its minimum (Page, 1954).</li>
 * </ul>
 *
 * <p>Papers: <a href="https://doi.org/10.1137/1.9781611972771.42">doi:10.1137/1.9781611972771.42</a>,
 * <a href="https://doi.org/10.1007/978-3-540-28645-5_29">doi:10.1007/978-3-540-28645-5_29</a>.
 */
public final class DriftDetectors {

    /** What a detector concluded after a value. */
    public enum Signal {
        /** No change. */
        NONE,
        /** A change may be starting. */
        WARNING,
        /** A change happened. */
        DRIFT
    }

    /** A drift detector. */
    public interface Detector {

        /**
         * Adds a value.
         *
         * @param value the next value, such as 1 for an error and 0 for a correct prediction
         * @return the detector's signal
         */
        Signal update(double value);

        /** Forgets everything. */
        void reset();

        /** @return a short name */
        String name();
    }

    private DriftDetectors() {
    }

    /**
     * Runs a detector over a series.
     *
     * @param detector the detector
     * @param values the series
     * @return the positions of drift signals
     */
    public static int[] detections(Detector detector, double[] values) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            if (detector.update(values[i]) == Signal.DRIFT) {
                out.add(i);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** ADWIN2 with an exponential histogram of at most {@code m} buckets per size. */
    public static final class Adwin implements Detector {

        private record Bucket(long count, double sum, double sumSq) {
            Bucket merge(Bucket o) {
                return new Bucket(count + o.count, sum + o.sum, sumSq + o.sumSq);
            }
        }

        private final double delta;
        private final int maxBuckets;
        private final int clock;
        /** rows.get(k): buckets of size 2^k, newest first. */
        private final List<Deque<Bucket>> rows = new ArrayList<>();
        private long width;
        private double total;
        private double totalSq;
        private long ticks;

        /**
         * Creates the detector.
         *
         * @param delta the confidence, such as 0.002
         * @param maxBuckets buckets per size, such as 5
         * @param clock check for change every this many values, such as 32
         */
        public Adwin(double delta, int maxBuckets, int clock) {
            this.delta = delta;
            this.maxBuckets = maxBuckets;
            this.clock = clock;
        }

        /** @return the window's length */
        public long width() {
            return width;
        }

        /** @return the window's mean */
        public double mean() {
            return width == 0 ? 0 : total / width;
        }

        @Override
        public Signal update(double value) {
            insert(new Bucket(1, value, value * value), 0);
            width++;
            total += value;
            totalSq += value * value;
            ticks++;
            if (ticks % clock != 0 || width < 10) {
                return Signal.NONE;
            }
            boolean changed = false;
            while (cut()) {
                changed = true;
            }
            return changed ? Signal.DRIFT : Signal.NONE;
        }

        private void insert(Bucket b, int row) {
            while (rows.size() <= row) {
                rows.add(new ArrayDeque<>());
            }
            Deque<Bucket> d = rows.get(row);
            d.addFirst(b);
            if (d.size() > maxBuckets) {
                Bucket oldest = d.removeLast();
                Bucket second = d.removeLast();
                insert(second.merge(oldest), row + 1);
            }
        }

        /** Tests every split from the oldest buckets; drops the oldest bucket on a change. */
        private boolean cut() {
            double variance = Math.max(0, totalSq / width - (total / width) * (total / width));
            double dPrime = delta / Math.log(Math.max(2, width));
            long n0 = 0;
            double s0 = 0;
            for (int r = rows.size() - 1; r >= 0; r--) {
                for (java.util.Iterator<Bucket> it = rows.get(r).descendingIterator(); it.hasNext();) {
                    Bucket b = it.next();
                    n0 += b.count();
                    s0 += b.sum();
                    long n1 = width - n0;
                    if (n1 < 5) {
                        return false;
                    }
                    if (n0 < 5) {
                        continue;
                    }
                    double m = 1 / (1.0 / n0 + 1.0 / n1);
                    double eps = Math.sqrt(2 / m * variance * Math.log(2 / dPrime)) + 2 / (3 * m) * Math.log(2 / dPrime);
                    if (Math.abs(s0 / n0 - (total - s0) / n1) > eps) {
                        dropOldest();
                        return width > 10;
                    }
                }
            }
            return false;
        }

        private void dropOldest() {
            for (int r = rows.size() - 1; r >= 0; r--) {
                Deque<Bucket> d = rows.get(r);
                if (!d.isEmpty()) {
                    Bucket b = d.removeLast();
                    width -= b.count();
                    total -= b.sum();
                    totalSq -= b.sumSq();
                    if (d.isEmpty() && r == rows.size() - 1) {
                        rows.remove(r);
                    }
                    return;
                }
            }
        }

        @Override
        public void reset() {
            rows.clear();
            width = 0;
            total = 0;
            totalSq = 0;
            ticks = 0;
        }

        @Override
        public String name() {
            return "ADWIN";
        }
    }

    /** The drift detection method on a 0/1 error stream. */
    public static final class Ddm implements Detector {

        private final int minSamples;
        private long n;
        private double p;
        private double pMin;
        private double sMin;

        /**
         * Creates the detector.
         *
         * @param minSamples samples before any signal, such as 30
         */
        public Ddm(int minSamples) {
            this.minSamples = minSamples;
            reset();
        }

        @Override
        public Signal update(double error) {
            n++;
            p += (error - p) / n;
            double s = Math.sqrt(p * (1 - p) / n);
            if (n < minSamples) {
                return Signal.NONE;
            }
            if (p + s < pMin + sMin) {
                pMin = p;
                sMin = s;
            }
            if (p + s >= pMin + 3 * sMin) {
                reset();
                return Signal.DRIFT;
            }
            return p + s >= pMin + 2 * sMin ? Signal.WARNING : Signal.NONE;
        }

        @Override
        public void reset() {
            n = 0;
            p = 0;
            pMin = Double.MAX_VALUE / 4;
            sMin = Double.MAX_VALUE / 4;
        }

        @Override
        public String name() {
            return "DDM";
        }
    }

    /** The early drift detection method: distances between consecutive errors. */
    public static final class Eddm implements Detector {

        private final int minErrors;
        private long t;
        private long lastError;
        private long errors;
        private double mean;
        private double m2;
        private double maxScore;

        /**
         * Creates the detector.
         *
         * @param minErrors errors before any signal, such as 30
         */
        public Eddm(int minErrors) {
            this.minErrors = minErrors;
            reset();
        }

        @Override
        public Signal update(double error) {
            t++;
            if (error < 0.5) {
                return Signal.NONE;
            }
            long distance = t - lastError;
            lastError = t;
            errors++;
            double delta = distance - mean;
            mean += delta / errors;
            m2 += delta * (distance - mean);
            double std = errors > 1 ? Math.sqrt(m2 / (errors - 1)) : 0;
            double score = mean + 2 * std;
            if (errors < minErrors) {
                maxScore = Math.max(maxScore, score);
                return Signal.NONE;
            }
            maxScore = Math.max(maxScore, score);
            double ratio = score / maxScore;
            if (ratio < 0.9) {
                reset();
                return Signal.DRIFT;
            }
            return ratio < 0.95 ? Signal.WARNING : Signal.NONE;
        }

        @Override
        public void reset() {
            t = 0;
            lastError = 0;
            errors = 0;
            mean = 0;
            m2 = 0;
            maxScore = 0;
        }

        @Override
        public String name() {
            return "EDDM";
        }
    }

    /** The Page-Hinkley test for an increase of the mean. */
    public static final class PageHinkley implements Detector {

        private final double delta;
        private final double threshold;
        private final int minSamples;
        private long n;
        private double mean;
        private double cumulative;
        private double minimum;

        /**
         * Creates the detector.
         *
         * @param delta the tolerated change, such as 0.005
         * @param threshold the alarm threshold {@code lambda}, such as 50
         * @param minSamples samples before any signal
         */
        public PageHinkley(double delta, double threshold, int minSamples) {
            this.delta = delta;
            this.threshold = threshold;
            this.minSamples = minSamples;
        }

        @Override
        public Signal update(double value) {
            n++;
            mean += (value - mean) / n;
            cumulative += value - mean - delta;
            minimum = Math.min(minimum, cumulative);
            if (n >= minSamples && cumulative - minimum > threshold) {
                reset();
                return Signal.DRIFT;
            }
            return Signal.NONE;
        }

        @Override
        public void reset() {
            n = 0;
            mean = 0;
            cumulative = 0;
            minimum = 0;
        }

        @Override
        public String name() {
            return "Page-Hinkley";
        }
    }
}
