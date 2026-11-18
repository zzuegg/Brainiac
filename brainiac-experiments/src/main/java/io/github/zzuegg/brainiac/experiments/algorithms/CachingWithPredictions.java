package io.github.zzuegg.brainiac.experiments.algorithms;

import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experimental.sketch.ClassicCaches;
import io.github.zzuegg.brainiac.experimental.sketch.ClassicCaches.PageCache;
import io.github.zzuegg.brainiac.experimental.sketch.SkiRental;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Online algorithms with predictions (item 11.2.17): paging rules that use next-arrival
 * predictions or an imitation-learned policy
 * ({@link io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions}) against
 * LRU, LFU, the heavy-hitter (H2O-style) rule and Belady on {@link KeyStreams} page
 * traces, and prediction-augmented ski rental against the classical rules.
 */
public final class CachingWithPredictions {

    private CachingWithPredictions() {
    }

    /**
     * A paging rule's result.
     *
     * @param method the rule
     * @param hitRate its hit rate
     * @param ofBelady the hit rate relative to Belady's
     */
    public record CacheRow(String method, double hitRate, double ofBelady) {
        @Override
        public String toString() {
            return String.format("%s %.3f (%.2f of Belady)", method, hitRate, ofBelady);
        }
    }

    /**
     * Replays one trace through every rule; Parrot is trained on another trace of the same
     * kind.
     *
     * @param kind the trace pattern
     * @param length requests
     * @param pages the page universe
     * @param loop the loop or scan length
     * @param capacity pages held
     * @param noises lognormal noise levels for the oracle predictor
     * @param seed the seed
     * @return one row per rule
     */
    public static List<CacheRow> paging(KeyStreams.TraceKind kind, int length, int pages, int loop, int capacity,
            double[] noises, long seed) {
        int[] trace = KeyStreams.pageTrace(kind, length, pages, loop, seed);
        int[] train = KeyStreams.pageTrace(kind, length, pages, loop, seed + 100);
        Map<String, PageCache> rules = new LinkedHashMap<>();
        rules.put("Belady", ClassicCaches.belady(capacity, trace));
        rules.put("LRU", ClassicCaches.lru(capacity));
        rules.put("LFU", ClassicCaches.lfu(capacity));
        rules.put("heavy hitters (H2O)", ClassicCaches.heavyHitters(capacity, capacity / 4));
        rules.put("marking", io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions.marking(capacity, seed));
        for (double n : noises) {
            rules.put("predictive marking, noise " + n, io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions
                    .predictiveMarking(capacity, io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions
                            .oracle(trace, n, seed + 1), seed));
            rules.put("follow predictions, noise " + n, io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions
                    .followPredictions(capacity, io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions
                            .oracle(trace, n, seed + 1)));
        }
        rules.put("predictive marking, learned gaps", io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions
                .predictiveMarking(capacity, io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions
                        .learnedGaps(0.3), seed));
        io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions.Parrot parrot =
                new io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions.Parrot(16, seed);
        parrot.train(train, capacity, 2, 2, seed);
        rules.put("Parrot imitation", io.github.zzuegg.brainiac.experimental.sketch.CachingWithPredictions.parrot(capacity,
                parrot));
        List<CacheRow> rows = new ArrayList<>();
        double belady = 0;
        for (Map.Entry<String, PageCache> e : rules.entrySet()) {
            double h = ClassicCaches.simulate(trace, e.getValue()).hitRate();
            if (e.getKey().equals("Belady")) {
                belady = h;
            }
            rows.add(new CacheRow(e.getKey(), h, h / Math.max(1e-9, belady)));
        }
        return rows;
    }

    /**
     * A ski-rental rule's empirical competitive ratios.
     *
     * @param rule the rule
     * @param noise the prediction noise (log-scale standard deviation)
     * @param lambda the trust parameter (NaN for the classical rules)
     * @param meanRatio mean cost over the optimum
     * @param maxRatio worst cost over the optimum
     */
    public record SkiRow(String rule, double noise, double lambda, double meanRatio, double maxRatio) {
        @Override
        public String toString() {
            return String.format("%s noise %.1f%s: mean %.3f, max %.3f", rule, noise,
                    Double.isNaN(lambda) ? "" : String.format(" lambda %.2f", lambda), meanRatio, maxRatio);
        }
    }

    /**
     * Ski rental with lognormal horizons around the buy price and predictions with
     * lognormal noise.
     *
     * @param b the buy price
     * @param instances horizons drawn
     * @param noises prediction noise levels
     * @param lambdas trust parameters
     * @param seed the seed
     * @return rows for break-even, Karlin's randomised rule, blind trust and the deterministic
     *     and randomised prediction rules
     */
    public static List<SkiRow> skiRental(int b, int instances, double[] noises, double[] lambdas, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] days = new int[instances];
        double[] z = new double[instances];
        for (int i = 0; i < instances; i++) {
            days[i] = Math.max(1, (int) Math.round(b * Math.exp(r.nextGaussian())));
            z[i] = r.nextGaussian();
        }
        List<SkiRow> rows = new ArrayList<>();
        double[] be = new double[2];
        double[] rnd = new double[2];
        for (int i = 0; i < instances; i++) {
            double opt = SkiRental.optimum(days[i], b);
            add(be, SkiRental.cost(SkiRental.breakEven(b), days[i], b) / opt);
            add(rnd, SkiRental.expectedClassicRandomized(days[i], b) / opt);
        }
        rows.add(new SkiRow("break-even", 0, Double.NaN, be[0] / instances, be[1]));
        rows.add(new SkiRow("Karlin randomised", 0, Double.NaN, rnd[0] / instances, rnd[1]));
        for (double noise : noises) {
            double[] blind = new double[2];
            for (int i = 0; i < instances; i++) {
                double pred = days[i] * Math.exp(noise * z[i]);
                double opt = SkiRental.optimum(days[i], b);
                add(blind, SkiRental.cost(pred >= b ? 1 : Integer.MAX_VALUE, days[i], b) / opt);
            }
            rows.add(new SkiRow("trust prediction", noise, Double.NaN, blind[0] / instances, blind[1]));
            for (double lambda : lambdas) {
                double[] det = new double[2];
                double[] ran = new double[2];
                for (int i = 0; i < instances; i++) {
                    double pred = days[i] * Math.exp(noise * z[i]);
                    double opt = SkiRental.optimum(days[i], b);
                    add(det, SkiRental.cost(SkiRental.deterministic(pred, b, lambda), days[i], b) / opt);
                    add(ran, SkiRental.expectedRandomized(pred, days[i], b, lambda) / opt);
                }
                rows.add(new SkiRow("deterministic", noise, lambda, det[0] / instances, det[1]));
                rows.add(new SkiRow("randomised", noise, lambda, ran[0] / instances, ran[1]));
            }
        }
        return rows;
    }

    private static void add(double[] acc, double ratio) {
        acc[0] += ratio;
        acc[1] = Math.max(acc[1], ratio);
    }
}
