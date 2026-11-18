package io.github.zzuegg.brainiac.experimental.sketch;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.sketch.FrequencySketches.FrequencySketch;
import io.github.zzuegg.brainiac.experimental.sketch.FrequencySketches.MisraGries;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.HashMap;
import java.util.Map;

/**
 * Learning-augmented frequency sketches: a heavy-hitter oracle routes the items it predicts
 * to be heavy away from the shared counters.
 *
 * <ul>
 *   <li>{@link Kind#LEARNED_COUNT_MIN} (Hsu, Indyk, Katabi and Vakilian, ICLR 2019): up to
 *       {@code reserved} predicted-heavy items get a unique bucket (an id and an exact
 *       count); every other item goes to a Count-Min over the remaining memory;</li>
 *   <li>{@link Kind#LEARNED_MISRA_GRIES} (after Aamand, Chen, Nguyen, Silwal and Vakilian,
 *       ICLR 2025): two Misra-Gries summaries, one for the predicted-heavy items and one for
 *       the rest, so each part keeps the classical {@code N_part / (k_part + 1)} bound
 *       whether or not the predictions are right.</li>
 * </ul>
 *
 * <p>{@link #classifier} trains the oracle, an MLP on item features labelled by the counts
 * in a prefix of the stream; {@link #noisy} flips a share of any oracle's answers, the
 * prediction-noise knob.
 */
public final class LearnedSketches {

    private LearnedSketches() {
    }

    /** The learned sketches. */
    public enum Kind {
        /** Count-Min with unique buckets for predicted heavy hitters */
        LEARNED_COUNT_MIN,
        /** Misra-Gries split into predicted-heavy and other items */
        LEARNED_MISRA_GRIES
    }

    /** Predicts whether an item is a heavy hitter. */
    @FunctionalInterface
    public interface HeavyOracle {
        /**
         * @param item an item
         * @return whether it is predicted heavy
         */
        boolean heavy(int item);
    }

    /**
     * Creates a learned sketch.
     *
     * @param kind the sketch
     * @param words memory in counters, the reserved part included
     * @param depth Count-Min rows
     * @param oracle the heavy-hitter oracle
     * @param reservedShare the share of memory reserved for predicted-heavy items
     * @param seed the hash seed
     * @return the sketch
     */
    public static FrequencySketch of(Kind kind, int words, int depth, HeavyOracle oracle, double reservedShare,
            long seed) {
        int reserved = (int) (words * reservedShare);
        return switch (kind) {
            case LEARNED_COUNT_MIN -> new LearnedCountMin(words, reserved / 2, depth, oracle, seed);
            case LEARNED_MISRA_GRIES -> new LearnedMisraGries(Math.max(1, reserved / 2),
                    Math.max(1, (words - reserved) / 2), oracle);
        };
    }

    /** Count-Min with unique buckets. */
    static final class LearnedCountMin implements FrequencySketch {
        private final int slots;
        private final Map<Integer, Long> exact = new HashMap<>();
        private final FrequencySketch rest;
        private final HeavyOracle oracle;
        private final int words;

        LearnedCountMin(int words, int slots, int depth, HeavyOracle oracle, long seed) {
            this.slots = slots;
            this.oracle = oracle;
            this.words = words;
            this.rest = FrequencySketches.of(FrequencySketches.Kind.COUNT_MIN, Math.max(depth, words - 2 * slots), depth,
                    seed);
        }

        @Override
        public void add(int item) {
            Long c = exact.get(item);
            if (c != null) {
                exact.put(item, c + 1);
            } else if (exact.size() < slots && oracle.heavy(item)) {
                exact.put(item, 1L);
            } else {
                rest.add(item);
            }
        }

        @Override
        public long estimate(int item) {
            Long c = exact.get(item);
            return c != null ? c : rest.estimate(item);
        }

        @Override
        public int words() {
            return words;
        }
    }

    /** Two Misra-Gries summaries. */
    static final class LearnedMisraGries implements FrequencySketch {
        private final MisraGries heavy;
        private final MisraGries rest;
        private final HeavyOracle oracle;

        LearnedMisraGries(int heavyCounters, int restCounters, HeavyOracle oracle) {
            this.heavy = new MisraGries(heavyCounters);
            this.rest = new MisraGries(restCounters);
            this.oracle = oracle;
        }

        @Override
        public void add(int item) {
            (oracle.heavy(item) ? heavy : rest).add(item);
        }

        @Override
        public long estimate(int item) {
            return (oracle.heavy(item) ? heavy : rest).estimate(item);
        }

        @Override
        public int words() {
            return heavy.words() + rest.words();
        }
    }

    /**
     * Flips a share of an oracle's answers (a fixed pseudo-random set of items).
     *
     * @param oracle the oracle
     * @param flip the share of items whose answer is flipped
     * @param seed the seed
     * @return the noisy oracle
     */
    public static HeavyOracle noisy(HeavyOracle oracle, double flip, long seed) {
        return item -> {
            double u = (FrequencySketches.hash(item, seed) >>> 11) * 0x1.0p-53;
            return u < flip ? !oracle.heavy(item) : oracle.heavy(item);
        };
    }

    /**
     * Trains a heavy-hitter classifier: an MLP on item features, labelled by whether the item
     * is among the {@code heavyCount} most frequent items of a stream prefix, trained with
     * class-balanced binary cross-entropy.
     *
     * @param features per item id, its feature vector
     * @param prefix the stream prefix the labels come from
     * @param heavyCount how many items count as heavy
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return the oracle (probability above one half)
     */
    public static HeavyOracle classifier(float[][] features, int[] prefix, int heavyCount, int steps, long seed) {
        int n = features.length;
        int d = features[0].length;
        long[] count = new long[n];
        for (int x : prefix) {
            count[x]++;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Long.compare(count[b], count[a]));
        float[] y = new float[n];
        for (int i = 0; i < Math.min(heavyCount, n); i++) {
            if (count[order[i]] > 0) {
                y[order[i]] = 1;
            }
        }
        float pos = 0;
        for (float v : y) {
            pos += v;
        }
        float wPos = n / (2f * Math.max(1, pos));
        float wNeg = n / (2f * Math.max(1, n - pos));
        float[] w = new float[n];
        float[] x = new float[n * d];
        for (int i = 0; i < n; i++) {
            w[i] = y[i] > 0 ? wPos : wNeg;
            System.arraycopy(features[i], 0, x, i * d, d);
        }
        Model mlp = Model.sequential().input(d).dense(32, Activation.RELU).dense(1).seed(seed).build();
        Optimizer opt = Optimizer.adam(1e-2f);
        Tensor xt = Tensor.of(Shape.of(n, d), x);
        Tensor yt = Tensor.of(Shape.of(n), y);
        Tensor wt = Tensor.of(Shape.of(n), w);
        for (int s = 0; s < steps; s++) {
            Var z = mlp.forward(Var.of(xt), true).reshape(n);
            z.softplus().sub(z.mul(yt)).mul(wt).mean().backward();
            opt.step(mlp.parameters());
        }
        float[] z = Var.noGrad(() -> mlp.forward(Var.of(xt), false)).value().toArray();
        boolean[] heavy = new boolean[n];
        for (int i = 0; i < n; i++) {
            heavy[i] = z[i] > 0;
        }
        return item -> item >= 0 && item < n && heavy[item];
    }
}
