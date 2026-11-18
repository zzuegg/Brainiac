package io.github.zzuegg.brainiac.experimental.sketch;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.sketch.ClassicCaches.PageCache;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Caching with predictions: online paging rules that use a predictor of each page's next
 * request, next to the classical rules in {@link ClassicCaches} (LRU, LFU, Belady).
 *
 * <ul>
 *   <li>{@link #marking}: randomised marking, {@code O(log k)}-competitive;</li>
 *   <li>{@link #followPredictions}: evict the page predicted to be requested furthest in the
 *       future (Belady on predictions; no worst-case guarantee);</li>
 *   <li>{@link #predictiveMarking}: Lykouris and Vassilvitskii's predictive marker (ICML 2018):
 *       marking phases, the victim among the unmarked pages chosen by prediction, but a chain
 *       of evictions that keeps coming back within a phase switches to random unmarked
 *       victims after {@code H_k} links, which keeps the {@code O(log k)} bound when
 *       predictions are bad;</li>
 *   <li>{@link #parrot}: a Parrot-style policy (Liu et al., ICML 2020) that scores cached
 *       pages from their access history and is trained by imitation of Belady's choices,
 *       on states its own policy visits (DAgger).</li>
 * </ul>
 *
 * <p>Predictors are {@link #oracle} (the true next request with multiplicative lognormal
 * noise, the noise knob) and {@link #learnedGaps} (an online per-page mean of
 * inter-arrival gaps).
 */
public final class CachingWithPredictions {

    private CachingWithPredictions() {
    }

    /** Predicts the time of a page's next request, asked when the page is requested. */
    public interface NextArrivalPredictor {
        /**
         * @param page the page requested now
         * @param t the current request index
         * @return the predicted index of its next request
         */
        double predict(int page, int t);
    }

    /**
     * The true next request, distorted by a lognormal factor on the gap.
     *
     * @param trace the trace that will be replayed
     * @param noise the standard deviation of the log factor (0 is exact)
     * @param seed the seed
     * @return the predictor
     */
    public static NextArrivalPredictor oracle(int[] trace, double noise, long seed) {
        int[] next = ClassicCaches.nextUse(trace);
        double[] pred = new double[trace.length];
        SplittableRandom r = new SplittableRandom(seed);
        for (int t = 0; t < trace.length; t++) {
            double gap = next[t] == Integer.MAX_VALUE ? 2.0 * trace.length : next[t] - t;
            pred[t] = t + gap * Math.exp(noise * r.nextGaussian());
        }
        return (page, t) -> pred[t];
    }

    /**
     * An online predictor: the page's last request plus an exponential moving average of its
     * inter-arrival gaps (the global mean gap for a page seen once).
     *
     * @param decay the weight of the newest gap
     * @return the predictor
     */
    public static NextArrivalPredictor learnedGaps(double decay) {
        Map<Integer, Integer> last = new HashMap<>();
        Map<Integer, Double> mean = new HashMap<>();
        double[] global = {0, 0};
        return (page, t) -> {
            Integer l = last.put(page, t);
            if (l != null) {
                double g = t - l;
                mean.merge(page, g, (old, x) -> (1 - decay) * old + decay * x);
                global[0] += g;
                global[1]++;
            }
            Double m = mean.get(page);
            return t + (m != null ? m : global[1] > 0 ? 4 * global[0] / global[1] : 1e9);
        };
    }

    /** Marking phases shared by the marking rules. */
    private abstract static class Marking implements PageCache {
        final int capacity;
        final Set<Integer> cached = new HashSet<>();
        final Set<Integer> marked = new HashSet<>();
        final SplittableRandom random;

        Marking(int capacity, long seed) {
            this.capacity = capacity;
            this.random = new SplittableRandom(seed);
        }

        abstract int victim(List<Integer> unmarked, int page, int t);

        void requested(int page, int t, boolean hit) {
        }

        void newPhase() {
        }

        @Override
        public boolean request(int page, int t) {
            boolean hit = cached.contains(page);
            if (!hit) {
                if (cached.size() >= capacity) {
                    if (marked.size() >= capacity) {
                        marked.clear();
                        newPhase();
                    }
                    List<Integer> unmarked = new ArrayList<>();
                    for (int p : cached) {
                        if (!marked.contains(p)) {
                            unmarked.add(p);
                        }
                    }
                    unmarked.sort(null);
                    int v = victim(unmarked, page, t);
                    cached.remove(v);
                }
                cached.add(page);
            }
            marked.add(page);
            requested(page, t, hit);
            return hit;
        }

        @Override
        public int size() {
            return cached.size();
        }
    }

    /**
     * Randomised marking.
     *
     * @param capacity pages held
     * @param seed the seed
     * @return the cache
     */
    public static PageCache marking(int capacity, long seed) {
        return new Marking(capacity, seed) {
            @Override
            int victim(List<Integer> unmarked, int page, int t) {
                return unmarked.get(random.nextInt(unmarked.size()));
            }
        };
    }

    /**
     * Evicts the page whose predicted next request is furthest away.
     *
     * @param capacity pages held
     * @param predictor the predictor
     * @return the cache
     */
    public static PageCache followPredictions(int capacity, NextArrivalPredictor predictor) {
        return new ClassicCaches.ScoredCache(capacity) {
            private final Map<Integer, Double> predicted = new HashMap<>();

            @Override
            void touched(int page, int t, boolean hit) {
                predicted.put(page, predictor.predict(page, t));
            }

            @Override
            int victim(int t) {
                int best = pages.get(0);
                for (int p : pages) {
                    if (predicted.get(p) > predicted.get(best)) {
                        best = p;
                    }
                }
                return best;
            }
        };
    }

    /**
     * Lykouris and Vassilvitskii's predictive marker.
     *
     * @param capacity pages held {@code k}
     * @param predictor the predictor
     * @param seed the seed for the random fallback
     * @return the cache
     */
    public static PageCache predictiveMarking(int capacity, NextArrivalPredictor predictor, long seed) {
        double harmonic = 0;
        for (int i = 1; i <= capacity; i++) {
            harmonic += 1.0 / i;
        }
        double hk = harmonic;
        return new Marking(capacity, seed) {
            private final Map<Integer, Double> predicted = new HashMap<>();
            private final Map<Integer, Integer> chainOf = new HashMap<>();
            private final List<Integer> chainLength = new ArrayList<>();

            @Override
            void newPhase() {
                chainOf.clear();
                chainLength.clear();
            }

            @Override
            int victim(List<Integer> unmarked, int page, int t) {
                int chain;
                Integer c = chainOf.remove(page);
                if (c != null) {
                    chain = c;
                    chainLength.set(chain, chainLength.get(chain) + 1);
                } else {
                    chain = chainLength.size();
                    chainLength.add(1);
                }
                int v;
                if (chainLength.get(chain) <= hk) {
                    v = unmarked.get(0);
                    for (int p : unmarked) {
                        if (predicted.getOrDefault(p, 0.0) > predicted.getOrDefault(v, 0.0)) {
                            v = p;
                        }
                    }
                } else {
                    v = unmarked.get(random.nextInt(unmarked.size()));
                }
                chainOf.put(v, chain);
                return v;
            }

            @Override
            void requested(int page, int t, boolean hit) {
                predicted.put(page, predictor.predict(page, t));
            }
        };
    }

    /** Access-history features of a cached page, for the imitation policy. */
    static final int FEATURES = 5;

    /** Per-page access statistics. */
    private static final class Stats {
        int last;
        int count;
        double meanGap;
        int lastGap;

        void touch(int t) {
            if (count > 0) {
                lastGap = t - last;
                meanGap = count == 1 ? lastGap : 0.8 * meanGap + 0.2 * lastGap;
            }
            last = t;
            count++;
        }

        void features(int t, float[] into, int offset) {
            double age = t - last;
            into[offset] = (float) Math.log1p(age);
            into[offset + 1] = (float) Math.log1p(count);
            into[offset + 2] = (float) Math.log1p(meanGap);
            into[offset + 3] = (float) Math.log1p(lastGap);
            into[offset + 4] = (float) Math.min(10, count > 1 ? age / Math.max(1, meanGap) : 10);
        }
    }

    /** A Parrot-style learned eviction policy. */
    public static final class Parrot {
        private final Model model;

        /**
         * Creates an untrained policy.
         *
         * @param hidden the scorer's hidden width
         * @param seed the seed
         */
        public Parrot(int hidden, long seed) {
            this.model = Model.sequential().input(FEATURES).dense(hidden, Activation.RELU).dense(1).seed(seed).build();
        }

        /**
         * DAgger-style imitation of Belady: replay the training trace with the expert in the
         * first round and the learned policy afterwards, record every eviction's cached pages
         * with Belady's victim as the label, and fit a softmax over the cached pages.
         *
         * @param trace the training trace
         * @param capacity pages held
         * @param rounds data-collection rounds
         * @param epochs passes over the collected data per round
         * @param seed the seed
         * @return the last round's agreement with Belady on the visited evictions
         */
        public double train(int[] trace, int capacity, int rounds, int epochs, long seed) {
            int[] next = ClassicCaches.nextUse(trace);
            List<float[]> states = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            double agree = 0;
            for (int round = 0; round < rounds; round++) {
                boolean expert = round == 0;
                Map<Integer, Stats> stats = new HashMap<>();
                Map<Integer, Integer> nextOf = new HashMap<>();
                List<Integer> cached = new ArrayList<>();
                int same = 0;
                int evictions = 0;
                for (int t = 0; t < trace.length; t++) {
                    int page = trace[t];
                    if (!nextOf.containsKey(page)) {
                        if (cached.size() >= capacity) {
                            float[] x = new float[capacity * FEATURES];
                            int best = 0;
                            for (int i = 0; i < capacity; i++) {
                                stats.get(cached.get(i)).features(t, x, i * FEATURES);
                                if (nextOf.get(cached.get(i)) > nextOf.get(cached.get(best))) {
                                    best = i;
                                }
                            }
                            states.add(x);
                            labels.add(best);
                            int pick = expert ? best : argmax(scores(x, capacity));
                            same += pick == best ? 1 : 0;
                            evictions++;
                            nextOf.remove(cached.remove(pick));
                        }
                        cached.add(page);
                    }
                    nextOf.put(page, next[t]);
                    stats.computeIfAbsent(page, p -> new Stats()).touch(t);
                }
                agree = same / (double) Math.max(1, evictions);
                fit(states, labels, capacity, epochs, opt, r);
            }
            return agree;
        }

        private void fit(List<float[]> states, List<Integer> labels, int capacity, int epochs, Optimizer opt,
                SplittableRandom r) {
            int n = states.size();
            int batch = 64;
            int[] order = new int[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            for (int e = 0; e < epochs; e++) {
                for (int i = n - 1; i > 0; i--) {
                    int j = r.nextInt(i + 1);
                    int t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
                for (int from = 0; from < n; from += batch) {
                    int b = Math.min(batch, n - from);
                    float[] x = new float[b * capacity * FEATURES];
                    float[] y = new float[b];
                    for (int i = 0; i < b; i++) {
                        System.arraycopy(states.get(order[from + i]), 0, x, i * capacity * FEATURES, capacity * FEATURES);
                        y[i] = labels.get(order[from + i]);
                    }
                    Var logits = model.forward(Var.of(Tensor.of(Shape.of(b * capacity, FEATURES), x)), true)
                            .reshape(b, capacity);
                    Functional.crossEntropy(logits, Tensor.of(Shape.of(b), y)).backward();
                    opt.step(model.parameters());
                }
            }
        }

        private float[] scores(float[] x, int count) {
            return Var.noGrad(() -> model.forward(Var.of(Tensor.of(Shape.of(count, FEATURES), x)), false)).value()
                    .toArray();
        }

        private static int argmax(float[] s) {
            int b = 0;
            for (int i = 1; i < s.length; i++) {
                b = s[i] > s[b] ? i : b;
            }
            return b;
        }
    }

    /**
     * A cache run by a trained {@link Parrot} policy.
     *
     * @param capacity pages held
     * @param policy the policy
     * @return the cache
     */
    public static PageCache parrot(int capacity, Parrot policy) {
        return new ClassicCaches.ScoredCache(capacity) {
            private final Map<Integer, Stats> stats = new HashMap<>();

            @Override
            void touched(int page, int t, boolean hit) {
                stats.computeIfAbsent(page, p -> new Stats()).touch(t);
            }

            @Override
            int victim(int t) {
                float[] x = new float[pages.size() * FEATURES];
                for (int i = 0; i < pages.size(); i++) {
                    stats.get(pages.get(i)).features(t, x, i * FEATURES);
                }
                return pages.get(Parrot.argmax(policy.scores(x, pages.size())));
            }
        };
    }
}
