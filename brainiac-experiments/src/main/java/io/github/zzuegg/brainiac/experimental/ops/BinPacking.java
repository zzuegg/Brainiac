package io.github.zzuegg.brainiac.experimental.ops;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Online bin packing: items arrive one at a time and must be placed immediately into bins of a
 * fixed capacity; the cost is the number of bins opened.
 *
 * <ul>
 *   <li>{@link Method#NEXT_FIT}, {@link Method#FIRST_FIT}, {@link Method#BEST_FIT} (Johnson et al.,
 *       SIAM J. Comput. 1974; First and Best Fit are 1.7-competitive) and {@link Method#HARMONIC}
 *       (Lee &amp; Lee, JACM 1985: items of size in {@code (C/(k+1), C/k]} share bins only with
 *       their own class);</li>
 *   <li>{@link Method#PROFILE_PACKING} (Angelopoulos, Kamali &amp; Shadkami, AAAI 2022,
 *       <a href="https://arxiv.org/abs/2102.03311">arXiv:2102.03311</a>): from predicted size
 *       frequencies build a profile of {@code m} items, pack it offline (First Fit Decreasing), and
 *       fill copies of that packing's placeholders online (an item whose size has no free
 *       placeholder takes the smallest larger free one before a new copy is opened); sizes absent
 *       from the prediction go to First Fit;</li>
 *   <li>{@link Method#HYBRID}: the robust variant: a share {@code lambda} of each size's items goes to
 *       profile packing and the rest to First Fit (separate bins), which keeps a worst-case guarantee
 *       when the predictions are wrong;</li>
 *   <li>{@link #firstFitDecreasing} and {@link #lowerBound}: the offline references;</li>
 *   <li>{@link RlPacker}: a placement policy learned with core {@link Dqn}: from the item size and a
 *       histogram of the open bins' residual capacities it chooses Best, Worst or First Fit, or a
 *       fresh bin, with reward -1 per opened bin.</li>
 * </ul>
 */
public final class BinPacking {

    private BinPacking() {
    }

    /** The online algorithm. */
    public enum Method {
        /** keep one open bin */
        NEXT_FIT,
        /** the first bin with room */
        FIRST_FIT,
        /** the fullest bin with room */
        BEST_FIT,
        /** size classes with their own bins */
        HARMONIC,
        /** placeholders of an offline packing of the predicted profile */
        PROFILE_PACKING,
        /** profile packing for a share of the items, First Fit for the rest */
        HYBRID
    }

    /**
     * Packs a stream.
     *
     * @param method the algorithm
     * @param items the sizes in arrival order
     * @param capacity the bin capacity
     * @param predicted predicted frequencies over {@link OperationsSims#DISCRETE_SIZES} (profile methods)
     * @param lambda the profile share of {@link Method#HYBRID}
     * @return bins used
     */
    public static int pack(Method method, int[] items, int capacity, double[] predicted, double lambda) {
        Objects.requireNonNull(method, "method");
        return switch (method) {
            case NEXT_FIT -> nextFit(items, capacity);
            case FIRST_FIT -> anyFit(items, capacity, false);
            case BEST_FIT -> anyFit(items, capacity, true);
            case HARMONIC -> harmonic(items, capacity, 6);
            case PROFILE_PACKING -> profile(items, capacity, predicted, 1.0);
            case HYBRID -> profile(items, capacity, predicted, lambda);
        };
    }

    private static int nextFit(int[] items, int capacity) {
        int bins = 0;
        int room = 0;
        for (int x : items) {
            if (x > room) {
                bins++;
                room = capacity;
            }
            room -= x;
        }
        return bins;
    }

    private static int anyFit(int[] items, int capacity, boolean best) {
        List<int[]> bins = new ArrayList<>();
        for (int x : items) {
            place(bins, x, capacity, best);
        }
        return bins.size();
    }

    private static void place(List<int[]> bins, int x, int capacity, boolean best) {
        int chosen = -1;
        for (int b = 0; b < bins.size(); b++) {
            int room = bins.get(b)[0];
            if (room >= x && (chosen < 0 || (best && room < bins.get(chosen)[0]))) {
                chosen = b;
                if (!best) {
                    break;
                }
            }
        }
        if (chosen < 0) {
            bins.add(new int[] {capacity - x});
        } else {
            bins.get(chosen)[0] -= x;
        }
    }

    private static int harmonic(int[] items, int capacity, int classes) {
        int[] inOpen = new int[classes + 1];
        int smallRoom = 0;
        int bins = 0;
        for (int x : items) {
            int k = Math.min(classes, Math.max(1, capacity / x));
            if (k < classes) {
                // class k: sizes in (C/(k+1), C/k], exactly k per bin
                if (inOpen[k] == 0 || inOpen[k] == k) {
                    bins++;
                    inOpen[k] = 0;
                }
                inOpen[k]++;
            } else {
                if (smallRoom < x) {
                    bins++;
                    smallRoom = capacity;
                }
                smallRoom -= x;
            }
        }
        return bins;
    }

    /**
     * First Fit Decreasing.
     *
     * @param items sizes
     * @param capacity capacity
     * @return bins used
     */
    public static int firstFitDecreasing(int[] items, int capacity) {
        int[] sorted = items.clone();
        Arrays.sort(sorted);
        List<int[]> bins = new ArrayList<>();
        for (int i = sorted.length - 1; i >= 0; i--) {
            place(bins, sorted[i], capacity, false);
        }
        return bins.size();
    }

    /**
     * @param items sizes
     * @param capacity capacity
     * @return {@code ceil(sum / capacity)}
     */
    public static int lowerBound(int[] items, int capacity) {
        long s = 0;
        for (int x : items) {
            s += x;
        }
        return (int) ((s + capacity - 1) / capacity);
    }

    /** The profile: an offline FFD packing of {@code m} items in the predicted proportions. */
    private static List<List<Integer>> profilePacking(int capacity, double[] predicted, int m) {
        List<Integer> profileItems = new ArrayList<>();
        for (int k = 0; k < predicted.length; k++) {
            int count = (int) Math.ceil(predicted[k] * m - 1e-9);
            int size = Math.max(1, OperationsSims.DISCRETE_SIZES[k] * capacity / 100);
            for (int c = 0; c < count; c++) {
                profileItems.add(size);
            }
        }
        profileItems.sort((a, b) -> b - a);
        List<List<Integer>> bins = new ArrayList<>();
        List<Integer> room = new ArrayList<>();
        for (int x : profileItems) {
            int chosen = -1;
            for (int b = 0; b < bins.size(); b++) {
                if (room.get(b) >= x) {
                    chosen = b;
                    break;
                }
            }
            if (chosen < 0) {
                bins.add(new ArrayList<>());
                room.add(capacity);
                chosen = bins.size() - 1;
            }
            bins.get(chosen).add(x);
            room.set(chosen, room.get(chosen) - x);
        }
        return bins;
    }

    private static int profile(int[] items, int capacity, double[] predicted, double lambda) {
        List<List<Integer>> profile = profilePacking(capacity, predicted, 60);
        // open placeholders by size: list of (bin id) with a free slot of that size
        Map<Integer, List<Integer>> free = new HashMap<>();
        List<Boolean> used = new ArrayList<>();
        List<int[]> ffBins = new ArrayList<>();
        Map<Integer, int[]> counts = new HashMap<>();
        java.util.Set<Integer> profileSizes = new java.util.HashSet<>();
        for (List<Integer> bin : profile) {
            profileSizes.addAll(bin);
        }
        for (int x : items) {
            int[] c = counts.computeIfAbsent(x, key -> new int[2]);
            c[0]++;
            boolean toProfile = profileSizes.contains(x) && c[1] < lambda * c[0] - 1e-9;
            if (!toProfile) {
                place(ffBins, x, capacity, false);
                continue;
            }
            c[1]++;
            List<Integer> slots = free.get(x);
            if (slots == null || slots.isEmpty()) {
                // the smallest larger placeholder already open, before opening a new profile copy
                int fit = Integer.MAX_VALUE;
                for (Map.Entry<Integer, List<Integer>> e : free.entrySet()) {
                    if (e.getKey() > x && e.getKey() < fit && !e.getValue().isEmpty()) {
                        fit = e.getKey();
                    }
                }
                if (fit != Integer.MAX_VALUE) {
                    slots = free.get(fit);
                }
            }
            if (slots == null || slots.isEmpty()) {
                int base = used.size();
                for (int b = 0; b < profile.size(); b++) {
                    used.add(false);
                    for (int s : profile.get(b)) {
                        free.computeIfAbsent(s, key -> new ArrayList<>()).add(base + b);
                    }
                }
                slots = free.get(x);
            }
            int bin = slots.removeFirst();
            used.set(bin, true);
        }
        int bins = ffBins.size();
        for (boolean u : used) {
            bins += u ? 1 : 0;
        }
        return bins;
    }

    // ------------------------------------------------------------------ RL packer

    /** A DQN that picks the placement rule per item. */
    public static final class RlPacker {
        private static final int BUCKETS = 10;
        private final Dqn dqn;
        private final int capacity;

        private RlPacker(int capacity, long seed) {
            this.capacity = capacity;
            this.dqn = Dqn.of(() -> Nets.mlp(BUCKETS + 2, 4, seed, 64, 64), 4).optimizer(Optimizer.adam(1e-3f))
                    .learningStarts(300).batchSize(32).targetUpdateEvery(200).discount(0.99f)
                    .epsilon(Schedule.linear(1, 0.05, 3000)).seed(seed);
        }

        /**
         * @param capacity the bin capacity
         * @param seed the seed
         * @return an untrained packer
         */
        public static RlPacker of(int capacity, long seed) {
            return new RlPacker(capacity, seed);
        }

        /**
         * Trains on item streams.
         *
         * @param streams the training streams, cycled through episode by episode
         * @param steps environment steps (items)
         * @return this packer
         */
        public RlPacker train(List<int[]> streams, long steps) {
            dqn.fit(new Env(streams, capacity), steps);
            return this;
        }

        /**
         * Packs one stream greedily.
         *
         * @param items sizes
         * @return bins used
         */
        public int pack(int[] items) {
            Env env = new Env(List.of(items), capacity);
            float[] obs = env.reset(new SplittableRandom(0));
            Environment.Step s;
            do {
                s = env.step(dqn.act(obs, false));
                obs = s.observation();
            } while (!s.done());
            return env.bins.size();
        }

        /** The packing environment. */
        private static final class Env implements Environment {
            private final List<int[]> streams;
            private final int capacity;
            private int stream = -1;
            private int index;
            private final List<int[]> bins = new ArrayList<>();

            Env(List<int[]> streams, int capacity) {
                this.streams = streams;
                this.capacity = capacity;
            }

            @Override
            public Space observationSpace() {
                return Space.box(BUCKETS + 2, 0, 1);
            }

            @Override
            public Space actionSpace() {
                return Space.discrete(4);
            }

            private float[] obs() {
                float[] o = new float[BUCKETS + 2];
                int x = streams.get(stream)[index];
                o[0] = x / (float) capacity;
                int fits = 0;
                for (int[] b : bins) {
                    int bucket = Math.min(BUCKETS - 1, b[0] * BUCKETS / (capacity + 1));
                    o[1 + bucket] += 1;
                    fits += b[0] >= x ? 1 : 0;
                }
                for (int k = 0; k < BUCKETS; k++) {
                    o[1 + k] /= Math.max(1, bins.size());
                }
                o[BUCKETS + 1] = bins.isEmpty() ? 0 : fits / (float) bins.size();
                return o;
            }

            @Override
            public float[] reset(SplittableRandom random) {
                stream = (stream + 1) % streams.size();
                index = 0;
                bins.clear();
                return obs();
            }

            @Override
            public Step step(float[] action) {
                int a = Math.round(action[0]);
                int x = streams.get(stream)[index];
                int before = bins.size();
                int chosen = -1;
                if (a != 3) {
                    for (int b = 0; b < bins.size(); b++) {
                        int room = bins.get(b)[0];
                        if (room < x) {
                            continue;
                        }
                        if (chosen < 0 || (a == 0 && room < bins.get(chosen)[0])
                                || (a == 1 && room > bins.get(chosen)[0])) {
                            chosen = b;
                            if (a == 2) {
                                break;
                            }
                        }
                    }
                }
                if (chosen < 0) {
                    bins.add(new int[] {capacity - x});
                } else {
                    bins.get(chosen)[0] -= x;
                }
                index++;
                boolean done = index >= streams.get(stream).length;
                float reward = bins.size() > before ? -1 : 0;
                if (done) {
                    index--;
                }
                return new Step(obs(), reward, false, done);
            }
        }
    }
}
