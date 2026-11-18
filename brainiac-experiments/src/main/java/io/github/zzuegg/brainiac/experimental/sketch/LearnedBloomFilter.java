package io.github.zzuegg.brainiac.experimental.sketch;

import io.github.zzuegg.brainiac.data.FeatureHasher;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Bloom filters over string keys, plain and learned:
 *
 * <ul>
 *   <li>{@link BloomFilter}: {@code m} bits and {@code k} hashes (double hashing over
 *       MurmurHash3), false-positive rate about {@code (1 - e^{-kn/m})^k};</li>
 *   <li>{@link #learned} (Kraska et al., SIGMOD 2018): a classifier answers yes above a
 *       threshold; the keys it misses go into a backup Bloom filter, so there are no false
 *       negatives;</li>
 *   <li>{@link #sandwiched} (Mitzenmacher, NeurIPS 2018): an initial Bloom filter over all
 *       keys in front of the learned filter, which then only sees what the first filter
 *       lets through.</li>
 * </ul>
 *
 * <p>Memory is compared in bits, the classifier's weights included (32 bits each).
 * {@link #tuneLearned} and {@link #tuneSandwiched} pick the threshold and the bit split
 * on validation non-keys.
 */
public final class LearnedBloomFilter {

    private LearnedBloomFilter() {
    }

    /** A membership filter with one-sided error. */
    public interface Filter {
        /**
         * @param key a key
         * @return {@code false} only if the key is certainly absent
         */
        boolean mightContain(String key);

        /** @return memory in bits */
        long bits();
    }

    /** Scores how likely a string is a key. */
    public interface Scorer {
        /**
         * @param key a string
         * @return a score in {@code [0, 1]}
         */
        double score(String key);

        /** @return the model's size in bits */
        long bits();
    }

    /** A standard Bloom filter. */
    public static final class BloomFilter implements Filter {
        private final long[] words;
        private final long m;
        private final int k;

        /**
         * Creates an empty filter.
         *
         * @param bits its size {@code m} (at least 1)
         * @param hashes its hash count {@code k}
         */
        public BloomFilter(long bits, int hashes) {
            this.m = Math.max(1, bits);
            this.k = Math.max(1, hashes);
            this.words = new long[(int) ((m + 63) / 64)];
        }

        /**
         * A filter with the optimal hash count {@code (m / n) ln 2} for {@code n} keys.
         *
         * @param bits its size
         * @param keys the keys to hold
         * @return the filled filter
         */
        public static BloomFilter of(long bits, List<String> keys) {
            int k = (int) Math.max(1, Math.round(bits / (double) Math.max(1, keys.size()) * Math.log(2)));
            BloomFilter f = new BloomFilter(bits, Math.min(k, 30));
            keys.forEach(f::add);
            return f;
        }

        /**
         * Adds a key.
         *
         * @param key the key
         */
        public void add(String key) {
            byte[] b = key.getBytes(StandardCharsets.UTF_8);
            long h1 = FeatureHasher.murmur3(b, 0) & 0xffffffffL;
            long h2 = FeatureHasher.murmur3(b, 0x9747b28c) & 0xffffffffL;
            for (int i = 0; i < k; i++) {
                long bit = Math.floorMod(h1 + i * h2, m);
                words[(int) (bit >>> 6)] |= 1L << (bit & 63);
            }
        }

        @Override
        public boolean mightContain(String key) {
            byte[] b = key.getBytes(StandardCharsets.UTF_8);
            long h1 = FeatureHasher.murmur3(b, 0) & 0xffffffffL;
            long h2 = FeatureHasher.murmur3(b, 0x9747b28c) & 0xffffffffL;
            for (int i = 0; i < k; i++) {
                long bit = Math.floorMod(h1 + i * h2, m);
                if ((words[(int) (bit >>> 6)] & (1L << (bit & 63))) == 0) {
                    return false;
                }
            }
            return true;
        }

        @Override
        public long bits() {
            return m;
        }
    }

    /**
     * A learned Bloom filter.
     *
     * @param keys the keys
     * @param scorer the classifier
     * @param threshold scores at or above it answer yes
     * @param backupBits bits of the backup filter for the classifier's false negatives
     * @return the filter
     */
    public static Filter learned(List<String> keys, Scorer scorer, double threshold, long backupBits) {
        List<String> missed = new ArrayList<>();
        for (String key : keys) {
            if (scorer.score(key) < threshold) {
                missed.add(key);
            }
        }
        BloomFilter backup = BloomFilter.of(Math.max(1, backupBits), missed);
        boolean empty = missed.isEmpty();
        return new Filter() {
            @Override
            public boolean mightContain(String key) {
                return scorer.score(key) >= threshold || !empty && backup.mightContain(key);
            }

            @Override
            public long bits() {
                return scorer.bits() + backup.bits();
            }
        };
    }

    /**
     * A sandwiched learned Bloom filter.
     *
     * @param keys the keys
     * @param scorer the classifier
     * @param threshold scores at or above it answer yes
     * @param firstBits bits of the initial filter
     * @param backupBits bits of the backup filter
     * @return the filter
     */
    public static Filter sandwiched(List<String> keys, Scorer scorer, double threshold, long firstBits,
            long backupBits) {
        BloomFilter first = BloomFilter.of(Math.max(1, firstBits), keys);
        Filter inner = learned(keys, scorer, threshold, backupBits);
        return new Filter() {
            @Override
            public boolean mightContain(String key) {
                return first.mightContain(key) && inner.mightContain(key);
            }

            @Override
            public long bits() {
                return first.bits() + inner.bits();
            }
        };
    }

    /**
     * The false-positive rate of a filter.
     *
     * @param f the filter
     * @param negatives strings that are not keys
     * @return the share it accepts
     */
    public static double falsePositiveRate(Filter f, List<String> negatives) {
        long fp = 0;
        for (String s : negatives) {
            fp += f.mightContain(s) ? 1 : 0;
        }
        return fp / (double) Math.max(1, negatives.size());
    }

    private static double[] thresholds(List<String> keys, Scorer scorer) {
        double[] s = keys.stream().mapToDouble(scorer::score).sorted().toArray();
        double[] q = {0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7};
        double[] t = new double[q.length + 1];
        for (int i = 0; i < q.length; i++) {
            t[i] = s[(int) Math.min(s.length - 1, q[i] * s.length)];
        }
        t[q.length] = 1.01;
        return t;
    }

    /**
     * The learned filter with the threshold that minimises validation false positives at a
     * total bit budget.
     *
     * @param keys the keys
     * @param scorer the classifier
     * @param totalBits the budget, the classifier included
     * @param validation non-keys for tuning
     * @return the best filter (a plain Bloom filter if the classifier alone exceeds the budget)
     */
    public static Filter tuneLearned(List<String> keys, Scorer scorer, long totalBits, List<String> validation) {
        long backup = totalBits - scorer.bits();
        if (backup <= 0) {
            return BloomFilter.of(totalBits, keys);
        }
        Filter best = null;
        double bestFpr = 2;
        for (double t : thresholds(keys, scorer)) {
            Filter f = learned(keys, scorer, t, backup);
            double fpr = falsePositiveRate(f, validation);
            if (fpr < bestFpr) {
                bestFpr = fpr;
                best = f;
            }
        }
        return best;
    }

    /**
     * The sandwiched filter with the threshold and first/backup split that minimise
     * validation false positives at a total bit budget.
     *
     * @param keys the keys
     * @param scorer the classifier
     * @param totalBits the budget, the classifier included
     * @param validation non-keys for tuning
     * @return the best filter
     */
    public static Filter tuneSandwiched(List<String> keys, Scorer scorer, long totalBits, List<String> validation) {
        long rest = totalBits - scorer.bits();
        if (rest <= 0) {
            return BloomFilter.of(totalBits, keys);
        }
        Filter best = null;
        double bestFpr = 2;
        for (double share : new double[] {0.1, 0.25, 0.5, 0.75, 0.9}) {
            long first = (long) (rest * share);
            for (double t : thresholds(keys, scorer)) {
                Filter f = sandwiched(keys, scorer, t, first, rest - first);
                double fpr = falsePositiveRate(f, validation);
                if (fpr < bestFpr) {
                    bestFpr = fpr;
                    best = f;
                }
            }
        }
        return best;
    }

    /**
     * Logistic regression on hashed character n-grams or tokens (the hashing trick of
     * {@link FeatureHasher}), trained by SGD.
     */
    public static final class NgramClassifier implements Scorer {
        private final FeatureHasher hasher;
        private final double[] w;
        private final int n;
        private double bias;

        /**
         * Creates an untrained classifier.
         *
         * @param columns hashed features (32 bits each)
         * @param n the n-gram length, or 0 for whole alphanumeric tokens
         */
        public NgramClassifier(int columns, int n) {
            this.hasher = new FeatureHasher(columns, false);
            this.w = new double[columns];
            this.n = n;
        }

        private int[] features(String s) {
            if (n <= 0) {
                String[] tokens = s.split("[^A-Za-z0-9]+");
                int[] f = new int[tokens.length];
                for (int i = 0; i < tokens.length; i++) {
                    f[i] = hasher.column(tokens[i]);
                }
                return f;
            }
            String p = "^" + s + "$";
            int count = Math.max(1, p.length() - n + 1);
            int[] f = new int[count];
            for (int i = 0; i < count; i++) {
                f[i] = hasher.column(p.substring(i, Math.min(p.length(), i + n)));
            }
            return f;
        }

        /**
         * Trains by SGD on the log loss.
         *
         * @param positives keys
         * @param negatives non-keys
         * @param epochs passes
         * @param rate the step size
         * @param seed shuffling
         * @return this
         */
        public NgramClassifier train(List<String> positives, List<String> negatives, int epochs, double rate,
                long seed) {
            List<int[]> xs = new ArrayList<>();
            List<Integer> ys = new ArrayList<>();
            positives.forEach(s -> {
                xs.add(features(s));
                ys.add(1);
            });
            negatives.forEach(s -> {
                xs.add(features(s));
                ys.add(0);
            });
            int[] order = new int[xs.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            SplittableRandom r = new SplittableRandom(seed);
            for (int e = 0; e < epochs; e++) {
                for (int i = order.length - 1; i > 0; i--) {
                    int j = r.nextInt(i + 1);
                    int t = order[i];
                    order[i] = order[j];
                    order[j] = t;
                }
                for (int i : order) {
                    int[] f = xs.get(i);
                    double z = bias;
                    for (int c : f) {
                        z += w[c];
                    }
                    double g = 1 / (1 + Math.exp(-z)) - ys.get(i);
                    bias -= rate * g;
                    for (int c : f) {
                        w[c] -= rate * g;
                    }
                }
            }
            return this;
        }

        @Override
        public double score(String key) {
            double z = bias;
            for (int c : features(key)) {
                z += w[c];
            }
            return 1 / (1 + Math.exp(-z));
        }

        @Override
        public long bits() {
            return 32L * (w.length + 1);
        }
    }
}
