package io.github.zzuegg.brainiac.experiments.algorithms;

import io.github.zzuegg.brainiac.experimental.data.KeyStreams;
import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus;
import io.github.zzuegg.brainiac.experimental.sketch.LearnedBloomFilter;
import io.github.zzuegg.brainiac.experimental.sketch.LearnedIndex;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Learned index structures and learned Bloom filters (item 11.2.16): RMI and error-bounded
 * piecewise-linear indexes against binary search and a B-tree-like index on sorted
 * {@link KeyStreams} key sets, and plain, learned and sandwiched Bloom filters on URL-like
 * strings from {@link SyntheticCorpus} documents with a planted "malicious" class.
 */
public final class LearnedStructures {

    private LearnedStructures() {
    }

    /**
     * An index's cost.
     *
     * @param method the index
     * @param nanos mean lookup time
     * @param comparisons mean key comparisons per lookup
     * @param bytes the index's own size
     * @param exact whether every key was found at its position
     */
    public record IndexRow(String method, double nanos, double comparisons, long bytes, boolean exact) {
        @Override
        public String toString() {
            return String.format("%s %.0f ns, %.1f cmp, %d B%s", method, nanos, comparisons, bytes,
                    exact ? "" : " (WRONG)");
        }
    }

    /**
     * Builds every index over a key set and times random lookups of existing keys.
     *
     * @param n keys
     * @param distribution the key distribution
     * @param lookups lookups to time
     * @param seed the seed
     * @return one row per index
     */
    public static List<IndexRow> indexes(int n, KeyStreams.KeyDistribution distribution, int lookups, long seed) {
        long[] keys = KeyStreams.sortedKeys(n, distribution, seed);
        Map<String, LearnedIndex.Index> all = new LinkedHashMap<>();
        all.put("binary search", LearnedIndex.binarySearch(keys));
        all.put("B-tree fan-out 64", LearnedIndex.bTree(keys, 64));
        all.put("RMI linear 1024", LearnedIndex.rmi(keys, 1024, LearnedIndex.Leaf.LINEAR));
        all.put("RMI cubic 1024", LearnedIndex.rmi(keys, 1024, LearnedIndex.Leaf.CUBIC));
        all.put("PLA eps 8", LearnedIndex.pla(keys, 8));
        all.put("PLA eps 64", LearnedIndex.pla(keys, 64));
        SplittableRandom r = new SplittableRandom(seed + 1);
        long[] probe = new long[lookups];
        for (int i = 0; i < lookups; i++) {
            probe[i] = keys[r.nextInt(n)];
        }
        List<IndexRow> rows = new ArrayList<>();
        for (Map.Entry<String, LearnedIndex.Index> e : all.entrySet()) {
            LearnedIndex.Index idx = e.getValue();
            boolean exact = LearnedIndex.exact(idx, keys);
            long before = idx.comparisons();
            long sink = 0;
            for (int rep = 0; rep < 2; rep++) {
                for (long k : probe) {
                    sink += idx.lookup(k);
                }
            }
            long t0 = System.nanoTime();
            for (long k : probe) {
                sink += idx.lookup(k);
            }
            long ns = System.nanoTime() - t0;
            double cmp = (idx.comparisons() - before) / (3.0 * lookups);
            rows.add(new IndexRow(e.getKey() + (sink == Long.MIN_VALUE ? "!" : ""), ns / (double) lookups, cmp,
                    idx.bytes(), exact));
        }
        return rows;
    }

    /**
     * URL-like strings: one per {@link SyntheticCorpus} document (30 aspect words per topic),
     * its tokens in base 36 as path segments; documents of every fourth topic are the
     * "malicious" keys.
     *
     * @param topics corpus topics
     * @param docsPerTopic documents per topic (a multiple of 3)
     * @param seed the seed
     * @return {@code {keys, non-keys}} as two lists
     */
    public static List<List<String>> urls(int topics, int docsPerTopic, long seed) {
        SyntheticCorpus c = SyntheticCorpus.generate(topics, 30, docsPerTopic, 0.5, seed);
        List<String> keys = new ArrayList<>();
        List<String> others = new ArrayList<>();
        for (int d = 0; d < c.size(); d++) {
            StringBuilder s = new StringBuilder("http://");
            int[] doc = c.document(d);
            for (int i = 0; i < doc.length; i++) {
                s.append(i == 0 ? "" : i == 1 ? ".com/" : "/").append(Integer.toString(doc[i], 36));
            }
            (c.topic(d) % 4 == 0 ? keys : others).add(s.toString());
        }
        return List.of(keys, others);
    }

    /**
     * Bloom filters at equal memory: the plain filter, the learned filter and the sandwiched
     * filter (thresholds and splits tuned on validation non-keys), reporting test
     * false-positive rates.
     *
     * @param keys the key set
     * @param others non-keys (split into classifier training, validation and test thirds)
     * @param bitsPerKey the budgets to try
     * @param columns the classifier's hashed token features
     * @param seed the seed
     * @return per budget: {plain, learned, sandwiched, classifier-only FPR at its best threshold}
     */
    public static Map<Double, double[]> bloomFilters(List<String> keys, List<String> others, double[] bitsPerKey,
            int columns, long seed) {
        List<String> shuffled = new ArrayList<>(others);
        java.util.Collections.shuffle(shuffled, new java.util.Random(seed));
        int third = shuffled.size() / 3;
        List<String> train = shuffled.subList(0, third);
        List<String> validation = shuffled.subList(third, 2 * third);
        List<String> test = shuffled.subList(2 * third, shuffled.size());
        LearnedBloomFilter.NgramClassifier model = new LearnedBloomFilter.NgramClassifier(columns, 0)
                .train(keys, train, 3, 0.1, seed);
        Map<String, Double> cache = new java.util.HashMap<>();
        LearnedBloomFilter.Scorer clf = new LearnedBloomFilter.Scorer() {
            @Override
            public double score(String key) {
                return cache.computeIfAbsent(key, model::score);
            }

            @Override
            public long bits() {
                return model.bits();
            }
        };
        Map<Double, double[]> out = new LinkedHashMap<>();
        for (double bpk : bitsPerKey) {
            long bits = (long) (bpk * keys.size());
            LearnedBloomFilter.Filter plain = LearnedBloomFilter.BloomFilter.of(bits, keys);
            LearnedBloomFilter.Filter learned = LearnedBloomFilter.tuneLearned(keys, clf, bits, validation);
            LearnedBloomFilter.Filter sandwich = LearnedBloomFilter.tuneSandwiched(keys, clf, bits, validation);
            for (LearnedBloomFilter.Filter f : List.of(plain, learned, sandwich)) {
                for (String k : keys) {
                    if (!f.mightContain(k)) {
                        throw new IllegalStateException("false negative");
                    }
                }
            }
            double negScore = 0;
            for (String s : test) {
                negScore += clf.score(s) >= 0.5 ? 1 : 0;
            }
            out.put(bpk, new double[] {LearnedBloomFilter.falsePositiveRate(plain, test),
                LearnedBloomFilter.falsePositiveRate(learned, test), LearnedBloomFilter.falsePositiveRate(sandwich, test),
                negScore / test.size()});
        }
        return out;
    }
}
