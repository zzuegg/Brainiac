package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Turns a clean labelled data set (rows {@code x}, classes {@code y}) into the imperfect
 * supervision settings of the learning-from-imperfect-labels literature, always keeping the
 * clean labels for scoring. It never changes the input arrays or how the input was sampled, so
 * it wraps any generator ({@code ToyTasks} digits, {@link ShapeImages}, {@link TabularStressSuite})
 * without changing it:
 *
 * <ul>
 *   <li>{@link #labelledSplit}: a few labels per class, the rest unlabelled (semi-supervised).</li>
 *   <li>{@link #symmetric}, {@link #pairFlip}, {@link #instanceDependent}: label noise.</li>
 *   <li>{@link #longTail}: an exponentially imbalanced subset with a given imbalance ratio.</li>
 *   <li>{@link #positiveUnlabelled}: a PU split with a known class prior.</li>
 *   <li>{@link #labellingFunctions}: noisy voters with planted accuracies, coverages and one
 *       correlated pair (weak supervision).</li>
 *   <li>{@link #bags}: multiple-instance bags labelled positive when they hold a key class.</li>
 *   <li>{@link #candidates}: candidate-label sets for partial-label learning.</li>
 *   <li>{@link #proportionBags}: groups whose class proportions are known.</li>
 * </ul>
 */
public final class LabelCorruption {

    private LabelCorruption() {
    }

    /**
     * A semi-supervised split.
     *
     * @param labelled row indices whose labels are kept
     * @param unlabelled row indices whose labels are hidden
     */
    public record Split(int[] labelled, int[] unlabelled) {
    }

    /**
     * Keeps {@code perClass} labels of every class (the first rows of each class in a seeded
     * shuffle) and hides the rest.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param perClass labels kept per class
     * @param seed the seed
     * @return the split
     */
    public static Split labelledSplit(int[] y, int classes, int perClass, long seed) {
        int[] order = permutation(y.length, new SplittableRandom(seed));
        int[] taken = new int[classes];
        List<Integer> lab = new ArrayList<>();
        List<Integer> unl = new ArrayList<>();
        for (int i : order) {
            if (taken[y[i]] < perClass) {
                taken[y[i]]++;
                lab.add(i);
            } else {
                unl.add(i);
            }
        }
        return new Split(toArray(lab), toArray(unl));
    }

    /**
     * Corrupted labels with the clean ones kept.
     *
     * @param noisy the labels a learner sees
     * @param clean the true labels
     * @param transition the class-conditional transition matrix {@code T[i][j] = P(noisy j |
     *     clean i)} that generated them (for instance-dependent noise, its empirical average)
     */
    public record Noisy(int[] noisy, int[] clean, double[][] transition) {

        /** @return which rows were flipped */
        public boolean[] flipped() {
            boolean[] out = new boolean[clean.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = noisy[i] != clean[i];
            }
            return out;
        }

        /** @return the share of flipped rows */
        public double rate() {
            int n = 0;
            for (int i = 0; i < clean.length; i++) {
                n += noisy[i] != clean[i] ? 1 : 0;
            }
            return n / (double) clean.length;
        }
    }

    /**
     * Symmetric noise: with probability {@code rate} a label is replaced by one of the other
     * classes, uniformly.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param rate the flip probability
     * @param seed the seed
     * @return the noisy labels
     */
    public static Noisy symmetric(int[] y, int classes, double rate, long seed) {
        double[][] t = new double[classes][classes];
        for (int i = 0; i < classes; i++) {
            for (int j = 0; j < classes; j++) {
                t[i][j] = i == j ? 1 - rate : rate / (classes - 1);
            }
        }
        return fromTransition(y, t, seed);
    }

    /**
     * Pair-flip (asymmetric) noise: with probability {@code rate} class {@code c} becomes
     * {@code c + 1 mod classes}.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param rate the flip probability, below 0.5 for the majority to stay correct
     * @param seed the seed
     * @return the noisy labels
     */
    public static Noisy pairFlip(int[] y, int classes, double rate, long seed) {
        double[][] t = new double[classes][classes];
        for (int i = 0; i < classes; i++) {
            t[i][i] = 1 - rate;
            t[i][(i + 1) % classes] += rate;
        }
        return fromTransition(y, t, seed);
    }

    /**
     * Labels drawn from a given class-conditional transition matrix.
     *
     * @param y the clean classes
     * @param transition {@code T[i][j] = P(noisy j | clean i)}, rows summing to 1
     * @param seed the seed
     * @return the noisy labels
     */
    public static Noisy fromTransition(int[] y, double[][] transition, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] out = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = draw(transition[y[i]], r);
        }
        return new Noisy(out, y.clone(), copy(transition));
    }

    /**
     * Instance-dependent noise in the style of Xia et al. (2020): every row gets its own flip
     * rate (a truncated normal around {@code rate}) and its own distribution over wrong classes
     * from a random projection of its features, so look-alike rows are flipped to the same
     * wrong classes and the transition matrix is no longer shared.
     *
     * @param x the rows
     * @param y the clean classes
     * @param classes the number of classes
     * @param rate the mean flip rate
     * @param seed the seed
     * @return the noisy labels, with the empirical average transition matrix
     */
    public static Noisy instanceDependent(float[][] x, int[] y, int classes, double rate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = x[0].length;
        double[][][] w = new double[classes][d][classes];
        for (double[][] wc : w) {
            for (double[] row : wc) {
                for (int k = 0; k < classes; k++) {
                    row[k] = r.nextGaussian();
                }
            }
        }
        int[] out = new int[y.length];
        double[][] t = new double[classes][classes];
        int[] count = new int[classes];
        for (int i = 0; i < y.length; i++) {
            double q = Math.clamp(rate + 0.1 * r.nextGaussian(), 0, 0.95);
            double[] s = new double[classes];
            for (int k = 0; k < classes; k++) {
                for (int f = 0; f < d; f++) {
                    s[k] += x[i][f] * w[y[i]][f][k];
                }
            }
            s[y[i]] = Double.NEGATIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (double v : s) {
                max = Math.max(max, v);
            }
            double sum = 0;
            for (int k = 0; k < classes; k++) {
                s[k] = k == y[i] ? 0 : Math.exp((s[k] - max) / Math.sqrt(d));
                sum += s[k];
            }
            double[] p = new double[classes];
            for (int k = 0; k < classes; k++) {
                p[k] = k == y[i] ? 1 - q : q * s[k] / sum;
            }
            out[i] = draw(p, r);
            count[y[i]]++;
            t[y[i]][out[i]]++;
        }
        for (int c = 0; c < classes; c++) {
            for (int k = 0; k < classes; k++) {
                t[c][k] /= Math.max(1, count[c]);
            }
        }
        return new Noisy(out, y.clone(), t);
    }

    /**
     * A long-tailed subset: class {@code c} keeps {@code max * ratio^(-c / (classes - 1))} rows,
     * so class 0 is the head and the last class has {@code max / ratio}.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param max rows kept of the head class
     * @param ratio the imbalance ratio, head over tail
     * @param seed the seed
     * @return the indices kept
     */
    public static int[] longTail(int[] y, int classes, int max, double ratio, long seed) {
        int[] quota = longTailCounts(classes, max, ratio);
        int[] taken = new int[classes];
        List<Integer> out = new ArrayList<>();
        for (int i : permutation(y.length, new SplittableRandom(seed))) {
            if (taken[y[i]] < quota[y[i]]) {
                taken[y[i]]++;
                out.add(i);
            }
        }
        return toArray(out);
    }

    /**
     * @param classes the number of classes
     * @param max the head class's count
     * @param ratio the imbalance ratio
     * @return the per-class counts of {@link #longTail}
     */
    public static int[] longTailCounts(int classes, int max, double ratio) {
        int[] out = new int[classes];
        for (int c = 0; c < classes; c++) {
            out[c] = Math.max(1, (int) Math.round(max * Math.pow(ratio, -c / (double) (classes - 1))));
        }
        return out;
    }

    /**
     * A balanced subset with {@code perClass} rows of each class, for test and validation sets.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param perClass rows per class
     * @param seed the seed
     * @return the indices kept
     */
    public static int[] balanced(int[] y, int classes, int perClass, long seed) {
        int[] quota = new int[classes];
        Arrays.fill(quota, perClass);
        int[] taken = new int[classes];
        List<Integer> out = new ArrayList<>();
        for (int i : permutation(y.length, new SplittableRandom(seed))) {
            if (taken[y[i]] < quota[y[i]]) {
                taken[y[i]]++;
                out.add(i);
            }
        }
        return toArray(out);
    }

    /**
     * A positive-unlabelled split.
     *
     * @param positive the true binary class of every row
     * @param labelled whether the row's positive label is observed (only positives can be)
     * @param prior the share of positives among the unlabelled rows
     * @param labelFrequency {@code c = P(labelled | positive)}
     */
    public record PuSplit(boolean[] positive, boolean[] labelled, double prior, double labelFrequency) {
    }

    /**
     * Builds a PU split: the classes in {@code positiveClasses} are positive, and each positive
     * is labelled with probability {@code labelFrequency} (selected completely at random).
     *
     * @param y the clean classes
     * @param positiveClasses which classes count as positive
     * @param labelFrequency the share of positives that are labelled
     * @param seed the seed
     * @return the split
     */
    public static PuSplit positiveUnlabelled(int[] y, int[] positiveClasses, double labelFrequency, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[] pos = new boolean[y.length];
        boolean[] lab = new boolean[y.length];
        int unl = 0;
        int unlPos = 0;
        for (int i = 0; i < y.length; i++) {
            for (int c : positiveClasses) {
                pos[i] |= y[i] == c;
            }
            lab[i] = pos[i] && r.nextDouble() < labelFrequency;
            if (!lab[i]) {
                unl++;
                unlPos += pos[i] ? 1 : 0;
            }
        }
        return new PuSplit(pos, lab, unlPos / (double) Math.max(1, unl), labelFrequency);
    }

    /**
     * Votes of synthetic labelling functions.
     *
     * @param votes {@code [rows][functions]} votes, a class or {@code -1} for abstain
     * @param accuracy each function's planted accuracy where it votes
     * @param coverage each function's planted vote probability
     * @param correlated the indices of the correlated pair (the second copies the first's vote
     *     most of the time), or an empty array
     */
    public record Votes(int[][] votes, double[] accuracy, double[] coverage, int[] correlated) {
    }

    /**
     * Synthesises labelling functions: function {@code j} votes with probability
     * {@code coverage[j]}, and its vote is correct with probability {@code accuracy[j]}, else a
     * uniformly random wrong class. If {@code correlatedPair} is true, the last function
     * copies the vote of the one before it 90% of the time (a duplicated heuristic).
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param accuracy per-function accuracies
     * @param coverage per-function coverages
     * @param correlatedPair whether the last two functions are correlated
     * @param seed the seed
     * @return the votes
     */
    public static Votes labellingFunctions(int[] y, int classes, double[] accuracy, double[] coverage,
            boolean correlatedPair, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int m = accuracy.length;
        int[][] v = new int[y.length][m];
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < m; j++) {
                if (correlatedPair && j == m - 1 && v[i][m - 2] >= 0 && r.nextDouble() < 0.9) {
                    v[i][j] = v[i][m - 2];
                    continue;
                }
                if (r.nextDouble() >= coverage[j]) {
                    v[i][j] = -1;
                } else if (r.nextDouble() < accuracy[j]) {
                    v[i][j] = y[i];
                } else {
                    int w = r.nextInt(classes - 1);
                    v[i][j] = w >= y[i] ? w + 1 : w;
                }
            }
        }
        return new Votes(v, accuracy.clone(), coverage.clone(),
                correlatedPair ? new int[] {m - 2, m - 1} : new int[0]);
    }

    /**
     * Multiple-instance bags.
     *
     * @param members each bag's row indices
     * @param label each bag's label (whether it holds a key instance)
     * @param key for every bag, which members are key instances
     */
    public record Bags(int[][] members, boolean[] label, boolean[][] key) {
    }

    /**
     * Draws multiple-instance bags in the style of MNIST-bags: sizes uniform in
     * {@code [minSize, maxSize]}, about half the bags positive, a bag positive when it holds at
     * least one row of class {@code keyClass}.
     *
     * @param y the clean classes
     * @param keyClass the class that makes a bag positive
     * @param count the number of bags
     * @param minSize the smallest bag
     * @param maxSize the largest bag
     * @param seed the seed
     * @return the bags
     */
    public static Bags bags(int[] y, int keyClass, int count, int minSize, int maxSize, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Integer> keys = new ArrayList<>();
        List<Integer> others = new ArrayList<>();
        for (int i = 0; i < y.length; i++) {
            (y[i] == keyClass ? keys : others).add(i);
        }
        int[][] members = new int[count][];
        boolean[] label = new boolean[count];
        boolean[][] key = new boolean[count][];
        for (int b = 0; b < count; b++) {
            int size = minSize + r.nextInt(maxSize - minSize + 1);
            label[b] = r.nextBoolean();
            int nKey = label[b] ? 1 + r.nextInt(Math.max(1, size / 5)) : 0;
            members[b] = new int[size];
            key[b] = new boolean[size];
            for (int k = 0; k < size; k++) {
                boolean isKey = k < nKey;
                members[b][k] = (isKey ? keys : others).get(r.nextInt((isKey ? keys : others).size()));
                key[b][k] = isKey;
            }
            for (int k = size - 1; k > 0; k--) {
                int j = r.nextInt(k + 1);
                int t = members[b][k];
                members[b][k] = members[b][j];
                members[b][j] = t;
                boolean kt = key[b][k];
                key[b][k] = key[b][j];
                key[b][j] = kt;
            }
        }
        return new Bags(members, label, key);
    }

    /**
     * Candidate-label sets: the true class plus {@code distractors} other classes, drawn
     * uniformly or (when {@code biased}) from the classes nearest the true one in a fixed
     * confusion order ({@code c + 1, c + 2, ...}), so the same wrong classes recur.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param distractors wrong candidates per row
     * @param biased whether distractors follow a confusion pattern
     * @param seed the seed
     * @return {@code [rows][classes]} candidate masks
     */
    public static boolean[][] candidates(int[] y, int classes, int distractors, boolean biased, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[][] out = new boolean[y.length][classes];
        for (int i = 0; i < y.length; i++) {
            out[i][y[i]] = true;
            int added = 0;
            while (added < distractors) {
                int c = biased ? (y[i] + 1 + (r.nextDouble() < 0.8 ? added : r.nextInt(classes - 1))) % classes
                        : r.nextInt(classes);
                if (!out[i][c]) {
                    out[i][c] = true;
                    added++;
                } else if (biased) {
                    int k = r.nextInt(classes);
                    if (!out[i][k]) {
                        out[i][k] = true;
                        added++;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Groups with known class proportions.
     *
     * @param members each bag's row indices
     * @param proportions each bag's class proportions
     */
    public record ProportionBags(int[][] members, double[][] proportions) {
    }

    /**
     * Partitions the rows into bags of {@code bagSize} and records their class proportions. With
     * {@code skewed}, rows are first sorted by a noisy class order, so bags lean towards a few
     * classes (as aggregate statistics of regions do); otherwise bags are random and their
     * proportions near uniform.
     *
     * @param y the clean classes
     * @param classes the number of classes
     * @param bagSize rows per bag
     * @param skewed whether bags are class-skewed
     * @param seed the seed
     * @return the bags (a trailing partial bag is dropped)
     */
    public static ProportionBags proportionBags(int[] y, int classes, int bagSize, boolean skewed, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] order = permutation(y.length, r);
        if (skewed) {
            double[] key = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                key[i] = y[i] + 2.5 * r.nextGaussian();
            }
            Integer[] boxed = new Integer[y.length];
            for (int i = 0; i < y.length; i++) {
                boxed[i] = i;
            }
            Arrays.sort(boxed, (a, b) -> Double.compare(key[a], key[b]));
            for (int i = 0; i < y.length; i++) {
                order[i] = boxed[i];
            }
        }
        int bags = y.length / bagSize;
        int[][] members = new int[bags][bagSize];
        double[][] prop = new double[bags][classes];
        for (int b = 0; b < bags; b++) {
            for (int k = 0; k < bagSize; k++) {
                members[b][k] = order[b * bagSize + k];
                prop[b][y[members[b][k]]] += 1.0 / bagSize;
            }
        }
        return new ProportionBags(members, prop);
    }

    /**
     * @param n the size
     * @param r the random source
     * @return a random permutation of {@code 0..n-1}
     */
    public static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static int draw(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        double acc = 0;
        for (int k = 0; k < p.length; k++) {
            acc += p[k];
            if (u < acc) {
                return k;
            }
        }
        return p.length - 1;
    }

    private static double[][] copy(double[][] a) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i].clone();
        }
        return out;
    }

    private static int[] toArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }
}
