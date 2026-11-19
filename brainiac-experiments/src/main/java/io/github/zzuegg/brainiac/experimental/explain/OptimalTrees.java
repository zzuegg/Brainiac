package io.github.zzuegg.brainiac.experimental.explain;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * Provably optimal sparse decision trees on binary features by dynamic programming with bounds:
 * DL8.5 (Aglin, Nijssen and Schaus, AAAI 2020) and the per-leaf penalised objective of GOSDT
 * (Lin et al., ICML 2020, <a href="https://arxiv.org/abs/2006.08690">arXiv:2006.08690</a>).
 *
 * <p>The objective of a tree is {@code mistakes / n + lambda * leaves} ({@code lambda = 0} is
 * DL8.5's depth-limited misclassification). A subproblem is the set of training rows reaching a
 * node, kept as a bit vector, with the remaining depth; its optimum is the better of a leaf and
 * the best split into two optimal subtrees. Solutions are cached by (bit vector, depth), so the
 * same row set reached by different paths is solved once, and the search prunes with upper bounds
 * (a child that alone exceeds the best known cost is abandoned; failed searches cache a lower
 * bound) and the split lower bound {@code 2 lambda}. Continuous features are binarised by
 * thresholds: all quantiles ({@link #quantileThresholds}) or only the thresholds a boosted
 * ensemble actually uses ({@link #guessedThresholds}, the threshold guessing of McTavish et al.,
 * AAAI 2022), which shrinks the search.
 *
 * <p>{@link #minimumCost} exposes the cache to {@link RashomonSet}; {@link #greedy} grows the
 * greedy top-down tree (CART-style, by misclassification-reducing splits) on the same features.
 */
public final class OptimalTrees {

    /**
     * A tree: a leaf when {@code feature < 0}; otherwise rows with the feature 0 go to
     * {@code left}, rows with 1 to {@code right}.
     *
     * @param feature the binary split feature, or -1
     * @param left the 0-branch
     * @param right the 1-branch
     * @param prediction a leaf's class
     */
    public record Tree(int feature, Tree left, Tree right, int prediction) {

        /**
         * @param x a binary row (values above 0.5 are 1)
         * @return the predicted class
         */
        public int predict(double[] x) {
            Tree t = this;
            while (t.feature >= 0) {
                t = x[t.feature] > 0.5 ? t.right : t.left;
            }
            return t.prediction;
        }

        /** @return the number of leaves */
        public int leaves() {
            return feature < 0 ? 1 : left.leaves() + right.leaves();
        }

        /** @return the depth */
        public int depth() {
            return feature < 0 ? 0 : 1 + Math.max(left.depth(), right.depth());
        }

        /** @return the features used, ascending */
        public int[] features() {
            TreeSet<Integer> s = new TreeSet<>();
            collect(s);
            return s.stream().mapToInt(Integer::intValue).toArray();
        }

        private void collect(TreeSet<Integer> s) {
            if (feature >= 0) {
                s.add(feature);
                left.collect(s);
                right.collect(s);
            }
        }

        static Tree leaf(int prediction) {
            return new Tree(-1, null, null, prediction);
        }
    }

    /**
     * A solved tree.
     *
     * @param tree the optimal tree
     * @param objective its objective
     * @param subproblems distinct subproblems cached
     * @param millis solve time
     */
    public record Result(Tree tree, double objective, long subproblems, double millis) {
    }

    private record Key(long[] bits, int depth) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && k.depth == depth && Arrays.equals(k.bits, bits);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(bits) * 31 + depth;
        }
    }

    private static final class Entry {
        double cost;
        boolean exact;
        Tree tree;
    }

    private final int n;
    private final int p;
    private final int words;
    private final long[][] featureBits;
    private final long[] positive;
    private final double lambda;
    private final Map<Key, Entry> cache = new HashMap<>();

    /**
     * Prepares a problem.
     *
     * @param binary binary features {@code [n][p]} (values above 0.5 are 1)
     * @param y 0/1 labels
     * @param lambda the per-leaf penalty (0 for pure misclassification)
     */
    public OptimalTrees(double[][] binary, double[] y, double lambda) {
        this.n = binary.length;
        this.p = binary[0].length;
        this.words = (n + 63) / 64;
        this.lambda = lambda;
        featureBits = new long[p][words];
        positive = new long[words];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                if (binary[i][j] > 0.5) {
                    featureBits[j][i >> 6] |= 1L << (i & 63);
                }
            }
            if (y[i] > 0.5) {
                positive[i >> 6] |= 1L << (i & 63);
            }
        }
    }

    /** @return all rows as a bit vector */
    public long[] all() {
        long[] s = new long[words];
        for (int i = 0; i < n; i++) {
            s[i >> 6] |= 1L << (i & 63);
        }
        return s;
    }

    static int count(long[] s) {
        int c = 0;
        for (long w : s) {
            c += Long.bitCount(w);
        }
        return c;
    }

    private int positives(long[] s) {
        int c = 0;
        for (int w = 0; w < words; w++) {
            c += Long.bitCount(s[w] & positive[w]);
        }
        return c;
    }

    /** {@code [rows with feature 0, rows with feature 1]}. */
    long[][] split(long[] s, int feature) {
        long[] zero = new long[words];
        long[] one = new long[words];
        for (int w = 0; w < words; w++) {
            one[w] = s[w] & featureBits[feature][w];
            zero[w] = s[w] & ~featureBits[feature][w];
        }
        return new long[][] {zero, one};
    }

    /** @return the number of rows */
    public int rows() {
        return n;
    }

    /** @return the number of binary features */
    public int features() {
        return p;
    }

    /** @return the per-leaf penalty */
    public double lambda() {
        return lambda;
    }

    /**
     * The cost of a leaf on a row set.
     *
     * @param s the rows
     * @return {@code min(pos, neg) / n + lambda}
     */
    public double leafCost(long[] s) {
        int c = count(s);
        int pos = positives(s);
        return Math.min(pos, c - pos) / (double) n + lambda;
    }

    /**
     * The majority class of a row set.
     *
     * @param s the rows
     * @return 1 if positives are the majority
     */
    public int majority(long[] s) {
        int c = count(s);
        return 2 * positives(s) > c ? 1 : 0;
    }

    /**
     * Solves for the optimal tree of at most the given depth.
     *
     * @param depth the maximum depth
     * @return the tree, objective and statistics
     */
    public Result solve(int depth) {
        long t0 = System.nanoTime();
        long[] all = all();
        double cost = search(all, depth, Double.POSITIVE_INFINITY);
        Entry e = cache.get(new Key(all, depth));
        return new Result(e.tree, cost, cache.size(), (System.nanoTime() - t0) / 1e6);
    }

    /**
     * The optimal cost of a subproblem (solving it if needed), for {@link RashomonSet}.
     *
     * @param s the rows
     * @param depth the remaining depth
     * @return the optimal objective contribution of a subtree on those rows
     */
    public double minimumCost(long[] s, int depth) {
        return search(s, depth, Double.POSITIVE_INFINITY);
    }

    private double search(long[] s, int depth, double ub) {
        Key key = new Key(s, depth);
        Entry e = cache.get(key);
        if (e != null && (e.exact || e.cost >= ub)) {
            return e.cost;
        }
        if (e == null) {
            e = new Entry();
            cache.put(key, e);
        }
        double leaf = leafCost(s);
        double best = leaf;
        Tree tree = Tree.leaf(majority(s));
        int c = count(s);
        int pos = positives(s);
        boolean pure = pos == 0 || pos == c;
        if (depth > 0 && !pure && 2 * lambda < Math.min(ub, leaf)) {
            for (int f = 0; f < p; f++) {
                long[][] parts = split(s, f);
                if (count(parts[0]) == 0 || count(parts[1]) == 0) {
                    continue;
                }
                double cur = Math.min(ub, best);
                double c0 = search(parts[0], depth - 1, cur - lambda);
                if (c0 >= cur - lambda) {
                    continue;
                }
                double c1 = search(parts[1], depth - 1, cur - c0);
                if (c1 < cur - c0 && c0 + c1 < best - 1e-12) {
                    best = c0 + c1;
                    Entry l = cache.get(new Key(parts[0], depth - 1));
                    Entry r = cache.get(new Key(parts[1], depth - 1));
                    tree = new Tree(f, l.tree, r.tree, majority(s));
                }
            }
        }
        if (best < ub) {
            e.cost = best;
            e.exact = true;
            e.tree = tree;
            return best;
        }
        e.cost = Math.max(e.cost, ub);
        return ub;
    }

    /**
     * The greedy top-down tree: at each node the split that most reduces misclassification
     * (ties by Gini), stopping at the depth or when no split helps.
     *
     * @param depth the maximum depth
     * @return the tree
     */
    public Tree greedy(int depth) {
        return greedy(all(), depth);
    }

    private Tree greedy(long[] s, int depth) {
        int c = count(s);
        int pos = positives(s);
        if (depth == 0 || pos == 0 || pos == c) {
            return Tree.leaf(majority(s));
        }
        int bestF = -1;
        double bestGini = gini(pos, c) * c;
        for (int f = 0; f < p; f++) {
            long[][] parts = split(s, f);
            int c0 = count(parts[0]);
            int c1 = count(parts[1]);
            if (c0 == 0 || c1 == 0) {
                continue;
            }
            double g = gini(positives(parts[0]), c0) * c0 + gini(positives(parts[1]), c1) * c1;
            if (g < bestGini - 1e-12) {
                bestGini = g;
                bestF = f;
            }
        }
        if (bestF < 0) {
            return Tree.leaf(majority(s));
        }
        long[][] parts = split(s, bestF);
        return new Tree(bestF, greedy(parts[0], depth - 1), greedy(parts[1], depth - 1), majority(s));
    }

    private static double gini(int pos, int c) {
        double q = pos / (double) c;
        return 2 * q * (1 - q);
    }

    /**
     * The objective of any tree on the training rows.
     *
     * @param tree the tree
     * @return {@code mistakes / n + lambda * leaves}
     */
    public double objective(Tree tree) {
        return cost(tree, all());
    }

    double cost(Tree t, long[] s) {
        if (t.feature() < 0) {
            int c = count(s);
            int pos = positives(s);
            return (t.prediction() == 1 ? c - pos : pos) / (double) n + lambda;
        }
        long[][] parts = split(s, t.feature());
        return cost(t.left(), parts[0]) + cost(t.right(), parts[1]);
    }

    // ------------------------------------------------------------------ binarisation

    /**
     * Thresholds at the {@code k} inner quantiles of each feature.
     *
     * @param x the data
     * @param k thresholds per feature
     * @return {@code [p][]} distinct thresholds
     */
    public static double[][] quantileThresholds(double[][] x, int k) {
        int d = x[0].length;
        double[][] out = new double[d][];
        for (int j = 0; j < d; j++) {
            double[] v = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                v[i] = x[i][j];
            }
            Arrays.sort(v);
            double[] t = new double[k];
            for (int q = 1; q <= k; q++) {
                t[q - 1] = v[(int) ((long) q * (v.length - 1) / (k + 1))];
            }
            out[j] = Arrays.stream(t).distinct().toArray();
        }
        return out;
    }

    /**
     * The thresholds a tree ensemble splits on, per feature.
     *
     * @param ensemble the boosted trees
     * @return {@code [p][]} distinct thresholds
     */
    public static double[][] guessedThresholds(TreeView ensemble) {
        List<TreeSet<Double>> sets = new ArrayList<>();
        for (int j = 0; j < ensemble.features(); j++) {
            sets.add(new TreeSet<>());
        }
        for (TreeView.Tree t : ensemble.trees()) {
            for (int k = 0; k < t.nodes(); k++) {
                if (t.feature()[k] >= 0) {
                    sets.get(t.feature()[k]).add(t.threshold()[k]);
                }
            }
        }
        double[][] out = new double[sets.size()][];
        for (int j = 0; j < out.length; j++) {
            out[j] = sets.get(j).stream().mapToDouble(Double::doubleValue).toArray();
        }
        return out;
    }

    /**
     * Binary features {@code x_j > t} for every threshold.
     *
     * @param x the data
     * @param thresholds per feature
     * @return the binary rows
     */
    public static double[][] binarise(double[][] x, double[][] thresholds) {
        int total = Arrays.stream(thresholds).mapToInt(t -> t.length).sum();
        double[][] out = new double[x.length][total];
        for (int i = 0; i < x.length; i++) {
            int k = 0;
            for (int j = 0; j < thresholds.length; j++) {
                for (double t : thresholds[j]) {
                    out[i][k++] = x[i][j] > t ? 1 : 0;
                }
            }
        }
        return out;
    }
}
