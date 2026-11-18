package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Random forests for regression with sample weights (Breiman 2001), the regression counterpart of
 * {@link RandomForest}: every CART tree grows on a subsample (bootstrap or half-sample without
 * replacement), choosing at each node the weighted squared-error split among a random subset of
 * features, and the forest averages the trees' leaf means.
 *
 * <p>Two options matter for causal inference: <em>sample weights</em> (the R-learner's weighted
 * regression of the Robinson pseudo-outcome) and <em>honesty</em> (Wager and Athey 2018,
 * <a href="https://arxiv.org/abs/1510.04342">arXiv:1510.04342</a>): each tree splits on one half of
 * its subsample and estimates its leaf values on the other half, so leaf values are not fitted to
 * the same noise that chose the splits. The grower is exposed through {@link #grow} so that other
 * forests (causal, survival, uplift) can reuse its node bookkeeping with their own split scores.
 */
public final class RegressionForest {

    /**
     * Forest settings.
     *
     * @param trees the number of trees
     * @param maxDepth the deepest a tree may grow
     * @param minLeaf the fewest samples in a leaf
     * @param featuresPerSplit candidate features per node; 0 means {@code max(1, d / 3)}
     * @param sampleFraction the share of rows each tree sees; 1 means a bootstrap sample
     * @param honest whether leaves are re-estimated on a held-out half of the subsample
     * @param seed the seed
     */
    public record Settings(int trees, int maxDepth, int minLeaf, int featuresPerSplit, double sampleFraction,
            boolean honest, long seed) {

        /** @return 100 bootstrap trees, depth 12, leaves of 5, {@code d / 3} features per split */
        public static Settings defaults() {
            return new Settings(100, 12, 5, 0, 1, false, 1);
        }

        /**
         * @param n trees
         * @return a copy with that many trees
         */
        public Settings withTrees(int n) {
            return new Settings(n, maxDepth, minLeaf, featuresPerSplit, sampleFraction, honest, seed);
        }

        /**
         * @param leaf the minimum leaf size
         * @return a copy with that leaf size
         */
        public Settings withMinLeaf(int leaf) {
            return new Settings(trees, maxDepth, leaf, featuresPerSplit, sampleFraction, honest, seed);
        }

        /**
         * @param on honesty on or off
         * @return a copy with half-sampling ({@code sampleFraction = 0.5}) and that honesty
         */
        public Settings withHonesty(boolean on) {
            return new Settings(trees, maxDepth, minLeaf, featuresPerSplit, 0.5, on, seed);
        }

        /**
         * @param s the seed
         * @return a copy with that seed
         */
        public Settings withSeed(long s) {
            return new Settings(trees, maxDepth, minLeaf, featuresPerSplit, sampleFraction, honest, s);
        }
    }

    /**
     * A tree in flat arrays; node {@code k} is a leaf when {@code feature[k] < 0}.
     *
     * @param feature the split feature per node
     * @param threshold the split threshold per node ({@code x <= threshold} goes left)
     * @param left the left child per node
     * @param right the right child per node
     * @param value the leaf value per node
     */
    public record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[] value) {

        /**
         * @param x a row
         * @return the index of the leaf it falls in
         */
        public int leaf(double[] x) {
            int k = 0;
            while (feature[k] >= 0) {
                k = x[feature[k]] <= threshold[k] ? left[k] : right[k];
            }
            return k;
        }

        /**
         * @param x a row
         * @return its leaf value
         */
        public double predict(double[] x) {
            return value[leaf(x)];
        }
    }

    /**
     * The score a split is chosen by.
     */
    @FunctionalInterface
    public interface SplitScore {
        /**
         * Scores the split of a node's rows into a left part {@code rows[0..cut)} and a right
         * part {@code rows[cut..)} (rows sorted by the candidate feature).
         *
         * @param rows the node's rows, sorted by the feature
         * @param cut the number of rows going left
         * @return the gain (higher is better); NaN or non-positive rejects the split
         */
        double gain(int[] rows, int cut);

        /**
         * Called once per node before its candidate splits are scored, so prefix sums can be
         * prepared; the default does nothing.
         *
         * @param rows the node's rows
         */
        default void node(int[] rows) {
        }
    }

    private final Tree[] trees;

    private RegressionForest(Tree[] trees) {
        this.trees = trees;
    }

    /**
     * Grows a weighted squared-error forest.
     *
     * @param x rows {@code [n][d]}
     * @param y targets
     * @param weights sample weights, or null for all ones
     * @param s the settings
     * @return the forest
     */
    public static RegressionForest fit(double[][] x, double[] y, double[] weights, Settings s) {
        int n = x.length;
        double[] w = weights != null ? weights : filled(n, 1);
        SplittableRandom random = new SplittableRandom(s.seed());
        Tree[] trees = new Tree[s.trees()];
        for (int b = 0; b < trees.length; b++) {
            int[][] halves = subsample(n, s.sampleFraction(), s.honest(), random);
            int[] grow = halves[0];
            MseScore score = new MseScore(y, w);
            Tree tree = grow(x, grow, s.maxDepth(), s.minLeaf(), s.featuresPerSplit(), score, random,
                    rows -> weightedMean(rows, y, w));
            if (s.honest()) {
                tree = reestimate(tree, x, halves[1], rows -> weightedMean(rows, y, w));
            }
            trees[b] = tree;
        }
        return new RegressionForest(trees);
    }

    /**
     * Draws a tree's rows.
     *
     * @param n rows available
     * @param fraction the share drawn without replacement, or 1 for a bootstrap sample
     * @param honest whether to split the draw into a growing half and an estimation half
     * @param random the source
     * @return {@code {grow, estimate}} (estimate is empty unless honest)
     */
    public static int[][] subsample(int n, double fraction, boolean honest, SplittableRandom random) {
        int[] rows;
        if (fraction >= 1) {
            rows = new int[n];
            for (int i = 0; i < n; i++) {
                rows[i] = random.nextInt(n);
            }
        } else {
            int[] perm = permutation(n, random);
            rows = Arrays.copyOf(perm, Math.max(2, (int) (n * fraction)));
        }
        if (!honest) {
            return new int[][] {rows, new int[0]};
        }
        int half = rows.length / 2;
        return new int[][] {Arrays.copyOfRange(rows, 0, half), Arrays.copyOfRange(rows, half, rows.length)};
    }

    /**
     * Grows one tree with a pluggable split score: the shared CART bookkeeping of the regression,
     * causal, survival and uplift forests.
     *
     * @param x all rows
     * @param rows the rows this tree grows on
     * @param maxDepth the depth limit
     * @param minLeaf the fewest rows per child
     * @param featuresPerSplit candidate features per node, 0 for {@code max(1, d / 3)}
     * @param score the split score
     * @param random the feature sampler
     * @param leafValue the value stored in a leaf from its rows
     * @return the tree
     */
    public static Tree grow(double[][] x, int[] rows, int maxDepth, int minLeaf, int featuresPerSplit,
            SplitScore score, SplittableRandom random, java.util.function.ToDoubleFunction<int[]> leafValue) {
        int d = x[0].length;
        int m = featuresPerSplit > 0 ? Math.min(d, featuresPerSplit) : Math.max(1, d / 3);
        Builder b = new Builder();
        build(b, x, rows, 0, maxDepth, minLeaf, m, score, random, leafValue);
        return b.tree();
    }

    private static int build(Builder b, double[][] x, int[] rows, int depth, int maxDepth, int minLeaf, int m,
            SplitScore score, SplittableRandom random, java.util.function.ToDoubleFunction<int[]> leafValue) {
        int node = b.add();
        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = 0;
        int[] bestOrder = null;
        int bestCut = 0;
        if (depth < maxDepth && rows.length >= 2 * minLeaf) {
            score.node(rows);
            int d = x[0].length;
            int[] features = permutation(d, random);
            for (int f = 0; f < m; f++) {
                int feature = features[f];
                Integer[] boxed = new Integer[rows.length];
                for (int i = 0; i < rows.length; i++) {
                    boxed[i] = rows[i];
                }
                Arrays.sort(boxed, (p, q) -> Double.compare(x[p][feature], x[q][feature]));
                int[] order = new int[rows.length];
                for (int i = 0; i < order.length; i++) {
                    order[i] = boxed[i];
                }
                for (int cut = minLeaf; cut <= rows.length - minLeaf; cut++) {
                    double lo = x[order[cut - 1]][feature];
                    double hi = x[order[cut]][feature];
                    if (lo == hi) {
                        continue;
                    }
                    double gain = score.gain(order, cut);
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = (lo + hi) / 2;
                        bestOrder = order;
                        bestCut = cut;
                    }
                }
            }
        }
        if (bestFeature < 0) {
            b.value[node] = leafValue.applyAsDouble(rows);
            return node;
        }
        b.feature[node] = bestFeature;
        b.threshold[node] = bestThreshold;
        int left = build(b, x, Arrays.copyOfRange(bestOrder, 0, bestCut), depth + 1, maxDepth, minLeaf, m, score,
                random, leafValue);
        b.left[node] = left;
        int right = build(b, x, Arrays.copyOfRange(bestOrder, bestCut, bestOrder.length), depth + 1, maxDepth,
                minLeaf, m, score, random, leafValue);
        b.right[node] = right;
        return node;
    }

    /**
     * Re-estimates a tree's leaves on other rows (honesty); leaves that receive no row keep their
     * old value.
     *
     * @param tree the tree
     * @param x all rows
     * @param rows the estimation rows
     * @param leafValue the leaf value from its rows
     * @return a tree with the same splits and new leaf values
     */
    public static Tree reestimate(Tree tree, double[][] x, int[] rows,
            java.util.function.ToDoubleFunction<int[]> leafValue) {
        int nodes = tree.feature().length;
        int[] count = new int[nodes];
        int[] leafOf = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            leafOf[i] = tree.leaf(x[rows[i]]);
            count[leafOf[i]]++;
        }
        int[][] members = new int[nodes][];
        for (int k = 0; k < nodes; k++) {
            members[k] = new int[count[k]];
        }
        int[] fill = new int[nodes];
        for (int i = 0; i < rows.length; i++) {
            members[leafOf[i]][fill[leafOf[i]]++] = rows[i];
        }
        double[] value = tree.value().clone();
        for (int k = 0; k < nodes; k++) {
            if (tree.feature()[k] < 0 && count[k] > 0) {
                value[k] = leafValue.applyAsDouble(members[k]);
            }
        }
        return new Tree(tree.feature(), tree.threshold(), tree.left(), tree.right(), value);
    }

    /**
     * @param x a row
     * @return the forest's prediction
     */
    public double predict(double[] x) {
        double s = 0;
        for (Tree t : trees) {
            s += t.predict(x);
        }
        return s / trees.length;
    }

    /**
     * @param x rows
     * @return one prediction per row
     */
    public double[] predict(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }

    /** @return the trees */
    public Tree[] trees() {
        return trees.clone();
    }

    /** Weighted squared-error reduction with prefix sums per node. */
    private static final class MseScore implements SplitScore {
        private final double[] y;
        private final double[] w;
        private double totalW;
        private double totalWy;

        MseScore(double[] y, double[] w) {
            this.y = y;
            this.w = w;
        }

        private double[] prefixW = new double[0];
        private double[] prefixWy = new double[0];
        private int[] cachedOrder;

        @Override
        public void node(int[] rows) {
            totalW = 0;
            totalWy = 0;
            for (int r : rows) {
                totalW += w[r];
                totalWy += w[r] * y[r];
            }
            cachedOrder = null;
        }

        @Override
        public double gain(int[] rows, int cut) {
            if (rows != cachedOrder) {
                prefixW = new double[rows.length + 1];
                prefixWy = new double[rows.length + 1];
                for (int i = 0; i < rows.length; i++) {
                    prefixW[i + 1] = prefixW[i] + w[rows[i]];
                    prefixWy[i + 1] = prefixWy[i] + w[rows[i]] * y[rows[i]];
                }
                cachedOrder = rows;
            }
            double wl = prefixW[cut];
            double wr = totalW - wl;
            if (wl <= 0 || wr <= 0) {
                return Double.NaN;
            }
            double sl = prefixWy[cut];
            double sr = totalWy - sl;
            return sl * sl / wl + sr * sr / wr - totalWy * totalWy / totalW;
        }
    }

    private static double weightedMean(int[] rows, double[] y, double[] w) {
        double s = 0;
        double z = 0;
        for (int r : rows) {
            s += w[r] * y[r];
            z += w[r];
        }
        return z > 0 ? s / z : 0;
    }

    private static double[] filled(int n, double v) {
        double[] a = new double[n];
        Arrays.fill(a, v);
        return a;
    }

    private static int[] permutation(int n, SplittableRandom random) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /** Growable flat node arrays. */
    private static final class Builder {
        int[] feature = new int[16];
        double[] threshold = new double[16];
        int[] left = new int[16];
        int[] right = new int[16];
        double[] value = new double[16];
        int size;

        int add() {
            if (size == feature.length) {
                int c = size * 2;
                feature = Arrays.copyOf(feature, c);
                threshold = Arrays.copyOf(threshold, c);
                left = Arrays.copyOf(left, c);
                right = Arrays.copyOf(right, c);
                value = Arrays.copyOf(value, c);
            }
            feature[size] = -1;
            return size++;
        }

        Tree tree() {
            return new Tree(Arrays.copyOf(feature, size), Arrays.copyOf(threshold, size), Arrays.copyOf(left, size),
                    Arrays.copyOf(right, size), Arrays.copyOf(value, size));
        }
    }
}
