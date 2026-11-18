package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Random forests (Breiman 2001, <a href="https://link.springer.com/article/10.1023/A:1010933404324">Random
 * forests</a>) and extremely randomised trees (Geurts, Ernst and Wehenkel 2006,
 * <a href="https://link.springer.com/article/10.1007/s10994-006-6226-1">Extremely randomized
 * trees</a>) for classification. Each CART tree grows on a bootstrap sample, choosing at every
 * node the best Gini split among a random subset of features; extra trees skip the bootstrap and
 * draw one random threshold per candidate feature. The forest averages the trees' class
 * frequencies.
 *
 * <p>Why it matters: an ensemble of deep, decorrelated trees is a strong tabular baseline with
 * almost no tuning, and the bootstrap gives a free out-of-bag estimate of the test error.
 */
public final class RandomForest {

    /** How the trees split. */
    public enum Mode {
        /** Bootstrap samples and the best threshold per candidate feature. */
        RANDOM_FOREST,
        /** The full sample and one uniformly random threshold per candidate feature. */
        EXTRA_TREES
    }

    /**
     * Forest settings.
     *
     * @param mode random forest or extra trees
     * @param trees the number of trees
     * @param maxDepth the deepest a tree may grow
     * @param minLeaf the fewest samples in a leaf
     * @param featuresPerSplit candidate features per node; 0 means {@code sqrt(d)}
     * @param seed the seed
     */
    public record Settings(Mode mode, int trees, int maxDepth, int minLeaf, int featuresPerSplit, long seed) {

        /**
         * Usual defaults: 100 deep trees, leaves of 1, {@code sqrt(d)} features per split.
         *
         * @param mode the mode
         * @return the settings
         */
        public static Settings defaults(Mode mode) {
            return new Settings(mode, 100, 20, 1, 0, 1);
        }

        /**
         * @param n the number of trees
         * @return a copy with that many trees
         */
        public Settings withTrees(int n) {
            return new Settings(mode, n, maxDepth, minLeaf, featuresPerSplit, seed);
        }
    }

    /** A node in flat arrays: leaf if {@code feature < 0}. */
    private record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[][] distribution) {
    }

    private final Tree[] trees;
    private final int classes;
    private final double oobError;

    private RandomForest(Tree[] trees, int classes, double oobError) {
        this.trees = trees;
        this.classes = classes;
        this.oobError = oobError;
    }

    /**
     * Grows a forest.
     *
     * @param x the training rows {@code [n][d]}
     * @param labels their classes
     * @param classes the number of classes
     * @param settings the settings
     * @return the forest
     */
    public static RandomForest fit(double[][] x, int[] labels, int classes, Settings settings) {
        int n = x.length;
        int d = x[0].length;
        int m = settings.featuresPerSplit() > 0 ? settings.featuresPerSplit() : Math.max(1, (int) Math.round(Math.sqrt(d)));
        SplittableRandom random = new SplittableRandom(settings.seed());
        Tree[] trees = new Tree[settings.trees()];
        double[][] oobVotes = new double[n][classes];
        for (int t = 0; t < trees.length; t++) {
            int[] sample = new int[n];
            boolean[] inBag = new boolean[n];
            for (int i = 0; i < n; i++) {
                sample[i] = settings.mode() == Mode.RANDOM_FOREST ? random.nextInt(n) : i;
                inBag[sample[i]] = true;
            }
            Builder b = new Builder(x, labels, classes, settings, m, random.split());
            b.grow(sample, 0);
            trees[t] = b.build();
            if (settings.mode() == Mode.RANDOM_FOREST) {
                for (int i = 0; i < n; i++) {
                    if (!inBag[i]) {
                        double[] p = leaf(trees[t], x[i]);
                        for (int c = 0; c < classes; c++) {
                            oobVotes[i][c] += p[c];
                        }
                    }
                }
            }
        }
        double oob = Double.NaN;
        if (settings.mode() == Mode.RANDOM_FOREST) {
            int wrong = 0;
            int counted = 0;
            for (int i = 0; i < n; i++) {
                double sum = Arrays.stream(oobVotes[i]).sum();
                if (sum > 0) {
                    counted++;
                    wrong += argmax(oobVotes[i]) != labels[i] ? 1 : 0;
                }
            }
            oob = counted == 0 ? Double.NaN : wrong / (double) counted;
        }
        return new RandomForest(trees, classes, oob);
    }

    /** Grows one tree depth-first into growable arrays. */
    private static final class Builder {
        private final double[][] x;
        private final int[] labels;
        private final int classes;
        private final Settings settings;
        private final int candidates;
        private final SplittableRandom random;
        private int size;
        private int[] feature = new int[64];
        private double[] threshold = new double[64];
        private int[] left = new int[64];
        private int[] right = new int[64];
        private double[][] distribution = new double[64][];

        Builder(double[][] x, int[] labels, int classes, Settings settings, int candidates, SplittableRandom random) {
            this.x = x;
            this.labels = labels;
            this.classes = classes;
            this.settings = settings;
            this.candidates = candidates;
            this.random = random;
        }

        private int newNode() {
            if (size == feature.length) {
                int cap = size * 2;
                feature = Arrays.copyOf(feature, cap);
                threshold = Arrays.copyOf(threshold, cap);
                left = Arrays.copyOf(left, cap);
                right = Arrays.copyOf(right, cap);
                distribution = Arrays.copyOf(distribution, cap);
            }
            return size++;
        }

        int grow(int[] rows, int depth) {
            int node = newNode();
            double[] counts = new double[classes];
            for (int r : rows) {
                counts[labels[r]]++;
            }
            feature[node] = -1;
            double[] dist = counts.clone();
            for (int c = 0; c < classes; c++) {
                dist[c] /= rows.length;
            }
            distribution[node] = dist;
            boolean pure = Arrays.stream(counts).filter(v -> v > 0).count() <= 1;
            if (pure || depth >= settings.maxDepth() || rows.length < 2 * settings.minLeaf()) {
                return node;
            }
            int d = x[0].length;
            int bestFeature = -1;
            double bestThreshold = 0;
            double bestImpurity = gini(counts, rows.length) * rows.length - 1e-12;
            int[] order = new int[d];
            for (int i = 0; i < d; i++) {
                order[i] = i;
            }
            for (int k = 0; k < candidates; k++) {
                int j = k + random.nextInt(d - k);
                int tmp = order[k];
                order[k] = order[j];
                order[j] = tmp;
                int f = order[k];
                double[] split = settings.mode() == Mode.EXTRA_TREES ? randomSplit(rows, f) : bestSplit(rows, f, counts);
                if (split != null && split[1] < bestImpurity) {
                    bestImpurity = split[1];
                    bestFeature = f;
                    bestThreshold = split[0];
                }
            }
            if (bestFeature < 0) {
                return node;
            }
            int nl = 0;
            for (int r : rows) {
                nl += x[r][bestFeature] <= bestThreshold ? 1 : 0;
            }
            if (nl < settings.minLeaf() || rows.length - nl < settings.minLeaf()) {
                return node;
            }
            int[] lr = new int[nl];
            int[] rr = new int[rows.length - nl];
            int a = 0;
            int b = 0;
            for (int r : rows) {
                if (x[r][bestFeature] <= bestThreshold) {
                    lr[a++] = r;
                } else {
                    rr[b++] = r;
                }
            }
            feature[node] = bestFeature;
            threshold[node] = bestThreshold;
            int l = grow(lr, depth + 1);
            int rgt = grow(rr, depth + 1);
            left[node] = l;
            right[node] = rgt;
            return node;
        }

        /** The best threshold of one feature: {threshold, weighted Gini}, or null. */
        private double[] bestSplit(int[] rows, int f, double[] total) {
            Integer[] idx = new Integer[rows.length];
            for (int i = 0; i < rows.length; i++) {
                idx[i] = rows[i];
            }
            Arrays.sort(idx, (p, q) -> Double.compare(x[p][f], x[q][f]));
            double[] lc = new double[classes];
            double[] rc = total.clone();
            double best = Double.POSITIVE_INFINITY;
            double thr = 0;
            int min = settings.minLeaf();
            for (int i = 0; i < idx.length - 1; i++) {
                int y = labels[idx[i]];
                lc[y]++;
                rc[y]--;
                double v = x[idx[i]][f];
                double next = x[idx[i + 1]][f];
                if (v == next || i + 1 < min || idx.length - i - 1 < min) {
                    continue;
                }
                double imp = gini(lc, i + 1) * (i + 1) + gini(rc, idx.length - i - 1) * (idx.length - i - 1);
                if (imp < best) {
                    best = imp;
                    thr = (v + next) / 2;
                }
            }
            return best == Double.POSITIVE_INFINITY ? null : new double[] {thr, best};
        }

        /** One uniformly random threshold between the feature's extremes. */
        private double[] randomSplit(int[] rows, int f) {
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (int r : rows) {
                lo = Math.min(lo, x[r][f]);
                hi = Math.max(hi, x[r][f]);
            }
            if (!(hi > lo)) {
                return null;
            }
            double thr = lo + random.nextDouble() * (hi - lo);
            double[] lc = new double[classes];
            double[] rc = new double[classes];
            int nl = 0;
            for (int r : rows) {
                if (x[r][f] <= thr) {
                    lc[labels[r]]++;
                    nl++;
                } else {
                    rc[labels[r]]++;
                }
            }
            int nr = rows.length - nl;
            if (nl == 0 || nr == 0) {
                return null;
            }
            return new double[] {thr, gini(lc, nl) * nl + gini(rc, nr) * nr};
        }

        Tree build() {
            return new Tree(Arrays.copyOf(feature, size), Arrays.copyOf(threshold, size), Arrays.copyOf(left, size),
                    Arrays.copyOf(right, size), Arrays.copyOf(distribution, size));
        }
    }

    private static double gini(double[] counts, double n) {
        double s = 1;
        for (double c : counts) {
            double p = c / n;
            s -= p * p;
        }
        return s;
    }

    private static double[] leaf(Tree t, double[] x) {
        int node = 0;
        while (t.feature()[node] >= 0) {
            node = x[t.feature()[node]] <= t.threshold()[node] ? t.left()[node] : t.right()[node];
        }
        return t.distribution()[node];
    }

    private static int argmax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    /**
     * The averaged class probabilities.
     *
     * @param x a row
     * @return the probability of each class
     */
    public double[] predictProbability(double[] x) {
        double[] p = new double[classes];
        for (Tree t : trees) {
            double[] l = leaf(t, x);
            for (int c = 0; c < classes; c++) {
                p[c] += l[c] / trees.length;
            }
        }
        return p;
    }

    /**
     * Predicts classes.
     *
     * @param x rows
     * @return their most probable classes
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = argmax(predictProbability(x[i]));
        }
        return out;
    }

    /** @return the out-of-bag error rate (NaN for extra trees, which use no bootstrap) */
    public double oobError() {
        return oobError;
    }

    /** @return the number of trees */
    public int size() {
        return trees.length;
    }
}
