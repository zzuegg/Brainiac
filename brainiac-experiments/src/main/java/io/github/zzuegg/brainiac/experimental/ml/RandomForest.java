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
 * <p>{@link #fitRegression} grows the same trees for a real-valued target (Breiman's regression
 * forest): splits minimise the summed squared error (variance reduction), leaves hold the mean
 * and variance of their targets, and {@link #predictValue} averages the leaf means. The
 * classification path is unchanged.
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
    private final boolean regression;

    private RandomForest(Tree[] trees, int classes, double oobError) {
        this(trees, classes, oobError, false);
    }

    private RandomForest(Tree[] trees, int classes, double oobError, boolean regression) {
        this.trees = trees;
        this.classes = classes;
        this.oobError = oobError;
        this.regression = regression;
    }

    /**
     * Grows a regression forest: bootstrap (or, for extra trees, the full sample), variance
     * reduction splits among {@code featuresPerSplit} candidates ({@code 0} means {@code d / 3}
     * as Breiman suggests for regression), mean leaves.
     *
     * @param x the training rows {@code [n][d]}
     * @param y their targets
     * @param settings the settings ({@code minLeaf} of about 5 is usual for regression)
     * @return the forest; {@link #oobError()} is the out-of-bag mean squared error
     */
    public static RandomForest fitRegression(double[][] x, double[] y, Settings settings) {
        int n = x.length;
        int d = x[0].length;
        int m = settings.featuresPerSplit() > 0 ? settings.featuresPerSplit() : Math.max(1, d / 3);
        SplittableRandom random = new SplittableRandom(settings.seed());
        Tree[] trees = new Tree[settings.trees()];
        double[] oobSum = new double[n];
        int[] oobCount = new int[n];
        for (int t = 0; t < trees.length; t++) {
            int[] sample = new int[n];
            boolean[] inBag = new boolean[n];
            for (int i = 0; i < n; i++) {
                sample[i] = settings.mode() == Mode.RANDOM_FOREST ? random.nextInt(n) : i;
                inBag[sample[i]] = true;
            }
            RegressionBuilder b = new RegressionBuilder(x, y, settings, m, random.split());
            b.grow(sample, 0);
            trees[t] = b.build();
            if (settings.mode() == Mode.RANDOM_FOREST) {
                for (int i = 0; i < n; i++) {
                    if (!inBag[i]) {
                        oobSum[i] += leaf(trees[t], x[i])[0];
                        oobCount[i]++;
                    }
                }
            }
        }
        double oob = Double.NaN;
        if (settings.mode() == Mode.RANDOM_FOREST) {
            double se = 0;
            int counted = 0;
            for (int i = 0; i < n; i++) {
                if (oobCount[i] > 0) {
                    double e = oobSum[i] / oobCount[i] - y[i];
                    se += e * e;
                    counted++;
                }
            }
            oob = counted == 0 ? Double.NaN : se / counted;
        }
        return new RandomForest(trees, 1, oob, true);
    }

    /** Grows one regression tree: leaves store {mean, variance} of their targets. */
    private static final class RegressionBuilder {
        private final double[][] x;
        private final double[] y;
        private final Settings settings;
        private final int candidates;
        private final SplittableRandom random;
        private int size;
        private int[] feature = new int[64];
        private double[] threshold = new double[64];
        private int[] left = new int[64];
        private int[] right = new int[64];
        private double[][] distribution = new double[64][];

        RegressionBuilder(double[][] x, double[] y, Settings settings, int candidates, SplittableRandom random) {
            this.x = x;
            this.y = y;
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
            double sum = 0;
            double sq = 0;
            for (int r : rows) {
                sum += y[r];
                sq += y[r] * y[r];
            }
            double mean = sum / rows.length;
            double sse = Math.max(0, sq - sum * mean);
            feature[node] = -1;
            distribution[node] = new double[] {mean, sse / rows.length};
            if (sse <= 1e-12 || depth >= settings.maxDepth() || rows.length < 2 * settings.minLeaf()) {
                return node;
            }
            int d = x[0].length;
            int bestFeature = -1;
            double bestThreshold = 0;
            double bestSse = sse - 1e-12;
            int[] order = new int[d];
            for (int i = 0; i < d; i++) {
                order[i] = i;
            }
            for (int k = 0; k < Math.min(candidates, d); k++) {
                int j = k + random.nextInt(d - k);
                int tmp = order[k];
                order[k] = order[j];
                order[j] = tmp;
                int f = order[k];
                double[] split = settings.mode() == Mode.EXTRA_TREES ? randomSplit(rows, f) : bestSplit(rows, f);
                if (split != null && split[1] < bestSse) {
                    bestSse = split[1];
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

        /** The best threshold of one feature: {threshold, summed squared error}, or null. */
        private double[] bestSplit(int[] rows, int f) {
            Integer[] idx = new Integer[rows.length];
            double total = 0;
            double totalSq = 0;
            for (int i = 0; i < rows.length; i++) {
                idx[i] = rows[i];
                total += y[rows[i]];
                totalSq += y[rows[i]] * y[rows[i]];
            }
            Arrays.sort(idx, (p, q) -> Double.compare(x[p][f], x[q][f]));
            double ls = 0;
            double lsq = 0;
            double best = Double.POSITIVE_INFINITY;
            double thr = 0;
            int min = settings.minLeaf();
            int n = idx.length;
            for (int i = 0; i < n - 1; i++) {
                double v = y[idx[i]];
                ls += v;
                lsq += v * v;
                double cur = x[idx[i]][f];
                double next = x[idx[i + 1]][f];
                if (cur == next || i + 1 < min || n - i - 1 < min) {
                    continue;
                }
                int nl = i + 1;
                int nr = n - nl;
                double rs = total - ls;
                double sse = lsq - ls * ls / nl + (totalSq - lsq) - rs * rs / nr;
                if (sse < best) {
                    best = sse;
                    thr = (cur + next) / 2;
                }
            }
            return best == Double.POSITIVE_INFINITY ? null : new double[] {thr, best};
        }

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
            double ls = 0;
            double lsq = 0;
            double rs = 0;
            double rsq = 0;
            int nl = 0;
            for (int r : rows) {
                if (x[r][f] <= thr) {
                    ls += y[r];
                    lsq += y[r] * y[r];
                    nl++;
                } else {
                    rs += y[r];
                    rsq += y[r] * y[r];
                }
            }
            int nr = rows.length - nl;
            if (nl == 0 || nr == 0) {
                return null;
            }
            return new double[] {thr, lsq - ls * ls / nl + rsq - rs * rs / nr};
        }

        Tree build() {
            return new Tree(Arrays.copyOf(feature, size), Arrays.copyOf(threshold, size), Arrays.copyOf(left, size),
                    Arrays.copyOf(right, size), Arrays.copyOf(distribution, size));
        }
    }

    /**
     * The regression forest's prediction: the mean of the trees' leaf means.
     *
     * @param x a row
     * @return the predicted value
     * @throws IllegalStateException for a classification forest
     */
    public double predictValue(double[] x) {
        return predictMeanVariance(x)[0];
    }

    /**
     * The regression forest's predictive mean and variance: the law of total variance over
     * the trees, each tree a Gaussian with its leaf's mean and within-leaf variance.
     *
     * @param x a row
     * @return {@code {mean, variance}}
     * @throws IllegalStateException for a classification forest
     */
    public double[] predictMeanVariance(double[] x) {
        if (!regression) {
            throw new IllegalStateException("not a regression forest");
        }
        double m = 0;
        double s = 0;
        for (Tree t : trees) {
            double[] l = leaf(t, x);
            m += l[0];
            s += l[1] + l[0] * l[0];
        }
        m /= trees.length;
        return new double[] {m, Math.max(0, s / trees.length - m * m)};
    }

    /** @return whether this is a regression forest */
    public boolean regression() {
        return regression;
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

    /** @return the out-of-bag error rate, or mean squared error for regression (NaN for extra trees) */
    public double oobError() {
        return oobError;
    }

    /** @return the number of trees */
    public int size() {
        return trees.length;
    }
}
