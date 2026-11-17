package io.github.zzuegg.brainiac.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Gradient-boosted decision trees for tabular data: the usual baseline next to a neural
 * network, and often the stronger one. Each tree fits the loss's gradient and curvature
 * (second-order boosting as XGBoost does) with splits on up to {@code bins} quantile
 * thresholds per feature, on the host.
 */
public final class GradientBoostedTrees {

    /** What the trees predict. */
    public enum Objective {
        /** Squared error; predictions are values. */
        REGRESSION,
        /** Logistic loss on 0/1 targets; {@link #predict} gives probabilities. */
        BINARY
    }

    /**
     * Training settings.
     *
     * @param trees how many trees
     * @param depth the most splits from root to leaf
     * @param learningRate the shrinkage of each tree
     * @param lambda L2 regularization of leaf values
     * @param minChildWeight the least summed curvature in a leaf
     * @param subsample the share of rows each tree sees
     * @param bins thresholds tried per feature
     * @param seed seeds the row sampling
     */
    public record Settings(int trees, int depth, double learningRate, double lambda, double minChildWeight,
            double subsample, int bins, long seed) {

        /** @return 100 trees of depth 4, learning rate 0.1 */
        public static Settings defaults() {
            return new Settings(100, 4, 0.1, 1, 1e-3, 1, 32, 1);
        }

        /**
         * @param n how many trees
         * @return a copy with that many trees
         */
        public Settings withTrees(int n) {
            return new Settings(n, depth, learningRate, lambda, minChildWeight, subsample, bins, seed);
        }

        /**
         * @param d the depth
         * @return a copy with that depth
         */
        public Settings withDepth(int d) {
            return new Settings(trees, d, learningRate, lambda, minChildWeight, subsample, bins, seed);
        }
    }

    /** A tree as arrays: a node splits when {@code feature >= 0}, else it is a leaf. */
    private record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[] value) {

        double predict(double[] x) {
            int n = 0;
            while (feature[n] >= 0) {
                n = x[feature[n]] < threshold[n] ? left[n] : right[n];
            }
            return value[n];
        }
    }

    private final Objective objective;
    private final double base;
    private final List<Tree> trees;
    private final double learningRate;
    private final double[] importance;

    private GradientBoostedTrees(Objective objective, double base, List<Tree> trees, double learningRate,
            double[] importance) {
        this.objective = objective;
        this.base = base;
        this.trees = trees;
        this.learningRate = learningRate;
        this.importance = importance;
    }

    /**
     * Trains trees.
     *
     * @param x rows of features
     * @param y one target per row
     * @param objective the loss
     * @param s the settings
     * @return the trained ensemble
     */
    public static GradientBoostedTrees fit(double[][] x, double[] y, Objective objective, Settings s) {
        int n = x.length;
        if (n == 0 || y.length != n) {
            throw new IllegalArgumentException(n + " rows and " + y.length + " targets");
        }
        int features = x[0].length;
        double mean = Arrays.stream(y).average().orElse(0);
        double base = objective == Objective.REGRESSION ? mean
                : Math.log(Math.clamp(mean, 1e-6, 1 - 1e-6) / (1 - Math.clamp(mean, 1e-6, 1 - 1e-6)));
        double[][] thresholds = new double[features][];
        for (int f = 0; f < features; f++) {
            thresholds[f] = thresholds(x, f, s.bins());
        }
        double[] raw = new double[n];
        Arrays.fill(raw, base);
        double[] g = new double[n];
        double[] h = new double[n];
        double[] importance = new double[features];
        List<Tree> trees = new ArrayList<>();
        SplittableRandom random = new SplittableRandom(s.seed());
        for (int t = 0; t < s.trees(); t++) {
            for (int i = 0; i < n; i++) {
                if (objective == Objective.REGRESSION) {
                    g[i] = raw[i] - y[i];
                    h[i] = 1;
                } else {
                    double p = 1 / (1 + Math.exp(-raw[i]));
                    g[i] = p - y[i];
                    h[i] = Math.max(p * (1 - p), 1e-12);
                }
            }
            int[] rows = sampleRows(n, s.subsample(), random);
            Builder b = new Builder(x, g, h, thresholds, s, importance);
            b.build(rows, 0);
            Tree tree = b.tree();
            trees.add(tree);
            for (int i = 0; i < n; i++) {
                raw[i] += s.learningRate() * tree.predict(x[i]);
            }
        }
        return new GradientBoostedTrees(objective, base, List.copyOf(trees), s.learningRate(), importance);
    }

    private static int[] sampleRows(int n, double share, SplittableRandom random) {
        if (share >= 1) {
            int[] all = new int[n];
            Arrays.setAll(all, i -> i);
            return all;
        }
        return random.ints(0, n).limit(Math.max(1, Math.round(n * share))).distinct().sorted().toArray();
    }

    /** Midpoints between quantiles of a feature's distinct values. */
    private static double[] thresholds(double[][] x, int f, int bins) {
        double[] v = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            v[i] = x[i][f];
        }
        double[] distinct = Arrays.stream(v).sorted().distinct().toArray();
        if (distinct.length <= 1) {
            return new double[0];
        }
        int count = Math.min(bins, distinct.length - 1);
        double[] t = new double[count];
        for (int k = 0; k < count; k++) {
            int at = (int) ((long) (k + 1) * (distinct.length - 1) / (count + 1));
            at = Math.min(at, distinct.length - 2);
            t[k] = (distinct[at] + distinct[at + 1]) / 2;
        }
        return Arrays.stream(t).distinct().toArray();
    }

    private static final class Builder {
        final double[][] x;
        final double[] g;
        final double[] h;
        final double[][] thresholds;
        final Settings s;
        final double[] importance;
        final List<int[]> nodes = new ArrayList<>();   // feature, left, right
        final List<double[]> values = new ArrayList<>();   // threshold, value

        Builder(double[][] x, double[] g, double[] h, double[][] thresholds, Settings s, double[] importance) {
            this.x = x;
            this.g = g;
            this.h = h;
            this.thresholds = thresholds;
            this.s = s;
            this.importance = importance;
        }

        int build(int[] rows, int depth) {
            int id = nodes.size();
            nodes.add(new int[] {-1, -1, -1});
            double gs = 0;
            double hs = 0;
            for (int r : rows) {
                gs += g[r];
                hs += h[r];
            }
            values.add(new double[] {0, -gs / (hs + s.lambda())});
            if (depth >= s.depth() || rows.length < 2) {
                return id;
            }
            double parent = gs * gs / (hs + s.lambda());
            double bestGain = 1e-12;
            int bestF = -1;
            double bestT = 0;
            for (int f = 0; f < thresholds.length; f++) {
                double[] th = thresholds[f];
                if (th.length == 0) {
                    continue;
                }
                double[] gb = new double[th.length + 1];
                double[] hb = new double[th.length + 1];
                for (int r : rows) {
                    int bin = Arrays.binarySearch(th, x[r][f]);
                    bin = bin >= 0 ? bin + 1 : -bin - 1;   // values equal to a threshold go right
                    gb[bin] += g[r];
                    hb[bin] += h[r];
                }
                double gl = 0;
                double hl = 0;
                for (int k = 0; k < th.length; k++) {
                    gl += gb[k];
                    hl += hb[k];
                    double gr = gs - gl;
                    double hr = hs - hl;
                    if (hl < s.minChildWeight() || hr < s.minChildWeight()) {
                        continue;
                    }
                    double gain = gl * gl / (hl + s.lambda()) + gr * gr / (hr + s.lambda()) - parent;
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestF = f;
                        bestT = th[k];
                    }
                }
            }
            if (bestF < 0) {
                return id;
            }
            importance[bestF] += bestGain;
            int f = bestF;
            double t = bestT;
            int[] left = Arrays.stream(rows).filter(r -> x[r][f] < t).toArray();
            int[] right = Arrays.stream(rows).filter(r -> x[r][f] >= t).toArray();
            int l = build(left, depth + 1);
            int r = build(right, depth + 1);
            nodes.set(id, new int[] {f, l, r});
            values.get(id)[0] = t;
            return id;
        }

        Tree tree() {
            int n = nodes.size();
            int[] feature = new int[n];
            int[] left = new int[n];
            int[] right = new int[n];
            double[] threshold = new double[n];
            double[] value = new double[n];
            for (int i = 0; i < n; i++) {
                feature[i] = nodes.get(i)[0];
                left[i] = nodes.get(i)[1];
                right[i] = nodes.get(i)[2];
                threshold[i] = values.get(i)[0];
                value[i] = values.get(i)[1];
            }
            return new Tree(feature, threshold, left, right, value);
        }
    }

    /**
     * @param x one row of features
     * @return the value (regression) or the probability of 1 (binary)
     */
    public double predict(double[] x) {
        double raw = base;
        for (Tree t : trees) {
            raw += learningRate * t.predict(x);
        }
        return objective == Objective.BINARY ? 1 / (1 + Math.exp(-raw)) : raw;
    }

    /**
     * @param x rows of features
     * @return one prediction per row
     */
    public double[] predict(double[][] x) {
        return Arrays.stream(x).mapToDouble(this::predict).toArray();
    }

    /** @return each feature's summed split gain, normalized to add up to 1 */
    public double[] featureImportance() {
        double sum = Arrays.stream(importance).sum();
        return Arrays.stream(importance).map(v -> sum > 0 ? v / sum : 0).toArray();
    }

    /** @return the number of trees */
    public int size() {
        return trees.size();
    }
}
