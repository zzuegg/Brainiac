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
        BINARY;

        /**
         * Returns this objective as a {@link Loss}, to wrap or to reuse its derivatives.
         *
         * @return the loss
         */
        public Loss loss() {
            return GradientBoostedTrees.loss(this);
        }
    }

    /**
     * A training loss given by its first and second derivatives with respect to the raw
     * score, as XGBoost's custom objectives: quantile, Huber, Poisson, ranking or any other.
     */
    public interface Loss {

        /**
         * Fills each row's gradient and curvature of the loss at the current raw scores.
         *
         * @param raw the current raw scores
         * @param y the targets
         * @param gradient receives the first derivatives
         * @param hessian receives the second derivatives; positive values keep leaves stable
         */
        void derivatives(double[] raw, double[] y, double[] gradient, double[] hessian);

        /**
         * @param y the targets
         * @return the raw score every row starts from
         */
        default double base(double[] y) {
            return 0;
        }

        /**
         * @param raw a raw score
         * @return the prediction for it, such as a probability or a rate
         */
        default double link(double raw) {
            return raw;
        }
    }

    private static Loss loss(Objective objective) {
        return switch (objective) {
            case REGRESSION -> new Loss() {
                @Override
                public void derivatives(double[] raw, double[] y, double[] g, double[] h) {
                    for (int i = 0; i < raw.length; i++) {
                        g[i] = raw[i] - y[i];
                        h[i] = 1;
                    }
                }

                @Override
                public double base(double[] y) {
                    return Arrays.stream(y).average().orElse(0);
                }
            };
            case BINARY -> new Loss() {
                @Override
                public void derivatives(double[] raw, double[] y, double[] g, double[] h) {
                    for (int i = 0; i < raw.length; i++) {
                        double p = 1 / (1 + Math.exp(-raw[i]));
                        g[i] = p - y[i];
                        h[i] = Math.max(p * (1 - p), 1e-12);
                    }
                }

                @Override
                public double base(double[] y) {
                    double mean = Math.clamp(Arrays.stream(y).average().orElse(0.5), 1e-6, 1 - 1e-6);
                    return Math.log(mean / (1 - mean));
                }

                @Override
                public double link(double raw) {
                    return 1 / (1 + Math.exp(-raw));
                }
            };
        };
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

    /**
     * A tree as arrays, node 0 the root: a node splits when {@code feature >= 0}, sending rows
     * with {@code x[feature] < threshold} to {@code left} and the rest to {@code right}; else it
     * is a leaf worth {@code value} (before the learning rate).
     *
     * @param feature the split feature per node, or -1 for a leaf
     * @param threshold the split value per node
     * @param left the left child per node
     * @param right the right child per node
     * @param value the leaf value per node
     */
    public record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[] value) {

        /** @return the number of nodes */
        public int nodes() {
            return feature.length;
        }

        private Tree copy() {
            return new Tree(feature.clone(), threshold.clone(), left.clone(), right.clone(), value.clone());
        }

        double predict(double[] x) {
            int n = 0;
            while (feature[n] >= 0) {
                n = x[feature[n]] < threshold[n] ? left[n] : right[n];
            }
            return value[n];
        }
    }

    private final Loss loss;
    private final double base;
    private final List<Tree> trees;
    private final double learningRate;
    private final double[] importance;

    private GradientBoostedTrees(Loss loss, double base, List<Tree> trees, double learningRate,
            double[] importance) {
        this.loss = loss;
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
        return fit(x, y, loss(objective), s);
    }

    /**
     * Trains trees with a weight per row; see {@link #fit(double[][], double[], double[], Loss, Settings)}.
     *
     * @param x rows of features
     * @param y one target per row
     * @param weights one non-negative weight per row
     * @param objective the loss
     * @param s the settings
     * @return the trained ensemble
     */
    public static GradientBoostedTrees fit(double[][] x, double[] y, double[] weights, Objective objective,
            Settings s) {
        return fit(x, y, weights, loss(objective), s);
    }

    /**
     * Trains trees on a custom loss.
     *
     * @param x rows of features
     * @param y one target per row
     * @param loss the loss's derivatives, starting score and link
     * @param s the settings
     * @return the trained ensemble
     */
    public static GradientBoostedTrees fit(double[][] x, double[] y, Loss loss, Settings s) {
        return fit(x, y, null, loss, s);
    }

    /**
     * Trains trees with a weight per row, as XGBoost's sample weights: each row's
     * gradient and curvature are scaled by its weight, so a row of weight 2 counts as two
     * copies of it. The starting score is still {@link Loss#base} of the unweighted targets.
     *
     * @param x rows of features
     * @param y one target per row
     * @param weights one non-negative weight per row, or {@code null} for all 1
     * @param loss the loss
     * @param s the settings
     * @return the trained ensemble
     */
    public static GradientBoostedTrees fit(double[][] x, double[] y, double[] weights, Loss loss, Settings s) {
        return fit(x, y, weights, null, loss, s);
    }

    /**
     * Trains trees that start from a given raw score per row, as XGBoost's {@code base_margin}:
     * the trees learn a correction to an earlier model's scores, or to an offset such as a
     * log exposure. The ensemble's own starting score is then 0, so add the row's margin to
     * {@link #raw} (or pass it to {@link #predict(double[], double)}) when predicting.
     *
     * @param x rows of features
     * @param y one target per row
     * @param weights one non-negative weight per row, or {@code null} for all 1
     * @param margin one starting raw score per row, or {@code null} for {@link Loss#base}
     * @param loss the loss
     * @param s the settings
     * @return the trained ensemble
     */
    public static GradientBoostedTrees fit(double[][] x, double[] y, double[] weights, double[] margin, Loss loss,
            Settings s) {
        return fit(x, y, weights, margin, loss, s, null);
    }

    /**
     * Trains trees whose predictions are monotone in some features, as XGBoost's
     * {@code monotone_constraints}: {@code +1} for a feature whose increase never lowers the raw
     * score, {@code -1} for one whose increase never raises it, {@code 0} for no constraint. A
     * split on a constrained feature is kept only when its two leaves are in order, and the
     * leaves below it are bounded by their midpoint.
     *
     * @param x rows of features
     * @param y one target per row
     * @param weights one non-negative weight per row, or {@code null} for all 1
     * @param margin one starting raw score per row, or {@code null} for {@link Loss#base}
     * @param loss the loss
     * @param s the settings
     * @param monotone one of -1, 0 or +1 per feature, or {@code null} for none
     * @return the trained ensemble
     */
    public static GradientBoostedTrees fit(double[][] x, double[] y, double[] weights, double[] margin, Loss loss,
            Settings s, int[] monotone) {
        int n = x.length;
        if (monotone != null) {
            if (n > 0 && monotone.length != x[0].length) {
                throw new IllegalArgumentException(x[0].length + " features and " + monotone.length + " constraints");
            }
            for (int c : monotone) {
                if (c < -1 || c > 1) {
                    throw new IllegalArgumentException("a monotone constraint is -1, 0 or 1, not " + c);
                }
            }
        }
        if (margin != null && margin.length != n) {
            throw new IllegalArgumentException(n + " rows and " + margin.length + " margins");
        }
        if (weights != null) {
            if (weights.length != n) {
                throw new IllegalArgumentException(n + " rows and " + weights.length + " weights");
            }
            for (double w : weights) {
                if (!(w >= 0)) {
                    throw new IllegalArgumentException("weights must be non-negative, got " + w);
                }
            }
        }
        if (n == 0 || y.length != n) {
            throw new IllegalArgumentException(n + " rows and " + y.length + " targets");
        }
        int features = x[0].length;
        double base = margin == null ? loss.base(y) : 0;
        double[][] thresholds = new double[features][];
        for (int f = 0; f < features; f++) {
            thresholds[f] = thresholds(x, f, s.bins());
        }
        double[] raw = margin == null ? new double[n] : margin.clone();
        if (margin == null) {
            Arrays.fill(raw, base);
        }
        double[] g = new double[n];
        double[] h = new double[n];
        double[] importance = new double[features];
        List<Tree> trees = new ArrayList<>();
        SplittableRandom random = new SplittableRandom(s.seed());
        for (int t = 0; t < s.trees(); t++) {
            loss.derivatives(raw, y, g, h);
            if (weights != null) {
                for (int i = 0; i < n; i++) {
                    g[i] *= weights[i];
                    h[i] *= weights[i];
                }
            }
            int[] rows = sampleRows(n, s.subsample(), random);
            Builder b = new Builder(x, g, h, thresholds, s, importance, monotone);
            b.build(rows, 0, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);
            Tree tree = b.tree();
            trees.add(tree);
            for (int i = 0; i < n; i++) {
                raw[i] += s.learningRate() * tree.predict(x[i]);
            }
        }
        return new GradientBoostedTrees(loss, base, List.copyOf(trees), s.learningRate(), importance);
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
        final int[] monotone;

        Builder(double[][] x, double[] g, double[] h, double[][] thresholds, Settings s, double[] importance,
                int[] monotone) {
            this.monotone = monotone;
            this.x = x;
            this.g = g;
            this.h = h;
            this.thresholds = thresholds;
            this.s = s;
            this.importance = importance;
        }

        // the leaf value -G / (H + lambda), kept within the bounds a monotone split set
        double weight(double gs, double hs, double lo, double hi) {
            return Math.clamp(-gs / (hs + s.lambda()), lo, hi);
        }

        // the loss reduction of a leaf at weight w: -(2 G w + (H + lambda) w^2), G^2 / (H + lambda) at the optimum
        double score(double gs, double hs, double w) {
            return -(2 * gs * w + (hs + s.lambda()) * w * w);
        }

        int build(int[] rows, int depth, double lo, double hi) {
            int id = nodes.size();
            nodes.add(new int[] {-1, -1, -1});
            double gs = 0;
            double hs = 0;
            for (int r : rows) {
                gs += g[r];
                hs += h[r];
            }
            double own = weight(gs, hs, lo, hi);
            values.add(new double[] {0, own});
            if (depth >= s.depth() || rows.length < 2) {
                return id;
            }
            double parent = score(gs, hs, own);
            boolean bounded = monotone != null || lo > Double.NEGATIVE_INFINITY || hi < Double.POSITIVE_INFINITY;
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
                    double gain;
                    if (bounded) {
                        double wl = weight(gl, hl, lo, hi);
                        double wr = weight(gr, hr, lo, hi);
                        int c = monotone == null ? 0 : monotone[f];
                        if (c != 0 && c * (wr - wl) < 0) {
                            continue;   // the leaves would be out of order
                        }
                        gain = score(gl, hl, wl) + score(gr, hr, wr) - parent;
                    } else {
                        gain = gl * gl / (hl + s.lambda()) + gr * gr / (hr + s.lambda()) - parent;
                    }
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
            double leftLo = lo;
            double leftHi = hi;
            double rightLo = lo;
            double rightHi = hi;
            int c = monotone == null ? 0 : monotone[f];
            if (c != 0) {
                double gl = 0;
                double hl = 0;
                for (int row : left) {
                    gl += g[row];
                    hl += h[row];
                }
                double mid = (weight(gl, hl, lo, hi) + weight(gs - gl, hs - hl, lo, hi)) / 2;
                if (c > 0) {
                    leftHi = Math.min(hi, mid);
                    rightLo = Math.max(lo, mid);
                } else {
                    leftLo = Math.max(lo, mid);
                    rightHi = Math.min(hi, mid);
                }
            }
            int l = build(left, depth + 1, leftLo, leftHi);
            int r = build(right, depth + 1, rightLo, rightHi);
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
     * @return the value (regression), the probability of 1 (binary), or a custom loss's link of the score
     */
    public double predict(double[] x) {
        return loss.link(raw(x));
    }

    /**
     * Predicts for a row that starts from a margin, as trained with one.
     *
     * @param x one row of features
     * @param margin the row's starting raw score
     * @return the loss's link of the margin plus the trees' score
     */
    public double predict(double[] x, double margin) {
        return loss.link(margin + raw(x));
    }

    /**
     * @param x one row of features
     * @return the raw score before the loss's link, such as a log-odds
     */
    public double raw(double[] x) {
        double raw = base;
        for (Tree t : trees) {
            raw += learningRate * t.predict(x);
        }
        return raw;
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

    /**
     * Returns a copy of one tree, for inspection or export.
     *
     * @param i the tree, from 0
     * @return the tree
     */
    public Tree tree(int i) {
        return trees.get(i).copy();
    }

    /** @return the raw score every row starts from, before the trees */
    public double base() {
        return base;
    }

    /** @return the factor each tree's leaf values are multiplied by */
    public double learningRate() {
        return learningRate;
    }
}
