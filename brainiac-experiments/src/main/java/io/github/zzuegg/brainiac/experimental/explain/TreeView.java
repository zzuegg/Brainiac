package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.experimental.ml.RandomForest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A read-only view of a tree ensemble as flat arrays, the input of {@link TreeShap},
 * {@link ShapleyInteractions} and other tree explainers: the model is
 * {@code base + scale * sum_t tree_t(x)}, every tree routing a row left when
 * {@code x[feature] <= threshold}. Each node also has a <em>cover</em>, the number (or weight)
 * of data rows that reach it, which path-dependent TreeSHAP uses as the conditional
 * distribution; {@link #withCovers} recomputes covers from any data set.
 *
 * <p>{@link #of(RandomForest, int, double[][])} adapts {@link RandomForest} through its read-only tree
 * accessor. The core {@code GradientBoostedTrees} keeps its trees private and offers no
 * read-only accessor, so {@link #boost} grows a stochastic gradient-boosted ensemble (Friedman,
 * CSDA 2002) whose stages are single bootstrap {@code RandomForest} regression trees on the
 * residuals; it stands in for the core ensemble until that accessor exists.
 */
public final class TreeView {

    /**
     * One tree.
     *
     * @param feature the split feature per node, negative at leaves
     * @param threshold the split threshold per node
     * @param left the left child per node ({@code x <= threshold})
     * @param right the right child per node
     * @param value the leaf value per node (unused at internal nodes)
     * @param cover the data weight reaching each node
     */
    public record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[] value, double[] cover) {

        /**
         * @param x a row
         * @return the leaf value reached by the row
         */
        public double predict(double[] x) {
            int k = 0;
            while (feature[k] >= 0) {
                k = x[feature[k]] <= threshold[k] ? left[k] : right[k];
            }
            return value[k];
        }

        /**
         * @param x a row
         * @return the index of the leaf the row reaches
         */
        public int leaf(double[] x) {
            int k = 0;
            while (feature[k] >= 0) {
                k = x[feature[k]] <= threshold[k] ? left[k] : right[k];
            }
            return k;
        }

        /** @return the number of nodes */
        public int nodes() {
            return feature.length;
        }

        /** @return the depth of the deepest leaf */
        public int depth() {
            return depth(0);
        }

        private int depth(int k) {
            return feature[k] < 0 ? 0 : 1 + Math.max(depth(left[k]), depth(right[k]));
        }
    }

    private final List<Tree> trees;
    private final double base;
    private final double scale;
    private final int features;

    /**
     * Creates a view.
     *
     * @param trees the trees
     * @param base the constant added to the sum
     * @param scale the factor of each tree's output
     * @param features the number of input features
     */
    public TreeView(List<Tree> trees, double base, double scale, int features) {
        this.trees = List.copyOf(trees);
        this.base = base;
        this.scale = scale;
        this.features = features;
    }

    /**
     * Views a random forest, covers taken from the given data.
     *
     * @param forest the forest
     * @param output the class whose probability is explained (ignored for regression)
     * @param data the rows whose counts become the covers, usually the training rows
     * @return the view: the mean of the trees' class frequency (or leaf mean)
     */
    public static TreeView of(RandomForest forest, int output, double[][] data) {
        List<Tree> out = new ArrayList<>();
        int p = data[0].length;
        for (int t = 0; t < forest.size(); t++) {
            RandomForest.TreeArrays a = forest.tree(t);
            double[] value = new double[a.feature().length];
            for (int k = 0; k < value.length; k++) {
                if (a.feature()[k] < 0) {
                    value[k] = forest.regression() ? a.value()[k][0] : a.value()[k][output];
                }
            }
            out.add(new Tree(a.feature(), a.threshold(), a.left(), a.right(), value, new double[value.length]));
        }
        return new TreeView(out, 0, 1.0 / forest.size(), p).withCovers(data);
    }

    /**
     * Views a random forest with covers from the given data and class 1 explained.
     *
     * @param forest the forest
     * @param data the cover rows
     * @return the view
     */
    public static TreeView of(RandomForest forest, double[][] data) {
        return of(forest, 1, data);
    }

    /**
     * Stochastic gradient boosting with squared loss: each stage fits one depth-limited
     * {@link RandomForest} regression tree (all features per split, a bootstrap sample) to the
     * residuals, shrunk by the learning rate.
     *
     * @param x the rows
     * @param y the targets (0/1 labels give a probability-like score)
     * @param stages the number of trees
     * @param depth the depth of each tree
     * @param learningRate the shrinkage
     * @param seed the seed
     * @return the ensemble, covers from {@code x}
     */
    public static TreeView boost(double[][] x, double[] y, int stages, int depth, double learningRate, long seed) {
        int n = x.length;
        int p = x[0].length;
        double base = Arrays.stream(y).average().orElse(0);
        double[] f = new double[n];
        Arrays.fill(f, base);
        List<Tree> out = new ArrayList<>();
        for (int s = 0; s < stages; s++) {
            double[] residual = new double[n];
            for (int i = 0; i < n; i++) {
                residual[i] = y[i] - f[i];
            }
            RandomForest stage = RandomForest.fitRegression(x, residual,
                    new RandomForest.Settings(RandomForest.Mode.RANDOM_FOREST, 1, depth, 3, p, seed * 7919 + s));
            Tree t = of(stage, 0, x).trees.get(0);
            out.add(t);
            for (int i = 0; i < n; i++) {
                f[i] += learningRate * t.predict(x[i]);
            }
        }
        return new TreeView(out, base, learningRate, p).withCovers(x);
    }

    /**
     * Recomputes every node's cover as the number of the given rows that reach it.
     *
     * @param data the rows
     * @return a new view with those covers
     */
    public TreeView withCovers(double[][] data) {
        List<Tree> out = new ArrayList<>();
        for (Tree t : trees) {
            double[] cover = new double[t.nodes()];
            for (double[] row : data) {
                int k = 0;
                cover[0]++;
                while (t.feature()[k] >= 0) {
                    k = row[t.feature()[k]] <= t.threshold()[k] ? t.left()[k] : t.right()[k];
                    cover[k]++;
                }
            }
            out.add(new Tree(t.feature(), t.threshold(), t.left(), t.right(), t.value(), cover));
        }
        return new TreeView(out, base, scale, features);
    }

    /** @return the trees */
    public List<Tree> trees() {
        return trees;
    }

    /** @return the constant term */
    public double base() {
        return base;
    }

    /** @return the factor of each tree's output */
    public double scale() {
        return scale;
    }

    /** @return the number of input features */
    public int features() {
        return features;
    }

    /**
     * @param x a row
     * @return {@code base + scale * sum_t tree_t(x)}
     */
    public double predict(double[] x) {
        double s = 0;
        for (Tree t : trees) {
            s += t.predict(x);
        }
        return base + scale * s;
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

    /** @return the total number of leaves */
    public int leaves() {
        int n = 0;
        for (Tree t : trees) {
            for (int f : t.feature()) {
                if (f < 0) {
                    n++;
                }
            }
        }
        return n;
    }

    /**
     * The split-count importance: how many internal nodes split on each feature, weighted by
     * cover and normalised to sum to 1, a gain-like global importance.
     *
     * @return one value per feature
     */
    public double[] splitImportance() {
        double[] imp = new double[features];
        for (Tree t : trees) {
            for (int k = 0; k < t.nodes(); k++) {
                if (t.feature()[k] >= 0) {
                    imp[t.feature()[k]] += t.cover()[k];
                }
            }
        }
        double s = Arrays.stream(imp).sum();
        for (int j = 0; j < features; j++) {
            imp[j] = s > 0 ? imp[j] / s : 0;
        }
        return imp;
    }
}
