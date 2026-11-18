package io.github.zzuegg.brainiac.experimental.train;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * LambdaMART (Burges, 2010, "From RankNet to LambdaRank to LambdaMART: An Overview",
 * MSR-TR-2010-82): gradient-boosted regression trees trained on the LambdaRank gradients of a
 * ranking metric. For every query and every pair of its documents with different grades, the
 * pair's RankNet gradient {@code rho = 1 / (1 + exp(s_i - s_j))} is scaled by {@code |delta NDCG@k|}
 * of swapping the two in the current ranking; summed per document these "lambdas" are the
 * targets of the next tree, whose leaves take a Newton step {@code sum(lambda) / sum(w)} with
 * {@code w = |delta NDCG| rho (1 - rho)}.
 *
 * <p>This is the ranking objective core {@code GradientBoostedTrees} lacks (it offers squared
 * error and logistic loss only, and no hook for custom gradients); it is written as a separate
 * booster with the same histogram-split trees so it could become a third objective there.
 */
public final class LambdaMart {

    /**
     * Training settings.
     *
     * @param trees how many trees
     * @param depth the most splits from root to leaf
     * @param learningRate the shrinkage of each tree
     * @param minLeaf the fewest rows in a leaf
     * @param bins thresholds tried per feature
     * @param k the NDCG cut-off of the lambdas
     */
    public record Settings(int trees, int depth, double learningRate, int minLeaf, int bins, int k) {

        /** @return 100 trees of depth 4, learning rate 0.1, NDCG@10 */
        public static Settings defaults() {
            return new Settings(100, 4, 0.1, 5, 32, 10);
        }
    }

    private record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[] value) {
        double predict(double[] x) {
            int n = 0;
            while (feature[n] >= 0) {
                n = x[feature[n]] < threshold[n] ? left[n] : right[n];
            }
            return value[n];
        }
    }

    private final List<Tree> trees;
    private final double learningRate;

    private LambdaMart(List<Tree> trees, double learningRate) {
        this.trees = trees;
        this.learningRate = learningRate;
    }

    /**
     * Trains the ranker.
     *
     * @param x feature rows of query-document pairs
     * @param grades the relevance grade of each row
     * @param query the query of each row; rows of one query need not be adjacent
     * @param s the settings
     * @return the ranker
     */
    public static LambdaMart fit(double[][] x, int[] grades, int[] query, Settings s) {
        int n = x.length;
        int features = x[0].length;
        int queries = Arrays.stream(query).max().orElse(0) + 1;
        List<List<Integer>> groups = new ArrayList<>();
        for (int q = 0; q < queries; q++) {
            groups.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            groups.get(query[i]).add(i);
        }
        double[][] thresholds = new double[features][];
        for (int f = 0; f < features; f++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = x[i][f];
            }
            Arrays.sort(col);
            double[] t = new double[s.bins()];
            for (int b = 0; b < s.bins(); b++) {
                t[b] = col[(int) ((b + 1) / (double) (s.bins() + 1) * (n - 1))];
            }
            thresholds[f] = Arrays.stream(t).distinct().toArray();
        }
        double[] score = new double[n];
        List<Tree> trees = new ArrayList<>();
        for (int t = 0; t < s.trees(); t++) {
            double[] lambda = new double[n];
            double[] weight = new double[n];
            for (List<Integer> g : groups) {
                lambdas(g, grades, score, s.k(), lambda, weight);
            }
            int[] rows = new int[n];
            for (int i = 0; i < n; i++) {
                rows[i] = i;
            }
            Tree tree = grow(x, lambda, weight, rows, thresholds, s);
            trees.add(tree);
            for (int i = 0; i < n; i++) {
                score[i] += s.learningRate() * tree.predict(x[i]);
            }
        }
        return new LambdaMart(trees, s.learningRate());
    }

    private static void lambdas(List<Integer> g, int[] grades, double[] score, int k, double[] lambda,
            double[] weight) {
        int m = g.size();
        Integer[] order = g.toArray(Integer[]::new);
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        double[] discount = new double[score.length];
        for (int r = 0; r < m; r++) {
            discount[order[r]] = r < k ? Math.log(2) / Math.log(r + 2) : 0;
        }
        int[] sorted = g.stream().mapToInt(i -> grades[i]).sorted().toArray();
        double idcg = 0;
        for (int r = 0; r < Math.min(k, m); r++) {
            idcg += (Math.pow(2, sorted[m - 1 - r]) - 1) * Math.log(2) / Math.log(r + 2);
        }
        if (idcg == 0) {
            return;
        }
        for (int i : g) {
            for (int j : g) {
                if (grades[i] <= grades[j]) {
                    continue;
                }
                double delta = Math.abs((Math.pow(2, grades[i]) - Math.pow(2, grades[j]))
                        * (discount[i] - discount[j]) / idcg);
                double rho = 1 / (1 + Math.exp(score[i] - score[j]));
                lambda[i] += delta * rho;
                lambda[j] -= delta * rho;
                double w = delta * rho * (1 - rho);
                weight[i] += w;
                weight[j] += w;
            }
        }
    }

    private static Tree grow(double[][] x, double[] g, double[] h, int[] rows, double[][] thresholds, Settings s) {
        List<Integer> feature = new ArrayList<>();
        List<Double> threshold = new ArrayList<>();
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        List<Double> value = new ArrayList<>();
        build(x, g, h, rows, 0, thresholds, s, feature, threshold, left, right, value);
        return new Tree(feature.stream().mapToInt(Integer::intValue).toArray(),
                threshold.stream().mapToDouble(Double::doubleValue).toArray(),
                left.stream().mapToInt(Integer::intValue).toArray(), right.stream().mapToInt(Integer::intValue).toArray(),
                value.stream().mapToDouble(Double::doubleValue).toArray());
    }

    private static int build(double[][] x, double[] g, double[] h, int[] rows, int depth, double[][] thresholds,
            Settings s, List<Integer> feature, List<Double> threshold, List<Integer> left, List<Integer> right,
            List<Double> value) {
        int node = feature.size();
        double sg = 0;
        double sh = 0;
        for (int r : rows) {
            sg += g[r];
            sh += h[r];
        }
        feature.add(-1);
        threshold.add(0.0);
        left.add(-1);
        right.add(-1);
        value.add(sg / (sh + 1e-9));
        if (depth >= s.depth() || rows.length < 2 * s.minLeaf()) {
            return node;
        }
        double base = sg * sg / rows.length;
        double best = 1e-12;
        int bestF = -1;
        double bestT = 0;
        for (int f = 0; f < thresholds.length; f++) {
            for (double t : thresholds[f]) {
                double lg = 0;
                int ln = 0;
                for (int r : rows) {
                    if (x[r][f] < t) {
                        lg += g[r];
                        ln++;
                    }
                }
                int rn = rows.length - ln;
                if (ln < s.minLeaf() || rn < s.minLeaf()) {
                    continue;
                }
                double rg = sg - lg;
                double gain = lg * lg / ln + rg * rg / rn - base;
                if (gain > best) {
                    best = gain;
                    bestF = f;
                    bestT = t;
                }
            }
        }
        if (bestF < 0) {
            return node;
        }
        final int f = bestF;
        final double t = bestT;
        int[] l = Arrays.stream(rows).filter(r -> x[r][f] < t).toArray();
        int[] rr = Arrays.stream(rows).filter(r -> x[r][f] >= t).toArray();
        feature.set(node, f);
        threshold.set(node, t);
        left.set(node, build(x, g, h, l, depth + 1, thresholds, s, feature, threshold, left, right, value));
        right.set(node, build(x, g, h, rr, depth + 1, thresholds, s, feature, threshold, left, right, value));
        return node;
    }

    /**
     * @param x a feature row
     * @return its ranking score (higher ranks first)
     */
    public double predict(double[] x) {
        double s = 0;
        for (Tree t : trees) {
            s += learningRate * t.predict(x);
        }
        return s;
    }

    /** @return the number of trees */
    public int size() {
        return trees.size();
    }
}
