package io.github.zzuegg.brainiac.experimental.boost;

import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Quantile regression forests (Meinshausen, "Quantile Regression Forests", JMLR 7, 2006,
 * <a href="https://jmlr.org/papers/v7/meinshausen06a.html">paper</a>). A random forest's leaves
 * hold training rows; instead of averaging their targets, the forest keeps them and weights each
 * training target {@code y_i} by how often it shares a leaf with the query, normalised per leaf:
 * {@code w_i(x) = (1/T) sum_t 1[i in leaf_t(x)] / |leaf_t(x)|}. The weighted empirical
 * distribution is a full conditional distribution, so any quantile, interval or CRPS comes from
 * one fitted forest with no distributional assumption (bimodal noise included).
 *
 * <p>The trees are {@link RegressionForest}'s; {@link #conformalize} adds conformalised quantile
 * regression (Romano et al. 2019, <a href="https://arxiv.org/abs/1905.03222">arXiv:1905.03222</a>)
 * on a calibration set for finite-sample coverage.
 */
public final class QuantileForest {

    private final RegressionForest forest;
    private final RegressionForest.Tree[] trees;
    private final int[][][] members;   // [tree][node] -> training rows in that leaf
    private final double[] y;

    private QuantileForest(RegressionForest forest, int[][][] members, double[] y) {
        this.forest = forest;
        this.trees = forest.trees();
        this.members = members;
        this.y = y;
    }

    /**
     * Grows the forest and records every training row's leaf in every tree.
     *
     * @param x rows
     * @param y targets
     * @param s the forest settings
     * @return the quantile forest
     */
    public static QuantileForest fit(double[][] x, double[] y, RegressionForest.Settings s) {
        RegressionForest forest = RegressionForest.fit(x, y, null, s);
        RegressionForest.Tree[] trees = forest.trees();
        int[][][] members = new int[trees.length][][];
        for (int t = 0; t < trees.length; t++) {
            int nodes = trees[t].feature().length;
            List<List<Integer>> lists = new ArrayList<>();
            for (int k = 0; k < nodes; k++) {
                lists.add(new ArrayList<>());
            }
            for (int i = 0; i < x.length; i++) {
                lists.get(trees[t].leaf(x[i])).add(i);
            }
            members[t] = new int[nodes][];
            for (int k = 0; k < nodes; k++) {
                members[t][k] = lists.get(k).stream().mapToInt(Integer::intValue).toArray();
            }
        }
        return new QuantileForest(forest, members, y.clone());
    }

    /**
     * The conditional distribution at a query.
     *
     * @param x a row
     * @return training targets sorted ascending ({@code [0]}) and their weights ({@code [1]}),
     *     summing to one
     */
    public double[][] distribution(double[] x) {
        double[] w = new double[y.length];
        int used = 0;
        for (int t = 0; t < trees.length; t++) {
            int[] leaf = members[t][trees[t].leaf(x)];
            if (leaf.length == 0) {
                continue;
            }
            used++;
            for (int i : leaf) {
                w[i] += 1.0 / leaf.length;
            }
        }
        int nonzero = 0;
        for (double v : w) {
            nonzero += v > 0 ? 1 : 0;
        }
        Integer[] idx = new Integer[nonzero];
        for (int i = 0, k = 0; i < w.length; i++) {
            if (w[i] > 0) {
                idx[k++] = i;
            }
        }
        Arrays.sort(idx, (a, b) -> Double.compare(y[a], y[b]));
        double[] vals = new double[nonzero];
        double[] ws = new double[nonzero];
        for (int k = 0; k < nonzero; k++) {
            vals[k] = y[idx[k]];
            ws[k] = w[idx[k]] / Math.max(1, used);
        }
        return new double[][] {vals, ws};
    }

    /**
     * @param x a row
     * @param p a probability
     * @return the weighted {@code p}-quantile of the conditional distribution
     */
    public double quantile(double[] x, double p) {
        double[][] d = distribution(x);
        double c = 0;
        for (int k = 0; k < d[0].length; k++) {
            c += d[1][k];
            if (c >= p - 1e-12) {
                return d[0][k];
            }
        }
        return d[0][d[0].length - 1];
    }

    /**
     * The CRPS of the weighted empirical distribution:
     * {@code sum_i w_i |y_i - y| - 1/2 sum_ij w_i w_j |y_i - y_j|}.
     *
     * @param x a row
     * @param target the observed value
     * @return the CRPS
     */
    public double crps(double[] x, double target) {
        double[][] d = distribution(x);
        double[] v = d[0];
        double[] w = d[1];
        double a = 0;
        for (int k = 0; k < v.length; k++) {
            a += w[k] * Math.abs(v[k] - target);
        }
        // sum_ij w_i w_j |v_i - v_j| for sorted v in O(m): 2 sum_j w_j (v_j W_<j - S_<j)
        double cw = 0;
        double cs = 0;
        double pair = 0;
        for (int k = 0; k < v.length; k++) {
            pair += w[k] * (v[k] * cw - cs);
            cw += w[k];
            cs += w[k] * v[k];
        }
        return a - pair;
    }

    /**
     * @param x a row
     * @return the forest's mean prediction
     */
    public double mean(double[] x) {
        return forest.predict(x);
    }

    /**
     * Conformalised quantile regression on a calibration set: the interval
     * {@code [q_lo - c, q_hi + c]} with {@code c} the {@code ceil((n + 1)(1 - alpha)) / n}
     * empirical quantile of the conformity scores {@code max(q_lo - y, y - q_hi)}.
     *
     * @param calibrationX calibration rows (not used for growing)
     * @param calibrationY their targets
     * @param alpha the miscoverage, such as 0.1 for a 90% interval
     * @return the correction {@code c} to add on each side
     */
    public double conformalize(double[][] calibrationX, double[] calibrationY, double alpha) {
        int n = calibrationY.length;
        double[] scores = new double[n];
        for (int i = 0; i < n; i++) {
            double lo = quantile(calibrationX[i], alpha / 2);
            double hi = quantile(calibrationX[i], 1 - alpha / 2);
            scores[i] = Math.max(lo - calibrationY[i], calibrationY[i] - hi);
        }
        Arrays.sort(scores);
        int k = (int) Math.ceil((n + 1) * (1 - alpha)) - 1;
        return scores[Math.clamp(k, 0, n - 1)];
    }
}
