package io.github.zzuegg.brainiac.experimental.synth;

import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.Arrays;

/**
 * Utility and privacy of a synthetic table (Kotelnikov et al., TabDDPM, ICML 2023,
 * <a href="https://arxiv.org/abs/2209.15421">arXiv:2209.15421</a>).
 * <ul>
 *   <li>{@link #utility}: train on synthetic, test on real ("TSTR"): gradient-boosted trees fitted
 *       on the synthetic rows predict a binary label column of real test rows; reports accuracy
 *       and the recall of the rare class;</li>
 *   <li>{@link #closestRecordRatio}: the median distance of each synthetic row to its closest real
 *       training row, over the same for real holdout rows; well below 1 means the generator
 *       copies training rows.</li>
 * </ul>
 */
public final class SyntheticDataAudit {

    private SyntheticDataAudit() {
    }

    /**
     * Train-on-synthetic, test-on-real scores.
     *
     * @param accuracy test accuracy
     * @param rareRecall recall of the positive (rare) class
     * @param balancedAccuracy the mean of both classes' recalls
     */
    public record Utility(double accuracy, double rareRecall, double balancedAccuracy) {
    }

    /**
     * Fits trees on one table and scores them on another.
     *
     * @param train the (synthetic) training table
     * @param test the real test table
     * @param label the categorical column holding the 0/1 label
     * @param stats numeric statistics for the encoding (from the real training table)
     * @param trees boosting rounds
     * @return the scores
     */
    public static Utility utility(MixedTable train, MixedTable test, int label, double[][] stats, int trees) {
        double[][] x = train.encode(stats, label);
        int[] yi = train.column(label);
        double[] y = new double[yi.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = yi[i];
        }
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(x, y, GradientBoostedTrees.Objective.BINARY,
                GradientBoostedTrees.Settings.defaults().withTrees(trees).withDepth(3));
        double[] p = gbt.predict(test.encode(stats, label));
        int[] truth = test.column(label);
        int ok = 0;
        int pos = 0;
        int tp = 0;
        int neg = 0;
        int tn = 0;
        for (int i = 0; i < p.length; i++) {
            int pred = p[i] > 0.5 ? 1 : 0;
            ok += pred == truth[i] ? 1 : 0;
            if (truth[i] == 1) {
                pos++;
                tp += pred;
            } else {
                neg++;
                tn += 1 - pred;
            }
        }
        double recall = pos == 0 ? 0 : tp / (double) pos;
        double spec = neg == 0 ? 0 : tn / (double) neg;
        return new Utility(ok / (double) p.length, recall, 0.5 * (recall + spec));
    }

    /**
     * Distances of rows to their closest record in a reference table (encoded features,
     * Euclidean).
     *
     * @param rows the rows
     * @param reference the reference rows
     * @return the closest distance per row
     */
    public static double[] closestRecordDistances(double[][] rows, double[][] reference) {
        double[] d = new double[rows.length];
        for (int i = 0; i < rows.length; i++) {
            double best = Double.MAX_VALUE;
            for (double[] r : reference) {
                double s = 0;
                for (int k = 0; k < r.length && s < best; k++) {
                    s += (rows[i][k] - r[k]) * (rows[i][k] - r[k]);
                }
                best = Math.min(best, s);
            }
            d[i] = Math.sqrt(best);
        }
        return d;
    }

    /**
     * The closest-record distance ratio: median distance of synthetic rows to the real training
     * rows over the median distance of real holdout rows to them.
     *
     * @param synthetic the synthetic table
     * @param train the real training table
     * @param holdout real rows not used for training
     * @param stats numeric statistics for the encoding
     * @return the ratio
     */
    public static double closestRecordRatio(MixedTable synthetic, MixedTable train, MixedTable holdout,
            double[][] stats) {
        double[][] ref = train.encode(stats, -1);
        double s = median(closestRecordDistances(synthetic.encode(stats, -1), ref));
        double h = median(closestRecordDistances(holdout.encode(stats, -1), ref));
        return h == 0 ? Double.POSITIVE_INFINITY : s / h;
    }

    private static double median(double[] v) {
        double[] c = v.clone();
        Arrays.sort(c);
        return c.length == 0 ? 0 : c[c.length / 2];
    }
}
