package io.github.zzuegg.brainiac.train;

import java.util.Arrays;

/**
 * Ranking scores for binary classifiers and detectors, on the host: the area under the ROC curve,
 * the ROC curve itself and average precision, as {@code sklearn.metrics}.
 */
public final class Scores {

    private Scores() {
    }

    /**
     * The area under the ROC curve: the chance that a random positive scores above a random
     * negative, ties counting half (the Mann–Whitney statistic).
     *
     * @param scores one score per sample, higher meaning more likely positive
     * @param positive whether each sample is positive
     * @return the area, in [0, 1]
     * @throws IllegalArgumentException if there are no positives or no negatives
     */
    public static double auroc(double[] scores, boolean[] positive) {
        int n = check(scores, positive);
        Integer[] order = order(scores);
        double rankSum = 0;
        long pos = 0;
        int i = 0;
        while (i < n) {
            int j = i;
            while (j + 1 < n && scores[order[j + 1]] == scores[order[i]]) {
                j++;
            }
            double rank = (i + j) / 2.0 + 1;   // ascending, ties share the mean rank
            for (int k = i; k <= j; k++) {
                if (positive[order[k]]) {
                    rankSum += rank;
                    pos++;
                }
            }
            i = j + 1;
        }
        long neg = n - pos;
        return (rankSum - pos * (pos + 1) / 2.0) / ((double) pos * neg);
    }

    /**
     * The ROC curve at every distinct threshold, from (0, 0) to (1, 1).
     *
     * @param scores one score per sample
     * @param positive whether each sample is positive
     * @return {false positive rates, true positive rates, thresholds}; the first threshold is
     *     {@code +Infinity}
     */
    public static double[][] roc(double[] scores, boolean[] positive) {
        int n = check(scores, positive);
        Integer[] order = order(scores);
        long pos = 0;
        for (boolean p : positive) {
            pos += p ? 1 : 0;
        }
        long neg = n - pos;
        double[] fpr = new double[n + 1];
        double[] tpr = new double[n + 1];
        double[] thr = new double[n + 1];
        thr[0] = Double.POSITIVE_INFINITY;
        int points = 1;
        long tp = 0;
        long fp = 0;
        for (int i = n - 1; i >= 0; i--) {
            if (positive[order[i]]) {
                tp++;
            } else {
                fp++;
            }
            if (i == 0 || scores[order[i - 1]] != scores[order[i]]) {
                fpr[points] = (double) fp / neg;
                tpr[points] = (double) tp / pos;
                thr[points] = scores[order[i]];
                points++;
            }
        }
        return new double[][] {Arrays.copyOf(fpr, points), Arrays.copyOf(tpr, points), Arrays.copyOf(thr, points)};
    }

    /**
     * Average precision: the precision at each positive's threshold, averaged over the
     * positives (the area under the precision–recall curve, as sklearn computes it).
     *
     * @param scores one score per sample
     * @param positive whether each sample is positive
     * @return the average precision, in [0, 1]
     */
    public static double averagePrecision(double[] scores, boolean[] positive) {
        int n = check(scores, positive);
        Integer[] order = order(scores);
        long pos = 0;
        for (boolean p : positive) {
            pos += p ? 1 : 0;
        }
        double ap = 0;
        long tp = 0;
        long seen = 0;
        double lastRecall = 0;
        for (int i = n - 1; i >= 0; i--) {
            seen++;
            tp += positive[order[i]] ? 1 : 0;
            if (i == 0 || scores[order[i - 1]] != scores[order[i]]) {
                double recall = (double) tp / pos;
                ap += (recall - lastRecall) * tp / seen;
                lastRecall = recall;
            }
        }
        return ap;
    }

    private static int check(double[] scores, boolean[] positive) {
        if (scores.length != positive.length) {
            throw new IllegalArgumentException(scores.length + " scores and " + positive.length + " labels");
        }
        int pos = 0;
        for (boolean p : positive) {
            pos += p ? 1 : 0;
        }
        if (pos == 0 || pos == positive.length) {
            throw new IllegalArgumentException("needs both positives and negatives");
        }
        return scores.length;
    }

    // ascending by score
    private static Integer[] order(double[] scores) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        return order;
    }
}
