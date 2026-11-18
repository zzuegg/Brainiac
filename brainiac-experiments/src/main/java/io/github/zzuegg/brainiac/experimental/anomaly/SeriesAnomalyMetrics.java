package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Scores for time-series anomaly detectors, including the point-adjustment pitfall.
 * <ul>
 *   <li>{@link #f1}: point-wise F1 of binary predictions;</li>
 *   <li>{@link #pointAdjust}: the protocol popular since 2018 that credits a whole anomalous
 *       segment as detected if any one of its points is flagged. Kim et al. (AAAI 2022,
 *       <a href="https://arxiv.org/abs/2109.05257">arXiv:2109.05257</a>) showed that it
 *       lets even random scores look state of the art, so report both;</li>
 *   <li>{@link #bestF1}: the best F1 over thresholds (the usual "oracle threshold" protocol),
 *       with or without point adjustment;</li>
 *   <li>{@link #rangeAuc}: the area under the ROC curve with labels softened by a linear
 *       ramp of {@code buffer} steps around each anomalous range (Paparrizos et al., "Volume
 *       under the surface", VLDB 2022), so detections just before or after a segment earn
 *       partial credit and no threshold is chosen;</li>
 *   <li>{@link #randomScores}: the random-score baseline of Kim et al.</li>
 * </ul>
 */
public final class SeriesAnomalyMetrics {

    private SeriesAnomalyMetrics() {
    }

    /**
     * Point-wise F1.
     *
     * @param predicted flagged steps
     * @param truth anomalous steps
     * @return the F1 score (0 when nothing is flagged or nothing is anomalous)
     */
    public static double f1(boolean[] predicted, boolean[] truth) {
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < truth.length; i++) {
            if (predicted[i] && truth[i]) {
                tp++;
            } else if (predicted[i]) {
                fp++;
            } else if (truth[i]) {
                fn++;
            }
        }
        return tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
    }

    /**
     * Point adjustment: every step of a true anomalous segment counts as flagged if any of
     * them is.
     *
     * @param predicted flagged steps
     * @param truth anomalous steps
     * @return the adjusted predictions
     */
    public static boolean[] pointAdjust(boolean[] predicted, boolean[] truth) {
        boolean[] out = predicted.clone();
        int i = 0;
        while (i < truth.length) {
            if (!truth[i]) {
                i++;
                continue;
            }
            int j = i;
            boolean hit = false;
            while (j < truth.length && truth[j]) {
                hit |= predicted[j];
                j++;
            }
            if (hit) {
                Arrays.fill(out, i, j, true);
            }
            i = j;
        }
        return out;
    }

    /**
     * The best F1 over thresholds at the score quantiles.
     *
     * @param scores higher means more anomalous
     * @param truth anomalous steps
     * @param adjusted whether to apply point adjustment before scoring
     * @return the best F1
     */
    public static double bestF1(double[] scores, boolean[] truth, boolean adjusted) {
        double[] sorted = scores.clone();
        Arrays.sort(sorted);
        double best = 0;
        boolean[] predicted = new boolean[scores.length];
        for (int k = 0; k < 200; k++) {
            double threshold = sorted[(int) Math.min(sorted.length - 1, (long) (sorted.length * (0.5 + 0.4999 * k
                    / 199.0)))];
            for (int i = 0; i < scores.length; i++) {
                predicted[i] = scores[i] >= threshold;
            }
            double f = f1(adjusted ? pointAdjust(predicted, truth) : predicted, truth);
            best = Math.max(best, f);
        }
        return best;
    }

    /**
     * Soft labels: 1 inside anomalous ranges, a linear ramp down to 0 over {@code buffer} steps
     * on each side.
     *
     * @param truth anomalous steps
     * @param buffer the ramp length
     * @return labels in {@code [0, 1]}
     */
    public static double[] softLabels(boolean[] truth, int buffer) {
        double[] w = new double[truth.length];
        for (int i = 0; i < truth.length; i++) {
            if (truth[i]) {
                w[i] = 1;
                for (int d = 1; d <= buffer; d++) {
                    double v = 1 - d / (buffer + 1.0);
                    if (i - d >= 0) {
                        w[i - d] = Math.max(w[i - d], v);
                    }
                    if (i + d < truth.length) {
                        w[i + d] = Math.max(w[i + d], v);
                    }
                }
            }
        }
        return w;
    }

    /**
     * The range AUC: the ROC area with soft labels (each step a positive with weight {@code w}
     * and a negative with weight {@code 1 - w}).
     *
     * @param scores higher means more anomalous
     * @param truth anomalous steps
     * @param buffer the ramp length (0 gives the plain AUROC)
     * @return the area in {@code [0, 1]}
     */
    public static double rangeAuc(double[] scores, boolean[] truth, int buffer) {
        double[] w = softLabels(truth, buffer);
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        double negBelow = 0;
        double area = 0;
        double totalPos = 0;
        double totalNeg = 0;
        int i = 0;
        while (i < order.length) {
            int j = i;
            double pos = 0;
            double neg = 0;
            while (j < order.length && scores[order[j]] == scores[order[i]]) {
                pos += w[order[j]];
                neg += 1 - w[order[j]];
                j++;
            }
            area += pos * (negBelow + neg / 2);
            negBelow += neg;
            totalPos += pos;
            totalNeg += neg;
            i = j;
        }
        return area / Math.max(1e-12, totalPos * totalNeg);
    }

    /**
     * Uniform random scores, the baseline that point adjustment flatters.
     *
     * @param n steps
     * @param seed the seed
     * @return scores in {@code [0, 1)}
     */
    public static double[] randomScores(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = r.nextDouble();
        }
        return s;
    }

    /**
     * Spreads per-window scores to the steps they cover (each step takes the largest score of
     * the windows containing it).
     *
     * @param windowScores one score per window start
     * @param window the window length
     * @param length the series length
     * @return per-step scores
     */
    public static double[] perStep(double[] windowScores, int window, int length) {
        double[] s = new double[length];
        Arrays.fill(s, Double.NEGATIVE_INFINITY);
        for (int i = 0; i < windowScores.length; i++) {
            for (int t = i; t < Math.min(length, i + window); t++) {
                s[t] = Math.max(s[t], windowScores[i]);
            }
        }
        double min = Arrays.stream(s).filter(Double::isFinite).min().orElse(0);
        for (int t = 0; t < length; t++) {
            if (!Double.isFinite(s[t])) {
                s[t] = min;
            }
        }
        return s;
    }
}
