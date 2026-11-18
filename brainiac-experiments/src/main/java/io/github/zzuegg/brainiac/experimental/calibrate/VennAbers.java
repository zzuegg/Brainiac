package io.github.zzuegg.brainiac.experimental.calibrate;

import java.util.Arrays;

/**
 * Venn–Abers predictors (Vovk &amp; Petej, UAI 2014,
 * <a href="https://arxiv.org/abs/1211.0025">arXiv:1211.0025</a>; Vovk, Petej &amp; Fedorova,
 * NeurIPS 2015): isotonic calibration made distribution-free. For a test score {@code s} the
 * inductive predictor fits isotonic regression to the calibration scores twice, once with
 * {@code (s, 0)} added and once with {@code (s, 1)}, and reads off {@code p0 <= p1} at
 * {@code s}; under exchangeability one of the two is perfectly calibrated. The width
 * {@code p1 - p0} is itself an uncertainty measure and {@code p1 / (1 - p0 + p1)} is the
 * single probability that minimises the worst-case log loss.
 *
 * <ul>
 *   <li>{@link #fit}: the inductive (IVAP) binary predictor;</li>
 *   <li>{@link #cross}: cross Venn–Abers (CVAP), merging the folds' intervals with geometric
 *       means, {@code GM(p1) / (GM(1 - p0) + GM(p1))};</li>
 *   <li>{@link #oneVsRest}: multiclass by one-vs-rest IVAPs on {@code p_k}, merged and
 *       renormalised, with the mean interval width per sample.</li>
 * </ul>
 */
public final class VennAbers {

    private final double[] xs;
    private final double[] sums;
    private final double[] counts;

    private VennAbers(double[] xs, double[] sums, double[] counts) {
        this.xs = xs;
        this.sums = sums;
        this.counts = counts;
    }

    /**
     * Fits the inductive predictor on calibration scores.
     *
     * @param scores calibration scores (any monotone score of the positive class)
     * @param labels 0/1 labels
     * @return the predictor
     */
    public static VennAbers fit(double[] scores, int[] labels) {
        int n = scores.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        double[] x = new double[n];
        double[] s = new double[n];
        double[] c = new double[n];
        int m = 0;
        for (int i = 0; i < n; i++) {
            double v = scores[order[i]];
            if (m > 0 && x[m - 1] == v) {
                s[m - 1] += labels[order[i]];
                c[m - 1]++;
            } else {
                x[m] = v;
                s[m] = labels[order[i]];
                c[m] = 1;
                m++;
            }
        }
        return new VennAbers(Arrays.copyOf(x, m), Arrays.copyOf(s, m), Arrays.copyOf(c, m));
    }

    /**
     * The Venn–Abers interval of a test score.
     *
     * @param score the test score
     * @return {@code {p0, p1}}
     */
    public double[] interval(double score) {
        return new double[] {valueWith(score, 0), valueWith(score, 1)};
    }

    /**
     * The merged (log-loss optimal) probability of a test score.
     *
     * @param score the test score
     * @return {@code p1 / (1 - p0 + p1)}
     */
    public double probability(double score) {
        double[] p = interval(score);
        return merge(p[0], p[1]);
    }

    /**
     * Merges an interval into one probability.
     *
     * @param p0 the lower probability
     * @param p1 the upper probability
     * @return {@code p1 / (1 - p0 + p1)}
     */
    public static double merge(double p0, double p1) {
        return p1 / (1 - p0 + p1);
    }

    private double valueWith(double score, int label) {
        int m = xs.length;
        double[] x = new double[m + 1];
        double[] y = new double[m + 1];
        double[] w = new double[m + 1];
        int pos = Arrays.binarySearch(xs, score);
        int at;
        int len;
        if (pos >= 0) {
            System.arraycopy(xs, 0, x, 0, m);
            for (int i = 0; i < m; i++) {
                y[i] = sums[i] / counts[i];
                w[i] = counts[i];
            }
            y[pos] = (sums[pos] + label) / (counts[pos] + 1);
            w[pos] = counts[pos] + 1;
            at = pos;
            len = m;
        } else {
            at = -pos - 1;
            for (int i = 0, j = 0; i <= m; i++) {
                if (i == at) {
                    x[i] = score;
                    y[i] = label;
                    w[i] = 1;
                } else {
                    x[i] = xs[j];
                    y[i] = sums[j] / counts[j];
                    w[i] = counts[j];
                    j++;
                }
            }
            len = m + 1;
        }
        // PAV, tracking which block holds the test point
        double[] by = new double[len];
        double[] bw = new double[len];
        int[] start = new int[len];
        int b = 0;
        for (int i = 0; i < len; i++) {
            by[b] = y[i] * w[i];
            bw[b] = w[i];
            start[b] = i;
            b++;
            while (b > 1 && by[b - 2] / bw[b - 2] >= by[b - 1] / bw[b - 1]) {
                by[b - 2] += by[b - 1];
                bw[b - 2] += bw[b - 1];
                b--;
            }
        }
        for (int q = b - 1; q >= 0; q--) {
            if (start[q] <= at) {
                return by[q] / bw[q];
            }
        }
        return by[0] / bw[0];
    }

    /**
     * Cross Venn–Abers: merges the intervals the fold predictors give for the same test points.
     *
     * @param folds the inductive predictors, one per fold (each calibrated on its fold, with its
     *     own model's scores)
     * @param testScores {@code [fold][test]}: each fold model's score of every test point
     * @return the merged probabilities of the test points
     */
    public static double[] cross(VennAbers[] folds, double[][] testScores) {
        int m = testScores[0].length;
        double[] out = new double[m];
        for (int i = 0; i < m; i++) {
            double lq = 0;
            double lp = 0;
            for (int f = 0; f < folds.length; f++) {
                double[] p = folds[f].interval(testScores[f][i]);
                lq += Math.log(Math.max(1e-12, 1 - p[0]));
                lp += Math.log(Math.max(1e-12, p[1]));
            }
            double gq = Math.exp(lq / folds.length);
            double gp = Math.exp(lp / folds.length);
            out[i] = gp / (gq + gp);
        }
        return out;
    }

    /**
     * Multiclass probabilities and interval widths by one-vs-rest inductive Venn–Abers.
     *
     * @param calibration calibration probabilities {@code [n][k]}
     * @param labels calibration labels
     * @param test test probabilities {@code [m][k]}
     * @return {@code {probabilities [m][k], {width per sample} [1][m]}}: renormalised merged
     *     probabilities, and the mean over classes of {@code p1 - p0}
     */
    public static double[][][] oneVsRest(double[][] calibration, int[] labels, double[][] test) {
        int k = calibration[0].length;
        int n = calibration.length;
        int m = test.length;
        double[][] raw = new double[m][k];
        double[][] width = new double[1][m];
        for (int c = 0; c < k; c++) {
            double[] s = new double[n];
            int[] y = new int[n];
            for (int i = 0; i < n; i++) {
                s[i] = calibration[i][c];
                y[i] = labels[i] == c ? 1 : 0;
            }
            VennAbers va = fit(s, y);
            for (int i = 0; i < m; i++) {
                double[] p = va.interval(test[i][c]);
                raw[i][c] = merge(p[0], p[1]);
                width[0][i] += (p[1] - p[0]) / k;
            }
        }
        return new double[][][] {AffineLogitFit.renormalise(raw, 1e-9), width};
    }
}
