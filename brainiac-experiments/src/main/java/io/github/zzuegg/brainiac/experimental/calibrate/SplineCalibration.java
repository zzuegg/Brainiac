package io.github.zzuegg.brainiac.experimental.calibrate;

import java.util.Arrays;

/**
 * Spline calibration of the top-label confidence (Gupta et al., ICLR 2021,
 * <a href="https://arxiv.org/abs/2006.12800">arXiv:2006.12800</a>), a bin-free
 * Kolmogorov–Smirnov view: sort the validation samples by confidence, let {@code t = i/n} be
 * the rank fraction, and fit a cubic spline {@code s(t)} by least squares to the cumulative
 * accuracy {@code (1/n) Σ_{j<=i} correct_j}. Its derivative {@code s'(t)} is the calibrated
 * accuracy at rank {@code t}; a test confidence is mapped to its rank fraction through the
 * validation confidences' empirical CDF (ranked by the top label's log-odds, which unlike the
 * probability does not saturate at 1 for overconfident models), its top probability set to {@code s'(t)} (clipped to
 * {@code [1/k, 1]}) and the other classes rescaled to the remaining mass.
 *
 * <p>The spline uses the truncated-power cubic basis with {@code knots} interior knots at
 * equal rank fractions and a tiny ridge.
 */
public final class SplineCalibration implements Recalibrator {

    private final double[] sortedConfidence;
    private final double[] knots;
    private final double[] coef;

    private SplineCalibration(double[] sortedConfidence, double[] knots, double[] coef) {
        this.sortedConfidence = sortedConfidence;
        this.knots = knots;
        this.coef = coef;
    }

    /**
     * Fits the spline with six interior knots.
     *
     * @param logits validation logits
     * @param labels validation labels
     * @return the map
     */
    public static SplineCalibration fit(double[][] logits, int[] labels) {
        return fit(logits, labels, 6);
    }

    /**
     * Fits the spline.
     *
     * @param logits validation logits
     * @param labels validation labels
     * @param interiorKnots the number of interior knots
     * @return the map
     */
    public static SplineCalibration fit(double[][] logits, int[] labels, int interiorKnots) {
        int n = logits.length;
        double[][] p = AffineLogitFit.softmaxRows(logits);
        double[] conf = new double[n];
        double[] correct = new double[n];
        for (int i = 0; i < n; i++) {
            int top = argmax(p[i]);
            conf[i] = topLogOdds(logits[i]);
            correct[i] = top == labels[i] ? 1 : 0;
        }
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(conf[a], conf[b]));
        double[] sorted = new double[n];
        double[] t = new double[n + 1];
        double[] cum = new double[n + 1];
        for (int i = 0; i < n; i++) {
            sorted[i] = conf[order[i]];
            t[i + 1] = (i + 1) / (double) n;
            cum[i + 1] = cum[i] + correct[order[i]] / n;
        }
        double[] knots = new double[interiorKnots];
        for (int j = 0; j < interiorKnots; j++) {
            knots[j] = (j + 1) / (interiorKnots + 1.0);
        }
        int d = 4 + interiorKnots;
        double[][] gram = new double[d][d];
        double[] rhs = new double[d];
        for (int i = 0; i <= n; i++) {
            double[] b = basis(t[i], knots);
            for (int a = 0; a < d; a++) {
                rhs[a] += b[a] * cum[i];
                for (int c = 0; c < d; c++) {
                    gram[a][c] += b[a] * b[c];
                }
            }
        }
        for (int a = 0; a < d; a++) {
            gram[a][a] += 1e-8;
        }
        return new SplineCalibration(sorted, knots, Recalibrators.solve(gram, rhs));
    }

    private static double[] basis(double t, double[] knots) {
        double[] b = new double[4 + knots.length];
        b[0] = 1;
        b[1] = t;
        b[2] = t * t;
        b[3] = t * t * t;
        for (int j = 0; j < knots.length; j++) {
            double u = Math.max(0, t - knots[j]);
            b[4 + j] = u * u * u;
        }
        return b;
    }

    /**
     * The calibrated accuracy at a rank fraction, {@code s'(t)}.
     *
     * @param t the rank fraction in {@code [0, 1]}
     * @return the derivative of the fitted cumulative accuracy
     */
    public double accuracyAt(double t) {
        double s = coef[1] + 2 * coef[2] * t + 3 * coef[3] * t * t;
        for (int j = 0; j < knots.length; j++) {
            double u = Math.max(0, t - knots[j]);
            s += 3 * coef[4 + j] * u * u;
        }
        return s;
    }

    private double rank(double c) {
        int n = sortedConfidence.length;
        int i = Arrays.binarySearch(sortedConfidence, c);
        int pos = i >= 0 ? i + 1 : -i - 1;
        return Math.min(1, pos / (double) n);
    }

    @Override
    public double[][] apply(double[][] logits) {
        double[][] p = AffineLogitFit.softmaxRows(logits);
        double[][] out = new double[p.length][];
        for (int i = 0; i < p.length; i++) {
            int k = p[i].length;
            int top = argmax(p[i]);
            double c = p[i][top];
            double a = Math.min(1 - 1e-6, Math.max(1.0 / k, accuracyAt(rank(topLogOdds(logits[i])))));
            out[i] = new double[k];
            double rest = 1 - c;
            for (int j = 0; j < k; j++) {
                out[i][j] = j == top ? a : rest > 1e-12 ? p[i][j] / rest * (1 - a) : (1 - a) / (k - 1);
            }
        }
        return out;
    }

    /** The log-odds of the top label, {@code z_top - logsumexp(other logits)}, which never saturates. */
    static double topLogOdds(double[] z) {
        int top = argmax(z);
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < z.length; i++) {
            if (i != top) {
                max = Math.max(max, z[i]);
            }
        }
        double s = 0;
        for (int i = 0; i < z.length; i++) {
            if (i != top) {
                s += Math.exp(z[i] - max);
            }
        }
        return z[top] - max - Math.log(s);
    }

    static int argmax(double[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }
}
