package io.github.zzuegg.brainiac.experimental.anomaly;

import java.util.Arrays;

/**
 * ECOD, empirical-CDF-based outlier detection (Li, Zhao, Hu, Botta, Ionescu and Chen, TKDE
 * 2022, <a href="https://arxiv.org/abs/2201.00382">arXiv:2201.00382</a>): every feature's
 * empirical left and right tail probabilities are estimated from the reference data, and a
 * point scores the summed negative log tail probabilities over features, using the left tail,
 * the right tail, or the tail the feature's skewness points to, whichever is largest. It has
 * no parameters and no training beyond sorting each column, so it is the fastest detector
 * here and scales linearly in the dimension, but it treats features independently and
 * cannot see anomalies that only show in feature interactions.
 */
public final class Ecod {

    private final double[][] sortedColumns;
    private final double[] skewness;

    private Ecod(double[][] sortedColumns, double[] skewness) {
        this.sortedColumns = sortedColumns;
        this.skewness = skewness;
    }

    /**
     * Fits the per-feature empirical distributions.
     *
     * @param x reference points {@code [n][d]}
     * @return the detector
     */
    public static Ecod fit(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[][] cols = new double[d][n];
        double[] skew = new double[d];
        for (int j = 0; j < d; j++) {
            double mean = 0;
            for (int i = 0; i < n; i++) {
                cols[j][i] = x[i][j];
                mean += x[i][j] / n;
            }
            double m2 = 0;
            double m3 = 0;
            for (int i = 0; i < n; i++) {
                double c = x[i][j] - mean;
                m2 += c * c / n;
                m3 += c * c * c / n;
            }
            skew[j] = m3 / Math.pow(Math.max(1e-12, m2), 1.5);
            Arrays.sort(cols[j]);
        }
        return new Ecod(cols, skew);
    }

    /**
     * Scores points.
     *
     * @param x points {@code [m][d]}
     * @return the ECOD scores, higher is more anomalous
     */
    public double[] score(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double left = 0;
            double right = 0;
            double auto = 0;
            for (int j = 0; j < sortedColumns.length; j++) {
                double[] c = sortedColumns[j];
                int n = c.length;
                // F_left(v) = P(X <= v), F_right(v) = P(X >= v), both kept away from 0
                double fl = (upperBound(c, x[i][j]) + 1.0) / (n + 1.0);
                double fr = (n - lowerBound(c, x[i][j]) + 1.0) / (n + 1.0);
                double ol = -Math.log(Math.min(1, fl));
                double or = -Math.log(Math.min(1, fr));
                left += ol;
                right += or;
                auto += skewness[j] < 0 ? ol : or;
            }
            out[i] = Math.max(auto, Math.max(left, right));
        }
        return out;
    }

    /** The number of entries {@code <= v}. */
    private static int upperBound(double[] c, double v) {
        int lo = 0;
        int hi = c.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (c[mid] <= v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /** The number of entries {@code < v}. */
    private static int lowerBound(double[] c, double v) {
        int lo = 0;
        int hi = c.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (c[mid] < v) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        return lo;
    }
}
