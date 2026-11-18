package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Robust location and scatter.
 *
 * <ul>
 *   <li>{@link #mcd}: FAST-MCD (Rousseeuw and Van Driessen, Technometrics 1999): the
 *       {@code h}-subset whose covariance has the smallest determinant, found by concentration
 *       steps (keep the {@code h} points with the smallest Mahalanobis distances) from random
 *       elemental starts, then a consistency factor and a reweighting step at the 97.5%
 *       chi-square quantile. Its robust distances unmask leverage outliers that the classical
 *       Mahalanobis distance hides.</li>
 *   <li>{@link #medianOfMeans}: median-of-means (Lugosi and Mendelson, 2019,
 *       <a href="https://arxiv.org/abs/1906.04280">arXiv:1906.04280</a>): split into blocks,
 *       average each, take the median (coordinate-wise for vectors, so also usable on
 *       per-example gradients), which gives sub-Gaussian deviations under heavy tails.</li>
 * </ul>
 */
public final class RobustCovariance {

    /**
     * A location/scatter estimate.
     *
     * @param location the centre
     * @param scatter the covariance
     */
    public record Estimate(double[] location, double[][] scatter) {

        /**
         * Squared Mahalanobis distances of rows.
         *
         * @param x rows
         * @return {@code (x - mu)^T S^-1 (x - mu)} per row
         */
        public double[] distances(double[][] x) {
            double[][] inv = MlMatrices.choleskyInverse(MlMatrices.cholesky(scatter));
            int p = location.length;
            double[] d = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                double s = 0;
                for (int a = 0; a < p; a++) {
                    double da = x[i][a] - location[a];
                    for (int b = 0; b < p; b++) {
                        s += da * inv[a][b] * (x[i][b] - location[b]);
                    }
                }
                d[i] = s;
            }
            return d;
        }
    }

    private RobustCovariance() {
    }

    /**
     * The classical mean and covariance (divisor {@code n - 1}).
     *
     * @param x rows {@code [n][p]}
     * @return the estimate
     */
    public static Estimate empirical(double[][] x) {
        boolean[] all = new boolean[x.length];
        Arrays.fill(all, true);
        return subset(x, all, 1);
    }

    /**
     * FAST-MCD with {@code h = (n + p + 1) / 2}.
     *
     * @param x rows {@code [n][p]}
     * @param starts random elemental starts
     * @param seed the seed
     * @return the reweighted MCD estimate
     */
    public static Estimate mcd(double[][] x, int starts, long seed) {
        int n = x.length;
        int p = x[0].length;
        int h = (n + p + 1) / 2;
        SplittableRandom r = new SplittableRandom(seed);
        Estimate[] cand = new Estimate[starts];
        double[] det = new double[starts];
        for (int s = 0; s < starts; s++) {
            boolean[] sel = new boolean[n];
            int[] perm = permutation(n, r);
            for (int k = 0; k < p + 1; k++) {
                sel[perm[k]] = true;
            }
            Estimate e = subset(x, sel, 1);
            for (int c = 0; c < 2; c++) {
                e = cStep(x, e, h);
            }
            cand[s] = e;
            det[s] = logDet(e.scatter());
        }
        Integer[] order = new Integer[starts];
        for (int s = 0; s < starts; s++) {
            order[s] = s;
        }
        Arrays.sort(order, (a, b) -> Double.compare(det[a], det[b]));
        Estimate best = null;
        double bestDet = Double.POSITIVE_INFINITY;
        for (int k = 0; k < Math.min(10, starts); k++) {
            Estimate e = cand[order[k]];
            double prev = Double.POSITIVE_INFINITY;
            for (int c = 0; c < 50; c++) {
                e = cStep(x, e, h);
                double d = logDet(e.scatter());
                if (d >= prev - 1e-10) {
                    break;
                }
                prev = d;
            }
            double d = logDet(e.scatter());
            if (d < bestDet) {
                bestDet = d;
                best = e;
            }
        }
        // consistency: scale so the median distance matches the chi-square median
        double[] dist = best.distances(x);
        double factor = median(dist) / chiSquareQuantile(0.5, p);
        double[][] sc = scale(best.scatter(), factor);
        Estimate raw = new Estimate(best.location(), sc);
        // reweighting at the 97.5% quantile
        double cut = chiSquareQuantile(0.975, p);
        double[] d2 = raw.distances(x);
        boolean[] keep = new boolean[n];
        for (int i = 0; i < n; i++) {
            keep[i] = d2[i] <= cut;
        }
        Estimate rw = subset(x, keep, 1);
        double f2 = median(rw.distances(x)) / chiSquareQuantile(0.5, p);
        return new Estimate(rw.location(), scale(rw.scatter(), f2));
    }

    /**
     * Median-of-means of a sample.
     *
     * @param v values
     * @param blocks the number of blocks (rows are assigned by a seeded shuffle)
     * @param seed the seed
     * @return the estimate
     */
    public static double medianOfMeans(double[] v, int blocks, long seed) {
        double[][] col = new double[v.length][1];
        for (int i = 0; i < v.length; i++) {
            col[i][0] = v[i];
        }
        return medianOfMeans(col, blocks, seed)[0];
    }

    /**
     * Coordinate-wise median-of-means of vectors (for example per-example gradients).
     *
     * @param x vectors {@code [n][d]}
     * @param blocks the number of blocks
     * @param seed the seed
     * @return the estimate
     */
    public static double[] medianOfMeans(double[][] x, int blocks, long seed) {
        int n = x.length;
        int d = x[0].length;
        int k = Math.max(1, Math.min(blocks, n));
        int[] perm = permutation(n, new SplittableRandom(seed));
        double[][] means = new double[d][k];
        int[] counts = new int[k];
        for (int a = 0; a < n; a++) {
            int b = a % k;
            counts[b]++;
            for (int j = 0; j < d; j++) {
                means[j][b] += x[perm[a]][j];
            }
        }
        double[] out = new double[d];
        for (int j = 0; j < d; j++) {
            for (int b = 0; b < k; b++) {
                means[j][b] /= counts[b];
            }
            out[j] = median(means[j]);
        }
        return out;
    }

    /**
     * A chi-square quantile by the Wilson-Hilferty approximation.
     *
     * @param prob the probability
     * @param df degrees of freedom
     * @return the quantile
     */
    public static double chiSquareQuantile(double prob, int df) {
        double z = NormalTails.quantile(prob);
        double a = 2.0 / (9 * df);
        double t = 1 - a + z * Math.sqrt(a);
        return df * t * t * t;
    }

    private static Estimate cStep(double[][] x, Estimate e, int h) {
        double[] d = e.distances(x);
        Integer[] o = new Integer[x.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        Arrays.sort(o, (a, b) -> Double.compare(d[a], d[b]));
        boolean[] sel = new boolean[x.length];
        for (int k = 0; k < h; k++) {
            sel[o[k]] = true;
        }
        return subset(x, sel, 1);
    }

    private static Estimate subset(double[][] x, boolean[] sel, int ddof) {
        int p = x[0].length;
        double[] m = new double[p];
        int cnt = 0;
        for (int i = 0; i < x.length; i++) {
            if (sel[i]) {
                cnt++;
                for (int j = 0; j < p; j++) {
                    m[j] += x[i][j];
                }
            }
        }
        for (int j = 0; j < p; j++) {
            m[j] /= cnt;
        }
        double[][] s = new double[p][p];
        for (int i = 0; i < x.length; i++) {
            if (sel[i]) {
                for (int a = 0; a < p; a++) {
                    for (int b = 0; b < p; b++) {
                        s[a][b] += (x[i][a] - m[a]) * (x[i][b] - m[b]);
                    }
                }
            }
        }
        for (int a = 0; a < p; a++) {
            for (int b = 0; b < p; b++) {
                s[a][b] /= Math.max(1, cnt - ddof);
            }
            s[a][a] += 1e-10;
        }
        return new Estimate(m, s);
    }

    private static double logDet(double[][] s) {
        return MlMatrices.logDet(MlMatrices.cholesky(s));
    }

    private static double[][] scale(double[][] s, double f) {
        double[][] o = new double[s.length][s.length];
        for (int a = 0; a < s.length; a++) {
            for (int b = 0; b < s.length; b++) {
                o[a][b] = s[a][b] * f;
            }
        }
        return o;
    }

    private static double median(double[] v) {
        return RobustRegression.median(v);
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }
}
