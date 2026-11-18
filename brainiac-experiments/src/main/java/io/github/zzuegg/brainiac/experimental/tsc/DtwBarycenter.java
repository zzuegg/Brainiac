package io.github.zzuegg.brainiac.experimental.tsc;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * DTW Barycenter Averaging (DBA; Petitjean, Ketterlin and Gançarski, Pattern Recognition 2011):
 * the average of series under DTW. Starting from a medoid, every iteration aligns each series to
 * the current average with DTW, collects for every average position the values aligned to it, and
 * replaces the position by their mean; the sum of squared DTW distances never increases. The
 * Euclidean mean of shifted or warped copies of a template blurs it; the DBA average keeps its
 * shape. {@link #kMeans} plugs DBA into k-means with DTW assignment.
 */
public final class DtwBarycenter {

    private DtwBarycenter() {
    }

    /**
     * The DTW alignment path between two series.
     *
     * @param a one series
     * @param b another
     * @param window the band half-width; negative for none
     * @return pairs {@code (i, j)} from the start to the end
     */
    public static List<int[]> path(double[] a, double[] b, int window) {
        int n = a.length;
        int m = b.length;
        int w = window < 0 ? Math.max(n, m) : Math.max(window, Math.abs(n - m));
        double[][] d = new double[n + 1][m + 1];
        for (double[] row : d) {
            java.util.Arrays.fill(row, Double.POSITIVE_INFINITY);
        }
        d[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = Math.max(1, i - w); j <= Math.min(m, i + w); j++) {
                double c = a[i - 1] - b[j - 1];
                d[i][j] = c * c + Math.min(d[i - 1][j - 1], Math.min(d[i - 1][j], d[i][j - 1]));
            }
        }
        List<int[]> p = new ArrayList<>();
        int i = n;
        int j = m;
        while (i > 0 && j > 0) {
            p.add(new int[] {i - 1, j - 1});
            double diag = d[i - 1][j - 1];
            double up = d[i - 1][j];
            double left = d[i][j - 1];
            if (diag <= up && diag <= left) {
                i--;
                j--;
            } else if (up <= left) {
                i--;
            } else {
                j--;
            }
        }
        java.util.Collections.reverse(p);
        return p;
    }

    /**
     * The medoid: the series with the smallest sum of squared DTW distances to the others.
     *
     * @param series the series
     * @param window the band half-width; negative for none
     * @return its index
     */
    public static int medoid(double[][] series, int window) {
        int best = 0;
        double bestCost = Double.POSITIVE_INFINITY;
        for (int i = 0; i < series.length; i++) {
            double c = 0;
            for (double[] s : series) {
                double d = ElasticDistances.dtw(series[i], s, window);
                c += d * d;
                if (c > bestCost) {
                    break;
                }
            }
            if (c < bestCost) {
                bestCost = c;
                best = i;
            }
        }
        return best;
    }

    /**
     * The DBA average.
     *
     * @param series the series to average
     * @param init the starting average (such as the medoid)
     * @param iterations refinement passes, such as 10
     * @param window the band half-width; negative for none
     * @return the average
     */
    public static double[] average(double[][] series, double[] init, int iterations, int window) {
        double[] avg = init.clone();
        for (int it = 0; it < iterations; it++) {
            double[] sum = new double[avg.length];
            int[] count = new int[avg.length];
            for (double[] s : series) {
                for (int[] p : path(avg, s, window)) {
                    sum[p[0]] += s[p[1]];
                    count[p[0]]++;
                }
            }
            for (int t = 0; t < avg.length; t++) {
                if (count[t] > 0) {
                    avg[t] = sum[t] / count[t];
                }
            }
        }
        return avg;
    }

    /**
     * k-means under DTW with DBA centroids, seeded by k-means++ on DTW distances.
     *
     * @param series the series
     * @param k clusters
     * @param iterations k-means iterations
     * @param window the band half-width; negative for none
     * @param seed the seed
     * @return cluster labels
     */
    public static int[] kMeans(double[][] series, int k, int iterations, int window, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = series.length;
        double[][] centres = new double[k][];
        centres[0] = series[r.nextInt(n)].clone();
        double[] dist = new double[n];
        for (int c = 1; c < k; c++) {
            double total = 0;
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                for (int q = 0; q < c; q++) {
                    best = Math.min(best, ElasticDistances.dtw(series[i], centres[q], window));
                }
                dist[i] = best;
                total += best;
            }
            double u = r.nextDouble() * total;
            int pick = n - 1;
            for (int i = 0; i < n; i++) {
                u -= dist[i];
                if (u <= 0) {
                    pick = i;
                    break;
                }
            }
            centres[c] = series[pick].clone();
        }
        int[] label = new int[n];
        for (int it = 0; it < iterations; it++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                int arg = 0;
                for (int c = 0; c < k; c++) {
                    double d = ElasticDistances.dtw(series[i], centres[c], window);
                    if (d < best) {
                        best = d;
                        arg = c;
                    }
                }
                changed |= label[i] != arg || it == 0;
                label[i] = arg;
            }
            if (!changed) {
                break;
            }
            for (int c = 0; c < k; c++) {
                List<double[]> members = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    if (label[i] == c) {
                        members.add(series[i]);
                    }
                }
                if (!members.isEmpty()) {
                    centres[c] = average(members.toArray(new double[0][]), centres[c], 3, window);
                }
            }
        }
        return label;
    }
}
