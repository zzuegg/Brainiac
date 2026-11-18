package io.github.zzuegg.brainiac.data;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * k-means clustering on the host, as {@code sklearn.cluster.KMeans}: k-means++ seeding, Lloyd
 * iterations, the best of several restarts by inertia.
 *
 * @param centroids the cluster centres, {@code k x d}
 * @param labels each training row's cluster
 * @param inertia the summed squared distance of the rows to their centres
 * @param iterations the Lloyd iterations of the best restart
 */
public record KMeans(double[][] centroids, int[] labels, double inertia, int iterations) {

    /**
     * Clusters rows.
     *
     * @param x the rows, all of one length
     * @param k the number of clusters, at most the number of rows
     * @param restarts how many seedings to try, such as 10
     * @param maxIterations the most Lloyd iterations per restart, such as 300
     * @param seed the random seed
     * @return the best clustering
     */
    public static KMeans fit(double[][] x, int k, int restarts, int maxIterations, long seed) {
        int n = x.length;
        if (k < 1 || k > n) {
            throw new IllegalArgumentException("k must be in [1, " + n + "], got " + k);
        }
        int d = x[0].length;
        for (double[] row : x) {
            if (row.length != d) {
                throw new IllegalArgumentException("rows of different lengths");
            }
        }
        SplittableRandom rng = new SplittableRandom(seed);
        KMeans best = null;
        for (int r = 0; r < Math.max(1, restarts); r++) {
            KMeans run = lloyd(x, plusPlus(x, k, rng), maxIterations, rng);
            if (best == null || run.inertia < best.inertia) {
                best = run;
            }
        }
        return best;
    }

    /**
     * Assigns rows to their nearest centre.
     *
     * @param x the rows
     * @return one cluster per row
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = nearest(x[i], centroids);
        }
        return out;
    }

    private static double[][] plusPlus(double[][] x, int k, SplittableRandom rng) {
        int n = x.length;
        double[][] c = new double[k][];
        c[0] = x[rng.nextInt(n)].clone();
        double[] dist = new double[n];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        for (int j = 1; j < k; j++) {
            double total = 0;
            for (int i = 0; i < n; i++) {
                dist[i] = Math.min(dist[i], squared(x[i], c[j - 1]));
                total += dist[i];
            }
            int pick = rng.nextInt(n);
            if (total > 0) {
                double u = rng.nextDouble() * total;
                for (int i = 0; i < n; i++) {
                    u -= dist[i];
                    if (u <= 0) {
                        pick = i;
                        break;
                    }
                }
            }
            c[j] = x[pick].clone();
        }
        return c;
    }

    private static KMeans lloyd(double[][] x, double[][] c, int maxIterations, SplittableRandom rng) {
        int n = x.length;
        int k = c.length;
        int d = x[0].length;
        int[] labels = new int[n];
        Arrays.fill(labels, -1);
        int it = 0;
        while (it < maxIterations) {
            it++;
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int l = nearest(x[i], c);
                if (l != labels[i]) {
                    labels[i] = l;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }
            double[][] sum = new double[k][d];
            int[] count = new int[k];
            for (int i = 0; i < n; i++) {
                count[labels[i]]++;
                for (int j = 0; j < d; j++) {
                    sum[labels[i]][j] += x[i][j];
                }
            }
            for (int j = 0; j < k; j++) {
                if (count[j] == 0) {
                    c[j] = x[rng.nextInt(n)].clone();   // an empty cluster restarts at a random row
                } else {
                    for (int l = 0; l < d; l++) {
                        c[j][l] = sum[j][l] / count[j];
                    }
                }
            }
        }
        double inertia = 0;
        for (int i = 0; i < n; i++) {
            labels[i] = nearest(x[i], c);
            inertia += squared(x[i], c[labels[i]]);
        }
        return new KMeans(c, labels, inertia, it);
    }

    private static int nearest(double[] p, double[][] c) {
        int best = 0;
        double bd = Double.POSITIVE_INFINITY;
        for (int j = 0; j < c.length; j++) {
            double dd = squared(p, c[j]);
            if (dd < bd) {
                bd = dd;
                best = j;
            }
        }
        return best;
    }

    private static double squared(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double t = a[i] - b[i];
            s += t * t;
        }
        return s;
    }
}
