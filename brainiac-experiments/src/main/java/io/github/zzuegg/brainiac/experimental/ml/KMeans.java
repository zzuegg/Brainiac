package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * k-means clustering by Lloyd's algorithm with k-means++ seeding (Arthur and Vassilvitskii 2007,
 * <a href="http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf">k-means++: the advantages of
 * careful seeding</a>). k-means++ picks each new centre with probability proportional to its
 * squared distance from the nearest centre so far, which is {@code O(log k)}-competitive with
 * the optimum before a single Lloyd step and avoids the bad local minima of random seeding.
 *
 * <p>A standalone clusterer (core uses k-means only inside vector quantisers), the baseline
 * for the GMM, spectral clustering, DBSCAN and the self-organising map, and the source of
 * inducing points for {@link SparseVariationalGp}.
 *
 * @param centres the cluster centres {@code [k][d]}
 * @param labels the cluster of each training point
 * @param inertia the sum of squared distances to the assigned centres
 * @param iterations the Lloyd iterations run
 */
public record KMeans(double[][] centres, int[] labels, double inertia, int iterations) {

    /** How the first centres are chosen. */
    public enum Init {
        /** k distinct training points drawn uniformly: the classic, fragile start. */
        RANDOM,
        /** k-means++ seeding. */
        PLUS_PLUS
    }

    /**
     * Clusters points, keeping the best of several restarts.
     *
     * @param x the points {@code [n][d]}
     * @param k the number of clusters
     * @param init the seeding
     * @param restarts independent runs; the lowest inertia wins
     * @param maxIterations Lloyd iterations per run
     * @param seed the seed
     * @return the clustering
     */
    public static KMeans fit(double[][] x, int k, Init init, int restarts, int maxIterations, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        KMeans best = null;
        for (int r = 0; r < restarts; r++) {
            KMeans run = lloyd(x, seed(x, k, init, random), maxIterations);
            if (best == null || run.inertia() < best.inertia()) {
                best = run;
            }
        }
        return best;
    }

    /**
     * Draws initial centres.
     *
     * @param x the points
     * @param k how many
     * @param init the seeding
     * @param random the source
     * @return {@code [k][d]}
     */
    public static double[][] seed(double[][] x, int k, Init init, SplittableRandom random) {
        int n = x.length;
        double[][] c = new double[k][];
        if (init == Init.RANDOM) {
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) {
                idx[i] = i;
            }
            for (int i = 0; i < k; i++) {
                int j = i + random.nextInt(n - i);
                int t = idx[i];
                idx[i] = idx[j];
                idx[j] = t;
                c[i] = x[idx[i]].clone();
            }
            return c;
        }
        c[0] = x[random.nextInt(n)].clone();
        double[] d2 = new double[n];
        for (int i = 0; i < n; i++) {
            d2[i] = MlMatrices.squaredDistance(x[i], c[0]);
        }
        for (int j = 1; j < k; j++) {
            double total = 0;
            for (double v : d2) {
                total += v;
            }
            double u = random.nextDouble() * total;
            int pick = n - 1;
            for (int i = 0; i < n; i++) {
                u -= d2[i];
                if (u <= 0) {
                    pick = i;
                    break;
                }
            }
            c[j] = x[pick].clone();
            for (int i = 0; i < n; i++) {
                d2[i] = Math.min(d2[i], MlMatrices.squaredDistance(x[i], c[j]));
            }
        }
        return c;
    }

    /**
     * Runs Lloyd's algorithm from given centres.
     *
     * @param x the points
     * @param start the initial centres (copied)
     * @param maxIterations the iteration limit
     * @return the clustering
     */
    public static KMeans lloyd(double[][] x, double[][] start, int maxIterations) {
        int n = x.length;
        int k = start.length;
        int d = x[0].length;
        double[][] c = new double[k][];
        for (int j = 0; j < k; j++) {
            c[j] = start[j].clone();
        }
        int[] labels = new int[n];
        java.util.Arrays.fill(labels, -1);
        int iter = 0;
        for (; iter < maxIterations; iter++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int a = nearest(c, x[i]);
                if (a != labels[i]) {
                    labels[i] = a;
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
                for (int t = 0; t < d; t++) {
                    sum[labels[i]][t] += x[i][t];
                }
            }
            for (int j = 0; j < k; j++) {
                if (count[j] > 0) {
                    for (int t = 0; t < d; t++) {
                        c[j][t] = sum[j][t] / count[j];
                    }
                }
            }
        }
        double inertia = 0;
        for (int i = 0; i < n; i++) {
            inertia += MlMatrices.squaredDistance(x[i], c[labels[i]]);
        }
        return new KMeans(c, labels, inertia, iter);
    }

    /**
     * The index of the nearest centre.
     *
     * @param centres the centres
     * @param point a point
     * @return the index
     */
    public static int nearest(double[][] centres, double[] point) {
        int best = 0;
        double bestD = Double.POSITIVE_INFINITY;
        for (int j = 0; j < centres.length; j++) {
            double d = MlMatrices.squaredDistance(centres[j], point);
            if (d < bestD) {
                bestD = d;
                best = j;
            }
        }
        return best;
    }

    /**
     * Assigns new points to the nearest centre.
     *
     * @param x points
     * @return their clusters
     */
    public int[] predict(double[][] x) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = nearest(centres, x[i]);
        }
        return out;
    }

    /**
     * The mean squared distance of points to their nearest centre (the quantisation error).
     *
     * @param x points
     * @return the mean squared error
     */
    public double quantizationError(double[][] x) {
        double s = 0;
        for (double[] p : x) {
            s += MlMatrices.squaredDistance(p, centres[nearest(centres, p)]);
        }
        return s / x.length;
    }
}
