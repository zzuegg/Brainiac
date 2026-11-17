package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;

/**
 * DBSCAN (Ester, Kriegel, Sander and Xu 1996,
 * <a href="https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf">A density-based algorithm for
 * discovering clusters in large spatial databases with noise</a>): a point with at least
 * {@code minPoints} neighbours within {@code eps} is a core point; clusters are the connected
 * sets of core points plus the border points they reach; everything else is noise.
 *
 * <p>Why it matters: it needs no number of clusters, follows arbitrary shapes, and is one of
 * the few clusterers that labels outliers as noise ({@link #NOISE}) instead of forcing them
 * into a cluster.
 */
public final class Dbscan {

    /** The label of a noise point. */
    public static final int NOISE = -1;

    private Dbscan() {
    }

    /**
     * Clusters points.
     *
     * @param x the points {@code [n][d]}
     * @param eps the neighbourhood radius
     * @param minPoints the neighbours (the point included) a core point needs
     * @return each point's cluster, or {@link #NOISE}
     */
    public static int[] cluster(double[][] x, double eps, int minPoints) {
        int n = x.length;
        double e2 = eps * eps;
        List<int[]> neighbours = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int[] buf = new int[n];
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (MlMatrices.squaredDistance(x[i], x[j]) <= e2) {
                    buf[c++] = j;
                }
            }
            neighbours.add(Arrays.copyOf(buf, c));
        }
        int[] labels = new int[n];
        Arrays.fill(labels, Integer.MIN_VALUE);   // unvisited
        int cluster = 0;
        for (int i = 0; i < n; i++) {
            if (labels[i] != Integer.MIN_VALUE) {
                continue;
            }
            if (neighbours.get(i).length < minPoints) {
                labels[i] = NOISE;
                continue;
            }
            labels[i] = cluster;
            Deque<Integer> queue = new ArrayDeque<>();
            for (int j : neighbours.get(i)) {
                queue.add(j);
            }
            while (!queue.isEmpty()) {
                int j = queue.poll();
                if (labels[j] == NOISE) {
                    labels[j] = cluster;   // border point
                }
                if (labels[j] != Integer.MIN_VALUE) {
                    continue;
                }
                labels[j] = cluster;
                if (neighbours.get(j).length >= minPoints) {
                    for (int t : neighbours.get(j)) {
                        queue.add(t);
                    }
                }
            }
            cluster++;
        }
        return labels;
    }

    /**
     * A heuristic radius: a quantile of every point's distance to its {@code k}-th nearest
     * neighbour (the "knee" of the k-distance plot).
     *
     * @param x the points
     * @param k the neighbour rank, usually {@code minPoints - 1}
     * @param quantile for example {@code 0.9}
     * @return a radius
     */
    public static double suggestEps(double[][] x, int k, double quantile) {
        double[][] d2 = MlMatrices.squaredDistances(x);
        double[] kd = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            int[] nn = MlMetrics.nearest(d2[i], i, k);
            kd[i] = Math.sqrt(d2[i][nn[k - 1]]);
        }
        Arrays.sort(kd);
        return kd[(int) Math.min(kd.length - 1, Math.floor(quantile * kd.length))];
    }
}
