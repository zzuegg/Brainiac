package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Objects;

/**
 * Agglomerative hierarchical clustering with Lance-Williams distance updates: starting from
 * singletons, repeatedly merge the closest pair of clusters {@code i, j} and update the distance
 * to every other cluster {@code k} by
 * {@code d(k, ij) = a_i d(k, i) + a_j d(k, j) + b d(i, j) + c |d(k, i) - d(k, j)|} (Lance and
 * Williams 1967). {@link Linkage#WARD} works on squared Euclidean distances, so each merge is the
 * one that least increases the within-cluster sum of squares (Ward, JASA 1963).
 */
public final class AgglomerativeClustering {

    /** The linkage criterion. */
    public enum Linkage {
        /** Nearest pair of points. */
        SINGLE,
        /** Farthest pair of points. */
        COMPLETE,
        /** Mean pairwise distance (UPGMA). */
        AVERAGE,
        /** Minimum increase of the within-cluster variance. */
        WARD
    }

    /**
     * The merge history.
     *
     * @param merges {@code [n-1][2]} the clusters merged at each step; ids below {@code n} are
     *     points, id {@code n + s} is the cluster created at step {@code s}
     * @param heights the merge distances (squared-Euclidean increase for Ward)
     */
    public record Dendrogram(int[][] merges, double[] heights) {

        /**
         * Cuts the tree into {@code k} clusters.
         *
         * @param k the number of clusters
         * @return labels {@code 0 .. k-1}
         */
        public int[] cut(int k) {
            int n = merges.length + 1;
            int[] uf = new int[2 * n - 1];
            for (int i = 0; i < uf.length; i++) {
                uf[i] = i;
            }
            for (int s = 0; s < n - k; s++) {
                uf[merges[s][0]] = n + s;
                uf[merges[s][1]] = n + s;
            }
            int[] labels = new int[n];
            int[] map = new int[2 * n - 1];
            java.util.Arrays.fill(map, -1);
            int next = 0;
            for (int i = 0; i < n; i++) {
                int r = i;
                while (uf[r] != r) {
                    r = uf[r];
                }
                if (map[r] < 0) {
                    map[r] = next++;
                }
                labels[i] = map[r];
            }
            return labels;
        }
    }

    private AgglomerativeClustering() {
    }

    /**
     * Builds the full dendrogram ({@code O(n^3)} time, {@code O(n^2)} memory).
     *
     * @param x the points {@code [n][d]}
     * @param linkage the linkage
     * @return the merge history
     */
    public static Dendrogram fit(double[][] x, Linkage linkage) {
        Objects.requireNonNull(linkage, "linkage");
        int n = x.length;
        double[][] d = MlMatrices.squaredDistances(x);
        if (linkage != Linkage.WARD) {
            for (double[] row : d) {
                for (int j = 0; j < n; j++) {
                    row[j] = Math.sqrt(Math.max(row[j], 0));
                }
            }
        }
        int[] id = new int[n];
        int[] size = new int[n];
        boolean[] active = new boolean[n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
            size[i] = 1;
            active[i] = true;
        }
        // nearest active neighbour of each row, refreshed lazily
        int[] nn = new int[n];
        double[] nd = new double[n];
        for (int i = 0; i < n; i++) {
            refresh(i, d, active, nn, nd);
        }
        int[][] merges = new int[n - 1][];
        double[] heights = new double[n - 1];
        for (int s = 0; s < n - 1; s++) {
            int a = -1;
            for (int i = 0; i < n; i++) {
                if (active[i] && (a < 0 || nd[i] < nd[a])) {
                    a = i;
                }
            }
            int b = nn[a];
            heights[s] = nd[a];
            merges[s] = new int[] {id[a], id[b]};
            double dab = d[a][b];
            for (int k = 0; k < n; k++) {
                if (!active[k] || k == a || k == b) {
                    continue;
                }
                double dka = d[k][a];
                double dkb = d[k][b];
                double v = switch (linkage) {
                    case SINGLE -> Math.min(dka, dkb);
                    case COMPLETE -> Math.max(dka, dkb);
                    case AVERAGE -> (size[a] * dka + size[b] * dkb) / (size[a] + size[b]);
                    case WARD -> ((size[a] + size[k]) * dka + (size[b] + size[k]) * dkb - size[k] * dab)
                            / (size[a] + size[b] + size[k]);
                };
                d[k][a] = v;
                d[a][k] = v;
            }
            active[b] = false;
            size[a] += size[b];
            id[a] = n + s;
            for (int k = 0; k < n; k++) {
                if (active[k] && (k == a || nn[k] == a || nn[k] == b || d[k][a] < nd[k])) {
                    refresh(k, d, active, nn, nd);
                }
            }
        }
        return new Dendrogram(merges, heights);
    }

    private static void refresh(int i, double[][] d, boolean[] active, int[] nn, double[] nd) {
        double best = Double.POSITIVE_INFINITY;
        int arg = -1;
        for (int j = 0; j < d.length; j++) {
            if (j != i && active[j] && d[i][j] < best) {
                best = d[i][j];
                arg = j;
            }
        }
        nn[i] = arg;
        nd[i] = best;
    }

    /**
     * Clusters into {@code k} groups.
     *
     * @param x the points
     * @param k the number of clusters
     * @param linkage the linkage
     * @return labels {@code 0 .. k-1}
     */
    public static int[] cluster(double[][] x, int k, Linkage linkage) {
        return fit(x, linkage).cut(k);
    }
}
