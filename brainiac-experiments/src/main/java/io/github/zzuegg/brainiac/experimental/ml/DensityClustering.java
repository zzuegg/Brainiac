package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Density-based clustering over all density levels.
 *
 * <ul>
 *   <li>{@link #optics}: OPTICS (Ankerst, Breunig, Kriegel and Sander, SIGMOD 1999) orders the
 *       points so that each one is reached from the nearest already-processed core point; the
 *       reachability plot shows clusters at every density as valleys, and
 *       {@link Optics#extractDbscan} cuts it at one level, which reproduces DBSCAN at that
 *       {@code eps} for core points.</li>
 *   <li>{@link #hdbscan}: HDBSCAN* (Campello, Moulavi and Sander, PAKDD 2013) builds the minimum
 *       spanning tree of the mutual-reachability distance
 *       {@code max(core_k(a), core_k(b), d(a, b))}, turns it into the single-linkage hierarchy,
 *       condenses that hierarchy to clusters of at least {@code minClusterSize} points and selects
 *       the clusters of maximal stability {@code sum_p (lambda_p - lambda_birth)},
 *       {@code lambda = 1 / distance} (excess of mass). Points outside every selected cluster are
 *       noise; GLOSH (Campello et al., TKDD 2015) scores every point by
 *       {@code 1 - lambda_p / lambda_max} of the cluster it last belonged to.</li>
 * </ul>
 * Distances are dense ({@code O(n^2)} memory), which suits a few thousand points.
 */
public final class DensityClustering {

    /** The noise label. */
    public static final int NOISE = -1;

    private DensityClustering() {
    }

    /**
     * An OPTICS ordering.
     *
     * @param order the processing order
     * @param reachability each point's reachability distance (infinite for the first of each
     *     connected region), indexed by point
     * @param core each point's core distance
     */
    public record Optics(int[] order, double[] reachability, double[] core) {

        /**
         * DBSCAN-equivalent labels at one density level.
         *
         * @param eps the cut
         * @return labels, {@link #NOISE} for noise
         */
        public int[] extractDbscan(double eps) {
            int[] labels = new int[order.length];
            int cluster = -1;
            for (int p : order) {
                if (reachability[p] > eps) {
                    if (core[p] <= eps) {
                        cluster++;
                        labels[p] = cluster;
                    } else {
                        labels[p] = NOISE;
                    }
                } else {
                    labels[p] = Math.max(cluster, 0);
                }
            }
            return labels;
        }

        /**
         * The reachability plot: reachability distances in processing order.
         *
         * @return {@code [n]}
         */
        public double[] plot() {
            double[] r = new double[order.length];
            for (int i = 0; i < order.length; i++) {
                r[i] = reachability[order[i]];
            }
            return r;
        }
    }

    /**
     * An HDBSCAN result.
     *
     * @param labels cluster labels {@code 0 ..}, {@link #NOISE} for noise
     * @param clusters the number of selected clusters
     * @param outlierScores GLOSH scores in {@code [0, 1]}, higher is more outlying
     * @param probabilities membership strengths {@code lambda_p / lambda_max} of clustered points
     */
    public record Hdbscan(int[] labels, int clusters, double[] outlierScores, double[] probabilities) {
    }

    private static double[][] distances(double[][] x) {
        double[][] d = MlMatrices.squaredDistances(x);
        for (double[] row : d) {
            for (int j = 0; j < row.length; j++) {
                row[j] = Math.sqrt(Math.max(row[j], 0));
            }
        }
        return d;
    }

    private static double[] coreDistances(double[][] d, int minSamples) {
        int n = d.length;
        double[] core = new double[n];
        int k = Math.min(minSamples, n - 1);
        for (int i = 0; i < n; i++) {
            double[] row = d[i].clone();
            row[i] = Double.POSITIVE_INFINITY;
            Arrays.sort(row);
            core[i] = row[k - 1];
        }
        return core;
    }

    /**
     * Computes the OPTICS ordering with an unbounded {@code eps}.
     *
     * @param x the points {@code [n][d]}
     * @param minSamples neighbours (excluding the point) that make a core point
     * @return the ordering
     */
    public static Optics optics(double[][] x, int minSamples) {
        int n = x.length;
        double[][] d = distances(x);
        double[] core = coreDistances(d, minSamples);
        double[] reach = new double[n];
        Arrays.fill(reach, Double.POSITIVE_INFINITY);
        boolean[] done = new boolean[n];
        int[] order = new int[n];
        int c = 0;
        for (int start = 0; start < n; start++) {
            if (done[start]) {
                continue;
            }
            int p = start;
            while (p >= 0) {
                done[p] = true;
                order[c++] = p;
                for (int q = 0; q < n; q++) {
                    if (!done[q]) {
                        double nr = Math.max(core[p], d[p][q]);
                        if (nr < reach[q]) {
                            reach[q] = nr;
                        }
                    }
                }
                // next: the unprocessed point with the smallest reachability
                int next = -1;
                double best = Double.POSITIVE_INFINITY;
                for (int q = 0; q < n; q++) {
                    if (!done[q] && reach[q] < best) {
                        best = reach[q];
                        next = q;
                    }
                }
                p = next;
            }
        }
        return new Optics(order, reach, core);
    }

    /**
     * Runs HDBSCAN* with excess-of-mass cluster selection.
     *
     * @param x the points {@code [n][d]}
     * @param minClusterSize the smallest cluster, the main knob
     * @param minSamples the core-distance neighbour count (often equal to {@code minClusterSize})
     * @return labels, GLOSH scores and membership strengths
     */
    public static Hdbscan hdbscan(double[][] x, int minClusterSize, int minSamples) {
        int n = x.length;
        double[][] d = distances(x);
        double[] core = coreDistances(d, minSamples);
        // Prim's MST on mutual reachability
        double[] best = new double[n];
        int[] from = new int[n];
        boolean[] in = new boolean[n];
        Arrays.fill(best, Double.POSITIVE_INFINITY);
        best[0] = 0;
        from[0] = -1;
        double[][] edges = new double[n - 1][];
        int e = 0;
        for (int it = 0; it < n; it++) {
            int u = -1;
            for (int v = 0; v < n; v++) {
                if (!in[v] && (u < 0 || best[v] < best[u])) {
                    u = v;
                }
            }
            in[u] = true;
            if (from[u] >= 0) {
                edges[e++] = new double[] {from[u], u, best[u]};
            }
            for (int v = 0; v < n; v++) {
                if (!in[v]) {
                    double m = Math.max(Math.max(core[u], core[v]), d[u][v]);
                    if (m < best[v]) {
                        best[v] = m;
                        from[v] = u;
                    }
                }
            }
        }
        Arrays.sort(edges, (a, b) -> Double.compare(a[2], b[2]));
        // single-linkage dendrogram: nodes n .. 2n-2
        int[] parent = new int[2 * n - 1];
        int[] left = new int[2 * n - 1];
        int[] right = new int[2 * n - 1];
        int[] size = new int[2 * n - 1];
        double[] height = new double[2 * n - 1];
        int[] uf = new int[2 * n - 1];
        for (int i = 0; i < 2 * n - 1; i++) {
            uf[i] = i;
            size[i] = i < n ? 1 : 0;
            parent[i] = -1;
        }
        for (int k = 0; k < n - 1; k++) {
            int a = find(uf, (int) edges[k][0]);
            int b = find(uf, (int) edges[k][1]);
            int node = n + k;
            left[node] = a;
            right[node] = b;
            size[node] = size[a] + size[b];
            height[node] = Math.max(edges[k][2], 1e-12);
            parent[a] = node;
            parent[b] = node;
            uf[a] = node;
            uf[b] = node;
        }
        int root = 2 * n - 2;
        // condensed tree
        List<Integer> clusterParent = new ArrayList<>();
        List<Double> birth = new ArrayList<>();
        List<Double> stability = new ArrayList<>();
        List<List<Integer>> children = new ArrayList<>();
        double[] pointLambda = new double[n];
        int[] pointCluster = new int[n];
        clusterParent.add(-1);
        birth.add(0.0);
        stability.add(0.0);
        children.add(new ArrayList<>());
        // stack of (dendrogram node, cluster id)
        int[] stackNode = new int[2 * n];
        int[] stackCluster = new int[2 * n];
        int sp = 0;
        stackNode[sp] = root;
        stackCluster[sp++] = 0;
        while (sp > 0) {
            int node = stackNode[--sp];
            int cl = stackCluster[sp];
            if (node < n) {
                // a single leaf reached while still in the cluster: it stays until lambda = infinity;
                // credit it at the cluster's last split height
                pointCluster[node] = cl;
                pointLambda[node] = Math.max(pointLambda[node], birth.get(cl));
                continue;
            }
            double lambda = 1 / height[node];
            int a = left[node];
            int b = right[node];
            boolean bigA = size[a] >= minClusterSize;
            boolean bigB = size[b] >= minClusterSize;
            if (bigA && bigB) {
                stability.set(cl, stability.get(cl) + size[node] * (lambda - birth.get(cl)));
                for (int child : new int[] {a, b}) {
                    int id = clusterParent.size();
                    clusterParent.add(cl);
                    birth.add(lambda);
                    stability.add(0.0);
                    children.add(new ArrayList<>());
                    children.get(cl).add(id);
                    stackNode[sp] = child;
                    stackCluster[sp++] = id;
                }
            } else {
                for (int child : new int[] {a, b}) {
                    boolean big = child == a ? bigA : bigB;
                    if (big) {
                        stackNode[sp] = child;
                        stackCluster[sp++] = cl;
                    } else {
                        for (int p : leaves(child, n, left, right)) {
                            pointCluster[p] = cl;
                            pointLambda[p] = lambda;
                            stability.set(cl, stability.get(cl) + (lambda - birth.get(cl)));
                        }
                    }
                }
            }
        }
        int nc = clusterParent.size();
        // excess of mass selection, bottom up (children have larger ids than parents)
        boolean[] selected = new boolean[nc];
        double[] score = new double[nc];
        for (int c = nc - 1; c >= 1; c--) {
            double sum = 0;
            for (int ch : children.get(c)) {
                sum += score[ch];
            }
            if (children.get(c).isEmpty() || stability.get(c) >= sum) {
                selected[c] = true;
                score[c] = stability.get(c);
                deselect(c, children, selected);
                selected[c] = true;
            } else {
                score[c] = sum;
            }
        }
        if (nc == 1) {
            // no split ever produced two large clusters: everything is one cluster
            selected[0] = true;
        }
        int[] labelOf = new int[nc];
        Arrays.fill(labelOf, -1);
        int clusters = 0;
        for (int c = 0; c < nc; c++) {
            if (selected[c]) {
                labelOf[c] = clusters++;
            }
        }
        // lambda_max per cluster subtree for GLOSH
        double[] lambdaMax = new double[nc];
        for (int p = 0; p < n; p++) {
            int c = pointCluster[p];
            while (c >= 0) {
                lambdaMax[c] = Math.max(lambdaMax[c], pointLambda[p]);
                c = clusterParent.get(c);
            }
        }
        int[] labels = new int[n];
        double[] glosh = new double[n];
        double[] prob = new double[n];
        for (int p = 0; p < n; p++) {
            int c = pointCluster[p];
            glosh[p] = lambdaMax[c] > 0 ? 1 - pointLambda[p] / lambdaMax[c] : 0;
            int up = c;
            while (up >= 0 && !selected[up]) {
                up = clusterParent.get(up);
            }
            labels[p] = up >= 0 ? labelOf[up] : NOISE;
            prob[p] = up >= 0 && lambdaMax[up] > 0 ? Math.min(1, pointLambda[p] / lambdaMax[up]) : 0;
        }
        return new Hdbscan(labels, clusters, glosh, prob);
    }

    private static void deselect(int c, List<List<Integer>> children, boolean[] selected) {
        for (int ch : children.get(c)) {
            selected[ch] = false;
            deselect(ch, children, selected);
        }
    }

    private static int find(int[] uf, int i) {
        while (uf[i] != i) {
            uf[i] = uf[uf[i]];
            i = uf[i];
        }
        return i;
    }

    private static List<Integer> leaves(int node, int n, int[] left, int[] right) {
        List<Integer> out = new ArrayList<>();
        int[] stack = new int[2 * n];
        int sp = 0;
        stack[sp++] = node;
        while (sp > 0) {
            int v = stack[--sp];
            if (v < n) {
                out.add(v);
            } else {
                stack[sp++] = left[v];
                stack[sp++] = right[v];
            }
        }
        return out;
    }
}
