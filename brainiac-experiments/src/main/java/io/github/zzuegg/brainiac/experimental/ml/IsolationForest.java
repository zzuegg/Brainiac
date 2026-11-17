package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * Isolation forest anomaly detection (Liu, Ting and Zhou 2008,
 * <a href="https://ieeexplore.ieee.org/document/4781136">Isolation forest</a>): random trees
 * split a random feature at a random value between its extremes until each point is alone.
 * Anomalies are few and different, so they are isolated after few splits; the anomaly score is
 * {@code 2^(-E[h(x)] / c(psi))}, where {@code h} is the path length and {@code c(psi)} the
 * average path length of an unsuccessful binary-search-tree lookup among {@code psi} points.
 *
 * <p>Why it matters: linear time, no distance computations, no density model, and it works on
 * small subsamples (256 points per tree is the classic choice).
 */
public final class IsolationForest {

    private record Node(int feature, double threshold, Node left, Node right, int size) {
    }

    private final Node[] trees;
    private final int sampleSize;

    private IsolationForest(Node[] trees, int sampleSize) {
        this.trees = trees;
        this.sampleSize = sampleSize;
    }

    /**
     * Grows the forest.
     *
     * @param x the (mostly normal) training rows {@code [n][d]}
     * @param trees the number of trees, for example 100
     * @param sampleSize rows per tree, for example 256
     * @param seed the seed
     * @return the forest
     */
    public static IsolationForest fit(double[][] x, int trees, int sampleSize, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int psi = Math.min(sampleSize, x.length);
        int limit = (int) Math.ceil(Math.log(psi) / Math.log(2));
        Node[] forest = new Node[trees];
        for (int t = 0; t < trees; t++) {
            int[] rows = new int[psi];
            int[] perm = new int[x.length];
            for (int i = 0; i < perm.length; i++) {
                perm[i] = i;
            }
            for (int i = 0; i < psi; i++) {
                int j = i + random.nextInt(perm.length - i);
                int tmp = perm[i];
                perm[i] = perm[j];
                perm[j] = tmp;
                rows[i] = perm[i];
            }
            forest[t] = grow(x, rows, 0, limit, random);
        }
        return new IsolationForest(forest, psi);
    }

    private static Node grow(double[][] x, int[] rows, int depth, int limit, SplittableRandom random) {
        if (depth >= limit || rows.length <= 1) {
            return new Node(-1, 0, null, null, rows.length);
        }
        int d = x[0].length;
        for (int attempt = 0; attempt < 2 * d; attempt++) {
            int f = random.nextInt(d);
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (int r : rows) {
                lo = Math.min(lo, x[r][f]);
                hi = Math.max(hi, x[r][f]);
            }
            if (!(hi > lo)) {
                continue;
            }
            double thr = lo + random.nextDouble() * (hi - lo);
            int nl = 0;
            for (int r : rows) {
                nl += x[r][f] < thr ? 1 : 0;
            }
            int[] l = new int[nl];
            int[] rr = new int[rows.length - nl];
            int a = 0;
            int b = 0;
            for (int r : rows) {
                if (x[r][f] < thr) {
                    l[a++] = r;
                } else {
                    rr[b++] = r;
                }
            }
            return new Node(f, thr, grow(x, l, depth + 1, limit, random), grow(x, rr, depth + 1, limit, random),
                    rows.length);
        }
        return new Node(-1, 0, null, null, rows.length);
    }

    /**
     * The average path length of an unsuccessful search among {@code n} points.
     *
     * @param n the number of points
     * @return {@code c(n)}
     */
    static double averagePath(double n) {
        if (n <= 1) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }
        return 2 * (Math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n;
    }

    private static double pathLength(Node node, double[] x, int depth) {
        if (node.feature() < 0) {
            return depth + averagePath(node.size());
        }
        return pathLength(x[node.feature()] < node.threshold() ? node.left() : node.right(), x, depth + 1);
    }

    /**
     * The anomaly score: near 1 for anomalies, well below 0.5 for normal points.
     *
     * @param x a row
     * @return {@code 2^(-E[h(x)] / c(psi))}
     */
    public double score(double[] x) {
        double h = 0;
        for (Node t : trees) {
            h += pathLength(t, x, 0);
        }
        h /= trees.length;
        return Math.pow(2, -h / averagePath(sampleSize));
    }

    /**
     * Scores many rows.
     *
     * @param x rows
     * @return their anomaly scores
     */
    public double[] score(double[][] x) {
        double[] s = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            s[i] = score(x[i]);
        }
        return s;
    }
}
