package io.github.zzuegg.brainiac.experimental.metric;

import java.util.Arrays;

/**
 * Evaluation of embeddings for retrieval and verification, by cosine similarity:
 * Recall@K (the share of queries with a same-class item among their K nearest neighbours,
 * leave-one-out), the equal error rate of same/different verification over all pairs, and
 * the mean intra- and inter-class angles.
 */
public final class RetrievalMetrics {

    private RetrievalMetrics() {
    }

    private static double[][] normalised(float[][] x) {
        double[][] u = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            double s = 1e-12;
            for (float v : x[i]) {
                s += (double) v * v;
            }
            s = Math.sqrt(s);
            u[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                u[i][j] = x[i][j] / s;
            }
        }
        return u;
    }

    private static double[][] cosines(float[][] x) {
        double[][] u = normalised(x);
        int n = u.length;
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                double s = 0;
                for (int k = 0; k < u[i].length; k++) {
                    s += u[i][k] * u[j][k];
                }
                c[i][j] = s;
                c[j][i] = s;
            }
        }
        return c;
    }

    /**
     * Recall@K by cosine similarity, leave-one-out.
     *
     * @param embeddings {@code [n][d]}
     * @param labels the class of each row
     * @param k the neighbours looked at
     * @return the share of rows with a same-class row among their {@code k} nearest
     */
    public static double recallAt(float[][] embeddings, int[] labels, int k) {
        double[][] c = cosines(embeddings);
        int n = c.length;
        int hit = 0;
        for (int i = 0; i < n; i++) {
            final int row = i;
            Integer[] order = new Integer[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            Arrays.sort(order, (a, b) -> Double.compare(c[row][b], c[row][a]));
            int seen = 0;
            for (int j = 0; j < n && seen < k; j++) {
                if (order[j] == i) {
                    continue;
                }
                seen++;
                if (labels[order[j]] == labels[i]) {
                    hit++;
                    break;
                }
            }
        }
        return (double) hit / n;
    }

    /**
     * The equal error rate of verifying every pair as same-class when its cosine passes a
     * threshold: the point where false accepts equal false rejects.
     *
     * @param embeddings {@code [n][d]}
     * @param labels the classes
     * @return the EER in {@code [0, 1]}
     */
    public static double equalErrorRate(float[][] embeddings, int[] labels) {
        double[][] c = cosines(embeddings);
        int n = c.length;
        int pairs = n * (n - 1) / 2;
        double[] score = new double[pairs];
        boolean[] same = new boolean[pairs];
        int p = 0;
        int positives = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                score[p] = c[i][j];
                same[p] = labels[i] == labels[j];
                positives += same[p] ? 1 : 0;
                p++;
            }
        }
        int negatives = pairs - positives;
        Integer[] order = new Integer[pairs];
        for (int i = 0; i < pairs; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        // threshold sweeps from high to low: accepted = prefix
        int acceptedPos = 0;
        int acceptedNeg = 0;
        double best = 1;
        for (int idx : order) {
            if (same[idx]) {
                acceptedPos++;
            } else {
                acceptedNeg++;
            }
            double far = (double) acceptedNeg / Math.max(1, negatives);
            double frr = 1 - (double) acceptedPos / Math.max(1, positives);
            if (far >= frr) {
                best = Math.min(best, (far + frr) / 2);
                break;
            }
        }
        return best;
    }

    /**
     * The mean angle in degrees between same-class pairs and between different-class pairs.
     *
     * @param embeddings {@code [n][d]}
     * @param labels the classes
     * @return {@code {intra, inter}}
     */
    public static double[] angles(float[][] embeddings, int[] labels) {
        double[][] c = cosines(embeddings);
        double intra = 0;
        double inter = 0;
        int ni = 0;
        int ne = 0;
        for (int i = 0; i < c.length; i++) {
            for (int j = i + 1; j < c.length; j++) {
                double a = Math.toDegrees(Math.acos(Math.max(-1, Math.min(1, c[i][j]))));
                if (labels[i] == labels[j]) {
                    intra += a;
                    ni++;
                } else {
                    inter += a;
                    ne++;
                }
            }
        }
        return new double[] {intra / Math.max(1, ni), inter / Math.max(1, ne)};
    }
}
