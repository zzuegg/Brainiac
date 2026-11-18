package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;

/**
 * External scores of a clustering against true classes: clustering accuracy (the best
 * one-to-one matching of clusters to classes, by the Hungarian algorithm of
 * {@link MinibatchOptimalTransport#assignCost}), normalised mutual information (arithmetic-mean
 * normalisation, as scikit-learn) and the adjusted Rand index ({@link MlMetrics}).
 */
public final class ClusteringMetrics {

    private ClusteringMetrics() {
    }

    private static int count(int[] x) {
        int m = 0;
        for (int v : x) {
            m = Math.max(m, v + 1);
        }
        return m;
    }

    /**
     * Clustering accuracy: the share of points whose cluster maps to their class under the best
     * one-to-one map; extra clusters map to nothing.
     *
     * @param truth the classes
     * @param predicted the clusters
     * @return the accuracy in {@code [0, 1]}
     */
    public static double accuracy(int[] truth, int[] predicted) {
        int size = Math.max(count(truth), count(predicted));
        double[][] cost = new double[size][size];
        for (int i = 0; i < truth.length; i++) {
            cost[predicted[i]][truth[i]] -= 1;
        }
        int[] map = MinibatchOptimalTransport.assignCost(cost);
        int hit = 0;
        for (int i = 0; i < truth.length; i++) {
            hit += map[predicted[i]] == truth[i] ? 1 : 0;
        }
        return (double) hit / truth.length;
    }

    /**
     * Normalised mutual information {@code I(T; P) / ((H(T) + H(P)) / 2)}.
     *
     * @param truth the classes
     * @param predicted the clusters
     * @return NMI in {@code [0, 1]}
     */
    public static double nmi(int[] truth, int[] predicted) {
        int a = count(truth);
        int b = count(predicted);
        double n = truth.length;
        double[][] joint = new double[a][b];
        double[] pa = new double[a];
        double[] pb = new double[b];
        for (int i = 0; i < truth.length; i++) {
            joint[truth[i]][predicted[i]] += 1 / n;
            pa[truth[i]] += 1 / n;
            pb[predicted[i]] += 1 / n;
        }
        double mi = 0;
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                if (joint[i][j] > 0) {
                    mi += joint[i][j] * Math.log(joint[i][j] / (pa[i] * pb[j]));
                }
            }
        }
        double h = entropy(pa) + entropy(pb);
        return h <= 0 ? 1 : 2 * mi / h;
    }

    private static double entropy(double[] p) {
        double h = 0;
        for (double v : p) {
            h -= v > 0 ? v * Math.log(v) : 0;
        }
        return h;
    }

    /**
     * The adjusted Rand index.
     *
     * @param truth the classes
     * @param predicted the clusters
     * @return ARI, 0 for chance, 1 for perfect
     */
    public static double ari(int[] truth, int[] predicted) {
        return MlMetrics.adjustedRandIndex(truth, predicted);
    }
}
