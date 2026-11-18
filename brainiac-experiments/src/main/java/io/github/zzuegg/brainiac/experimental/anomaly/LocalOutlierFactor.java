package io.github.zzuegg.brainiac.experimental.anomaly;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * The local outlier factor (Breunig, Kriegel, Ng and Sander, SIGMOD 2000,
 * <a href="https://doi.org/10.1145/335191.335388">doi:10.1145/335191.335388</a>): a point's
 * local reachability density (the inverse mean reachability distance to its {@code k}
 * nearest neighbours, where the reachability distance to {@code o} is
 * {@code max(k-distance(o), d(p, o))}) compared with its neighbours' densities. A score near
 * 1 means as dense as the neighbourhood; well above 1 means locally sparse, so a point next
 * to a dense cluster counts as an outlier even when it would be ordinary next to a sparse
 * one, which a global density or distance threshold cannot express.
 */
public final class LocalOutlierFactor {

    private final double[][] train;
    private final int k;
    private final double[] kDistance;
    private final double[] lrd;
    private final double[] trainScores;

    private LocalOutlierFactor(double[][] train, int k) {
        this.train = train;
        this.k = k;
        int n = train.length;
        int[][] nn = new int[n][];
        double[][] dist = new double[n][];
        kDistance = new double[n];
        for (int i = 0; i < n; i++) {
            double[] d = distances(train[i]);
            d[i] = Double.POSITIVE_INFINITY;
            nn[i] = nearest(d, k);
            dist[i] = d;
            kDistance[i] = d[nn[i][k - 1]];
        }
        lrd = new double[n];
        for (int i = 0; i < n; i++) {
            lrd[i] = density(dist[i], nn[i]);
        }
        trainScores = new double[n];
        for (int i = 0; i < n; i++) {
            trainScores[i] = factor(lrd[i], nn[i]);
        }
    }

    /**
     * Fits the detector.
     *
     * @param x the reference points {@code [n][d]}, {@code n > k}
     * @param k the neighbourhood size, for example 20
     * @return the detector
     */
    public static LocalOutlierFactor fit(double[][] x, int k) {
        if (x.length <= k) {
            throw new IllegalArgumentException("need more than k points");
        }
        return new LocalOutlierFactor(x, k);
    }

    private double[] distances(double[] p) {
        double[] d = new double[train.length];
        for (int j = 0; j < train.length; j++) {
            d[j] = Math.sqrt(MlMatrices.squaredDistance(p, train[j]));
        }
        return d;
    }

    private static int[] nearest(double[] d, int k) {
        int[] idx = new int[k];
        double[] best = new double[k];
        java.util.Arrays.fill(best, Double.POSITIVE_INFINITY);
        for (int j = 0; j < d.length; j++) {
            if (d[j] < best[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && best[pos - 1] > d[j]) {
                    best[pos] = best[pos - 1];
                    idx[pos] = idx[pos - 1];
                    pos--;
                }
                best[pos] = d[j];
                idx[pos] = j;
            }
        }
        return idx;
    }

    private double density(double[] d, int[] nn) {
        double reach = 0;
        for (int o : nn) {
            reach += Math.max(kDistance[o], d[o]);
        }
        return nn.length / Math.max(1e-12, reach);
    }

    private double factor(double ownDensity, int[] nn) {
        double s = 0;
        for (int o : nn) {
            s += lrd[o];
        }
        return s / nn.length / Math.max(1e-12, ownDensity);
    }

    /** @return the LOF of each reference point (the transductive, outlier-detection use) */
    public double[] trainingScores() {
        return trainScores.clone();
    }

    /**
     * Scores new points against the reference set (the novelty-detection use).
     *
     * @param x points {@code [m][d]}
     * @return their LOF, higher is more anomalous
     */
    public double[] score(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double[] d = distances(x[i]);
            int[] nn = nearest(d, k);
            out[i] = factor(density(d, nn), nn);
        }
        return out;
    }
}
