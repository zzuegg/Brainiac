package io.github.zzuegg.brainiac.experiments.science;

import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.ml.Dbscan;
import io.github.zzuegg.brainiac.experimental.ml.DensityClustering;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.science.NeuralDecoders;
import io.github.zzuegg.brainiac.experimental.science.SpikeSorting;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Neural population decoding and spike sorting (wave 13, items 13.3.16–13.3.17).
 */
public final class Neuroscience {

    private Neuroscience() {
    }

    private static double[][] rows(double[][] m, int from, int to) {
        return Arrays.copyOfRange(m, from, to);
    }

    // ================================================================== 13.3.16 decoding

    /**
     * Decodes cursor velocity from cosine-tuned Poisson neurons with every decoder: held-out
     * R² against the number of training reaches (50 ms bins), with 30% of the neurons silenced at
     * test time, after a drift of the preferred directions over the session, and at 20 ms bins.
     *
     * @param neurons neurons
     * @param trainSizes training reaches, ascending (the largest is used for dropout and drift)
     * @param testTrials held-out reaches
     * @param lstmSteps LSTM Adam steps
     * @param seed the seed
     * @return decoder to {R² per training size..., R² with dropout, R² under drift, R² at 20 ms}
     */
    public static Map<String, double[]> decoders(int neurons, int[] trainSizes, int testTrials, int lstmSteps,
            long seed) {
        int maxTrain = trainSizes[trainSizes.length - 1];
        ScienceSims.NeuralRecording rec = ScienceSims.neuralPopulation(neurons, maxTrain + testTrials, 0.05, 10, 0,
                seed);
        ScienceSims.NeuralRecording drift = ScienceSims.neuralPopulation(neurons, maxTrain + testTrials, 0.05, 10,
                0.8, seed + 1);
        int fineTrain = trainSizes[Math.min(1, trainSizes.length - 1)];
        ScienceSims.NeuralRecording fine = ScienceSims.neuralPopulation(neurons, fineTrain + testTrials, 0.02, 10, 0,
                seed + 2);
        int per = 20;
        int finePer = 50;
        int testFrom = maxTrain * per;
        int end = (maxTrain + testTrials) * per;
        double[][] testCounts = rows(rec.counts(), testFrom, end);
        double[][] testVel = rows(rec.velocity(), testFrom, end);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] dropped = new double[testCounts.length][];
        boolean[] silent = new boolean[neurons];
        for (int j = 0; j < neurons; j++) {
            silent[j] = r.nextDouble() < 0.3;
        }
        for (int i = 0; i < dropped.length; i++) {
            dropped[i] = testCounts[i].clone();
            for (int j = 0; j < neurons; j++) {
                if (silent[j]) {
                    dropped[i][j] = 0;
                }
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (NeuralDecoders.Method m : NeuralDecoders.Method.values()) {
            double[] row = new double[trainSizes.length + 3];
            NeuralDecoders last = null;
            for (int k = 0; k < trainSizes.length; k++) {
                int to = trainSizes[k] * per;
                NeuralDecoders d = new NeuralDecoders(m, 3).recurrent(lstmSteps, 24)
                        .fit(rows(rec.counts(), 0, to), rows(rec.velocity(), 0, to), 0.05, seed);
                row[k] = NeuralDecoders.r2(d.decode(testCounts), testVel);
                last = d;
            }
            row[trainSizes.length] = NeuralDecoders.r2(last.decode(dropped), testVel);
            NeuralDecoders dd = new NeuralDecoders(m, 3).recurrent(lstmSteps, 24)
                    .fit(rows(drift.counts(), 0, testFrom), rows(drift.velocity(), 0, testFrom), 0.05, seed);
            row[trainSizes.length + 1] = NeuralDecoders.r2(dd.decode(rows(drift.counts(), testFrom, end)),
                    rows(drift.velocity(), testFrom, end));
            if (m == NeuralDecoders.Method.RECURRENT) {
                row[trainSizes.length + 2] = Double.NaN;
            } else {
                int fTo = fineTrain * finePer;
                int fEnd = (fineTrain + testTrials) * finePer;
                NeuralDecoders df = new NeuralDecoders(m, 8).fit(rows(fine.counts(), 0, fTo),
                        rows(fine.velocity(), 0, fTo), 0.02, seed);
                row[trainSizes.length + 2] = NeuralDecoders.r2(df.decode(rows(fine.counts(), fTo, fEnd)),
                        rows(fine.velocity(), fTo, fEnd));
            }
            out.put(m.toString(), row);
        }
        return out;
    }

    // ================================================================== 13.3.17 spike sorting

    /**
     * Sorts synthetic recordings with the three sorters and the clustering baselines.
     *
     * @param channels channels
     * @param units units
     * @param seconds recording length
     * @param noise noise standard deviation (template troughs about 1)
     * @param similar whether templates come in similar pairs
     * @param seed the seed
     * @return sorter to {mean unit accuracy, false-positive rate, false-negative rate, recall on
     *     overlapping spikes, refractory-violation rate}
     */
    public static Map<String, double[]> sorting(int channels, int units, double seconds, double noise,
            boolean similar, long seed) {
        double rate = 20000;
        ScienceSims.SpikeRecording rec = ScienceSims.spikeTrace(channels, units, seconds, rate, noise, 0.15, similar,
                seed);
        // two spare clusters absorb overlaps and outliers; every cluster is then mapped to its
        // majority unit (an oracle merge, the same for every method)
        int clusters = units + 2;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (SpikeSorting.Method m : SpikeSorting.Method.values()) {
            SpikeSorting.Sorted s = SpikeSorting.sort(rec.trace(), clusters, m, rate, seed);
            out.put(m.toString(), score(rec, s, rate));
        }
        // baselines on the same detections
        int width = (int) (rate * 0.0015);
        int before = width / 3;
        double[] sd = SpikeSorting.noiseSd(rec.trace());
        int[] times = SpikeSorting.detect(rec.trace(), sd, 4.5, Math.max(2, (int) (rate * 0.0004)), width);
        double[][] snips = SpikeSorting.snippets(rec.trace(), times, width, before);
        int[] km = kMeans(snips, clusters, 50, seed);
        out.put("k-means on raw snippets", score(rec, new SpikeSorting.Sorted(times, km, clusters), rate));
        int[] gm = GaussianMixture.fit(snips, clusters, 100, 1e-2, seed).predict(snips);
        out.put("GMM on raw snippets", score(rec, new SpikeSorting.Sorted(times, gm, clusters), rate));
        double[][] pcs = ComponentAnalysis.pca(snips, 5).transform(snips);
        int[] db = Dbscan.cluster(pcs, Dbscan.suggestEps(pcs, 5, 0.9), 5);
        out.put("PCA + DBSCAN", score(rec, new SpikeSorting.Sorted(times, db, max(db) + 1), rate));
        DensityClustering.Hdbscan h = DensityClustering.hdbscan(pcs, 15, 5);
        out.put("PCA + HDBSCAN", score(rec, new SpikeSorting.Sorted(times, h.labels(), h.clusters()), rate));
        return out;
    }

    private static int max(int[] v) {
        int m = -1;
        for (int x : v) {
            m = Math.max(m, x);
        }
        return m;
    }

    static int[] kMeans(double[][] x, int k, int iterations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.length;
        int d = x[0].length;
        double[][] c = new double[k][];
        c[0] = x[r.nextInt(n)].clone();
        double[] dist = new double[n];
        for (int j = 1; j < k; j++) {
            double total = 0;
            for (int i = 0; i < n; i++) {
                dist[i] = Double.POSITIVE_INFINITY;
                for (int a = 0; a < j; a++) {
                    dist[i] = Math.min(dist[i], sq(x[i], c[a]));
                }
                total += dist[i];
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
            c[j] = x[pick].clone();
        }
        int[] label = new int[n];
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                for (int a = 0; a < k; a++) {
                    double v = sq(x[i], c[a]);
                    if (v < best) {
                        best = v;
                        label[i] = a;
                    }
                }
            }
            double[][] s = new double[k][d];
            int[] cnt = new int[k];
            for (int i = 0; i < n; i++) {
                cnt[label[i]]++;
                for (int j = 0; j < d; j++) {
                    s[label[i]][j] += x[i][j];
                }
            }
            for (int a = 0; a < k; a++) {
                if (cnt[a] > 0) {
                    for (int j = 0; j < d; j++) {
                        c[a][j] = s[a][j] / cnt[a];
                    }
                }
            }
        }
        return label;
    }

    private static double sq(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    /**
     * Scores a sorting: each cluster is mapped to the true unit it shares most matched spikes
     * with (within 0.5 ms), so clusters of one unit merge; per true unit, accuracy {@code TP / (TP + FN + FP)} over the
     * clusters mapped to it.
     */
    private static double[] score(ScienceSims.SpikeRecording rec, SpikeSorting.Sorted s, double rate) {
        int units = rec.spikeTimes().length;
        int tol = (int) Math.round(0.0005 * rate);
        int clusters = Math.max(1, s.clusters());
        // nearest true spike of each sorted spike, per unit
        int[][] overlap = new int[clusters][units];
        for (int k = 0; k < s.times().length; k++) {
            if (s.labels()[k] < 0) {
                continue;
            }
            for (int u = 0; u < units; u++) {
                if (near(rec.spikeTimes()[u], s.times()[k], tol)) {
                    overlap[s.labels()[k]][u]++;
                }
            }
        }
        int[] map = new int[clusters];
        for (int c = 0; c < clusters; c++) {
            int best = 0;
            for (int u = 1; u < units; u++) {
                if (overlap[c][u] > overlap[c][best]) {
                    best = u;
                }
            }
            map[c] = best;
        }
        double acc = 0;
        double fpRate = 0;
        double fnRate = 0;
        int overlapping = 0;
        int overlapHit = 0;
        for (int u = 0; u < units; u++) {
            int[] truth = rec.spikeTimes()[u];
            List2 sorted = new List2();
            for (int k = 0; k < s.times().length; k++) {
                if (s.labels()[k] >= 0 && map[s.labels()[k]] == u) {
                    sorted.add(s.times()[k]);
                }
            }
            int[] st = sorted.toSortedArray();
            int tp = 0;
            for (int t : truth) {
                boolean hit = near(st, t, tol);
                tp += hit ? 1 : 0;
                boolean isOverlap = false;
                for (int v = 0; v < units; v++) {
                    if (v != u && near(rec.spikeTimes()[v], t, (int) (0.001 * rate))) {
                        isOverlap = true;
                        break;
                    }
                }
                if (isOverlap) {
                    overlapping++;
                    overlapHit += hit ? 1 : 0;
                }
            }
            int fn = truth.length - tp;
            int fp = 0;
            for (int t : st) {
                fp += near(truth, t, tol) ? 0 : 1;
            }
            acc += tp / (double) Math.max(1, tp + fn + fp) / units;
            fpRate += fp / (double) Math.max(1, st.length) / units;
            fnRate += fn / (double) Math.max(1, truth.length) / units;
        }
        // refractory violations of the sorted clusters (ISI < 1.5 ms)
        int isis = 0;
        int viol = 0;
        for (int c = 0; c < clusters; c++) {
            List2 l = new List2();
            for (int k = 0; k < s.times().length; k++) {
                if (s.labels()[k] == c) {
                    l.add(s.times()[k]);
                }
            }
            int[] t = l.toSortedArray();
            for (int k = 1; k < t.length; k++) {
                isis++;
                viol += t[k] - t[k - 1] < 0.0015 * rate ? 1 : 0;
            }
        }
        return new double[] {acc, fpRate, fnRate, overlapHit / (double) Math.max(1, overlapping),
            viol / (double) Math.max(1, isis)};
    }

    private static boolean near(int[] sorted, int t, int tol) {
        int i = Arrays.binarySearch(sorted, t);
        if (i >= 0) {
            return true;
        }
        int ins = -i - 1;
        return (ins < sorted.length && sorted[ins] - t <= tol) || (ins > 0 && t - sorted[ins - 1] <= tol);
    }

    private static final class List2 {
        private int[] v = new int[16];
        private int n;

        void add(int x) {
            if (n == v.length) {
                v = Arrays.copyOf(v, 2 * n);
            }
            v[n++] = x;
        }

        int[] toSortedArray() {
            int[] a = Arrays.copyOf(v, n);
            Arrays.sort(a);
            return a;
        }
    }
}
