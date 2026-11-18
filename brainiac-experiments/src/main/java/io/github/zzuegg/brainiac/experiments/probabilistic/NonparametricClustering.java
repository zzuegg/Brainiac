package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.Dbscan;
import io.github.zzuegg.brainiac.experimental.ml.DirichletProcessMixture;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.VariationalGaussianMixture;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Clustering with an unknown number of clusters: Dirichlet-process mixtures against model
 * selection.
 *
 * <ul>
 *   <li>{@link #unknownK}: 2-D Gaussian blobs, {@code K} from 2 to 12, unequal sizes and
 *       spreads. {@link DirichletProcessMixture} (collapsed Gibbs, stick-breaking VI, DP-means)
 *       against {@link GaussianMixture} selected by BIC, {@link KMeans} with the elbow
 *       (kneedle) rule and {@link Dbscan} with its suggested radius. The DP models get the same
 *       weak prior: an expected cluster standard deviation of 0.6 and {@code alpha = 1}. Scores:
 *       adjusted Rand index, absolute error in {@code K} and held-out log-likelihood, averaged
 *       over {@code K}. A {@link VariationalGaussianMixture} row (truncation 16, Dirichlet
 *       concentration 0.01) was added in wave 11; it reports no held-out likelihood.</li>
 *   <li>{@link #stream}: points arrive in five chunks and every chunk brings two new clusters.
 *       Warm-started Gibbs (ten sweeps per chunk) and single-pass online DP-means against
 *       k-means with the {@code K} its elbow picked on the first chunk, and BIC-GMM refitted on
 *       everything seen. Scores after every chunk, averaged.</li>
 * </ul>
 */
public final class NonparametricClustering {

    private static final double SIGMA = 0.6;

    private NonparametricClustering() {
    }

    private record Blobs(double[][] centres, double[] spreads, double[] weights) {

        double[][] sample(int n, int[] labels, int clusters, SplittableRandom r) {
            double total = 0;
            for (int c = 0; c < clusters; c++) {
                total += weights[c];
            }
            double[][] x = new double[n][];
            for (int i = 0; i < n; i++) {
                double u = r.nextDouble() * total;
                int c = 0;
                while (c < clusters - 1 && (u -= weights[c]) > 0) {
                    c++;
                }
                labels[i] = c;
                x[i] = new double[] {centres[c][0] + spreads[c] * r.nextGaussian(),
                    centres[c][1] + spreads[c] * r.nextGaussian()};
            }
            return x;
        }
    }

    private static Blobs blobs(int k, SplittableRandom r) {
        double[][] c = new double[k][];
        for (int i = 0; i < k; i++) {
            boolean ok;
            do {
                c[i] = new double[] {-10 + 20 * r.nextDouble(), -10 + 20 * r.nextDouble()};
                ok = true;
                for (int j = 0; j < i; j++) {
                    ok &= MlMatrices.squaredDistance(c[i], c[j]) > 16;
                }
            } while (!ok);
        }
        double[] s = new double[k];
        double[] w = new double[k];
        for (int i = 0; i < k; i++) {
            s[i] = SIGMA * (0.7 + 0.6 * r.nextDouble());
            w[i] = 0.3 + r.nextDouble();
        }
        return new Blobs(c, s, w);
    }

    private static int elbow(double[][] x, int maxK, long seed) {
        double[] inertia = new double[maxK];
        for (int k = 1; k <= maxK; k++) {
            inertia[k - 1] = KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 2, 50, seed).inertia();
        }
        // kneedle: the point furthest below the chord from (1, I_1) to (maxK, I_maxK), normalised
        int best = 1;
        double bestGap = Double.NEGATIVE_INFINITY;
        for (int k = 1; k <= maxK; k++) {
            double xn = (k - 1.0) / (maxK - 1);
            double yn = (inertia[k - 1] - inertia[maxK - 1]) / (inertia[0] - inertia[maxK - 1]);
            double gap = (1 - xn) - yn;
            if (gap > bestGap) {
                bestGap = gap;
                best = k;
            }
        }
        return best;
    }

    private static int count(int[] labels) {
        return (int) Arrays.stream(labels).filter(v -> v >= 0).distinct().count();
    }

    private static double[] score(int[] truth, int[] found, int k, double heldOut) {
        return new double[] {MlMetrics.adjustedRandIndex(truth, found), Math.abs(count(found) - k), heldOut};
    }

    private static void add(Map<String, double[]> out, String key, double[] v, int runs) {
        double[] acc = out.computeIfAbsent(key, s -> new double[v.length]);
        for (int i = 0; i < v.length; i++) {
            acc[i] += v[i] / runs;
        }
    }

    /**
     * Unknown-K blobs.
     *
     * @param ks the true cluster counts
     * @param n points per data set
     * @param seed the seed
     * @return method to mean {@code {ARI, |K error|, held-out log-likelihood (NaN if none)}}
     */
    public static Map<String, double[]> unknownK(int[] ks, int n, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        for (int k : ks) {
            Blobs b = blobs(k, r);
            int[] y = new int[n];
            double[][] x = b.sample(n, y, k, r);
            double[][] test = b.sample(n, new int[n], k, r);
            DirichletProcessMixture.Prior prior = DirichletProcessMixture.Prior.of(x, SIGMA * SIGMA);
            DirichletProcessMixture.Result g = DirichletProcessMixture.gibbs(x, 1, prior, 30, null, seed);
            add(out, "DP Gibbs", score(y, g.labels(), k, g.meanLogLikelihood(test)), ks.length);
            DirichletProcessMixture.Result v = DirichletProcessMixture.variational(x, 1, prior, 20, 300, seed);
            add(out, "DP stick-breaking VI", score(y, v.labels(), k, v.meanLogLikelihood(test)), ks.length);
            DirichletProcessMixture.Result dm = DirichletProcessMixture.dpMeans(x, Math.pow(5 * SIGMA, 2), 50);
            add(out, "DP-means", score(y, dm.labels(), k, dm.meanLogLikelihood(test)), ks.length);
            GaussianMixture gmm = GaussianMixture.selectByBic(x, 14, 60, seed);
            add(out, "GMM + BIC", score(y, gmm.predict(x), k, gmm.meanLogLikelihood(test)), ks.length);
            int ke = elbow(x, 14, seed);
            add(out, "k-means + elbow", score(y, KMeans.fit(x, ke, KMeans.Init.PLUS_PLUS, 3, 50, seed).labels(), k,
                    Double.NaN), ks.length);
            add(out, "DBSCAN", score(y, Dbscan.cluster(x, Dbscan.suggestEps(x, 4, 0.9), 5), k, Double.NaN),
                    ks.length);
            // wave 11 (11.3.13): one variational Bayesian fit with a generous truncation
            VariationalGaussianMixture vb = VariationalGaussianMixture.fit(x, 16,
                    VariationalGaussianMixture.Prior.FINITE_DIRICHLET, 1e-2, SIGMA * SIGMA, 300, seed);
            add(out, "VB-GMM (Dirichlet 0.01)", score(y, vb.labels(), k, Double.NaN), ks.length);
        }
        return out;
    }

    /**
     * A stream in which new clusters keep appearing.
     *
     * @param chunks the number of chunks (two new clusters each)
     * @param perChunk points per chunk
     * @param seed the seed
     * @return method to {@code {ARI, |K error|}} averaged over chunks
     */
    public static Map<String, double[]> stream(int chunks, int perChunk, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Blobs b = blobs(2 * chunks, r);
        List<double[]> seen = new ArrayList<>();
        List<Integer> truth = new ArrayList<>();
        Map<String, double[]> out = new LinkedHashMap<>();
        int[] gibbsLabels = null;
        List<double[]> centres = new ArrayList<>();
        List<Integer> sizes = new ArrayList<>();
        List<Integer> onlineLabels = new ArrayList<>();
        double lambda = Math.pow(5 * SIGMA, 2);
        int firstK = -1;
        for (int c = 0; c < chunks; c++) {
            int k = 2 * (c + 1);
            int[] y = new int[perChunk];
            double[][] chunk = b.sample(perChunk, y, k, r);
            for (int i = 0; i < perChunk; i++) {
                seen.add(chunk[i]);
                truth.add(y[i]);
                // online DP-means: nearest centre or a new one, then a running-mean update
                int best = -1;
                double bd = lambda;
                for (int j = 0; j < centres.size(); j++) {
                    double dist = MlMatrices.squaredDistance(chunk[i], centres.get(j));
                    if (dist <= bd) {
                        bd = dist;
                        best = j;
                    }
                }
                if (best < 0) {
                    centres.add(chunk[i].clone());
                    sizes.add(1);
                    best = centres.size() - 1;
                } else {
                    int m = sizes.get(best) + 1;
                    sizes.set(best, m);
                    double[] ce = centres.get(best);
                    for (int a = 0; a < 2; a++) {
                        ce[a] += (chunk[i][a] - ce[a]) / m;
                    }
                }
                onlineLabels.add(best);
            }
            double[][] x = seen.toArray(new double[0][]);
            int[] t = truth.stream().mapToInt(Integer::intValue).toArray();
            int[] start = new int[x.length];
            int fresh = gibbsLabels == null ? 0 : Arrays.stream(gibbsLabels).max().orElse(0) + 1;
            for (int i = 0; i < x.length; i++) {
                start[i] = gibbsLabels != null && i < gibbsLabels.length ? gibbsLabels[i] : fresh;
            }
            gibbsLabels = DirichletProcessMixture.gibbs(x, 1, DirichletProcessMixture.Prior.of(x, SIGMA * SIGMA), 10,
                    start, seed + c).labels();
            add(out, "DP Gibbs (warm start)", score(t, gibbsLabels, k, 0), chunks);
            add(out, "online DP-means", score(t, onlineLabels.stream().mapToInt(Integer::intValue).toArray(), k, 0),
                    chunks);
            if (firstK < 0) {
                firstK = elbow(x, 14, seed);
            }
            add(out, "k-means, K from chunk 1", score(t, KMeans.fit(x, firstK, KMeans.Init.PLUS_PLUS, 3, 50, seed)
                    .labels(), k, 0), chunks);
            add(out, "GMM + BIC refit", score(t, GaussianMixture.selectByBic(x, 14, 60, seed).predict(x), k, 0),
                    chunks);
        }
        out.replaceAll((key, v) -> Arrays.copyOf(v, 2));
        return out;
    }
}
