package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeasonalSeries;
import io.github.zzuegg.brainiac.experimental.data.SeriesArchive;
import io.github.zzuegg.brainiac.experimental.data.SeriesClassificationTasks;
import io.github.zzuegg.brainiac.experimental.ml.ClusteringMetrics;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.ml.SpectralClustering;
import io.github.zzuegg.brainiac.experimental.tsc.DtwBarycenter;
import io.github.zzuegg.brainiac.experimental.tsc.ElasticDistances;
import io.github.zzuegg.brainiac.experimental.tsc.KShape;
import io.github.zzuegg.brainiac.experimental.tsc.SoftDtw;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Elastic clustering, averaging and losses (13.6.7):
 * <ul>
 *   <li>{@link #clustering}: k-Shape ({@link KShape}) and DBA k-means ({@link DtwBarycenter#kMeans})
 *       against Euclidean {@link KMeans}, DTW k-medoids and {@link SpectralClustering} on DTW
 *       affinities, by adjusted Rand index on warped templates and phase-shifted motifs;</li>
 *   <li>{@link #averaging}: the DTW distance to the true template of the Euclidean mean, the DBA
 *       average and the soft-DTW barycentre ({@link SoftDtw#barycenter}) of warped copies;</li>
 *   <li>{@link #forecasting}: the same MLP forecaster trained with the mean squared error and with
 *       the soft-DTW loss ({@link SoftDtw#loss}) on a seasonal series with timing-jittered spikes,
 *       scored by test MSE, DTW error and the height of the predicted peak.</li>
 * </ul>
 */
public final class ElasticClustering {

    private ElasticClustering() {
    }

    private static double[][] toDouble(float[][] x) {
        return SeriesClassifiers.toDouble(x);
    }

    /**
     * Adjusted Rand index of each clustering method.
     *
     * @param task "warped" or "phase" ({@link SeriesArchive#named})
     * @param n series
     * @param length steps
     * @param seed the seed
     * @return method to ARI
     */
    public static Map<String, Double> clustering(String task, int n, int length, long seed) {
        SeriesClassificationTasks.Task t = SeriesArchive.named(task, n, length, seed);
        double[][] x = toDouble(t.series());
        int k = t.classes();
        int[] truth = t.labels();
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("k-means (Euclidean)", ClusteringMetrics.ari(truth,
                KMeans.fit(x, k, KMeans.Init.PLUS_PLUS, 5, 100, seed).labels()));
        int window = Math.max(1, length / 5);
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                d[i][j] = ElasticDistances.dtw(x[i], x[j], window);
                d[j][i] = d[i][j];
            }
        }
        out.put("DTW k-medoids", ClusteringMetrics.ari(truth, kMedoids(d, k, 5, seed)));
        double median = median(d);
        double[][] aff = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aff[i][j] = i == j ? 0 : Math.exp(-d[i][j] / median);
            }
        }
        out.put("spectral (DTW affinity)", ClusteringMetrics.ari(truth, SpectralClustering.cluster(aff, k, seed)));
        out.put("DBA k-means", ClusteringMetrics.ari(truth, DtwBarycenter.kMeans(x, k, 8, window, seed)));
        int[] best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int restart = 0; restart < 3; restart++) {
            KShape.Clustering c = KShape.fit(x, k, 15, seed + restart);
            double score = -withinSbd(x, c);
            if (score > bestScore) {
                bestScore = score;
                best = c.labels();
            }
        }
        out.put("k-Shape", ClusteringMetrics.ari(truth, best));
        return out;
    }

    private static double withinSbd(double[][] x, KShape.Clustering c) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            s += KShape.sbd(c.centroids()[c.labels()[i]], KShape.zNormalise(x[i]))[0];
        }
        return s;
    }

    private static double median(double[][] d) {
        int n = d.length;
        double[] v = new double[n * (n - 1) / 2];
        int k = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                v[k++] = d[i][j];
            }
        }
        java.util.Arrays.sort(v);
        return Math.max(1e-12, v[v.length / 2]);
    }

    /** k-medoids (alternating assignment and medoid update) on a distance matrix, best of restarts. */
    static int[] kMedoids(double[][] d, int k, int restarts, long seed) {
        int n = d.length;
        SplittableRandom r = new SplittableRandom(seed);
        int[] bestLabels = null;
        double bestCost = Double.POSITIVE_INFINITY;
        for (int rs = 0; rs < restarts; rs++) {
            int[] med = new int[k];
            java.util.Set<Integer> used = new java.util.HashSet<>();
            for (int c = 0; c < k; c++) {
                int m;
                do {
                    m = r.nextInt(n);
                } while (!used.add(m));
                med[c] = m;
            }
            int[] label = new int[n];
            double cost = 0;
            for (int it = 0; it < 20; it++) {
                cost = 0;
                for (int i = 0; i < n; i++) {
                    int arg = 0;
                    for (int c = 1; c < k; c++) {
                        if (d[i][med[c]] < d[i][med[arg]]) {
                            arg = c;
                        }
                    }
                    label[i] = arg;
                    cost += d[i][med[arg]];
                }
                boolean changed = false;
                for (int c = 0; c < k; c++) {
                    double best = Double.POSITIVE_INFINITY;
                    int arg = med[c];
                    for (int i = 0; i < n; i++) {
                        if (label[i] != c) {
                            continue;
                        }
                        double s = 0;
                        for (int j = 0; j < n; j++) {
                            if (label[j] == c) {
                                s += d[i][j];
                            }
                        }
                        if (s < best) {
                            best = s;
                            arg = i;
                        }
                    }
                    changed |= arg != med[c];
                    med[c] = arg;
                }
                if (!changed) {
                    break;
                }
            }
            if (cost < bestCost) {
                bestCost = cost;
                bestLabels = label.clone();
            }
        }
        return bestLabels;
    }

    /**
     * DTW distance from each average of warped copies of one template to the noise-free template.
     *
     * @param copies warped copies averaged
     * @param length steps
     * @param seed the seed
     * @return method to DTW distance (lower is better)
     */
    public static Map<String, Double> averaging(int copies, int length, long seed) {
        int classes = 6;
        SeriesClassificationTasks.Task t = SeriesArchive.warpedTemplates(copies * classes, length, classes, 0.45, seed);
        double[][] all = toDouble(t.series());
        double[][] x = new double[copies][];
        int k = 0;
        for (int i = 0; i < all.length; i++) {
            if (t.labels()[i] == 0) {
                x[k++] = all[i];
            }
        }
        double[] truth = SeriesArchive.warpedTemplate(length, classes, 0);
        double[] mean = new double[length];
        for (double[] s : x) {
            for (int j = 0; j < length; j++) {
                mean[j] += s[j] / x.length;
            }
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("Euclidean mean", ElasticDistances.dtw(mean, truth, -1));
        double[] medoid = x[DtwBarycenter.medoid(x, -1)];
        out.put("medoid", ElasticDistances.dtw(medoid, truth, -1));
        out.put("DBA", ElasticDistances.dtw(DtwBarycenter.average(x, medoid, 10, -1), truth, -1));
        out.put("soft-DTW barycentre", ElasticDistances.dtw(SoftDtw.barycenter(x, medoid, 0.1, 60), truth, -1));
        out.put("peak of Euclidean mean", max(mean));
        out.put("peak of template", max(truth));
        return out;
    }

    private static double max(double[] x) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            m = Math.max(m, v);
        }
        return m;
    }

    /** A seasonal series with narrow spikes whose timing jitters around every 12th step. */
    static float[] spikySeries(int length, long seed) {
        float[] base = SeasonalSeries.generate(length, 24, 0, 0.3, 0.05, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int c = 6; c < length; c += 12) {
            int t = c + r.nextInt(5) - 2;
            if (t >= 0 && t < length) {
                base[t] += 2f;
                if (t + 1 < length) {
                    base[t + 1] += 1f;
                }
            }
        }
        return base;
    }

    /**
     * MSE versus soft-DTW training of the same MLP forecaster.
     *
     * @param steps training steps
     * @param seed the seed
     * @return loss name to {test MSE, test DTW error, mean predicted peak, mean true peak}
     */
    public static Map<String, double[]> forecasting(int steps, long seed) {
        int lookback = 24;
        int horizon = 12;
        float[] series = spikySeries(1600, seed);
        SeasonalSeries.Windows train = SeasonalSeries.windows(series, 0, 1200, lookback, horizon, 1);
        SeasonalSeries.Windows test = SeasonalSeries.windows(series, 1200, 1600, lookback, horizon, horizon);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String loss : new String[] {"MSE", "soft-DTW"}) {
            Model m = Model.sequential().input(lookback).dense(64, Activation.RELU).dense(horizon).seed(seed).build();
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            int n = train.inputs().shape().dim(0);
            for (int s = 0; s < steps; s++) {
                float[] idx = new float[32];
                for (int i = 0; i < idx.length; i++) {
                    idx[i] = r.nextInt(n);
                }
                Tensor sel = Tensor.of(Shape.of(idx.length), idx);
                Tensor xb = train.inputs().gather(0, sel);
                Tensor yb = train.targets().gather(0, sel);
                opt.minimize(m.parameters(), () -> {
                    Var pred = m.forward(Var.of(xb), true);
                    return loss.equals("MSE") ? pred.sub(yb).square().mean() : SoftDtw.loss(pred, yb, 0.1);
                });
            }
            float[] p = Var.noGrad(() -> m.forward(Var.of(test.inputs()), false)).value().toArray();
            float[] y = test.targets().toArray();
            int rows = test.inputs().shape().dim(0);
            double mse = 0;
            double dtw = 0;
            double peak = 0;
            double truePeak = 0;
            for (int i = 0; i < rows; i++) {
                double[] a = new double[horizon];
                double[] b = new double[horizon];
                for (int h = 0; h < horizon; h++) {
                    a[h] = p[i * horizon + h];
                    b[h] = y[i * horizon + h];
                    mse += (a[h] - b[h]) * (a[h] - b[h]) / (rows * horizon);
                }
                dtw += ElasticDistances.dtw(a, b, -1) / rows;
                peak += max(a) / rows;
                truePeak += max(b) / rows;
            }
            out.put(loss, new double[] {mse, dtw, peak, truePeak});
        }
        return out;
    }
}
