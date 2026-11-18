package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.anomaly.AnomalyTransformer;
import io.github.zzuegg.brainiac.experimental.anomaly.DeepSvdd;
import io.github.zzuegg.brainiac.experimental.anomaly.Ecod;
import io.github.zzuegg.brainiac.experimental.anomaly.LocalOutlierFactor;
import io.github.zzuegg.brainiac.experimental.anomaly.MatrixProfile;
import io.github.zzuegg.brainiac.experimental.anomaly.OneClassSvm;
import io.github.zzuegg.brainiac.experimental.anomaly.SeriesAnomalyMetrics;
import io.github.zzuegg.brainiac.experimental.data.AnomalySeries;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.ml.IsolationForest;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Anomaly detection (8.2.11-8.2.14), tabular, image and series rows.
 * <ul>
 *   <li>{@link #tabular}: the {@code classic.ForestBenchmarks} task (a correlated two-cluster
 *       cloud with 5% uniform outliers), optionally with 20 uniform noise features, scored by
 *       AUROC and fit-plus-score time (the autoencoder gets 400 full-batch steps on the
 *       original task, 150 on the variants): the isolation forest, the k-NN distance and the
 *       autoencoder (the baselines) against {@link LocalOutlierFactor}, {@link OneClassSvm},
 *       {@link Ecod}, {@link DeepSvdd} and Deep SAD with 1% of the points labelled anomalies;</li>
 *   <li>{@link #localDensity}: a dense and a sparse cluster with anomalies just outside the
 *       dense one, where global distance or isolation scores cannot tell them from the sparse
 *       cluster's ordinary points;</li>
 *   <li>{@link #images}: one-class {@link ShapeImages} (one shape normal, the other five
 *       anomalous);</li>
 *   <li>{@link #series}: {@link AnomalySeries} with point, contextual, collective and
 *       seasonal-shift anomalies after a clean prefix: random scores, the
 *       {@link MatrixProfile} (self-join and against the clean prefix), the isolation forest,
 *       an autoencoder and Deep SVDD on windows, and the {@link AnomalyTransformer}, each
 *       scored by best point-wise F1, best point-adjusted F1 ({@link SeriesAnomalyMetrics}) and
 *       range AUC, plus the top-k discord hit rate on the collective and shift anomalies.</li>
 * </ul>
 */
public final class AnomalyBenchmark {

    private AnomalyBenchmark() {
    }

    /** Points with ground truth. */
    private record Labelled(double[][] x, boolean[] anomalous) {
    }

    /** The {@code ForestBenchmarks} cloud, with optional uniform noise features. */
    private static Labelled cloud(long seed, int noiseFeatures) {
        SplittableRandom r = new SplittableRandom(seed);
        int normal = 950;
        int outliers = 50;
        int d = 4;
        double[][] x = new double[normal + outliers][d + noiseFeatures];
        boolean[] anomalous = new boolean[x.length];
        for (int i = 0; i < normal; i++) {
            double shift = i % 2 == 0 ? -2 : 2;
            double z = r.nextGaussian();
            for (int j = 0; j < d; j++) {
                x[i][j] = shift + z + 0.3 * r.nextGaussian();
            }
        }
        for (int i = normal; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = r.nextDouble(-5, 5);
            }
            anomalous[i] = true;
        }
        for (double[] row : x) {
            for (int j = d; j < d + noiseFeatures; j++) {
                row[j] = r.nextDouble(-3, 3);
            }
        }
        return new Labelled(x, anomalous);
    }

    /**
     * Unsupervised AUROC and time of every detector on the tabular task.
     *
     * @param noiseFeatures extra uniform noise features (0 for the original task)
     * @param seed the seed
     * @return detector to {AUROC, milliseconds}
     */
    public static Map<String, double[]> tabular(int noiseFeatures, long seed) {
        Labelled data = cloud(seed, noiseFeatures);
        return detectors(data.x(), data.x(), data.anomalous(), seed, 1.0 / (data.x()[0].length),
                noiseFeatures == 0 ? 400 : 150);
    }

    /**
     * The local-density task: 500 points in a tight cluster ({@code sigma = 0.3}), 300 in a
     * wide one ({@code sigma = 2}, 8 units away), and 20 anomalies on a ring of radius 1.5
     * around the tight cluster.
     *
     * @param seed the seed
     * @return detector to {AUROC, milliseconds}
     */
    public static Map<String, double[]> localDensity(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[820][2];
        boolean[] anomalous = new boolean[820];
        for (int i = 0; i < 820; i++) {
            if (i < 500) {
                x[i] = new double[] {0.3 * r.nextGaussian(), 0.3 * r.nextGaussian()};
            } else if (i < 800) {
                x[i] = new double[] {8 + 2 * r.nextGaussian(), 2 * r.nextGaussian()};
            } else {
                double a = r.nextDouble(0, 2 * Math.PI);
                x[i] = new double[] {1.5 * Math.cos(a), 1.5 * Math.sin(a)};
                anomalous[i] = true;
            }
        }
        return detectors(x, x, anomalous, seed, 0.5, 150);
    }

    private static Map<String, double[]> detectors(double[][] train, double[][] test, boolean[] anomalous, long seed,
            double gamma, int autoencoderSteps) {
        Map<String, double[]> out = new LinkedHashMap<>();
        timed(out, "isolation forest", () -> IsolationForest.fit(train, 100, 256, seed).score(test), anomalous);
        timed(out, "kNN distance", () -> knnDistance(train, test, 10), anomalous);
        timed(out, "autoencoder", () -> autoencoder(train, test, autoencoderSteps, seed), anomalous);
        timed(out, "LOF", () -> LocalOutlierFactor.fit(train, 20).score(test), anomalous);
        timed(out, "one-class SVM", () -> OneClassSvm.fit(train, KernelFunction.rbf(gamma), 0.05, 1e-3)
                .score(test), anomalous);
        timed(out, "ECOD", () -> Ecod.fit(train).score(test), anomalous);
        timed(out, "Deep SVDD", () -> svdd(train, test, null, new int[] {32, 8}, seed), anomalous);
        if (train == test) {
            // Deep SAD: 1% of the points are labelled anomalies (drawn from the true ones)
            boolean[] labelled = new boolean[train.length];
            int want = Math.max(1, train.length / 100);
            SplittableRandom r = new SplittableRandom(seed + 9);
            for (int k = 0; k < 10 * train.length && want > 0; k++) {
                int i = r.nextInt(train.length);
                if (anomalous[i] && !labelled[i]) {
                    labelled[i] = true;
                    want--;
                }
            }
            timed(out, "Deep SAD (1% labels)", () -> svdd(train, test, labelled, new int[] {32, 8}, seed),
                    anomalous);
        }
        return out;
    }

    /** Deep SVDD (or Deep SAD with labels) after autoencoder pretraining. */
    private static double[] svdd(double[][] train, double[][] test, boolean[] labelled, int[] widths, long seed) {
        DeepSvdd model = new DeepSvdd(train[0].length, widths, seed);
        model.pretrain(train, 300, 3e-3f, seed);
        model.fit(train, labelled, 1f, 200, 1e-3f, seed);
        return model.score(test);
    }

    private static void timed(Map<String, double[]> out, String name, java.util.function.Supplier<double[]> scorer,
            boolean[] anomalous) {
        long t = System.nanoTime();
        double[] s = scorer.get();
        out.put(name, new double[] {MlMetrics.auroc(s, anomalous), (System.nanoTime() - t) / 1e6});
    }

    private static double[] knnDistance(double[][] train, double[][] test, int k) {
        double[] out = new double[test.length];
        for (int i = 0; i < test.length; i++) {
            double[] d = new double[train.length];
            for (int j = 0; j < train.length; j++) {
                d[j] = train[j] == test[i] ? Double.POSITIVE_INFINITY : MlMatrices.squaredDistance(train[j], test[i]);
            }
            Arrays.sort(d);
            out[i] = d[k - 1];
        }
        return out;
    }

    private static double[] autoencoder(double[][] train, double[][] test, int steps, long seed) {
        int d = train[0].length;
        int code = Math.max(1, Math.min(4, d / 4));
        Model ae = Model.sequential().input(d).dense(32, Activation.TANH).dense(code).dense(32, Activation.TANH)
                .dense(d).seed(seed).build();
        Tensor tx = Tensor.of(MlMatrices.toFloat(train));
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            ae.forward(Var.of(tx), true).sub(tx).square().mean().backward();
            opt.step(ae.parameters());
        }
        float[] rec = ae.predict(Tensor.of(MlMatrices.toFloat(test))).toArray();
        double[] err = new double[test.length];
        for (int i = 0; i < test.length; i++) {
            for (int j = 0; j < d; j++) {
                double e = rec[i * d + j] - test[i][j];
                err[i] += e * e;
            }
        }
        return err;
    }

    /**
     * One-class images: shape 0 is normal (trained on {@code train} images of it), scored on
     * 100 normal and 100 anomalous test images.
     *
     * @param train normal training images
     * @param seed the seed
     * @return detector to {AUROC, milliseconds}
     */
    public static Map<String, double[]> images(int train, long seed) {
        ShapeImages gen = new ShapeImages(10, 0.1f);
        double[][] normal = imagesOf(gen, train, seed, true);
        double[][] testNormal = imagesOf(gen, 100, seed + 1, true);
        double[][] testOdd = imagesOf(gen, 100, seed + 2, false);
        double[][] test = new double[200][];
        boolean[] anomalous = new boolean[200];
        for (int i = 0; i < 100; i++) {
            test[i] = testNormal[i];
            test[100 + i] = testOdd[i];
            anomalous[100 + i] = true;
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        timed(out, "isolation forest", () -> IsolationForest.fit(normal, 100, 256, seed).score(test), anomalous);
        timed(out, "kNN distance", () -> knnDistance(normal, test, 5), anomalous);
        timed(out, "autoencoder", () -> autoencoder(normal, test, 300, seed), anomalous);
        timed(out, "Deep SVDD", () -> svdd(normal, test, null, new int[] {64, 16}, seed), anomalous);
        return out;
    }

    private static double[][] imagesOf(ShapeImages gen, int n, long seed, boolean normalClass) {
        double[][] out = new double[n][];
        int filled = 0;
        long s = seed * 1000;
        while (filled < n) {
            Tensor[] batch = gen.generate(64, s++);
            float[] ids = batch[1].toArray();
            float[] px = batch[0].toArray();
            int size = px.length / ids.length;
            for (int i = 0; i < ids.length && filled < n; i++) {
                if ((ids[i] == 0) == normalClass) {
                    double[] row = new double[size];
                    for (int j = 0; j < size; j++) {
                        row[j] = px[i * size + j];
                    }
                    out[filled++] = row;
                }
            }
        }
        return out;
    }

    /**
     * Series anomaly scores.
     *
     * @param pointF1 best point-wise F1
     * @param adjustedF1 best point-adjusted F1
     * @param rangeAuc range AUC with a 10-step buffer
     */
    public record SeriesScore(double pointF1, double adjustedF1, double rangeAuc) {
        @Override
        public String toString() {
            return String.format("F1 %.2f PA-F1 %.2f R-AUC %.3f", pointF1, adjustedF1, rangeAuc);
        }
    }

    /**
     * The series rows, averaged over seeds.
     *
     * @param seeds how many series
     * @param transformerSteps Anomaly Transformer training steps
     * @return method to scores, plus {@code "discord hit rate"} (in {@code pointF1}) for the
     *     matrix profile's top-k discords on the collective and shift anomalies
     */
    public static Map<String, SeriesScore> series(int seeds, int transformerSteps) {
        Map<String, double[]> acc = new LinkedHashMap<>();
        int period = 32;
        int clean = 640;
        for (int s = 0; s < seeds; s++) {
            AnomalySeries.Labelled data = AnomalySeries.anomalies(1600, period, clean, 2,
                    List.of(AnomalySeries.Kind.values()), 40 + s);
            float[] y = data.values();
            double[] yd = new double[y.length];
            for (int t = 0; t < y.length; t++) {
                yd[t] = y[t];
            }
            int n = y.length;
            boolean[] truth = Arrays.copyOfRange(data.labels(), clean, n);
            double[] trainSeries = Arrays.copyOf(yd, clean);
            double[] testSeries = Arrays.copyOfRange(yd, clean, n);
            Map<String, double[]> scores = new LinkedHashMap<>();
            scores.put("random", SeriesAnomalyMetrics.randomScores(truth.length, s));
            MatrixProfile self = MatrixProfile.stomp(yd, period);
            scores.put("matrix profile (self-join)", Arrays.copyOfRange(self.perStep(n), clean, n));
            scores.put("matrix profile (vs clean prefix)",
                    MatrixProfile.abJoin(testSeries, trainSeries, period).perStep(truth.length));
            double[][] trainWin = windows(trainSeries, period, 2);
            double[][] testWin = windows(testSeries, period, 1);
            scores.put("isolation forest (windows)", SeriesAnomalyMetrics.perStep(
                    IsolationForest.fit(trainWin, 100, 256, s).score(testWin), period, truth.length));
            scores.put("autoencoder (windows)", SeriesAnomalyMetrics.perStep(autoencoder(trainWin, testWin, 300, s),
                    period, truth.length));
            scores.put("Deep SVDD (windows)", SeriesAnomalyMetrics.perStep(svdd(trainWin, testWin, null,
                    new int[] {32, 8}, s), period, truth.length));
            AnomalyTransformer at = new AnomalyTransformer(1, period, 16, 1, new SplittableRandom(s));
            at.fit(Arrays.copyOf(y, clean), transformerSteps, 3f, s);
            scores.put("Anomaly Transformer", at.score(Arrays.copyOfRange(y, clean, n)));
            for (Map.Entry<String, double[]> e : scores.entrySet()) {
                double[] a = acc.computeIfAbsent(e.getKey(), k -> new double[3]);
                a[0] += SeriesAnomalyMetrics.bestF1(e.getValue(), truth, false) / seeds;
                a[1] += SeriesAnomalyMetrics.bestF1(e.getValue(), truth, true) / seeds;
                a[2] += SeriesAnomalyMetrics.rangeAuc(e.getValue(), truth, 10) / seeds;
            }
            // top-k discords on the collective and seasonal-shift anomalies
            List<AnomalySeries.Event> events = data.events().stream().filter(ev -> ev.kind()
                    == AnomalySeries.Kind.COLLECTIVE || ev.kind() == AnomalySeries.Kind.SEASONAL_SHIFT).toList();
            int[] discords = self.discords(events.size());
            int hits = 0;
            for (AnomalySeries.Event ev : events) {
                for (int dsc : discords) {
                    if (dsc + period > ev.start() && dsc < ev.end()) {
                        hits++;
                        break;
                    }
                }
            }
            acc.computeIfAbsent("discord hit rate", k -> new double[3])[0] += hits / (double) events.size() / seeds;
        }
        Map<String, SeriesScore> out = new LinkedHashMap<>();
        acc.forEach((k, a) -> out.put(k, new SeriesScore(a[0], a[1], a[2])));
        return out;
    }

    private static double[][] windows(double[] y, int m, int stride) {
        int count = (y.length - m) / stride + 1;
        double[][] w = new double[count][];
        for (int i = 0; i < count; i++) {
            w[i] = Arrays.copyOfRange(y, i * stride, i * stride + m);
        }
        return w;
    }
}
