package io.github.zzuegg.brainiac.experiments.streaming;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.anomaly.BayesianOnlineChangePoint;
import io.github.zzuegg.brainiac.experimental.anomaly.Pelt;
import io.github.zzuegg.brainiac.experimental.data.AnomalySeries;
import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.nn.FactorizationMachine;
import io.github.zzuegg.brainiac.experimental.stream.AdaptiveRandomForest;
import io.github.zzuegg.brainiac.experimental.stream.AsymmetricReplay;
import io.github.zzuegg.brainiac.experimental.stream.CoinBetting;
import io.github.zzuegg.brainiac.experimental.stream.DriftDetectors;
import io.github.zzuegg.brainiac.experimental.stream.FtrlProximal;
import io.github.zzuegg.brainiac.experimental.stream.HoeffdingTree;
import io.github.zzuegg.brainiac.experimental.stream.OnlineClassifier;
import io.github.zzuegg.brainiac.experimental.stream.OnlineLogistic;
import io.github.zzuegg.brainiac.experimental.stream.OnlineNaiveBayes;
import io.github.zzuegg.brainiac.experimental.stream.Prequential;
import io.github.zzuegg.brainiac.experimental.train.ContinualBackprop;
import io.github.zzuegg.brainiac.experimental.train.DAdaptation;
import io.github.zzuegg.brainiac.experimental.train.Prodigy;
import io.github.zzuegg.brainiac.experimental.train.ScheduleFree;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.experiments.optim.ClassIncrementalDigits;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Online learning on drifting streams, all evaluated prequentially (test, then train):
 * <ul>
 *   <li>9.6.8 FTRL-Proximal on a drifting click stream with hashed sparse features, against online
 *       SGD and AdaGrad logistic regression, a logistic regression retrained daily, and a
 *       {@link FactorizationMachine} trained online;</li>
 *   <li>9.6.9 Hoeffding trees and adaptive random forests on SEA (abrupt), hyperplane (gradual)
 *       and RBF (recurring) streams, against a {@link RandomForest} retrained on a sliding window,
 *       an online MLP, naive Bayes, no-change and majority-class predictors;</li>
 *   <li>9.6.10 drift detectors (ADWIN, DDM, EDDM, Page-Hinkley) on the error streams of a frozen
 *       classifier and on {@link AnomalySeries} mean shifts, against wave 8's
 *       {@link BayesianOnlineChangePoint} and {@link Pelt} and a fixed-window two-sample test;</li>
 *   <li>9.6.11 ER-ACE and ER-AML in online class-incremental learning on split digits and split
 *       shapes, against experience replay, DER++, iCaRL-style nearest-mean classification, A-GEM
 *       and replay with continual backprop;</li>
 *   <li>9.6.12 COCOB and KT coin betting against D-Adaptation, Prodigy, schedule-free AdamW and
 *       AdamW at its best and at a 10x-off learning rate on three tasks.</li>
 * </ul>
 */
public final class StreamShootout {

    private StreamShootout() {
    }

    // ---------------------------------------------------------------- 9.6.8

    /**
     * A click-prediction result.
     *
     * @param method the learner
     * @param logLoss progressive log loss
     * @param nonZeros non-zero weights at the end
     * @param bytes model memory at the end
     * @param lossAfterDrift progressive log loss over the first tenth after each drift
     */
    public record ClickRow(String method, double logLoss, long nonZeros, long bytes, double lossAfterDrift) {
    }

    /** Logistic regression retrained from scratch each "day" on the previous day's impressions. */
    private static final class DailyBatch implements OnlineClassifier {
        private final int space;
        private final int day;
        private OnlineLogistic model;
        private final List<DriftingStreams.Sample> today = new ArrayList<>();

        DailyBatch(int space, int day) {
            this.space = space;
            this.day = day;
        }

        @Override
        public double[] predict(DriftingStreams.Sample s) {
            return model == null ? new double[] {0.5, 0.5} : model.predict(s);
        }

        @Override
        public void learn(DriftingStreams.Sample s) {
            today.add(s);
            if (today.size() == day) {
                OnlineLogistic m = new OnlineLogistic(space, 0.1, true, 0);
                for (int epoch = 0; epoch < 2; epoch++) {
                    for (DriftingStreams.Sample x : today) {
                        m.learn(x);
                    }
                }
                model = m;
                today.clear();
            }
        }

        @Override
        public long bytes() {
            return model == null ? 0 : model.bytes();
        }

        long nonZeros() {
            return model == null ? 0 : model.nonZeros();
        }
    }

    /**
     * A factorization machine over the hashed features folded into {@code 2^10} slots, predicted
     * per impression from its current weights and trained with Adam on mini-batches of 256.
     */
    private static final class OnlineFm implements OnlineClassifier {
        private static final int SLOTS = 1024;
        private final FactorizationMachine fm;
        private final Optimizer opt = Optimizer.adam(0.01f);
        private final List<DriftingStreams.Sample> pending = new ArrayList<>();
        private float[] linear;
        private float[] factors;
        private float bias;
        private final int rank;

        OnlineFm(int rank, long seed) {
            this.rank = rank;
            this.fm = new FactorizationMachine(SLOTS, rank, new SplittableRandom(seed));
            refresh();
        }

        private void refresh() {
            List<Var> p = fm.parameters();
            bias = p.get(0).value().toArray()[0];
            linear = p.get(1).value().toArray();
            factors = p.get(2).value().toArray();
        }

        private static int[] slots(int[] active) {
            int[] s = new int[active.length];
            for (int i = 0; i < s.length; i++) {
                s[i] = active[i] % SLOTS;
            }
            return s;
        }

        @Override
        public double[] predict(DriftingStreams.Sample s) {
            int[] a = slots(s.active());
            double z = bias;
            for (int i : a) {
                z += linear[i];
            }
            for (int f = 0; f < rank; f++) {
                double sum = 0;
                double sq = 0;
                for (int i : a) {
                    double v = factors[i * rank + f];
                    sum += v;
                    sq += v * v;
                }
                z += 0.5 * (sum * sum - sq);
            }
            double p = 1 / (1 + Math.exp(-Math.max(-35, Math.min(35, z))));
            return new double[] {1 - p, p};
        }

        @Override
        public void learn(DriftingStreams.Sample s) {
            pending.add(s);
            if (pending.size() < 256) {
                return;
            }
            int b = pending.size();
            float[] x = new float[b * SLOTS];
            float[] y = new float[b];
            for (int r = 0; r < b; r++) {
                for (int i : slots(pending.get(r).active())) {
                    x[r * SLOTS + i] = 1;
                }
                y[r] = pending.get(r).label();
            }
            Var z = fm.forward(Var.of(Tensor.of(Shape.of(b, SLOTS), x)));
            Tensor yt = Tensor.of(Shape.of(b, 1), y);
            z.softplus().sub(z.mul(yt)).mean().backward();
            opt.step(fm.parameters());
            pending.clear();
            refresh();
        }

        @Override
        public long bytes() {
            return 4L * SLOTS * (rank + 1) + 4;
        }

        long nonZeros() {
            long c = 0;
            for (float v : linear) {
                c += v != 0 ? 1 : 0;
            }
            return c + (long) SLOTS * rank;
        }
    }

    /**
     * FTRL-Proximal and its baselines on a drifting click stream (drifts at a third and two
     * thirds).
     *
     * @param impressions stream length
     * @param seed the seed
     * @return one row per learner
     */
    public static List<ClickRow> clicks(int impressions, long seed) {
        int bits = 16;
        int space = 1 << bits;
        int[] drifts = {impressions / 3, 2 * impressions / 3};
        DriftingStreams.Sample[] data = DriftingStreams.take(
                DriftingStreams.clicks(6, 24, bits, DriftingStreams.Drift.ABRUPT, drifts, 1, seed), impressions);
        Map<String, OnlineClassifier> learners = new LinkedHashMap<>();
        learners.put("FTRL-Proximal (l1 1)", new FtrlProximal(space, 0.1, 1, 1, 1, 0, false));
        learners.put("FTRL-Proximal + Bloom(2) + 16-bit", new FtrlProximal(space, 0.1, 1, 1, 1, 2, true));
        learners.put("online SGD", new OnlineLogistic(space, 0.5, false, 0));
        learners.put("online AdaGrad", new OnlineLogistic(space, 0.1, true, 0));
        learners.put("logistic regression retrained daily", new DailyBatch(space, impressions / 10));
        learners.put("FactorizationMachine (online, batches of 256)", new OnlineFm(4, seed));
        List<ClickRow> out = new ArrayList<>();
        for (Map.Entry<String, OnlineClassifier> e : learners.entrySet()) {
            OnlineClassifier c = e.getValue();
            double[] perSample = new double[impressions];
            for (int i = 0; i < impressions; i++) {
                double[] p = c.predict(data[i]);
                perSample[i] = -Math.log(Math.max(1e-12, p[data[i].label()]));
                c.learn(data[i]);
            }
            double total = Arrays.stream(perSample).sum() / impressions;
            double after = 0;
            int count = 0;
            for (int d : drifts) {
                for (int i = d; i < Math.min(impressions, d + impressions / 10); i++) {
                    after += perSample[i];
                    count++;
                }
            }
            long nz = switch (c) {
                case FtrlProximal f -> f.nonZeros();
                case OnlineLogistic l -> l.nonZeros();
                case DailyBatch d -> d.nonZeros();
                case OnlineFm f -> f.nonZeros();
                default -> 0;
            };
            out.add(new ClickRow(e.getKey(), total, nz, c.bytes(), after / count));
        }
        return out;
    }

    // ---------------------------------------------------------------- 9.6.9

    /**
     * A stream-classification result.
     *
     * @param stream the stream
     * @param method the learner
     * @param accuracy prequential accuracy
     * @param recovery mean samples to recover after a drift
     * @param bytes model memory
     * @param seconds wall time
     */
    public record TreeRow(String stream, String method, double accuracy, double recovery, long bytes,
            double seconds) {
    }

    private static final class NoChange implements OnlineClassifier {
        private final int classes;
        private int last;

        NoChange(int classes) {
            this.classes = classes;
        }

        @Override
        public double[] predict(DriftingStreams.Sample s) {
            double[] p = new double[classes];
            p[last] = 0.9;
            for (int c = 0; c < classes; c++) {
                if (c != last) {
                    p[c] = 0.1 / (classes - 1);
                }
            }
            return p;
        }

        @Override
        public void learn(DriftingStreams.Sample s) {
            last = s.label();
        }
    }

    private static final class Majority implements OnlineClassifier {
        private final double[] count;

        Majority(int classes) {
            this.count = new double[classes];
        }

        @Override
        public double[] predict(DriftingStreams.Sample s) {
            double total = Arrays.stream(count).sum();
            double[] p = new double[count.length];
            for (int c = 0; c < p.length; c++) {
                p[c] = (count[c] + 1) / (total + p.length);
            }
            return p;
        }

        @Override
        public void learn(DriftingStreams.Sample s) {
            count[s.label()]++;
        }
    }

    /** A random forest refitted every {@code every} samples on the last {@code window}. */
    private static final class WindowedForest implements OnlineClassifier {
        private final int classes;
        private final int window;
        private final int every;
        private final int trees;
        private final List<DriftingStreams.Sample> recent = new ArrayList<>();
        private RandomForest forest;
        private int since;

        WindowedForest(int classes, int window, int every, int trees) {
            this.classes = classes;
            this.window = window;
            this.every = every;
            this.trees = trees;
        }

        @Override
        public double[] predict(DriftingStreams.Sample s) {
            if (forest == null) {
                double[] p = new double[classes];
                Arrays.fill(p, 1.0 / classes);
                return p;
            }
            return forest.predictProbability(s.x());
        }

        @Override
        public void learn(DriftingStreams.Sample s) {
            recent.add(s);
            if (recent.size() > window) {
                recent.removeFirst();
            }
            if (++since >= every) {
                since = 0;
                double[][] x = new double[recent.size()][];
                int[] y = new int[recent.size()];
                for (int i = 0; i < y.length; i++) {
                    x[i] = recent.get(i).x();
                    y[i] = recent.get(i).label();
                }
                forest = RandomForest.fit(x, y, classes, RandomForest.Settings
                        .defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(trees));
            }
        }

        @Override
        public long bytes() {
            return forest == null ? 0 : 24L * forest.size() + (long) window * 8 * (recent.isEmpty() ? 0
                    : recent.getFirst().x().length + 1);
        }
    }

    /** A one-hidden-layer MLP trained by SGD on every sample, inputs standardised online. */
    private static final class OnlineMlp implements OnlineClassifier {
        private final int in;
        private final int hidden;
        private final int classes;
        private final double[][] w1;
        private final double[] b1;
        private final double[][] w2;
        private final double[] b2;
        private final double[] mean;
        private final double[] m2;
        private final double rate;
        private long n;

        OnlineMlp(int in, int hidden, int classes, double rate, long seed) {
            this.in = in;
            this.hidden = hidden;
            this.classes = classes;
            this.rate = rate;
            SplittableRandom r = new SplittableRandom(seed);
            w1 = new double[in][hidden];
            w2 = new double[hidden][classes];
            b1 = new double[hidden];
            b2 = new double[classes];
            for (double[] row : w1) {
                for (int j = 0; j < hidden; j++) {
                    row[j] = r.nextGaussian() * Math.sqrt(2.0 / in);
                }
            }
            for (double[] row : w2) {
                for (int k = 0; k < classes; k++) {
                    row[k] = r.nextGaussian() * Math.sqrt(1.0 / hidden);
                }
            }
            mean = new double[in];
            m2 = new double[in];
        }

        private double[] scaled(double[] x) {
            double[] z = new double[in];
            for (int i = 0; i < in; i++) {
                double sd = n > 1 ? Math.sqrt(m2[i] / (n - 1)) : 1;
                z[i] = (x[i] - mean[i]) / Math.max(sd, 1e-3);
            }
            return z;
        }

        private double[] forward(double[] z, double[] h) {
            for (int j = 0; j < hidden; j++) {
                double s = b1[j];
                for (int i = 0; i < in; i++) {
                    s += z[i] * w1[i][j];
                }
                h[j] = Math.max(0, s);
            }
            double[] p = new double[classes];
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < classes; k++) {
                double s = b2[k];
                for (int j = 0; j < hidden; j++) {
                    s += h[j] * w2[j][k];
                }
                p[k] = s;
                max = Math.max(max, s);
            }
            double sum = 0;
            for (int k = 0; k < classes; k++) {
                p[k] = Math.exp(p[k] - max);
                sum += p[k];
            }
            for (int k = 0; k < classes; k++) {
                p[k] /= sum;
            }
            return p;
        }

        @Override
        public double[] predict(DriftingStreams.Sample s) {
            return forward(scaled(s.x()), new double[hidden]);
        }

        @Override
        public void learn(DriftingStreams.Sample s) {
            double[] x = s.x();
            n++;
            for (int i = 0; i < in; i++) {
                double d = x[i] - mean[i];
                mean[i] += d / n;
                m2[i] += d * (x[i] - mean[i]);
            }
            double[] z = scaled(x);
            double[] h = new double[hidden];
            double[] p = forward(z, h);
            p[s.label()] -= 1;
            for (int j = 0; j < hidden; j++) {
                double dh = 0;
                for (int k = 0; k < classes; k++) {
                    dh += w2[j][k] * p[k];
                    w2[j][k] -= rate * h[j] * p[k];
                }
                if (h[j] > 0) {
                    b1[j] -= rate * dh;
                    for (int i = 0; i < in; i++) {
                        w1[i][j] -= rate * z[i] * dh;
                    }
                }
            }
            for (int k = 0; k < classes; k++) {
                b2[k] -= rate * p[k];
            }
        }

        @Override
        public long bytes() {
            return 8L * (in * hidden + hidden + hidden * classes + classes + 2L * in);
        }
    }

    /**
     * Stream classifiers on the three drifting streams.
     *
     * @param n samples per stream
     * @param seed the seed
     * @return rows
     */
    public static List<TreeRow> trees(int n, long seed) {
        int[] abrupt = {n / 4, n / 2, 3 * n / 4};
        Map<String, DriftingStreams.Stream> streams = new LinkedHashMap<>();
        streams.put("SEA abrupt", DriftingStreams.sea(DriftingStreams.Drift.ABRUPT, abrupt, 1, 0.1, seed));
        streams.put("hyperplane gradual", DriftingStreams.hyperplane(10, DriftingStreams.Drift.GRADUAL, abrupt,
                n / 20, 0.05, seed));
        streams.put("RBF recurring", DriftingStreams.rbf(5, 20, 4, DriftingStreams.Drift.RECURRING, abrupt, 1, seed));
        List<TreeRow> out = new ArrayList<>();
        for (Map.Entry<String, DriftingStreams.Stream> e : streams.entrySet()) {
            DriftingStreams.Stream s = e.getValue();
            DriftingStreams.Sample[] data = DriftingStreams.take(s, n);
            int d = s.features();
            int k = s.classes();
            Map<String, OnlineClassifier> learners = new LinkedHashMap<>();
            learners.put("adaptive random forest (10 trees)", new AdaptiveRandomForest(d, k, 10, 6, 0.01, 0.001,
                    seed));
            learners.put("Hoeffding tree", HoeffdingTree.of(d, k));
            learners.put("windowed RandomForest (2000, refit every 1000)", new WindowedForest(k, 2000, 1000, 20));
            learners.put("online MLP", new OnlineMlp(d, 32, k, 0.02, seed));
            learners.put("naive Bayes", new OnlineNaiveBayes(d, k));
            learners.put("no-change", new NoChange(k));
            learners.put("majority class", new Majority(k));
            for (Map.Entry<String, OnlineClassifier> l : learners.entrySet()) {
                Prequential.Result r = Prequential.evaluate(data, s.driftPoints(), l.getValue(), 500);
                out.add(new TreeRow(e.getKey(), l.getKey(), r.accuracy(), r.meanRecovery(), r.bytes(), r.seconds()));
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- 9.6.10

    /**
     * A drift-detection result.
     *
     * @param series the series
     * @param method the detector
     * @param meanDelay mean samples from a drift to its first detection
     * @param falseAlarmsPer10k detections matching no drift, per 10,000 samples
     * @param missed drifts not detected before the next one
     * @param drifts drifts in the series
     */
    public record DetectionRow(String series, String method, double meanDelay, double falseAlarmsPer10k, int missed,
            int drifts) {
    }

    /** A fixed-window two-sample z-test on the means of the last two windows. */
    static int[] windowTest(double[] y, int w, double z) {
        List<Integer> out = new ArrayList<>();
        int last = -10 * w;
        for (int t = 2 * w; t <= y.length; t++) {
            if (t - last < 2 * w) {
                continue;
            }
            double m1 = 0;
            double m2 = 0;
            double v1 = 0;
            double v2 = 0;
            for (int i = t - 2 * w; i < t - w; i++) {
                m1 += y[i];
            }
            for (int i = t - w; i < t; i++) {
                m2 += y[i];
            }
            m1 /= w;
            m2 /= w;
            for (int i = t - 2 * w; i < t - w; i++) {
                v1 += (y[i] - m1) * (y[i] - m1);
            }
            for (int i = t - w; i < t; i++) {
                v2 += (y[i] - m2) * (y[i] - m2);
            }
            double se = Math.sqrt((v1 + v2) / (w - 1.0) / w + 1e-12);
            if (Math.abs(m2 - m1) / se > z) {
                out.add(t - 1);
                last = t;
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Scores detections against known drift positions: the first detection after each drift
     * and before the next counts; every other detection is a false alarm.
     *
     * @param series the series name
     * @param method the detector name
     * @param detections detection positions
     * @param drifts drift positions
     * @param n the series length
     * @return the scores
     */
    static DetectionRow score(String series, String method, int[] detections, int[] drifts, int n) {
        int[] sorted = detections.clone();
        Arrays.sort(sorted);
        boolean[] used = new boolean[sorted.length];
        double delay = 0;
        int found = 0;
        int missed = 0;
        for (int d = 0; d < drifts.length; d++) {
            int end = d + 1 < drifts.length ? drifts[d + 1] : n;
            int hit = -1;
            for (int i = 0; i < sorted.length; i++) {
                if (sorted[i] >= drifts[d] && sorted[i] < end) {
                    hit = i;
                    break;
                }
            }
            if (hit < 0) {
                missed++;
            } else {
                used[hit] = true;
                delay += sorted[hit] - drifts[d];
                found++;
            }
        }
        int false_ = 0;
        for (boolean u : used) {
            false_ += u ? 0 : 1;
        }
        return new DetectionRow(series, method, found == 0 ? Double.NaN : delay / found, false_ * 10_000.0 / n, missed,
                drifts.length);
    }

    /**
     * Drift detectors on three series with known change points: the 0/1 errors of a naive Bayes
     * frozen after the first concept on abrupt SEA (4 drifts) and on gradual hyperplane, and
     * {@link AnomalySeries} mean shifts.
     *
     * @param n samples per stream series
     * @param seed the seed
     * @return rows
     */
    public static List<DetectionRow> detection(int n, long seed) {
        Map<String, double[]> series = new LinkedHashMap<>();
        Map<String, int[]> truth = new LinkedHashMap<>();
        // SEA's thresholds 8 and 9 differ on only 8.5% of the input square, so a frozen model's
        // error rate moves by about 0.085; with delta 0.002 ADWIN needs several hundred samples on
        // each side of a change to see that, so the concepts last n / 5 samples (8 drifts in n
        // left about 1000 per concept, and ADWIN, DDM and the window test missed most of them)
        int[] points = new int[4];
        for (int i = 0; i < points.length; i++) {
            points[i] = (i + 1) * n / (points.length + 1);
        }
        // abrupt: SEA switching between thresholds 8 and 9 (the recurring schedule), errors of a frozen model
        DriftingStreams.Stream sea = DriftingStreams.sea(DriftingStreams.Drift.RECURRING, points, 1, 0.05, seed);
        series.put("SEA abrupt (errors)", frozenErrors(sea, n, points[0]));
        truth.put("SEA abrupt (errors)", points);
        int[] few = {n / 4, n / 2, 3 * n / 4};
        DriftingStreams.Stream hyper = DriftingStreams.hyperplane(5, DriftingStreams.Drift.GRADUAL, few, n / 40, 0.0,
                seed);
        series.put("hyperplane gradual (errors)", frozenErrors(hyper, n, few[0]));
        truth.put("hyperplane gradual (errors)", few);
        AnomalySeries.Segmented seg = AnomalySeries.changePoints(n, 50, List.of(AnomalySeries.Change.MEAN), seed);
        double[] values = new double[n];
        for (int i = 0; i < n; i++) {
            values[i] = seg.values()[i];
        }
        series.put("AnomalySeries mean shifts", values);
        truth.put("AnomalySeries mean shifts", seg.changePoints());
        List<DetectionRow> out = new ArrayList<>();
        for (Map.Entry<String, double[]> e : series.entrySet()) {
            double[] y = e.getValue();
            int[] cps = truth.get(e.getKey());
            boolean errors = e.getKey().contains("errors");
            double sd = std(y);
            List<DriftDetectors.Detector> ds = new ArrayList<>();
            ds.add(new DriftDetectors.Adwin(0.002, 5, 8));
            if (errors) {
                ds.add(new DriftDetectors.Ddm(30));
                ds.add(new DriftDetectors.Eddm(30));
            }
            ds.add(new DriftDetectors.PageHinkley(0.05 * sd, 25 * sd, 30));
            for (DriftDetectors.Detector d : ds) {
                long t0 = System.nanoTime();
                int[] det = DriftDetectors.detections(d, y);
                double us = (System.nanoTime() - t0) / 1e3 / y.length;
                out.add(new DetectionRow(e.getKey(), d.name() + String.format(" (%.2f us/step)", us),
                        score(e.getKey(), d.name(), det, cps, y.length).meanDelay(),
                        score(e.getKey(), d.name(), det, cps, y.length).falseAlarmsPer10k(),
                        score(e.getKey(), d.name(), det, cps, y.length).missed(), cps.length));
            }
            // wave 8's change-point methods, read-only
            double mean = Arrays.stream(y).limit(Math.max(10, cps.length > 0 ? cps[0] : y.length)).average().orElse(0);
            BayesianOnlineChangePoint bocpd = new BayesianOnlineChangePoint(Math.max(50, y.length / (cps.length + 1.0)),
                    mean, 1, 1, sd * sd + 1e-6);
            long t0 = System.nanoTime();
            int[] det = bocpd.run(y, 20, 0.5).detectedAt();
            double us = (System.nanoTime() - t0) / 1e3 / y.length;
            DetectionRow b = score(e.getKey(), "BOCPD", det, cps, y.length);
            out.add(new DetectionRow(b.series(), String.format("BOCPD (%.2f us/step)", us), b.meanDelay(),
                    b.falseAlarmsPer10k(), b.missed(), b.drifts()));
            int[] pelt = new Pelt(Pelt.Cost.MEAN, 2 * Math.log(y.length) * sd * sd * 4, 30).segment(y);
            DetectionRow p = score(e.getKey(), "Pelt (offline)", pelt, cps, y.length);
            out.add(p);
            out.add(score(e.getKey(), "fixed-window z-test (200)", windowTest(y, 200, 4), cps, y.length));
        }
        return out;
    }

    private static double std(double[] y) {
        double m = Arrays.stream(y).average().orElse(0);
        double s = 0;
        for (double v : y) {
            s += (v - m) * (v - m);
        }
        return Math.sqrt(s / Math.max(1, y.length - 1)) + 1e-9;
    }

    /** Trains naive Bayes on the first {@code warm} samples, then records its 0/1 errors. */
    private static double[] frozenErrors(DriftingStreams.Stream s, int n, int warm) {
        OnlineNaiveBayes nb = new OnlineNaiveBayes(s.features(), s.classes());
        DriftingStreams.Sample[] data = DriftingStreams.take(s, n);
        for (int i = 0; i < warm / 2; i++) {
            nb.learn(data[i]);
        }
        double[] err = new double[n];
        for (int i = 0; i < n; i++) {
            err[i] = OnlineClassifier.argmax(nb.predict(data[i])) == data[i].label() ? 0 : 1;
        }
        return err;
    }

    // ---------------------------------------------------------------- 9.6.11

    /**
     * An online class-incremental result.
     *
     * @param dataset split digits or split shapes
     * @param method the method
     * @param buffer the replay buffer size
     * @param finalAccuracy accuracy over all classes at the end
     * @param forgetting mean drop from each task's best accuracy to its final accuracy
     * @param afterSwitch accuracy on earlier tasks just after each task switch (mean)
     */
    public record ContinualRow(String dataset, String method, int buffer, double finalAccuracy, double forgetting,
            double afterSwitch) {
    }

    /** An MLP made of core Dense layers. */
    private static final class Net {
        final List<Dense> body = new ArrayList<>();
        final Dense head;

        Net(int in, int hidden, int classes, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            body.add(new Dense(in, hidden, Activation.RELU, Init.HE, r));
            body.add(new Dense(hidden, hidden, Activation.RELU, Init.HE, r));
            head = new Dense(hidden, classes, Activation.IDENTITY, Init.XAVIER, r);
        }

        Var features(Var x) {
            Var h = x;
            for (Dense d : body) {
                h = d.forward(h);
            }
            return h;
        }

        Var logits(Var x) {
            return head.forward(features(x));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            body.forEach(d -> p.addAll(d.parameters()));
            p.addAll(head.parameters());
            return p;
        }
    }

    private static Tensor rows(ToyTasks.Data d, int from, int to) {
        return Tensor.of(Arrays.copyOfRange(d.x(), from, to));
    }

    private static Tensor ids(ToyTasks.Data d, int from, int to) {
        float[] v = new float[to - from];
        for (int i = from; i < to; i++) {
            v[i - from] = d.y()[i];
        }
        return Tensor.of(Shape.of(v.length), v);
    }

    private static double accuracy(Function<Tensor, int[]> classify, ToyTasks.Data d) {
        if (d.size() == 0) {
            return 0;
        }
        int[] pred = classify.apply(d.inputs());
        int ok = 0;
        for (int i = 0; i < pred.length; i++) {
            ok += pred[i] == d.y()[i] ? 1 : 0;
        }
        return ok / (double) pred.length;
    }

    /**
     * Online class-incremental learning: one pass over each task in order, batches of 10.
     *
     * @param dataset "digits" (5 tasks of 2 classes) or "shapes" (3 tasks of 2 classes)
     * @param buffers buffer sizes
     * @param seed the seed
     * @return rows
     */
    public static List<ContinualRow> continual(String dataset, int[] buffers, long seed) {
        List<ClassIncrementalDigits.Task> tasks = dataset.equals("shapes") ? shapeTasks(seed)
                : ClassIncrementalDigits.split(60, seed);
        String[] methods = {"fine-tune (no replay)", "ER", "ER-ACE", "ER-AML", "DER++", "iCaRL-style NCM", "A-GEM",
            "ER + ContinualBackprop"};
        List<ContinualRow> out = new ArrayList<>();
        for (int buffer : buffers) {
            for (String m : methods) {
                if (m.startsWith("fine-tune") && buffer != buffers[0]) {
                    continue;
                }
                out.add(runContinual(dataset, tasks, m, buffer, seed));
            }
        }
        return out;
    }

    private static List<ClassIncrementalDigits.Task> shapeTasks(long seed) {
        ShapeImages gen = new ShapeImages(8, 0.1f);
        Tensor[] train = gen.generate(600, seed);
        Tensor[] test = gen.generate(240, seed + 1);
        ToyTasks.Data tr = toData(train);
        ToyTasks.Data te = toData(test);
        List<ClassIncrementalDigits.Task> out = new ArrayList<>();
        for (int t = 0; t < 3; t++) {
            int[] c = {2 * t, 2 * t + 1};
            out.add(new ClassIncrementalDigits.Task(tr.onlyClasses(c), te.onlyClasses(c), c));
        }
        return out;
    }

    private static ToyTasks.Data toData(Tensor[] d) {
        int n = d[0].shape().dim(0);
        int f = Math.toIntExact(d[0].size() / n);
        float[] all = d[0].toArray();
        float[] ids = d[1].toArray();
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            x[i] = Arrays.copyOfRange(all, i * f, (i + 1) * f);
            y[i] = (int) ids[i];
        }
        return new ToyTasks.Data(x, y, ShapeImages.CLASSES);
    }

    private static ContinualRow runContinual(String dataset, List<ClassIncrementalDigits.Task> tasks, String method,
            int bufferSize, long seed) {
        int classes = tasks.getFirst().train().classes();
        int in = tasks.getFirst().train().features();
        Net net = new Net(in, 64, classes, seed);
        List<Var> params = net.parameters();
        Optimizer opt = Optimizer.sgd(0.1f);
        AsymmetricReplay memory = new AsymmetricReplay(bufferSize, classes, seed);
        // DER++ keeps logits next to labels in the target vector: [label, logits...]
        io.github.zzuegg.brainiac.experimental.data.ReservoirReplay der =
                new io.github.zzuegg.brainiac.experimental.data.ReservoirReplay(bufferSize, seed + 1);
        ContinualBackprop cbp = method.contains("ContinualBackprop")
                ? new ContinualBackprop(net.body, 1e-3, 50, seed) : null;
        int batch = 10;
        int t = tasks.size();
        double[][] acc = new double[t][t];
        double afterSwitch = 0;
        int switches = 0;
        Function<Tensor, int[]> argmaxClassify = x -> {
            float[] z = Var.noGrad(() -> net.logits(Var.of(x))).value().toArray();
            int[] pred = new int[z.length / classes];
            for (int i = 0; i < pred.length; i++) {
                int best = 0;
                for (int c = 1; c < classes; c++) {
                    if (z[i * classes + c] > z[i * classes + best]) {
                        best = c;
                    }
                }
                pred[i] = best;
            }
            return pred;
        };
        Function<Tensor, int[]> ncm = x -> nearestMean(net, memory, x, classes, argmaxClassify);
        Function<Tensor, int[]> classify = method.startsWith("iCaRL") ? ncm : argmaxClassify;
        for (int k = 0; k < t; k++) {
            ToyTasks.Data d = tasks.get(k).train();
            int steps = d.size() / batch;
            for (int s = 0; s < steps; s++) {
                Tensor x = rows(d, s * batch, (s + 1) * batch);
                Tensor y = ids(d, s * batch, (s + 1) * batch);
                for (Var p : params) {
                    p.zeroGrad();
                }
                Var loss = switch (method) {
                    case "ER-ACE" -> memory.aceLoss(net::logits, x, y);
                    case "ER-AML" -> memory.amlLoss(net::features, net.head::forward, x, y);
                    case "DER++" -> derLoss(net, der, x, y, classes);
                    case "fine-tune (no replay)" -> Functional.crossEntropy(net.logits(Var.of(x)), y);
                    default -> {
                        if (method.equals("A-GEM") || memory.buffer().size() == 0) {
                            yield Functional.crossEntropy(net.logits(Var.of(x)), y);
                        }
                        Tensor[] old = memory.buffer().sample(batch);
                        yield Functional.crossEntropy(net.logits(Var.of(Tensor.concat(0, x, old[0]))),
                                Tensor.concat(0, y, old[1].reshape(batch)));
                    }
                };
                loss.backward();
                if (method.equals("A-GEM") && memory.buffer().size() > 0) {
                    project(net, params, memory, batch);
                }
                opt.step(params);
                if (cbp != null) {
                    cbp.update(x);
                }
                if (method.equals("DER++")) {
                    float[] z = Var.noGrad(() -> net.logits(Var.of(x))).value().toArray();
                    float[] yy = y.toArray();
                    float[] target = new float[batch * (classes + 1)];
                    for (int i = 0; i < batch; i++) {
                        target[i * (classes + 1)] = yy[i];
                        System.arraycopy(z, i * classes, target, i * (classes + 1) + 1, classes);
                    }
                    der.add(x, Tensor.of(Shape.of(batch, classes + 1), target));
                }
                memory.remember(x, y);
                if (k > 0 && s == 2) {
                    double old = 0;
                    for (int j = 0; j < k; j++) {
                        old += accuracy(classify, tasks.get(j).test());
                    }
                    afterSwitch += old / k;
                    switches++;
                }
            }
            for (int j = 0; j <= k; j++) {
                acc[k][j] = accuracy(classify, tasks.get(j).test());
            }
        }
        double fin = 0;
        double forget = 0;
        for (int j = 0; j < t; j++) {
            fin += acc[t - 1][j];
            if (j < t - 1) {
                double best = 0;
                for (int k = j; k < t - 1; k++) {
                    best = Math.max(best, acc[k][j]);
                }
                forget += best - acc[t - 1][j];
            }
        }
        return new ContinualRow(dataset, method, bufferSize, fin / t, forget / Math.max(1, t - 1),
                switches == 0 ? Double.NaN : afterSwitch / switches);
    }

    private static Var derLoss(Net net, io.github.zzuegg.brainiac.experimental.data.ReservoirReplay der, Tensor x,
            Tensor y, int classes) {
        Var loss = Functional.crossEntropy(net.logits(Var.of(x)), y);
        if (der.size() == 0) {
            return loss;
        }
        int b = y.shape().dim(0);
        Tensor[] a = der.sample(b);
        Tensor logits = a[1].slice(1, 1, classes + 1);
        // alpha * logit matching on one replayed batch, beta * labels on another (alpha 0.3, beta 0.5)
        loss = loss.add(net.logits(Var.of(a[0])).sub(logits).square().mean().mul(0.3f));
        Tensor[] c = der.sample(b);
        return loss.add(Functional.crossEntropy(net.logits(Var.of(c[0])), c[1].slice(1, 0, 1).reshape(b)).mul(0.5f));
    }

    /** A-GEM: if the gradient conflicts with the replay gradient, removes the conflicting part. */
    private static void project(Net net, List<Var> params, AsymmetricReplay memory, int batch) {
        List<Tensor> g = new ArrayList<>();
        for (Var p : params) {
            g.add(p.grad());
            p.zeroGrad();
        }
        Tensor[] old = memory.buffer().sample(batch);
        Functional.crossEntropy(net.logits(Var.of(old[0])), old[1].reshape(batch)).backward();
        double dot = 0;
        double norm = 0;
        List<Tensor> ref = new ArrayList<>();
        for (int i = 0; i < params.size(); i++) {
            Tensor r = params.get(i).grad();
            ref.add(r);
            dot += g.get(i).mul(r).sum().item();
            norm += r.square().sum().item();
        }
        for (int i = 0; i < params.size(); i++) {
            Tensor gi = g.get(i);
            if (dot < 0 && norm > 0) {
                gi = gi.sub(ref.get(i).mul((float) (dot / norm)));
            }
            params.get(i).setGrad(gi);
        }
    }

    /** iCaRL's nearest-mean-of-exemplars classifier over the replay buffer's features. */
    private static int[] nearestMean(Net net, AsymmetricReplay memory, Tensor x, int classes,
            Function<Tensor, int[]> fallback) {
        if (memory.buffer().size() == 0) {
            return fallback.apply(x);
        }
        Tensor[] all = memory.buffer().sample(Math.min(256, Math.max(memory.buffer().size() * 2, 32)));
        float[] f = Var.noGrad(() -> net.features(Var.of(all[0]))).value().toArray();
        float[] ys = all[1].toArray();
        int dim = f.length / ys.length;
        double[][] mean = new double[classes][dim];
        int[] count = new int[classes];
        for (int i = 0; i < ys.length; i++) {
            int c = (int) ys[i];
            count[c]++;
            double norm = 0;
            for (int j = 0; j < dim; j++) {
                norm += f[i * dim + j] * f[i * dim + j];
            }
            norm = Math.sqrt(norm) + 1e-8;
            for (int j = 0; j < dim; j++) {
                mean[c][j] += f[i * dim + j] / norm;
            }
        }
        float[] q = Var.noGrad(() -> net.features(Var.of(x))).value().toArray();
        int n = q.length / dim;
        int[] pred = new int[n];
        for (int i = 0; i < n; i++) {
            double norm = 0;
            for (int j = 0; j < dim; j++) {
                norm += q[i * dim + j] * q[i * dim + j];
            }
            norm = Math.sqrt(norm) + 1e-8;
            double best = Double.POSITIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                if (count[c] == 0) {
                    continue;
                }
                double dist = 0;
                for (int j = 0; j < dim; j++) {
                    double v = q[i * dim + j] / norm - mean[c][j] / count[c];
                    dist += v * v;
                }
                if (dist < best) {
                    best = dist;
                    pred[i] = c;
                }
            }
        }
        return pred;
    }

    // ---------------------------------------------------------------- 9.6.12

    /**
     * An optimizer result.
     *
     * @param task the task
     * @param method the optimizer
     * @param loss held-out loss after the budget
     */
    public record OptimizerRow(String task, String method, double loss) {
    }

    /** A small task: parameters, a training loss on a batch, a held-out loss. */
    private interface Task {
        List<Var> parameters();

        Var loss(SplittableRandom r);

        double heldOut();
    }

    private static Task logisticTask(long seed) {
        SyntheticRecsys.Table t = SyntheticRecsys.clicks(4000, 6, 12, seed);
        SyntheticRecsys.Table train = t.rows(0, 3000);
        SyntheticRecsys.Table test = t.rows(3000, 4000);
        int f = t.features().shape().dim(1);
        Var w = Var.param(Tensor.zeros(Shape.of(f, 1)));
        Var b = Var.param(Tensor.zeros(Shape.of(1)));
        Function<Tensor[], Var> bce = xy -> {
            Var z = Var.of(xy[0]).matmul(w).add(b);
            return z.softplus().sub(z.mul(xy[1])).mean();
        };
        return new Task() {
            @Override
            public List<Var> parameters() {
                return List.of(w, b);
            }

            @Override
            public Var loss(SplittableRandom r) {
                int s = r.nextInt(3000 - 32);
                SyntheticRecsys.Table m = train.rows(s, s + 32);
                return bce.apply(new Tensor[] {m.features(), m.labels()});
            }

            @Override
            public double heldOut() {
                return Var.noGrad(() -> bce.apply(new Tensor[] {test.features(), test.labels()})).value().item();
            }
        };
    }

    private static Task mlpTask(long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 60, 8, 0.05, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.05, seed, seed + 1000);
        Net net = new Net(64, 32, 10, seed);
        return new Task() {
            @Override
            public List<Var> parameters() {
                return net.parameters();
            }

            @Override
            public Var loss(SplittableRandom r) {
                int s = r.nextInt(train.size() - 32);
                return Functional.crossEntropy(net.logits(Var.of(rows(train, s, s + 32))), ids(train, s, s + 32));
            }

            @Override
            public double heldOut() {
                return Var.noGrad(() -> Functional.crossEntropy(net.logits(Var.of(test.inputs())), test.ids()))
                        .value().item();
            }
        };
    }

    private static Task regressionTask(long seed) {
        SplittableRandom data = new SplittableRandom(seed);
        int n = 1024;
        float[] x = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) (data.nextDouble() * 4 - 2);
            y[i] = (float) (Math.sin(2 * x[i]) + 0.1 * data.nextGaussian());
        }
        SplittableRandom r0 = new SplittableRandom(seed + 1);
        Dense a = new Dense(1, 32, Activation.TANH, Init.XAVIER, r0);
        Dense b = new Dense(32, 1, Activation.IDENTITY, Init.XAVIER, r0);
        Tensor xs = Tensor.of(Shape.of(n, 1), x);
        Tensor ys = Tensor.of(Shape.of(n, 1), y);
        return new Task() {
            @Override
            public List<Var> parameters() {
                List<Var> p = new ArrayList<>(a.parameters());
                p.addAll(b.parameters());
                return p;
            }

            @Override
            public Var loss(SplittableRandom r) {
                int s = r.nextInt(768 - 32);
                return b.forward(a.forward(Var.of(xs.slice(0, s, s + 32)))).sub(ys.slice(0, s, s + 32)).square()
                        .mean();
            }

            @Override
            public double heldOut() {
                return Var.noGrad(() -> b.forward(a.forward(Var.of(xs.slice(0, 768, n)))).sub(ys.slice(0, 768, n))
                        .square().mean()).value().item();
            }
        };
    }

    private static double train(Task task, Optimizer opt, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            task.loss(r).backward();
            opt.step(task.parameters());
        }
        if (opt instanceof ScheduleFree sf) {
            sf.evalMode(task.parameters());
        }
        double l = task.heldOut();
        return Double.isFinite(l) ? l : 1e9;
    }

    /**
     * Every optimizer on the three tasks at a fixed budget; AdamW's best rate is chosen per task
     * from {@code 1e-3 .. 1e-1}.
     *
     * @param steps the budget
     * @param seed the seed
     * @return rows
     */
    public static List<OptimizerRow> optimizers(int steps, long seed) {
        Map<String, Supplier<Task>> tasks = new LinkedHashMap<>();
        tasks.put("click logistic regression", () -> logisticTask(seed));
        tasks.put("digits MLP", () -> mlpTask(seed));
        tasks.put("sine regression MLP", () -> regressionTask(seed));
        List<OptimizerRow> out = new ArrayList<>();
        float[] grid = {1e-3f, 3e-3f, 1e-2f, 3e-2f, 1e-1f};
        for (Map.Entry<String, Supplier<Task>> e : tasks.entrySet()) {
            double bestLoss = Double.POSITIVE_INFINITY;
            float bestRate = grid[0];
            for (float lr : grid) {
                double l = train(e.getValue().get(), Optimizer.adamW(lr, 0.01f), steps, seed);
                if (l < bestLoss) {
                    bestLoss = l;
                    bestRate = lr;
                }
            }
            out.add(new OptimizerRow(e.getKey(), "AdamW (best lr " + bestRate + ")", bestLoss));
            float off = bestRate * 10;
            out.add(new OptimizerRow(e.getKey(), "AdamW (10x lr " + off + ")",
                    train(e.getValue().get(), Optimizer.adamW(off, 0.01f), steps, seed)));
            out.add(new OptimizerRow(e.getKey(), "AdamW (lr/10 " + bestRate / 10 + ")",
                    train(e.getValue().get(), Optimizer.adamW(bestRate / 10, 0.01f), steps, seed)));
            Map<String, Supplier<Optimizer>> free = new LinkedHashMap<>();
            free.put("COCOB-Backprop", CoinBetting::cocob);
            free.put("KT coin betting", CoinBetting::kt);
            free.put("D-Adaptation (SGD)", DAdaptation::sgd);
            free.put("Prodigy", Prodigy::of);
            free.put("schedule-free AdamW (lr 1e-2)", () -> ScheduleFree.adamW(1e-2f, 0.01f, 20));
            for (Map.Entry<String, Supplier<Optimizer>> o : free.entrySet()) {
                out.add(new OptimizerRow(e.getKey(), o.getKey(), train(e.getValue().get(), o.getValue().get(), steps,
                        seed)));
            }
        }
        return out;
    }
}
