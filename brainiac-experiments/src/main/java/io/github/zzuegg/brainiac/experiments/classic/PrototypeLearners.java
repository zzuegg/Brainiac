package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.metric.PairLosses;
import io.github.zzuegg.brainiac.experimental.ml.AdaptiveResonance;
import io.github.zzuegg.brainiac.experimental.ml.GrowingNeuralGas;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.SelfOrganizingMap;
import io.github.zzuegg.brainiac.experimental.ml.VectorQuantization;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 13 items 13.6.15 and 13.6.17: prototype learners.
 * <ul>
 *   <li>{@link #classIncremental} and {@link #stream}: Fuzzy ARTMAP ({@link AdaptiveResonance})
 *       trained in one pass, class-incrementally (the classes arrive in tasks) and on a drifting
 *       stream, against k-NN on everything seen ({@link KnnIndex}), an MLP fine-tuned task by task
 *       (forgetting), the same MLP with a replay buffer, and a SOM and a growing neural gas with a
 *       labelled read-out (trained offline on all data: the unsupervised prototype reference).</li>
 *   <li>{@link #lvq}: LVQ1, LVQ2.1, GLVQ, GMLVQ and localised GMLVQ ({@link VectorQuantization})
 *       against k-NN, the nearest class mean, a SOM read-out, logistic regression, an MLP, and k-NN
 *       after a linear map trained with the triplet loss ({@link PairLosses}), on tasks with
 *       irrelevant rescaled features, a {@link TabularStressSuite} task, toy digits and an
 *       X-shaped task whose classes differ only in their correlation (a single global metric cannot
 *       separate them).</li>
 * </ul>
 */
public final class PrototypeLearners {

    private PrototypeLearners() {
    }

    /**
     * Train and test data.
     *
     * @param name the name
     * @param trainX training inputs
     * @param trainY training classes
     * @param testX test inputs
     * @param testY test classes
     * @param classes classes
     * @param relevant which features carry signal (null when not planted)
     */
    public record Data(String name, double[][] trainX, int[] trainY, double[][] testX, int[] testY, int classes,
            boolean[] relevant) {
    }

    /** An incremental learner. */
    private interface Learner {
        void learn(double[][] x, int[] y);

        int[] predict(double[][] x);

        int memory();
    }

    // ------------------------------------------------------------------ data

    /**
     * Class-incremental source data: {@code "digits"} (10 classes of 8x8) or {@code "spirals"} (3 arms).
     *
     * @param name the task
     * @param seed the seed
     * @return the data, features in about {@code [0, 1]} or {@code [-1, 1]}
     */
    public static Data toy(String name, long seed) {
        if (name.equals("digits")) {
            ToyTasks.Data tr = ToyTasks.digits(10, 50, 8, 0.12, seed, seed * 31 + 1);
            ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.12, seed, seed * 31 + 2);
            return new Data(name, MlMatrices.toDouble(tr.x()), tr.y(), MlMatrices.toDouble(te.x()), te.y(), 10, null);
        }
        ToyTasks.Data d = ToyTasks.spiral(200, 3, 0.15, seed);
        double[][] x = MlMatrices.toDouble(d.x());
        int cut = 400;
        return new Data(name, java.util.Arrays.copyOfRange(x, 0, cut), java.util.Arrays.copyOfRange(d.y(), 0, cut),
                java.util.Arrays.copyOfRange(x, cut, x.length), java.util.Arrays.copyOfRange(d.y(), cut, x.length), 3,
                null);
    }

    /**
     * LVQ tasks: {@code "irrelevant"} (3 Gaussian classes in 2 features plus 8 irrelevant features
     * of standard deviation 2.5), {@code "stress"} (TabularStressSuite, 2 informative of 5),
     * {@code "digits"}, and {@code "correlated"} (two classes centred at the origin, one stretched
     * along {@code (1, 1)} and one along {@code (1, -1)} in features 0 and 1, plus 4 noise features).
     *
     * @param name the task
     * @param seed the seed
     * @return the data
     */
    public static Data lvqTask(String name, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        switch (name) {
            case "irrelevant" -> {
                int[] rows = {450, 450};
                double[][][] xs = new double[2][][];
                int[][] ys = new int[2][];
                for (int s = 0; s < 2; s++) {
                    xs[s] = new double[rows[s]][10];
                    ys[s] = new int[rows[s]];
                    for (int i = 0; i < rows[s]; i++) {
                        int c = i % 3;
                        double angle = 2 * Math.PI * c / 3;
                        xs[s][i][0] = 1.2 * Math.cos(angle) + 0.6 * r.nextGaussian();
                        xs[s][i][1] = 1.2 * Math.sin(angle) + 0.6 * r.nextGaussian();
                        for (int j = 2; j < 10; j++) {
                            xs[s][i][j] = 2.5 * r.nextGaussian();
                        }
                        ys[s][i] = c;
                    }
                }
                boolean[] rel = new boolean[10];
                rel[0] = true;
                rel[1] = true;
                return new Data(name, xs[0], ys[0], xs[1], ys[1], 3, rel);
            }
            case "stress" -> {
                TabularStressSuite suite = new TabularStressSuite(2, 0.6, false, seed);
                double[][][] xs = new double[2][][];
                int[][] ys = new int[2][];
                int[] rows = {1200, 400};
                for (int s = 0; s < 2; s++) {
                    SyntheticRecsys.Table t = suite.table(rows[s], seed * 13 + s);
                    xs[s] = MlMatrices.matrix(t.features());
                    float[] l = t.labels().toArray();
                    ys[s] = new int[l.length];
                    for (int i = 0; i < l.length; i++) {
                        ys[s][i] = l[i] > 0.5f ? 1 : 0;
                    }
                }
                boolean[] rel = new boolean[xs[0][0].length];
                rel[0] = true;
                rel[1] = true;
                return new Data(name, xs[0], ys[0], xs[1], ys[1], 2, rel);
            }
            case "digits" -> {
                Data d = toy("digits", seed);
                return d;
            }
            case "correlated" -> {
                int[] rows = {400, 400};
                double[][][] xs = new double[2][][];
                int[][] ys = new int[2][];
                double h = Math.sqrt(0.5);
                for (int s = 0; s < 2; s++) {
                    xs[s] = new double[rows[s]][6];
                    ys[s] = new int[rows[s]];
                    for (int i = 0; i < rows[s]; i++) {
                        int c = i % 2;
                        double t = 1.5 * r.nextGaussian();
                        double o = 0.25 * r.nextGaussian();
                        xs[s][i][0] = h * (t + o);
                        xs[s][i][1] = c == 0 ? h * (t - o) : -h * (t - o);
                        for (int j = 2; j < 6; j++) {
                            xs[s][i][j] = r.nextGaussian();
                        }
                        ys[s][i] = c;
                    }
                }
                boolean[] rel = new boolean[6];
                rel[0] = true;
                rel[1] = true;
                return new Data(name, xs[0], ys[0], xs[1], ys[1], 2, rel);
            }
            default -> throw new IllegalArgumentException(name);
        }
    }

    // ------------------------------------------------------------------ helpers

    private static double accuracy(int[] p, int[] y) {
        int hit = 0;
        for (int i = 0; i < p.length; i++) {
            hit += p[i] == y[i] ? 1 : 0;
        }
        return (double) hit / Math.max(1, p.length);
    }

    private static float[] f(double[] v) {
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = (float) v[i];
        }
        return out;
    }

    private static int[] knn(KnnIndex index, double[][] x, int k, int classes) {
        int[] out = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            KnnIndex.Hit[] hits = index.search(f(x[i]), k);
            double[] votes = new double[classes];
            for (KnnIndex.Hit h : hits) {
                votes[(int) index.value(h.row())[0]]++;
            }
            int best = 0;
            for (int c = 1; c < classes; c++) {
                best = votes[c] > votes[best] ? c : best;
            }
            out[i] = best;
        }
        return out;
    }

    private static KnnIndex index(double[][] x, int[] y) {
        KnnIndex index = KnnIndex.create(x[0].length, 1, KnnIndex.Metric.L2);
        for (int i = 0; i < x.length; i++) {
            index.add(f(x[i]), new float[] {y[i]});
        }
        return index;
    }

    /** Labels prototypes by the majority class of the training points they win, then classifies by the nearest. */
    private static int[] prototypeReadout(double[][] prototypes, double[][] trainX, int[] trainY, double[][] testX,
            int classes) {
        int p = prototypes.length;
        int[][] votes = new int[p][classes];
        for (int i = 0; i < trainX.length; i++) {
            votes[nearest(prototypes, trainX[i])][trainY[i]]++;
        }
        int[] label = new int[p];
        for (int k = 0; k < p; k++) {
            for (int c = 1; c < classes; c++) {
                label[k] = votes[k][c] > votes[k][label[k]] ? c : label[k];
            }
        }
        int[] out = new int[testX.length];
        for (int i = 0; i < testX.length; i++) {
            out[i] = label[nearest(prototypes, testX[i])];
        }
        return out;
    }

    private static int nearest(double[][] prototypes, double[] x) {
        int best = 0;
        double bd = Double.POSITIVE_INFINITY;
        for (int k = 0; k < prototypes.length; k++) {
            double d = MlMatrices.squaredDistance(prototypes[k], x);
            if (d < bd) {
                bd = d;
                best = k;
            }
        }
        return best;
    }

    /**
     * A small MLP (or, with {@code hidden == 0}, logistic regression) trained by Adam, with an
     * optional replay buffer.
     */
    private static final class Mlp implements Learner {

        private final Model model;
        private final Optimizer opt = Optimizer.adam(3e-3f);
        private final int steps;
        private final int replayPerClass;
        private final int classes;
        private final SplittableRandom random;
        private final List<double[]> bufferX = new ArrayList<>();
        private final List<Integer> bufferY = new ArrayList<>();
        private final int[] seen;

        Mlp(int inputs, int hidden, int classes, int steps, int replayPerClass, long seed) {
            this.model = hidden > 0
                    ? Model.sequential().input(inputs).dense(hidden, Activation.RELU).dense(classes).seed(seed).build()
                    : Model.sequential().input(inputs).dense(classes).seed(seed).build();
            this.steps = steps;
            this.replayPerClass = replayPerClass;
            this.classes = classes;
            this.random = new SplittableRandom(seed);
            this.seen = new int[classes];
        }

        @Override
        public void learn(double[][] x, int[] y) {
            int n = x.length;
            int b = Math.min(32, n);
            for (int s = 0; s < steps; s++) {
                int extra = bufferX.isEmpty() ? 0 : b;
                float[] xv = new float[(b + extra) * x[0].length];
                float[] yv = new float[b + extra];
                for (int i = 0; i < b + extra; i++) {
                    double[] row;
                    int label;
                    if (i < b) {
                        int j = random.nextInt(n);
                        row = x[j];
                        label = y[j];
                    } else {
                        int j = random.nextInt(bufferX.size());
                        row = bufferX.get(j);
                        label = bufferY.get(j);
                    }
                    for (int k = 0; k < row.length; k++) {
                        xv[i * row.length + k] = (float) row[k];
                    }
                    yv[i] = label;
                }
                Var o = model.forward(Var.of(Tensor.of(Shape.of(b + extra, x[0].length), xv)), true);
                Functional.crossEntropy(o, Tensor.of(Shape.of(b + extra), yv)).backward();
                opt.step(model.parameters());
            }
            if (replayPerClass > 0) {
                // reservoir sampling per class
                for (int i = 0; i < n; i++) {
                    int c = y[i];
                    seen[c]++;
                    int have = 0;
                    List<Integer> slots = new ArrayList<>();
                    for (int j = 0; j < bufferY.size(); j++) {
                        if (bufferY.get(j) == c) {
                            have++;
                            slots.add(j);
                        }
                    }
                    if (have < replayPerClass) {
                        bufferX.add(x[i]);
                        bufferY.add(c);
                    } else {
                        int r = random.nextInt(seen[c]);
                        if (r < replayPerClass) {
                            bufferX.set(slots.get(r), x[i]);
                        }
                    }
                }
            }
        }

        @Override
        public int[] predict(double[][] x) {
            float[] o = Var.noGrad(() -> model.forward(Var.of(MlMatrices.tensor(x)))).value().toArray();
            int[] out = new int[x.length];
            for (int i = 0; i < x.length; i++) {
                int best = 0;
                for (int c = 1; c < classes; c++) {
                    best = o[i * classes + c] > o[i * classes + best] ? c : best;
                }
                out[i] = best;
            }
            return out;
        }

        @Override
        public int memory() {
            return bufferX.size();
        }
    }

    private static Learner artmap(double vigilance, double[][] range, int inputs) {
        AdaptiveResonance art = new AdaptiveResonance(AdaptiveResonance.Mode.FUZZY_ARTMAP, inputs, vigilance, 0.001,
                1, 0.001);
        return new Learner() {
            @Override
            public void learn(double[][] x, int[] y) {
                for (int i = 0; i < x.length; i++) {
                    art.learn(AdaptiveResonance.scale(x[i], range), y[i]);
                }
            }

            @Override
            public int[] predict(double[][] x) {
                int[] out = new int[x.length];
                for (int i = 0; i < x.length; i++) {
                    out[i] = art.predict(AdaptiveResonance.scale(x[i], range));
                }
                return out;
            }

            @Override
            public int memory() {
                return art.categories();
            }
        };
    }

    private static Learner knnLearner(int inputs, int k, int classes) {
        KnnIndex index = KnnIndex.create(inputs, 1, KnnIndex.Metric.L2);
        return new Learner() {
            @Override
            public void learn(double[][] x, int[] y) {
                for (int i = 0; i < x.length; i++) {
                    index.add(f(x[i]), new float[] {y[i]});
                }
            }

            @Override
            public int[] predict(double[][] x) {
                return knn(index, x, k, classes);
            }

            @Override
            public int memory() {
                return index.size();
            }
        };
    }

    // ------------------------------------------------------------------ ART

    /**
     * Class-incremental training: the classes arrive in tasks of {@code perTask}, each seen once
     * (one pass for ARTMAP and k-NN; {@code mlpSteps} Adam steps per task for the MLPs).
     *
     * @param d the data
     * @param perTask classes per task
     * @param mlpSteps Adam steps per task
     * @param seed the seed
     * @return learner to {final accuracy on all classes, average forgetting, stored prototypes or
     *     samples}
     */
    public static Map<String, double[]> classIncremental(Data d, int perTask, int mlpSteps, long seed) {
        int inputs = d.trainX()[0].length;
        double[][] range = AdaptiveResonance.range(d.trainX());
        Map<String, Learner> learners = new LinkedHashMap<>();
        learners.put("Fuzzy ARTMAP (rho 0)", artmap(0, range, inputs));
        learners.put("Fuzzy ARTMAP (rho 0.6)", artmap(0.6, range, inputs));
        learners.put("Fuzzy ARTMAP (rho 0.8)", artmap(0.8, range, inputs));
        learners.put("k-NN (k 3, all samples)", knnLearner(inputs, 3, d.classes()));
        learners.put("MLP, sequential", new Mlp(inputs, 64, d.classes(), mlpSteps, 0, seed));
        learners.put("MLP + replay (20 per class)", new Mlp(inputs, 64, d.classes(), mlpSteps, 20, seed));
        int tasks = (d.classes() + perTask - 1) / perTask;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, Learner> e : learners.entrySet()) {
            Learner l = e.getValue();
            double[][] accAfter = new double[tasks][tasks];
            for (int t = 0; t < tasks; t++) {
                int lo = t * perTask;
                int hi = Math.min(d.classes(), lo + perTask);
                List<Integer> rows = new ArrayList<>();
                for (int i = 0; i < d.trainY().length; i++) {
                    if (d.trainY()[i] >= lo && d.trainY()[i] < hi) {
                        rows.add(i);
                    }
                }
                double[][] tx = new double[rows.size()][];
                int[] ty = new int[rows.size()];
                for (int i = 0; i < ty.length; i++) {
                    tx[i] = d.trainX()[rows.get(i)];
                    ty[i] = d.trainY()[rows.get(i)];
                }
                l.learn(tx, ty);
                int[] p = l.predict(d.testX());
                for (int s = 0; s <= t; s++) {
                    int a = 0;
                    int c = 0;
                    for (int i = 0; i < p.length; i++) {
                        int task = d.testY()[i] / perTask;
                        if (task == s) {
                            c++;
                            a += p[i] == d.testY()[i] ? 1 : 0;
                        }
                    }
                    accAfter[t][s] = (double) a / Math.max(1, c);
                }
            }
            double finalAcc = accuracy(l.predict(d.testX()), d.testY());
            double forgetting = 0;
            for (int s = 0; s < tasks - 1; s++) {
                double best = 0;
                for (int t = s; t < tasks; t++) {
                    best = Math.max(best, accAfter[t][s]);
                }
                forgetting += (best - accAfter[tasks - 1][s]) / (tasks - 1);
            }
            out.put(e.getKey(), new double[] {finalAcc, forgetting, l.memory()});
        }
        // offline unsupervised prototype learners with a labelled read-out
        SelfOrganizingMap som = SelfOrganizingMap.fit(d.trainX(), 6, 6, 20, seed);
        out.put("SOM 6x6 + read-out (offline)", new double[] {accuracy(prototypeReadout(som.prototypes(), d.trainX(),
            d.trainY(), d.testX(), d.classes()), d.testY()), Double.NaN, 36});
        GrowingNeuralGas gng = GrowingNeuralGas.fit(d.trainX(), 40, 20 * d.trainX().length, seed);
        out.put("GNG (40 units) + read-out (offline)", new double[] {accuracy(prototypeReadout(gng.prototypes(),
            d.trainX(), d.trainY(), d.testX(), d.classes()), d.testY()), Double.NaN, gng.prototypes().length});
        return out;
    }

    /**
     * Prequential (test-then-train) accuracy on a drifting RBF stream with one abrupt drift in the
     * middle.
     *
     * @param n samples
     * @param seed the seed
     * @return learner to {accuracy before the drift, accuracy in the last quarter, overall, memory}
     */
    public static Map<String, double[]> stream(int n, long seed) {
        DriftingStreams.Stream s = DriftingStreams.rbf(4, 12, 3, DriftingStreams.Drift.ABRUPT, new int[] {n / 2}, 0,
                seed);
        DriftingStreams.Sample[] data = DriftingStreams.take(s, n);
        double[][] range = {{-0.3, -0.3, -0.3, -0.3}, {1.3, 1.3, 1.3, 1.3}};
        Map<String, Learner> learners = new LinkedHashMap<>();
        learners.put("Fuzzy ARTMAP (rho 0)", artmap(0, range, 4));
        learners.put("Fuzzy ARTMAP (rho 0.8)", artmap(0.8, range, 4));
        learners.put("k-NN (k 3, all samples)", knnLearner(4, 3, 3));
        learners.put("MLP, online (1 step per sample)", new Mlp(4, 32, 3, 1, 0, seed));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, Learner> e : learners.entrySet()) {
            Learner l = e.getValue();
            int before = 0;
            int late = 0;
            int all = 0;
            for (int i = 0; i < n; i++) {
                double[][] x = {data[i].x()};
                int y = data[i].label();
                boolean hit = i > 0 && l.predict(x)[0] == y;
                all += hit ? 1 : 0;
                before += hit && i < n / 2 ? 1 : 0;
                late += hit && i >= 3 * n / 4 ? 1 : 0;
                l.learn(x, new int[] {y});
            }
            out.put(e.getKey(), new double[] {before / (n / 2.0), late / (n / 4.0), all / (double) n, l.memory()});
        }
        return out;
    }

    // ------------------------------------------------------------------ LVQ

    private static Data standardise(Data d) {
        int dim = d.trainX()[0].length;
        double[] mean = new double[dim];
        double[] sd = new double[dim];
        int n = d.trainX().length;
        for (double[] row : d.trainX()) {
            for (int j = 0; j < dim; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (double[] row : d.trainX()) {
            for (int j = 0; j < dim; j++) {
                sd[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        for (int j = 0; j < dim; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-8;
        }
        return new Data(d.name(), z(d.trainX(), mean, sd), d.trainY(), z(d.testX(), mean, sd), d.testY(), d.classes(),
                d.relevant());
    }

    private static double[][] z(double[][] x, double[] mean, double[] sd) {
        double[][] out = new double[x.length][mean.length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < mean.length; j++) {
                out[i][j] = (x[i][j] - mean[j]) / sd[j];
            }
        }
        return out;
    }

    private static double correlation(double[] a, boolean[] b) {
        int n = a.length;
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < n; i++) {
            ma += a[i] / n;
            mb += (b[i] ? 1 : 0) / (double) n;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < n; i++) {
            double x = a[i] - ma;
            double y = (b[i] ? 1 : 0) - mb;
            sab += x * y;
            saa += x * x;
            sbb += y * y;
        }
        return sab / Math.sqrt(Math.max(1e-300, saa * sbb));
    }

    /**
     * The LVQ table. Inputs are used raw (the irrelevant features keep their larger scale), so
     * Euclidean methods see the rescaled noise as a metric learner would before learning.
     *
     * @param raw the task
     * @param perClass prototypes per class of the LVQ variants
     * @param epochs LVQ epochs
     * @param seed the seed
     * @return method to {accuracy, prototypes or stored samples, correlation of the relevance
     *     profile with the planted features (NaN if none)}
     */
    public static Map<String, double[]> lvq(Data raw, int perClass, int epochs, long seed) {
        Data d = raw.name().equals("irrelevant") ? raw : standardise(raw);
        Map<String, double[]> out = new LinkedHashMap<>();
        int dim = d.trainX()[0].length;
        for (VectorQuantization.Kind kind : VectorQuantization.Kind.values()) {
            VectorQuantization m = VectorQuantization.fit(kind, d.trainX(), d.trainY(), d.classes(), perClass, epochs,
                    0.05, seed);
            double rel = d.relevant() == null || kind.ordinal() < VectorQuantization.Kind.GMLVQ.ordinal() ? Double.NaN
                    : correlation(m.relevanceProfile(), d.relevant());
            out.put(kind.toString(), new double[] {accuracy(m.predict(d.testX()), d.testY()),
                d.classes() * perClass, rel});
        }
        out.put("k-NN (k 5)", new double[] {accuracy(knn(index(d.trainX(), d.trainY()), d.testX(), 5, d.classes()),
            d.testY()), d.trainX().length, Double.NaN});
        double[][] means = new double[d.classes()][dim];
        int[] count = new int[d.classes()];
        for (int i = 0; i < d.trainY().length; i++) {
            count[d.trainY()[i]]++;
            for (int j = 0; j < dim; j++) {
                means[d.trainY()[i]][j] += d.trainX()[i][j];
            }
        }
        for (int c = 0; c < d.classes(); c++) {
            for (int j = 0; j < dim; j++) {
                means[c][j] /= Math.max(1, count[c]);
            }
        }
        int[] ncm = new int[d.testX().length];
        for (int i = 0; i < ncm.length; i++) {
            ncm[i] = nearest(means, d.testX()[i]);
        }
        out.put("nearest class mean", new double[] {accuracy(ncm, d.testY()), d.classes(), Double.NaN});
        SelfOrganizingMap som = SelfOrganizingMap.fit(d.trainX(), 5, 5, 20, seed);
        out.put("SOM 5x5 + read-out", new double[] {accuracy(prototypeReadout(som.prototypes(), d.trainX(),
            d.trainY(), d.testX(), d.classes()), d.testY()), 25, Double.NaN});
        Mlp logistic = new Mlp(dim, 0, d.classes(), 600, 0, seed);
        logistic.learn(d.trainX(), d.trainY());
        out.put("logistic regression", new double[] {accuracy(logistic.predict(d.testX()), d.testY()), Double.NaN,
            Double.NaN});
        Mlp mlp = new Mlp(dim, 64, d.classes(), 600, 0, seed);
        mlp.learn(d.trainX(), d.trainY());
        out.put("MLP", new double[] {accuracy(mlp.predict(d.testX()), d.testY()), Double.NaN, Double.NaN});
        double[][] map = tripletMap(d, 400, seed);
        double[][] tr = applyMap(d.trainX(), map);
        double[][] te = applyMap(d.testX(), map);
        out.put("triplet linear map + k-NN", new double[] {accuracy(knn(index(tr, d.trainY()), te, 5, d.classes()),
            d.testY()), d.trainX().length, Double.NaN});
        return out;
    }

    /** A {@code d x d} linear map trained with the semi-hard triplet loss (normalised embeddings). */
    private static double[][] tripletMap(Data d, int steps, long seed) {
        int dim = d.trainX()[0].length;
        float[] init = new float[dim * dim];
        for (int j = 0; j < dim; j++) {
            init[j * dim + j] = 1;
        }
        Var w = Var.param(Tensor.of(Shape.of(dim, dim), init));
        Optimizer opt = Optimizer.adam(1e-2f);
        Loss loss = PairLosses.triplet(0.2f, PairLosses.Mining.SEMI_HARD, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.trainX().length;
        int b = Math.min(64, n);
        for (int s = 0; s < steps; s++) {
            float[] xv = new float[b * dim];
            float[] yv = new float[b];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(n);
                for (int k = 0; k < dim; k++) {
                    xv[i * dim + k] = (float) d.trainX()[j][k];
                }
                yv[i] = d.trainY()[j];
            }
            loss.compute(Var.of(Tensor.of(Shape.of(b, dim), xv)).matmul(w), Tensor.of(Shape.of(b), yv)).backward();
            opt.step(List.of(w));
        }
        float[] v = w.value().toArray();
        double[][] out = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                out[i][j] = v[i * dim + j];
            }
        }
        return out;
    }

    private static double[][] applyMap(double[][] x, double[][] map) {
        int dim = map[0].length;
        double[][] out = new double[x.length][dim];
        for (int i = 0; i < x.length; i++) {
            double norm = 0;
            for (int j = 0; j < dim; j++) {
                double s = 0;
                for (int k = 0; k < x[i].length; k++) {
                    s += x[i][k] * map[k][j];
                }
                out[i][j] = s;
                norm += s * s;
            }
            norm = Math.sqrt(Math.max(1e-12, norm));
            for (int j = 0; j < dim; j++) {
                out[i][j] /= norm;
            }
        }
        return out;
    }
}
