package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelSvm;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.MultipleKernelLearning;
import io.github.zzuegg.brainiac.experimental.ml.NystromKernelRidge;
import io.github.zzuegg.brainiac.experimental.ml.RandomFourierFeatures;
import io.github.zzuegg.brainiac.experimental.ml.RandomWeightNetworks;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
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
 * Wave 13 items 13.6.13 and 13.6.14: randomised-weight networks against kernel machines, and
 * multiple kernel learning against a tuned single kernel.
 * <ul>
 *   <li>{@link #randomWeights}: ELM, RVFL, deep RVFL and ensemble deep RVFL
 *       ({@link RandomWeightNetworks}) against random Fourier features with a ridge read-out, an RBF
 *       {@link KernelSvm}, Nyström kernel ridge, an Adam-trained MLP and a linear ridge on the raw
 *       features, on a {@link TabularStressSuite} classification task, a Friedman regression,
 *       spirals and toy digits.</li>
 *   <li>{@link #mkl}: SimpleMKL, EasyMKL, ALIGNF and the uniform average
 *       ({@link MultipleKernelLearning}) against an RBF SVM with the bandwidth chosen by 3-fold
 *       cross-validation and a Gaussian process with ARD lengthscales fitted by marginal
 *       likelihood, on a task with one informative feature group out of four (one kernel per group)
 *       and a two-scale task (one kernel per bandwidth).</li>
 * </ul>
 */
public final class KernelLearners {

    private KernelLearners() {
    }

    /**
     * A standardised task.
     *
     * @param name the name
     * @param trainX training inputs
     * @param trainY training classes (empty for regression)
     * @param trainT training targets (regression; empty otherwise)
     * @param testX test inputs
     * @param testY test classes
     * @param testT test targets
     * @param classes classes, or 0 for regression
     */
    public record Task(String name, double[][] trainX, int[] trainY, double[] trainT, double[][] testX, int[] testY,
            double[] testT, int classes) {

        /** @return whether the task is a regression */
        public boolean regression() {
            return classes == 0;
        }
    }

    /**
     * Builds a task: {@code "stress"} (TabularStressSuite, 2 informative of 4 columns, 1200 rows),
     * {@code "friedman"} (Friedman #1, 10 inputs), {@code "spirals"} (3 arms) or {@code "digits"} (10 classes of 8x8).
     *
     * @param name the task
     * @param seed the seed
     * @return the task
     */
    public static Task task(String name, long seed) {
        switch (name) {
            case "stress" -> {
                TabularStressSuite suite = new TabularStressSuite(2, 0.5, false, seed);
                double[][][] xs = new double[2][][];
                int[][] ys = new int[2][];
                int[] rows = {1200, 400};
                for (int s = 0; s < 2; s++) {
                    SyntheticRecsys.Table t = suite.table(rows[s], seed * 7 + s);
                    xs[s] = MlMatrices.matrix(t.features());
                    float[] l = t.labels().toArray();
                    ys[s] = new int[l.length];
                    for (int i = 0; i < l.length; i++) {
                        ys[s][i] = l[i] > 0.5f ? 1 : 0;
                    }
                }
                return standardise(new Task(name, xs[0], ys[0], new double[0], xs[1], ys[1], new double[0], 2));
            }
            case "friedman" -> {
                double[][][] xs = new double[2][][];
                double[][] ts = new double[2][];
                int[] rows = {800, 400};
                SplittableRandom r = new SplittableRandom(seed);
                for (int s = 0; s < 2; s++) {
                    xs[s] = new double[rows[s]][10];
                    ts[s] = new double[rows[s]];
                    for (int i = 0; i < rows[s]; i++) {
                        double[] v = xs[s][i];
                        for (int j = 0; j < 10; j++) {
                            v[j] = r.nextDouble();
                        }
                        ts[s][i] = 10 * Math.sin(Math.PI * v[0] * v[1]) + 20 * (v[2] - 0.5) * (v[2] - 0.5) + 10 * v[3]
                                + 5 * v[4] + r.nextGaussian();
                    }
                }
                return standardise(new Task(name, xs[0], new int[0], ts[0], xs[1], new int[0], ts[1], 0));
            }
            case "spirals" -> {
                ToyTasks.Data d = ToyTasks.spiral(150, 3, 0.15, seed);
                return standardise(split(name, d, 300));
            }
            case "digits" -> {
                ToyTasks.Data tr = ToyTasks.digits(10, 50, 8, 0.12, seed, seed * 31 + 1);
                ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.12, seed, seed * 31 + 2);
                return standardise(new Task(name, MlMatrices.toDouble(tr.x()), tr.y(), new double[0],
                        MlMatrices.toDouble(te.x()), te.y(), new double[0], 10));
            }
            default -> throw new IllegalArgumentException(name);
        }
    }

    private static Task split(String name, ToyTasks.Data d, int train) {
        double[][] x = MlMatrices.toDouble(d.x());
        double[][] a = java.util.Arrays.copyOfRange(x, 0, train);
        double[][] b = java.util.Arrays.copyOfRange(x, train, x.length);
        return new Task(name, a, java.util.Arrays.copyOfRange(d.y(), 0, train), new double[0], b,
                java.util.Arrays.copyOfRange(d.y(), train, x.length), new double[0], d.classes());
    }

    private static Task standardise(Task t) {
        int d = t.trainX()[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        int n = t.trainX().length;
        for (double[] row : t.trainX()) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / n;
            }
        }
        for (double[] row : t.trainX()) {
            for (int j = 0; j < d; j++) {
                sd[j] += (row[j] - mean[j]) * (row[j] - mean[j]) / n;
            }
        }
        for (int j = 0; j < d; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-8;
        }
        return new Task(t.name(), z(t.trainX(), mean, sd), t.trainY(), t.trainT(), z(t.testX(), mean, sd), t.testY(),
                t.testT(), t.classes());
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

    private static double[][] targets(Task t) {
        int n = t.trainX().length;
        if (t.regression()) {
            double[][] y = new double[n][1];
            for (int i = 0; i < n; i++) {
                y[i][0] = t.trainT()[i];
            }
            return y;
        }
        double[][] y = new double[n][t.classes()];
        for (int i = 0; i < n; i++) {
            y[i][t.trainY()[i]] = 1;
        }
        return y;
    }

    /** Accuracy for classification, RMSE for regression, from {@code [n][k]} outputs. */
    private static double score(Task t, double[][] out) {
        int n = out.length;
        if (t.regression()) {
            double s = 0;
            for (int i = 0; i < n; i++) {
                double e = out[i][0] - t.testT()[i];
                s += e * e / n;
            }
            return Math.sqrt(s);
        }
        int hit = 0;
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int k = 1; k < out[i].length; k++) {
                best = out[i][k] > out[i][best] ? k : best;
            }
            hit += best == t.testY()[i] ? 1 : 0;
        }
        return (double) hit / n;
    }

    private static double accuracy(int[] p, int[] y) {
        int hit = 0;
        for (int i = 0; i < p.length; i++) {
            hit += p[i] == y[i] ? 1 : 0;
        }
        return (double) hit / p.length;
    }

    /**
     * The random-weight table.
     *
     * @param t the task
     * @param width the random width (hidden units, random features, Nyström landmarks, MLP width)
     * @param mlpSteps Adam steps of the MLP
     * @param seed the seed
     * @return method to {metric (accuracy, or RMSE for regression), training milliseconds}
     */
    public static Map<String, double[]> randomWeights(Task t, int width, int mlpSteps, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] y = targets(t);
        int d = t.trainX()[0].length;
        double gamma = 0.5 / d;
        for (RandomWeightNetworks.Kind kind : RandomWeightNetworks.Kind.values()) {
            long start = System.nanoTime();
            boolean deep = kind == RandomWeightNetworks.Kind.DEEP_RVFL
                    || kind == RandomWeightNetworks.Kind.ENSEMBLE_DEEP_RVFL;
            RandomWeightNetworks.Settings s = RandomWeightNetworks.Settings.of(kind, deep ? width / 2 : width,
                    deep ? 3 : 1, seed);
            RandomWeightNetworks net = RandomWeightNetworks.fit(t.trainX(), y, s);
            double ms = (System.nanoTime() - start) / 1e6;
            String name = switch (kind) {
                case ELM -> "ELM";
                case RVFL -> "RVFL";
                case DEEP_RVFL -> "deep RVFL (3 x " + width / 2 + ")";
                case ENSEMBLE_DEEP_RVFL -> "edRVFL (3 x " + width / 2 + ")";
            };
            out.put(name, new double[] {score(t, net.outputs(t.testX())), ms});
        }
        long start = System.nanoTime();
        RandomFourierFeatures rff = new RandomFourierFeatures(d, width, gamma, new SplittableRandom(seed));
        double[][] beta = RandomWeightNetworks.ridgeReadout(rff.transform(t.trainX()), y, 0, seed);
        double ms = (System.nanoTime() - start) / 1e6;
        double[][] rffOut = RandomWeightNetworks.applyReadout(rff.transform(t.testX()), beta);
        out.put("RFF + ridge", new double[] {score(t, rffOut), ms});
        start = System.nanoTime();
        double[][] lin = RandomWeightNetworks.ridgeReadout(t.trainX(), y, 0, seed);
        ms = (System.nanoTime() - start) / 1e6;
        out.put("linear ridge", new double[] {score(t, RandomWeightNetworks.applyReadout(t.testX(), lin)), ms});
        if (!t.regression()) {
            start = System.nanoTime();
            KernelSvm svm = KernelSvm.fit(t.trainX(), t.trainY(), t.classes(), KernelFunction.rbf(gamma), 10, 1e-3);
            ms = (System.nanoTime() - start) / 1e6;
            out.put("kernel SVM (RBF)", new double[] {accuracy(svm.predict(t.testX()), t.testY()), ms});
        }
        start = System.nanoTime();
        int k = y[0].length;
        double[][] ny = new double[t.testX().length][k];
        for (int c = 0; c < k; c++) {
            double[] col = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                col[i] = y[i][c];
            }
            double[] p = NystromKernelRidge.nystrom(t.trainX(), col, KernelFunction.rbf(gamma),
                    Math.min(width, y.length), 1e-4, seed).predict(t.testX());
            for (int i = 0; i < p.length; i++) {
                ny[i][c] = p[i];
            }
        }
        ms = (System.nanoTime() - start) / 1e6;
        out.put("Nystrom KRR", new double[] {score(t, ny), ms});
        start = System.nanoTime();
        double[][] mlp = mlp(t, y, width, mlpSteps, seed);
        ms = (System.nanoTime() - start) / 1e6;
        out.put("MLP (Adam)", new double[] {score(t, mlp), ms});
        return out;
    }

    private static double[][] mlp(Task t, double[][] y, int width, int steps, long seed) {
        int d = t.trainX()[0].length;
        int k = y[0].length;
        Model m = Model.sequential().input(d).dense(width, Activation.RELU).dense(k).seed(seed).build();
        Tensor x = MlMatrices.tensor(t.trainX());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int n = y.length;
        int b = Math.min(64, n);
        float[] yv = new float[n];
        float[] tv = new float[n * k];
        for (int i = 0; i < n; i++) {
            yv[i] = t.regression() ? (float) y[i][0] : t.trainY()[i];
            for (int c = 0; c < k; c++) {
                tv[i * k + c] = (float) y[i][c];
            }
        }
        double ym = 0;
        double ys = 0;
        if (t.regression()) {
            for (float v : yv) {
                ym += v / n;
            }
            for (float v : yv) {
                ys += (v - ym) * (v - ym) / n;
            }
            ys = Math.sqrt(ys);
            for (int i = 0; i < n; i++) {
                yv[i] = (float) ((yv[i] - ym) / ys);
            }
        }
        Tensor labels = Tensor.of(Shape.of(n), yv);
        for (int s = 0; s < steps; s++) {
            float[] idx = new float[b];
            for (int i = 0; i < b; i++) {
                idx[i] = r.nextInt(n);
            }
            Tensor sel = Tensor.of(Shape.of(b), idx);
            Var o = m.forward(Var.of(x.gather(0, sel)), true);
            Var loss = t.regression() ? o.reshape(b).sub(labels.gather(0, sel)).square().mean()
                    : Functional.crossEntropy(o, labels.gather(0, sel));
            loss.backward();
            opt.step(m.parameters());
        }
        float[] o = Var.noGrad(() -> m.forward(Var.of(MlMatrices.tensor(t.testX())))).value().toArray();
        double[][] out = new double[t.testX().length][k];
        for (int i = 0; i < out.length; i++) {
            for (int c = 0; c < k; c++) {
                out[i][c] = t.regression() ? o[i * k + c] * ys + ym : o[i * k + c];
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ MKL

    /**
     * A binary MKL task with its base kernels.
     *
     * @param task the data (binary)
     * @param kernels the base kernels
     * @param kernelNames one name per kernel
     * @param informative which kernels carry the signal
     */
    public record KernelTask(Task task, List<KernelFunction> kernels, List<String> kernelNames,
            boolean[] informative) {
    }

    /**
     * Four feature groups of 3 Gaussian columns, only the first informative: label
     * {@code sin(1.5 x0) + 0.8 x1 x2 > 0} with 5% flipped training labels; one RBF kernel per group.
     * (The step functions of {@link TabularStressSuite} left every kernel near 0.6 accuracy at 300
     * rows, too weak a signal to read kernel selection from.)
     *
     * @param train training rows
     * @param seed the seed
     * @return the task
     */
    public static KernelTask groups(int train, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] xs = new double[2][][];
        int[][] ys = new int[2][];
        int[] rows = {train, 400};
        for (int s = 0; s < 2; s++) {
            xs[s] = new double[rows[s]][12];
            ys[s] = new int[rows[s]];
            for (int i = 0; i < rows[s]; i++) {
                double[] v = xs[s][i];
                for (int j = 0; j < 12; j++) {
                    v[j] = r.nextGaussian();
                }
                boolean pos = Math.sin(1.5 * v[0]) + 0.8 * v[1] * v[2] > 0;
                ys[s][i] = (pos ^ (s == 0 && r.nextDouble() < 0.05)) ? 1 : 0;
            }
        }
        Task task = standardise(new Task("groups", xs[0], ys[0], new double[0], xs[1], ys[1], new double[0], 2));
        List<KernelFunction> kernels = new ArrayList<>();
        List<String> names = new ArrayList<>();
        int cols = xs[0][0].length;
        for (int g = 0; g * 3 < cols; g++) {
            int from = 3 * g;
            int to = Math.min(cols, from + 3);
            kernels.add(groupRbf(from, to, 1.0 / 3));
            names.add("group " + g + (g == 0 ? " (informative)" : ""));
        }
        boolean[] informative = new boolean[kernels.size()];
        informative[0] = true;
        return new KernelTask(task, kernels, names, informative);
    }

    private static KernelFunction groupRbf(int from, int to, double gamma) {
        return (a, b) -> {
            double s = 0;
            for (int j = from; j < to; j++) {
                double e = a[j] - b[j];
                s += e * e;
            }
            return Math.exp(-gamma * s);
        };
    }

    /**
     * A two-scale task in 2-D: label {@code sin(1.5 x0) + 0.8 sin(6 x1) > 0} on {@code [-2, 2]^2}
     * with 5% flipped labels; RBF kernels at 5 bandwidths.
     *
     * @param train training rows
     * @param seed the seed
     * @return the task
     */
    public static KernelTask multiScale(int train, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] rows = {train, 400};
        double[][][] xs = new double[2][][];
        int[][] ys = new int[2][];
        for (int s = 0; s < 2; s++) {
            xs[s] = new double[rows[s]][2];
            ys[s] = new int[rows[s]];
            for (int i = 0; i < rows[s]; i++) {
                xs[s][i][0] = -2 + 4 * r.nextDouble();
                xs[s][i][1] = -2 + 4 * r.nextDouble();
                boolean pos = Math.sin(1.5 * xs[s][i][0]) + 0.8 * Math.sin(6 * xs[s][i][1]) > 0;
                ys[s][i] = (pos ^ (s == 0 && r.nextDouble() < 0.05)) ? 1 : 0;
            }
        }
        Task task = standardise(new Task("multi-scale", xs[0], ys[0], new double[0], xs[1], ys[1], new double[0], 2));
        double[] gammas = {0.03, 0.1, 0.3, 1, 3, 10};
        List<KernelFunction> kernels = new ArrayList<>();
        List<String> names = new ArrayList<>();
        for (double g : gammas) {
            kernels.add(KernelFunction.rbf(g));
            names.add("rbf gamma " + g);
        }
        return new KernelTask(task, kernels, names, new boolean[gammas.length]);
    }

    /**
     * The MKL table.
     *
     * @param kt the task and its base kernels
     * @param c the SVM box constraint
     * @param seed the seed
     * @return method to {accuracy, training milliseconds, weight share on the informative kernels (NaN when
     *     not applicable), kernel weights...}
     */
    public static Map<String, double[]> mkl(KernelTask kt, double c, long seed) {
        Task t = kt.task();
        Map<String, double[]> out = new LinkedHashMap<>();
        int d = t.trainX()[0].length;
        // single RBF on all features, bandwidth by 3-fold cross-validation
        long start = System.nanoTime();
        double[] grid = {0.03, 0.1, 0.3, 1, 3, 10};
        double best = -1;
        double bestGamma = 1;
        for (double g : grid) {
            double gamma = g / d;
            double acc = crossValidate(t.trainX(), t.trainY(), KernelFunction.rbf(gamma), c, 3, seed);
            if (acc > best) {
                best = acc;
                bestGamma = gamma;
            }
        }
        KernelSvm single = KernelSvm.fit(t.trainX(), t.trainY(), 2, KernelFunction.rbf(bestGamma), c, 1e-3);
        double ms = (System.nanoTime() - start) / 1e6;
        out.put(String.format("single RBF (3-fold CV, gamma %.3g)", bestGamma),
                new double[] {accuracy(single.predict(t.testX()), t.testY()), ms, Double.NaN});
        for (MultipleKernelLearning.Method m : MultipleKernelLearning.Method.values()) {
            start = System.nanoTime();
            MultipleKernelLearning model = MultipleKernelLearning.fit(m, t.trainX(), t.trainY(), kt.kernels(), c, 0.1,
                    20);
            ms = (System.nanoTime() - start) / 1e6;
            double[] w = model.weights();
            double[] row = new double[3 + w.length];
            row[0] = accuracy(model.predict(t.testX()), t.testY());
            row[1] = ms;
            row[2] = informativeShare(w, kt.informative());
            System.arraycopy(w, 0, row, 3, w.length);
            String name = switch (m) {
                case UNIFORM -> "uniform average";
                case SIMPLE_MKL -> "SimpleMKL (" + model.svmSolves() + " SVM solves)";
                case EASY_MKL -> "EasyMKL";
                case ALIGNF -> "ALIGNF";
            };
            out.put(name, row);
        }
        start = System.nanoTime();
        ArdGp gp = ArdGp.fit(t.trainX(), t.trainY(), 30);
        ms = (System.nanoTime() - start) / 1e6;
        double[] rel = gp.relevance();
        double[] groupRel = new double[kt.kernels().size()];
        double[] row = new double[3 + groupRel.length];
        row[0] = accuracy(gp.predict(t.testX()), t.testY());
        row[1] = ms;
        if (kt.task().name().equals("groups")) {
            double total = 0;
            for (double v : rel) {
                total += v;
            }
            for (int j = 0; j < rel.length; j++) {
                groupRel[Math.min(groupRel.length - 1, j / 3)] += rel[j] / total;
            }
            row[2] = informativeShare(groupRel, kt.informative());
        } else {
            row[2] = Double.NaN;
        }
        System.arraycopy(groupRel, 0, row, 3, groupRel.length);
        out.put("ARD Gaussian process", row);
        return out;
    }

    private static double informativeShare(double[] w, boolean[] informative) {
        boolean any = false;
        double s = 0;
        double total = 0;
        for (int k = 0; k < w.length; k++) {
            total += w[k];
            if (informative[k]) {
                s += w[k];
                any = true;
            }
        }
        return any ? s / Math.max(1e-300, total) : Double.NaN;
    }

    private static double crossValidate(double[][] x, int[] y, KernelFunction k, double c, int folds, long seed) {
        int n = x.length;
        int[] fold = new int[n];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < n; i++) {
            fold[i] = r.nextInt(folds);
        }
        int hit = 0;
        for (int f = 0; f < folds; f++) {
            List<double[]> tx = new ArrayList<>();
            List<Integer> ty = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (fold[i] != f) {
                    tx.add(x[i]);
                    ty.add(y[i]);
                }
            }
            int[] labels = ty.stream().mapToInt(Integer::intValue).toArray();
            KernelSvm svm = KernelSvm.fit(tx.toArray(new double[0][]), labels, 2, k, c, 1e-3);
            for (int i = 0; i < n; i++) {
                if (fold[i] == f) {
                    hit += svm.predict(x[i]) == y[i] ? 1 : 0;
                }
            }
        }
        return (double) hit / n;
    }

    /**
     * A Gaussian process with ARD squared-exponential kernel on {@code +-1} labels (regression
     * likelihood), hyperparameters by Adam ascent of the log marginal likelihood
     * ({@code GaussianProcess} has one lengthscale only).
     */
    static final class ArdGp {

        private final double[][] x;
        private final double[] alpha;
        private final double[] lengthscale;
        private final double variance;

        private ArdGp(double[][] x, double[] alpha, double[] lengthscale, double variance) {
            this.x = x;
            this.alpha = alpha;
            this.lengthscale = lengthscale;
            this.variance = variance;
        }

        static ArdGp fit(double[][] x, int[] labels, int iterations) {
            int n = x.length;
            int d = x[0].length;
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = labels[i] == 1 ? 1 : -1;
            }
            int p = d + 2;
            double[] theta = new double[p];   // log lengthscales, log variance, log noise
            java.util.Arrays.fill(theta, 0, d, Math.log(Math.sqrt(d)));
            theta[d + 1] = Math.log(0.3);
            double[] m = new double[p];
            double[] v = new double[p];
            double[] alpha = null;
            for (int it = 1; it <= iterations + 1; it++) {
                double[][] kf = new double[n][n];
                for (int i = 0; i < n; i++) {
                    for (int j = i; j < n; j++) {
                        double s = 0;
                        for (int a = 0; a < d; a++) {
                            double e = (x[i][a] - x[j][a]) / Math.exp(theta[a]);
                            s += e * e;
                        }
                        kf[i][j] = Math.exp(theta[d]) * Math.exp(-0.5 * s);
                        kf[j][i] = kf[i][j];
                    }
                }
                double[][] k = new double[n][];
                for (int i = 0; i < n; i++) {
                    k[i] = kf[i].clone();
                    k[i][i] += Math.exp(theta[d + 1]) + 1e-6;
                }
                double[][] l = MlMatrices.cholesky(k);
                alpha = MlMatrices.choleskySolve(l, y);
                if (it > iterations) {
                    break;
                }
                double[][] inv = MlMatrices.choleskyInverse(l);
                // W = alpha alpha^T - K^-1; dL/dtheta = 1/2 tr(W dK/dtheta)
                double[] g = new double[p];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        double w = alpha[i] * alpha[j] - inv[i][j];
                        g[d] += 0.5 * w * kf[i][j];
                        if (i == j) {
                            g[d + 1] += 0.5 * w * Math.exp(theta[d + 1]);
                        }
                        if (j > i) {
                            for (int a = 0; a < d; a++) {
                                double e = (x[i][a] - x[j][a]) / Math.exp(theta[a]);
                                g[a] += w * kf[i][j] * e * e; // both triangles: 2 * 1/2
                            }
                        }
                    }
                }
                for (int q = 0; q < p; q++) {
                    m[q] = 0.9 * m[q] + 0.1 * g[q];
                    v[q] = 0.999 * v[q] + 0.001 * g[q] * g[q];
                    double mh = m[q] / (1 - Math.pow(0.9, it));
                    double vh = v[q] / (1 - Math.pow(0.999, it));
                    theta[q] += 0.1 * mh / (Math.sqrt(vh) + 1e-8);
                }
                theta[d + 1] = Math.max(theta[d + 1], Math.log(1e-3));
            }
            double[] ls = new double[d];
            for (int a = 0; a < d; a++) {
                ls[a] = Math.exp(theta[a]);
            }
            return new ArdGp(x, alpha, ls, Math.exp(theta[d]));
        }

        int[] predict(double[][] xs) {
            int[] out = new int[xs.length];
            for (int q = 0; q < xs.length; q++) {
                double f = 0;
                for (int i = 0; i < x.length; i++) {
                    double s = 0;
                    for (int a = 0; a < lengthscale.length; a++) {
                        double e = (x[i][a] - xs[q][a]) / lengthscale[a];
                        s += e * e;
                    }
                    f += alpha[i] * variance * Math.exp(-0.5 * s);
                }
                out[q] = f > 0 ? 1 : 0;
            }
            return out;
        }

        /** @return the inverse squared lengthscales */
        double[] relevance() {
            double[] out = new double[lengthscale.length];
            for (int a = 0; a < out.length; a++) {
                out[a] = 1 / (lengthscale[a] * lengthscale[a]);
            }
            return out;
        }
    }
}
