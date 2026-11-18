package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularPrior;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.nn.CrossNetwork;
import io.github.zzuegg.brainiac.experimental.nn.FeatureTokenizer;
import io.github.zzuegg.brainiac.experimental.nn.FtTransformer;
import io.github.zzuegg.brainiac.experimental.nn.NeuralAdditiveModel;
import io.github.zzuegg.brainiac.experimental.nn.ObliviousTreeEnsemble;
import io.github.zzuegg.brainiac.experimental.nn.PriorFittedNetwork;
import io.github.zzuegg.brainiac.experimental.nn.RealMlp;
import io.github.zzuegg.brainiac.experimental.nn.TabM;
import io.github.zzuegg.brainiac.experimental.nn.TabNetLite;
import io.github.zzuegg.brainiac.experimental.train.PopulationEnsemble;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Tabular deep learning against trees (W7 2.15 to 2.18), all binary classification through
 * {@link BinaryBench} at an equal step budget:
 * <ul>
 *   <li>{@link #shootout}: FT-Transformer, MLP with PLR embeddings, TabM, RealMLP, NODE and NAM
 *       against MLP, a 5-MLP deep ensemble, TabNet-lite, DCN-V2, gradient-boosted trees and a
 *       random forest on {@link SyntheticRecsys#tabular} (pairwise interactions);</li>
 *   <li>{@link #highFrequency}: a label {@code sin(6 x_0) + sin(6 x_1) > 0}, where periodic
 *       embeddings matter (PLR ablation);</li>
 *   <li>{@link #stress}: {@link TabularStressSuite} at 0, 50 and 90 % uninformative columns,
 *       with and without a random rotation, plus how well NAM recovers the planted shape
 *       functions;</li>
 *   <li>{@link #priorFitted}: a {@link PriorFittedNetwork} pre-trained on {@link TabularPrior}
 *       against per-task logistic regression, MLP, random forest, GBT and a GP classifier on
 *       held-out families with at most 200 rows.</li>
 * </ul>
 */
public final class TabularShootout {

    private TabularShootout() {
    }

    private static Model mlp(int cols, long seed) {
        return Model.sequential().input(cols).dense(32, Activation.RELU).dense(32, Activation.RELU).dense(1)
                .seed(seed).build();
    }

    private static BinaryBench.Result model(Model m, SyntheticRecsys.Table train, SyntheticRecsys.Table test,
            int steps, float lr, long seed) {
        return BinaryBench.train((x, t) -> m.forward(x, t), m.parameters(), train, test, steps, lr, seed);
    }

    private static BinaryBench.Result gbt(SyntheticRecsys.Table train, SyntheticRecsys.Table test, int trees) {
        long start = System.currentTimeMillis();
        double[] y = new double[train.rows()];
        int[] l = BinaryBench.labels(train);
        for (int i = 0; i < y.length; i++) {
            y[i] = l[i];
        }
        GradientBoostedTrees g = GradientBoostedTrees.fit(BinaryBench.features(train), y,
                GradientBoostedTrees.Objective.BINARY, GradientBoostedTrees.Settings.defaults().withTrees(trees)
                        .withDepth(4));
        return BinaryBench.scoreProbabilities(g.predict(BinaryBench.features(test)), test.labels().toArray(),
                System.currentTimeMillis() - start);
    }

    private static BinaryBench.Result forest(SyntheticRecsys.Table train, SyntheticRecsys.Table test, int trees) {
        long start = System.currentTimeMillis();
        RandomForest rf = RandomForest.fit(BinaryBench.features(train), BinaryBench.labels(train), 2,
                RandomForest.Settings.defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(trees));
        double[][] x = BinaryBench.features(test);
        double[] p = new double[x.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = rf.predictProbability(x[i])[1];
        }
        return BinaryBench.scoreProbabilities(p, test.labels().toArray(), System.currentTimeMillis() - start);
    }

    private static BinaryBench.Result node(SyntheticRecsys.Table train, SyntheticRecsys.Table test, int steps,
            long seed) {
        int cols = train.features().shape().dim(1);
        SplittableRandom r = new SplittableRandom(seed);
        ObliviousTreeEnsemble trees = new ObliviousTreeEnsemble(cols, 16, 4, 1, r);
        trees.initialize(train.features(), r);
        return BinaryBench.train((x, t) -> trees.forward(x, t), trees.parameters(), train, test, steps, 0.03f, seed);
    }

    private static BinaryBench.Result realMlp(SyntheticRecsys.Table train, SyntheticRecsys.Table test, int steps,
            long seed) {
        RealMlp m = new RealMlp(train.features(), 64, 2, 1, 8, new SplittableRandom(seed));
        return BinaryBench.train(x -> m.forward(x, true), m.parameters(), train, test, steps,
                s -> RealMlp.learningRate(s, steps, 0.01f), seed, x -> Var.noGrad(() -> m.forward(x, false)));
    }

    private static BinaryBench.Result plrMlp(SyntheticRecsys.Table train, SyntheticRecsys.Table test, int steps,
            float sigma, long seed) {
        int cols = train.features().shape().dim(1);
        SplittableRandom r = new SplittableRandom(seed);
        FeatureTokenizer plr = FeatureTokenizer.plr(cols, 8, 8, sigma, false, r);
        Model head = mlp(cols * 8, seed);
        List<Var> params = new ArrayList<>(plr.parameters());
        params.addAll(head.parameters());
        return BinaryBench.train((x, t) -> head.forward(plr.forward(x, t).reshape(x.shape().dim(0), cols * 8), t),
                params, train, test, steps, 0.01f, seed);
    }

    /**
     * The main comparison on the pairwise-interaction table.
     *
     * @param steps gradient steps of batch 64 per neural model
     * @param seed the seed
     * @return model name to test metrics
     */
    public static Map<String, BinaryBench.Result> shootout(int steps, long seed) {
        SyntheticRecsys.Table train = SyntheticRecsys.tabular(2000, 10, seed);
        SyntheticRecsys.Table test = SyntheticRecsys.tabular(1000, 10, seed + 1);
        int cols = 10;
        Map<String, BinaryBench.Result> out = new LinkedHashMap<>();
        out.put("MLP", model(mlp(cols, seed), train, test, steps, 0.01f, seed));
        List<Model> members = new ArrayList<>();
        long start = System.currentTimeMillis();
        for (int k = 0; k < 5; k++) {
            Model m = mlp(cols, seed + 100 + k);
            model(m, train, test, steps, 0.01f, seed + 100 + k);
            members.add(m);
        }
        List<Function<float[][], double[][]>> fns = new ArrayList<>();
        for (Model m : members) {
            fns.add(x -> probabilities(m, x));
        }
        double[][] mean = PopulationEnsemble.ofFunctions(fns).predict(rows(test.features())).mean();
        double[] p = new double[mean.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = mean[i][1];
        }
        out.put("deep ensemble (5 MLPs)", BinaryBench.scoreProbabilities(p, test.labels().toArray(),
                System.currentTimeMillis() - start));
        out.put("MLP + PLR", plrMlp(train, test, steps, 0.1f, seed));
        SplittableRandom r = new SplittableRandom(seed);
        FtTransformer ft = new FtTransformer(FeatureTokenizer.linear(cols, 16, r), 1, 2, 1, r);
        out.put("FT-Transformer", BinaryBench.train((x, t) -> ft.forward(x, t), ft.parameters(), train, test, steps,
                3e-3f, seed));
        TabM tabm = new TabM(cols, 64, 2, 8, 1, null, new SplittableRandom(seed));
        out.put("TabM (k=8)", BinaryBench.train(x -> {
            Var m = tabm.members(x, true);
            return m.reshape(m.shape().dim(0), tabm.size());
        }, tabm.parameters(), train, test, steps, 3e-3f, seed, x -> Var.noGrad(() -> tabm.forward(x, false))));
        out.put("RealMLP", realMlp(train, test, steps, seed));
        out.put("NODE", node(train, test, steps, seed));
        NeuralAdditiveModel nam = new NeuralAdditiveModel(cols, 16, new SplittableRandom(seed));
        out.put("NAM", BinaryBench.train((x, t) -> nam.forward(x, t), nam.parameters(), train, test, steps, 0.01f,
                seed));
        TabNetLite tab = new TabNetLite(cols, 1, 3, 16, 1.5f, new SplittableRandom(seed));
        out.put("TabNet-lite", BinaryBench.train((x, t) -> tab.forward(x, t), tab.parameters(), train, test, steps,
                0.01f, seed));
        CrossNetwork cross = new CrossNetwork(cols, 2, new SplittableRandom(seed));
        Model head = Model.sequential().input(cols).dense(1).seed(seed).build();
        List<Var> dcn = new ArrayList<>(cross.parameters());
        dcn.addAll(head.parameters());
        out.put("DCN-V2", BinaryBench.train((x, t) -> head.forward(cross.forward(x, t), t), dcn, train, test, steps,
                0.01f, seed));
        out.put("GBT", gbt(train, test, 150));
        out.put("random forest", forest(train, test, 30));
        return out;
    }

    /**
     * A high-frequency target: {@code sin(6 x_0) + sin(6 x_1) > 0} with 6 Gaussian columns.
     *
     * @param steps gradient steps per neural model
     * @param seed the seed
     * @return model name to test metrics
     */
    public static Map<String, BinaryBench.Result> highFrequency(int steps, long seed) {
        SyntheticRecsys.Table train = sine(3000, seed);
        SyntheticRecsys.Table test = sine(1500, seed + 1);
        Map<String, BinaryBench.Result> out = new LinkedHashMap<>();
        out.put("MLP", model(mlp(6, seed), train, test, steps, 0.01f, seed));
        out.put("MLP + PLR (sigma 0.1)", plrMlp(train, test, steps, 0.1f, seed));
        out.put("MLP + PLR (sigma 1)", plrMlp(train, test, steps, 1f, seed));
        out.put("RealMLP", realMlp(train, test, steps, seed));
        out.put("GBT", gbt(train, test, 150));
        return out;
    }

    private static SyntheticRecsys.Table sine(int rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[rows * 6];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < 6; c++) {
                x[i * 6 + c] = (float) r.nextGaussian();
            }
            y[i] = Math.sin(6 * x[i * 6]) + Math.sin(6 * x[i * 6 + 1]) > 0 ? 1 : 0;
        }
        return new SyntheticRecsys.Table(Tensor.of(Shape.of(rows, 6), x), Tensor.of(Shape.of(rows, 1), y));
    }

    /**
     * The trees-favouring stress suite.
     *
     * @param steps gradient steps per neural model
     * @param seed the seed
     * @param shapeCorrelation receives NAM's mean Pearson correlation between learned and planted
     *     shape functions on the plain (0 %, unrotated) setting
     * @return setting ("0% noise", "50% noise rotated", ...) to model name to test accuracy
     */
    public static Map<String, Map<String, Double>> stress(int steps, long seed, double[] shapeCorrelation) {
        Map<String, Map<String, Double>> out = new LinkedHashMap<>();
        for (boolean rotate : new boolean[] {false, true}) {
            for (double share : new double[] {0, 0.5, 0.9}) {
                TabularStressSuite suite = new TabularStressSuite(4, share, rotate, seed);
                SyntheticRecsys.Table train = suite.table(1500, seed + 1);
                SyntheticRecsys.Table test = suite.table(800, seed + 2);
                int cols = suite.columns();
                Map<String, Double> row = new LinkedHashMap<>();
                row.put("MLP", model(mlp(cols, seed), train, test, steps, 0.01f, seed).accuracy());
                row.put("GBT", gbt(train, test, 80).accuracy());
                row.put("random forest", forest(train, test, 25).accuracy());
                row.put("NODE", node(train, test, steps, seed).accuracy());
                NeuralAdditiveModel nam = new NeuralAdditiveModel(cols, 16, new SplittableRandom(seed));
                row.put("NAM", BinaryBench.train((x, t) -> nam.forward(x, t), nam.parameters(), train, test, steps,
                        0.01f, seed).accuracy());
                out.put(Math.round(share * 100) + "% noise" + (rotate ? " rotated" : ""), row);
                if (!rotate && share == 0) {
                    shapeCorrelation[0] = shapeCorrelation(nam, suite);
                }
            }
        }
        return out;
    }

    private static double shapeCorrelation(NeuralAdditiveModel nam, TabularStressSuite suite) {
        double[] grid = new double[81];
        for (int i = 0; i < grid.length; i++) {
            grid[i] = -2 + 4.0 * i / (grid.length - 1);
        }
        double total = 0;
        for (int j = 0; j < suite.informative(); j++) {
            double[] learned = nam.shape(j, grid);
            double[] planted = new double[grid.length];
            for (int i = 0; i < grid.length; i++) {
                planted[i] = suite.effect(j, grid[i]);
            }
            total += pearson(learned, planted);
        }
        return total / suite.informative();
    }

    private static double pearson(double[] a, double[] b) {
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < a.length; i++) {
            ma += a[i] / a.length;
            mb += b[i] / b.length;
        }
        double ab = 0;
        double aa = 0;
        double bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += (a[i] - ma) * (b[i] - mb);
            aa += (a[i] - ma) * (a[i] - ma);
            bb += (b[i] - mb) * (b[i] - mb);
        }
        return ab / Math.sqrt(aa * bb + 1e-12);
    }

    /**
     * In-context classification of small tables by a prior-fitted network.
     *
     * @param pretrainSteps pre-training steps (8 tasks of 64 rows each)
     * @param tasks evaluation tasks per family
     * @param seed the seed
     * @return evaluation set ("SCM/MLP prior", "trees (held out)", "interaction table") to method
     *     name to mean test accuracy
     */
    public static Map<String, Map<String, Double>> priorFitted(int pretrainSteps, int tasks, long seed) {
        int features = 5;
        PriorFittedNetwork pfn = new PriorFittedNetwork(features, 32, 4, 1, 3e-3f, new SplittableRandom(seed));
        long start = System.currentTimeMillis();
        pfn.pretrain(pretrainSteps, 8, 64, seed);
        long pretrainMillis = System.currentTimeMillis() - start;
        SplittableRandom r = new SplittableRandom(seed + 7);
        Map<String, List<TabularPrior.Task>> sets = new LinkedHashMap<>();
        List<TabularPrior.Task> prior = new ArrayList<>();
        List<TabularPrior.Task> trees = new ArrayList<>();
        List<TabularPrior.Task> table = new ArrayList<>();
        for (int t = 0; t < tasks; t++) {
            prior.add(TabularPrior.sample(200, features, r));
            trees.add(TabularPrior.sample(TabularPrior.Family.TREES, 200, features, r));
            SyntheticRecsys.Table sub = SyntheticRecsys.tabular(200, features, seed + 100 + t);
            table.add(new TabularPrior.Task(BinaryBench.features(sub), BinaryBench.labels(sub)));
        }
        sets.put("SCM/MLP prior", prior);
        sets.put("trees (held out)", trees);
        sets.put("interaction table", table);
        Map<String, Map<String, Double>> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<TabularPrior.Task>> e : sets.entrySet()) {
            Map<String, Double> row = new LinkedHashMap<>();
            for (TabularPrior.Task task : e.getValue()) {
                accumulate(row, task, 100, pfn, seed);
            }
            row.replaceAll((k, v) -> v / e.getValue().size());
            out.put(e.getKey(), row);
        }
        out.put("pre-training", Map.of("millis", (double) pretrainMillis));
        return out;
    }

    private static void accumulate(Map<String, Double> row, TabularPrior.Task task, int context,
            PriorFittedNetwork pfn, long seed) {
        int n = task.rows();
        double[][] trainX = java.util.Arrays.copyOfRange(task.x(), 0, context);
        int[] trainY = java.util.Arrays.copyOfRange(task.y(), 0, context);
        double[][] testX = java.util.Arrays.copyOfRange(task.x(), context, n);
        int[] testY = java.util.Arrays.copyOfRange(task.y(), context, n);
        row.merge("PFN (in context)", accuracy(pfn.predictProbability(trainX, trainY, testX), testY), Double::sum);
        SyntheticRecsys.Table train = table(trainX, trainY);
        SyntheticRecsys.Table test = table(testX, testY);
        int cols = trainX[0].length;
        row.merge("logistic", model(Model.sequential().input(cols).dense(1).seed(seed).build(), train, test, 200,
                0.05f, seed).accuracy(), Double::sum);
        row.merge("MLP", model(mlp(cols, seed), train, test, 200, 0.01f, seed).accuracy(), Double::sum);
        row.merge("random forest", forest(train, test, 50).accuracy(), Double::sum);
        row.merge("GBT", gbt(train, test, 50).accuracy(), Double::sum);
        GaussianProcess.Classifier gp = GaussianProcess.classifier(trainX, trainY,
                (a, b) -> 4 * Math.exp(-0.1 * MlMatrices.squaredDistance(a, b)));
        row.merge("GP classifier", accuracy(gp.predictProbability(testX), testY), Double::sum);
    }

    private static double accuracy(double[] p, int[] y) {
        int right = 0;
        for (int i = 0; i < p.length; i++) {
            right += (p[i] > 0.5 ? 1 : 0) == y[i] ? 1 : 0;
        }
        return right / (double) p.length;
    }

    private static SyntheticRecsys.Table table(double[][] x, int[] y) {
        int rows = x.length;
        int cols = x[0].length;
        float[] f = new float[rows * cols];
        float[] l = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                f[i * cols + j] = (float) x[i][j];
            }
            l[i] = y[i];
        }
        return new SyntheticRecsys.Table(Tensor.of(Shape.of(rows, cols), f), Tensor.of(Shape.of(rows, 1), l));
    }

    private static float[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int c = t.shape().dim(1);
        float[] a = t.toArray();
        float[][] out = new float[n][c];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * c, out[i], 0, c);
        }
        return out;
    }

    private static double[][] probabilities(Model m, float[][] x) {
        int n = x.length;
        int c = x[0].length;
        float[] flat = new float[n * c];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x[i], 0, flat, i * c, c);
        }
        float[] l = Var.noGrad(() -> m.forward(Var.of(Tensor.of(Shape.of(n, c), flat)), false)).value().toArray();
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            double p = 1 / (1 + Math.exp(-l[i]));
            out[i] = new double[] {1 - p, p};
        }
        return out;
    }
}
