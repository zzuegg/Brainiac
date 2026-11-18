package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.experimental.boost.BoostLosses;
import io.github.zzuegg.brainiac.experimental.boost.ExplainableBoosting;
import io.github.zzuegg.brainiac.experimental.boost.LightBoost;
import io.github.zzuegg.brainiac.experimental.boost.NaturalGradientBoosting;
import io.github.zzuegg.brainiac.experimental.boost.OrderedBoosting;
import io.github.zzuegg.brainiac.experimental.boost.PriorBoost;
import io.github.zzuegg.brainiac.experimental.boost.QuantileForest;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularPrior;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;
import io.github.zzuegg.brainiac.experimental.nn.NeuralAdditiveModel;
import io.github.zzuegg.brainiac.experimental.nn.PriorFittedNetwork;
import io.github.zzuegg.brainiac.experiments.recsys.BinaryBench;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Boosting variants against core {@link GradientBoostedTrees} and the repo's forests (wave 10,
 * items 10.5.1 to 10.5.6):
 * <ul>
 *   <li>{@link #lightGbm}: LightGBM's GOSS and leaf-wise growth ({@link LightBoost}) on
 *       {@link TabularStressSuite} tables, and exclusive feature bundling on a one-hot-heavy
 *       sparse table;</li>
 *   <li>{@link #categorical}: CatBoost's ordered target statistics and ordered boosting
 *       ({@link OrderedBoosting}) on high-cardinality Zipf categories with a planted interaction,
 *       against one-hot, leaky and out-of-fold target encodings;</li>
 *   <li>{@link #probabilistic}: NGBoost ({@link NaturalGradientBoosting}) and quantile regression
 *       forests ({@link QuantileForest}) on heteroscedastic and bimodal regression, against
 *       quantile GBT and a forest with a Gaussian from its tree spread;</li>
 *   <li>{@link #interpretable}: an explainable boosting machine ({@link ExplainableBoosting})
 *       against GBT, logistic regression and the neural additive model, with shape-function and
 *       planted-pair recovery;</li>
 *   <li>{@link #priorBoost}: PFN-Boost ({@link PriorBoost}) on {@link TabularPrior} tables of
 *       growing size, against the PFN alone, GBT alone and stacking.</li>
 * </ul>
 */
public final class BoostingShootout {

    private BoostingShootout() {
    }

    /** Features and 0/1 labels. */
    record Data(double[][] x, double[] y) {
    }

    private static Data data(SyntheticRecsys.Table t) {
        double[][] x = BinaryBench.features(t);
        int[] l = BinaryBench.labels(t);
        return new Data(x, Arrays.stream(l).asDoubleStream().toArray());
    }

    private static float[] floats(double[] y) {
        float[] f = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            f[i] = (float) y[i];
        }
        return f;
    }

    private static double[] score(Data test, java.util.function.ToDoubleFunction<double[]> p, long millis) {
        double[] prob = new double[test.y().length];
        for (int i = 0; i < prob.length; i++) {
            prob[i] = p.applyAsDouble(test.x()[i]);
        }
        BinaryBench.Result r = BinaryBench.scoreProbabilities(prob, floats(test.y()), millis);
        return new double[] {r.accuracy(), r.auc(), r.logLoss(), millis};
    }

    /**
     * GOSS, leaf-wise growth and EFB (10.5.1).
     *
     * @param rows training rows (test rows are half as many)
     * @param trees boosting rounds
     * @param seed the seed
     * @return model to {@code {accuracy, AUC, log-loss, millis, leaves or bundles}}; rows prefixed
     *     {@code "sparse "} are on the one-hot table, where the last column is the number of
     *     histogram columns
     */
    public static Map<String, double[]> lightGbm(int rows, int trees, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        TabularStressSuite suite = new TabularStressSuite(6, 0.5, false, seed);
        Data train = data(suite.table(rows, seed + 1));
        Data test = data(suite.table(rows / 2, seed + 2));
        long t0 = System.currentTimeMillis();
        GradientBoostedTrees core = GradientBoostedTrees.fit(train.x(), train.y(), GradientBoostedTrees.Objective.BINARY,
                GradientBoostedTrees.Settings.defaults().withTrees(trees).withDepth(5));
        out.put("core GBT (level-wise, depth 5)", append(score(test, core::predict, System.currentTimeMillis() - t0),
                trees * 32));
        LightBoost.Settings base = LightBoost.Settings.defaults().withTrees(trees).withLeaves(32, 0);
        light(out, "level-wise (depth 5)", train, test, base.withLeafWise(false).withLeaves(32, 5));
        light(out, "leaf-wise (32 leaves)", train, test, base);
        light(out, "leaf-wise (8 leaves)", train, test, base.withLeaves(8, 0));
        light(out, "leaf-wise + GOSS (a=0.2, b=0.1)", train, test, base.withGoss(0.2, 0.1));
        light(out, "level-wise + GOSS (a=0.2, b=0.1)", train, test, base.withLeafWise(false).withLeaves(32, 5)
                .withGoss(0.2, 0.1));
        t0 = System.currentTimeMillis();
        int[] labels = Arrays.stream(train.y()).mapToInt(v -> (int) v).toArray();
        RandomForest rf = RandomForest.fit(train.x(), labels, 2,
                RandomForest.Settings.defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(50));
        out.put("random forest (50 trees)", append(score(test, x -> rf.predictProbability(x)[1],
                System.currentTimeMillis() - t0), 0));
        // a one-hot-heavy sparse table: 6 categorical columns of 40 levels, one-hot encoded
        Data[] sparse = oneHot(rows, 6, 40, seed);
        LightBoost.Settings sb = base.withLeaves(16, 0);
        light(out, "sparse no bundling", sparse[0], sparse[1], sb);
        light(out, "sparse EFB (exclusive bundles)", sparse[0], sparse[1], sb.withBundling(true, 0));
        light(out, "sparse EFB + GOSS", sparse[0], sparse[1], sb.withBundling(true, 0).withGoss(0.2, 0.1));
        t0 = System.currentTimeMillis();
        GradientBoostedTrees sparseCore = GradientBoostedTrees.fit(sparse[0].x(), sparse[0].y(),
                GradientBoostedTrees.Objective.BINARY, GradientBoostedTrees.Settings.defaults().withTrees(trees));
        out.put("sparse core GBT", append(score(sparse[1], sparseCore::predict, System.currentTimeMillis() - t0),
                sparse[0].x()[0].length));
        return out;
    }

    private static void light(Map<String, double[]> out, String name, Data train, Data test, LightBoost.Settings s) {
        long t0 = System.currentTimeMillis();
        LightBoost b = LightBoost.fit(train.x(), train.y(), BoostLosses.logistic(), s);
        long ms = System.currentTimeMillis() - t0;
        out.put(name, append(score(test, b::predict, ms), name.startsWith("sparse") ? b.bundles() : b.leaves()));
    }

    private static double[] append(double[] a, double v) {
        double[] r = Arrays.copyOf(a, a.length + 1);
        r[a.length] = v;
        return r;
    }

    /** One-hot columns of Zipf categories whose levels carry additive effects. */
    private static Data[] oneHot(int rows, int columns, int levels, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 77);
        double[][] effect = new double[columns][levels];
        for (double[] e : effect) {
            for (int l = 0; l < levels; l++) {
                e[l] = r.nextGaussian();
            }
        }
        Data[] out = new Data[2];
        for (int part = 0; part < 2; part++) {
            int n = part == 0 ? rows : rows / 2;
            double[][] x = new double[n][columns * levels];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                for (int c = 0; c < columns; c++) {
                    int l = zipf(levels, r);
                    x[i][c * levels + l] = 1;
                    s += effect[c][l];
                }
                y[i] = s + 0.5 * r.nextGaussian() > 0 ? 1 : 0;
            }
            out[part] = new Data(x, y);
        }
        return out;
    }

    private static int zipf(int levels, SplittableRandom r) {
        double h = 0;
        for (int k = 1; k <= levels; k++) {
            h += 1.0 / k;
        }
        double u = r.nextDouble() * h;
        for (int k = 1; k <= levels; k++) {
            u -= 1.0 / k;
            if (u <= 0) {
                return k - 1;
            }
        }
        return levels - 1;
    }

    /** A table with two high-cardinality categoricals, a planted pair effect and two numeric columns. */
    record Categorical(double[][] numeric, int[][] cats, double[] y) {
    }

    static Categorical[] categoricalData(int rows, int levels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] effect = new double[2][levels];
        for (double[] e : effect) {
            for (int l = 0; l < levels; l++) {
                e[l] = 0.8 * r.nextGaussian();
            }
        }
        Categorical[] out = new Categorical[2];
        for (int part = 0; part < 2; part++) {
            int n = rows;
            double[][] num = new double[n][2];
            int[][] cats = new int[n][2];
            double[] y = new double[n];
            for (int i = 0; i < n; i++) {
                num[i][0] = r.nextDouble() * 4 - 2;
                num[i][1] = r.nextDouble() * 4 - 2;
                cats[i][0] = zipf(levels, r);
                cats[i][1] = zipf(levels, r);
                // a planted interaction between the two categories: the sign depends on both levels
                double inter = cats[i][0] % 2 == cats[i][1] % 2 ? 0.8 : -0.8;
                double s = effect[0][cats[i][0]] + effect[1][cats[i][1]] + inter + Math.sin(2 * num[i][0]);
                y[i] = s + 0.5 * r.nextGaussian() > 0 ? 1 : 0;
            }
            out[part] = new Categorical(num, cats, y);
        }
        return out;
    }

    /**
     * Ordered target statistics and ordered boosting (10.5.2).
     *
     * @param rows training rows (as many test rows)
     * @param levels levels per categorical column (Zipf frequencies)
     * @param trees boosting rounds
     * @param seed the seed
     * @return model to {@code {test log-loss, test AUC, train log-loss, train-test log-loss gap}}
     */
    public static Map<String, double[]> categorical(int rows, int levels, int trees, long seed) {
        Categorical[] d = categoricalData(rows, levels, seed);
        Categorical train = d[0];
        Categorical test = d[1];
        Map<String, double[]> out = new LinkedHashMap<>();
        OrderedBoosting.Settings os = OrderedBoosting.Settings.defaults().withTrees(trees);
        for (OrderedBoosting.Mode mode : OrderedBoosting.Mode.values()) {
            OrderedBoosting m = OrderedBoosting.fit(train.numeric(), train.cats(), train.y(), BoostLosses.logistic(),
                    os.withMode(mode));
            out.put("CatBoost-style " + mode.name().toLowerCase(java.util.Locale.ROOT) + " boosting",
                    gap(i -> m.predict(train.numeric()[i], train.cats()[i]), i -> m.predict(test.numeric()[i],
                            test.cats()[i]), train.y(), test.y()));
        }
        OrderedBoosting comb = OrderedBoosting.fit(train.numeric(), train.cats(), train.y(), BoostLosses.logistic(),
                os.withCombinations(true));
        out.put("CatBoost-style ordered + pair combinations", gap(i -> comb.predict(train.numeric()[i],
                train.cats()[i]), i -> comb.predict(test.numeric()[i], test.cats()[i]), train.y(), test.y()));
        GradientBoostedTrees.Settings gs = GradientBoostedTrees.Settings.defaults().withTrees(trees).withDepth(4);
        double[][] leakyTrain = join(train.numeric(), OrderedBoosting.targetEncode(train.cats(), train.y(),
                train.cats(), 1));
        double[][] leakyTest = join(test.numeric(), OrderedBoosting.targetEncode(train.cats(), train.y(), test.cats(),
                1));
        GradientBoostedTrees leaky = GradientBoostedTrees.fit(leakyTrain, train.y(),
                GradientBoostedTrees.Objective.BINARY, gs);
        out.put("GBT + leaky target encoding", gap(i -> leaky.predict(leakyTrain[i]), i -> leaky.predict(leakyTest[i]),
                train.y(), test.y()));
        double[][] oofTrain = join(train.numeric(), OrderedBoosting.outOfFoldEncode(train.cats(), train.y(), 5, 1,
                seed));
        GradientBoostedTrees oof = GradientBoostedTrees.fit(oofTrain, train.y(), GradientBoostedTrees.Objective.BINARY,
                gs);
        out.put("GBT + out-of-fold target encoding", gap(i -> oof.predict(oofTrain[i]), i -> oof.predict(leakyTest[i]),
                train.y(), test.y()));
        double[][] ohTrain = join(train.numeric(), oneHotCats(train.cats(), levels));
        double[][] ohTest = join(test.numeric(), oneHotCats(test.cats(), levels));
        GradientBoostedTrees oh = GradientBoostedTrees.fit(ohTrain, train.y(), GradientBoostedTrees.Objective.BINARY,
                gs);
        out.put("GBT + one-hot", gap(i -> oh.predict(ohTrain[i]), i -> oh.predict(ohTest[i]), train.y(), test.y()));
        int[] labels = Arrays.stream(train.y()).mapToInt(v -> (int) v).toArray();
        RandomForest rf = RandomForest.fit(oofTrain, labels, 2,
                RandomForest.Settings.defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(50));
        out.put("random forest + out-of-fold encoding", gap(i -> rf.predictProbability(oofTrain[i])[1],
                i -> rf.predictProbability(leakyTest[i])[1], train.y(), test.y()));
        return out;
    }

    private static double[] gap(java.util.function.IntToDoubleFunction trainP,
            java.util.function.IntToDoubleFunction testP, double[] trainY, double[] testY) {
        double trainLl = logLoss(trainP, trainY);
        double testLl = logLoss(testP, testY);
        double[] p = new double[testY.length];
        boolean[] pos = new boolean[testY.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = testP.applyAsDouble(i);
            pos[i] = testY[i] > 0.5;
        }
        return new double[] {testLl, io.github.zzuegg.brainiac.experimental.ml.MlMetrics.auroc(p, pos), trainLl,
            testLl - trainLl};
    }

    private static double logLoss(java.util.function.IntToDoubleFunction p, double[] y) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double q = Math.clamp(p.applyAsDouble(i), 1e-7, 1 - 1e-7);
            s -= y[i] > 0.5 ? Math.log(q) : Math.log(1 - q);
        }
        return s / y.length;
    }

    private static double[][] join(double[][] a, double[][] b) {
        double[][] out = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            out[i] = new double[a[i].length + b[i].length];
            System.arraycopy(a[i], 0, out[i], 0, a[i].length);
            System.arraycopy(b[i], 0, out[i], a[i].length, b[i].length);
        }
        return out;
    }

    private static double[][] oneHotCats(int[][] cats, int levels) {
        double[][] out = new double[cats.length][cats[0].length * levels];
        for (int i = 0; i < cats.length; i++) {
            for (int c = 0; c < cats[i].length; c++) {
                out[i][c * levels + cats[i][c]] = 1;
            }
        }
        return out;
    }

    /** Heteroscedastic (or bimodal) regression: {@code y = f(x) + sigma(x) * noise}. */
    static Data regression(int rows, boolean bimodal, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[rows][4];
        double[] y = new double[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < 4; j++) {
                x[i][j] = r.nextDouble() * 4 - 2;
            }
            double f = Math.sin(1.5 * x[i][0]) + 0.5 * x[i][1];
            double sigma = 0.1 + 0.4 * Math.abs(x[i][2]);
            double noise = bimodal ? (r.nextBoolean() ? 1.5 : -1.5) + 0.3 * r.nextGaussian() : r.nextGaussian();
            y[i] = f + sigma * noise;
        }
        return new Data(x, y);
    }

    /**
     * Probabilistic regression (10.5.3 and 10.5.6).
     *
     * @param rows training rows (test rows: as many; calibration rows: a quarter)
     * @param bimodal whether the noise is a two-component mixture
     * @param stages NGBoost stages
     * @param seed the seed
     * @return model to {@code {NLL (NaN if none), CRPS, mean pinball at 5/50/95%, 90% coverage,
     *     mean 90% width, RMSE}}
     */
    public static Map<String, double[]> probabilistic(int rows, boolean bimodal, int stages, long seed) {
        Data train = regression(rows, bimodal, seed);
        Data test = regression(rows, bimodal, seed + 1);
        Data cal = regression(rows / 4, bimodal, seed + 2);
        int n = test.y().length;
        Map<String, double[]> out = new LinkedHashMap<>();
        NaturalGradientBoosting.Settings ns = NaturalGradientBoosting.Settings.defaults().withStages(stages);
        record Variant(String name, NaturalGradientBoosting.Family family, NaturalGradientBoosting.Score score,
                boolean natural) {
        }
        for (Variant v : new Variant[] {
            new Variant("NGBoost Normal, log score", NaturalGradientBoosting.Family.NORMAL,
                    NaturalGradientBoosting.Score.LOG, true),
            new Variant("NGBoost Normal, CRPS", NaturalGradientBoosting.Family.NORMAL,
                    NaturalGradientBoosting.Score.CRPS, true),
            new Variant("NGBoost Laplace, log score", NaturalGradientBoosting.Family.LAPLACE,
                    NaturalGradientBoosting.Score.LOG, true),
            new Variant("NGBoost Normal, ordinary gradient", NaturalGradientBoosting.Family.NORMAL,
                    NaturalGradientBoosting.Score.LOG, false)}) {
            NaturalGradientBoosting m = NaturalGradientBoosting.fit(train.x(), train.y(), v.family(), v.score(),
                    ns.withNatural(v.natural()));
            double[][] q = new double[3][n];
            double nll = 0;
            double crps = 0;
            double se = 0;
            for (int i = 0; i < n; i++) {
                NaturalGradientBoosting.Distribution d = m.predict(test.x()[i]);
                q[0][i] = d.quantile(0.05);
                q[1][i] = d.quantile(0.5);
                q[2][i] = d.quantile(0.95);
                nll += d.nll(test.y()[i]) / n;
                crps += d.crps(test.y()[i]) / n;
                se += Math.pow(d.mean() - test.y()[i], 2) / n;
            }
            out.put(v.name(), row(nll, crps, q, test.y(), Math.sqrt(se)));
        }
        RegressionForest.Settings fs = RegressionForest.Settings.defaults().withTrees(60).withMinLeaf(10);
        QuantileForest qrf = QuantileForest.fit(train.x(), train.y(), fs);
        double[][] q = new double[3][n];
        double crps = 0;
        double se = 0;
        for (int i = 0; i < n; i++) {
            q[0][i] = qrf.quantile(test.x()[i], 0.05);
            q[1][i] = qrf.quantile(test.x()[i], 0.5);
            q[2][i] = qrf.quantile(test.x()[i], 0.95);
            crps += qrf.crps(test.x()[i], test.y()[i]) / n;
            se += Math.pow(qrf.mean(test.x()[i]) - test.y()[i], 2) / n;
        }
        out.put("quantile regression forest", row(Double.NaN, crps, q, test.y(), Math.sqrt(se)));
        double c = qrf.conformalize(cal.x(), cal.y(), 0.1);
        double[][] qc = {q[0].clone(), q[1].clone(), q[2].clone()};
        for (int i = 0; i < n; i++) {
            qc[0][i] -= c;
            qc[2][i] += c;
        }
        out.put("quantile forest + CQR", row(Double.NaN, crps, qc, test.y(), Math.sqrt(se)));
        // quantile GBT: three core boosters on the pinball loss
        double[] taus = {0.05, 0.5, 0.95};
        double[][] gq = new double[3][];
        for (int k = 0; k < 3; k++) {
            GradientBoostedTrees g = GradientBoostedTrees.fit(train.x(), train.y(), BoostLosses.quantile(taus[k]),
                    GradientBoostedTrees.Settings.defaults().withTrees(200).withDepth(3));
            gq[k] = g.predict(test.x());
        }
        double gse = 0;
        for (int i = 0; i < n; i++) {
            gse += Math.pow(gq[1][i] - test.y()[i], 2) / n;
        }
        out.put("quantile GBT (3 models)", row(Double.NaN, Double.NaN, gq, test.y(), Math.sqrt(gse)));
        // a random forest with a Gaussian from the spread of its trees
        RegressionForest rf = RegressionForest.fit(train.x(), train.y(), null, fs);
        double[] mean = new double[n];
        double[] var = new double[n];
        for (int i = 0; i < n; i++) {
            RegressionForest.Tree[] trees = rf.trees();
            double s = 0;
            double s2 = 0;
            for (RegressionForest.Tree t : trees) {
                double v = t.predict(test.x()[i]);
                s += v;
                s2 += v * v;
            }
            mean[i] = s / trees.length;
            var[i] = Math.max(s2 / trees.length - mean[i] * mean[i], 1e-4);
        }
        double[][] rq = new double[3][n];
        for (int i = 0; i < n; i++) {
            double sd = Math.sqrt(var[i]);
            rq[0][i] = mean[i] - 1.645 * sd;
            rq[1][i] = mean[i];
            rq[2][i] = mean[i] + 1.645 * sd;
        }
        double rse = 0;
        for (int i = 0; i < n; i++) {
            rse += Math.pow(mean[i] - test.y()[i], 2) / n;
        }
        out.put("forest Gaussian (tree spread)", row(
                io.github.zzuegg.brainiac.experimental.ml.MlMetrics.gaussianNll(mean, var, test.y()),
                io.github.zzuegg.brainiac.experimental.ml.MlMetrics.gaussianCrps(mean, var, test.y()), rq, test.y(),
                Math.sqrt(rse)));
        return out;
    }

    private static double[] row(double nll, double crps, double[][] q, double[] y, double rmse) {
        double pin = (BoostLosses.pinball(y, q[0], 0.05) + BoostLosses.pinball(y, q[1], 0.5)
                + BoostLosses.pinball(y, q[2], 0.95)) / 3;
        double cover = 0;
        double width = 0;
        for (int i = 0; i < y.length; i++) {
            cover += (y[i] >= q[0][i] && y[i] <= q[2][i] ? 1.0 : 0) / y.length;
            width += (q[2][i] - q[0][i]) / y.length;
        }
        return new double[] {nll, crps, pin, cover, width, rmse};
    }

    /**
     * Interpretable additive models (10.5.4): planted step shape functions on four informative
     * columns (from {@link TabularStressSuite}) plus two planted pairs.
     *
     * @param rows training rows (test rows: half as many)
     * @param steps NAM and logistic-regression gradient steps
     * @param seed the seed
     * @return model to {@code {accuracy, AUC, shape correlation (NaN if none), millis}}, plus
     *     {@code "EBM planted pairs"} to {@code {rank of pair 1, rank of pair 2}} in FAST's
     *     ranking (1 = best)
     */
    public static Map<String, double[]> interpretable(int rows, int steps, long seed) {
        TabularStressSuite suite = new TabularStressSuite(4, 0.33, false, seed);
        int cols = suite.columns();
        int[][] planted = {{0, 4}, {1, 5}};
        Data train = additive(suite, planted, rows, seed + 1);
        Data test = additive(suite, planted, rows / 2, seed + 2);
        Map<String, double[]> out = new LinkedHashMap<>();
        long t0 = System.currentTimeMillis();
        ExplainableBoosting ebm = ExplainableBoosting.fit(train.x(), train.y(), BoostLosses.logistic(),
                ExplainableBoosting.Settings.defaults());
        long ms = System.currentTimeMillis() - t0;
        double[] s = score(test, ebm::predict, ms);
        out.put("EBM (GA2M, 2 pairs)", new double[] {s[0], s[1], shapeCorrelation(suite, ebmShape(ebm)), ms});
        ExplainableBoosting ga = ExplainableBoosting.fit(train.x(), train.y(), BoostLosses.logistic(),
                ExplainableBoosting.Settings.defaults().withInteractions(0));
        double[] sa = score(test, ga::predict, 0);
        out.put("EBM main effects only (GAM)", new double[] {sa[0], sa[1], shapeCorrelation(suite, ebmShape(ga)), 0});
        int[][] ranking = ebm.pairRanking();
        double[] ranks = new double[planted.length];
        for (int p = 0; p < planted.length; p++) {
            for (int k = 0; k < ranking.length; k++) {
                if (ranking[k][0] == planted[p][0] && ranking[k][1] == planted[p][1]) {
                    ranks[p] = k + 1;
                }
            }
        }
        out.put("EBM planted pairs", ranks);
        t0 = System.currentTimeMillis();
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(train.x(), train.y(), GradientBoostedTrees.Objective.BINARY,
                GradientBoostedTrees.Settings.defaults().withTrees(150));
        double[] sg = score(test, gbt::predict, System.currentTimeMillis() - t0);
        out.put("core GBT (black box)", new double[] {sg[0], sg[1], Double.NaN, sg[3]});
        SyntheticRecsys.Table tt = table(train);
        SyntheticRecsys.Table te = table(test);
        Model logistic = Model.sequential().input(cols).dense(1).seed(seed).build();
        BinaryBench.Result lr = BinaryBench.train((x, t) -> logistic.forward(x, t), logistic.parameters(), tt, te,
                steps, 0.05f, seed);
        out.put("logistic regression", new double[] {lr.accuracy(), lr.auc(), Double.NaN, lr.millis()});
        NeuralAdditiveModel nam = new NeuralAdditiveModel(cols, 16, new SplittableRandom(seed));
        BinaryBench.Result nr = BinaryBench.train((x, t) -> nam.forward(x, t), nam.parameters(), tt, te, steps, 0.01f,
                seed);
        out.put("NAM", new double[] {nr.accuracy(), nr.auc(), shapeCorrelation(suite, nam::shape), nr.millis()});
        return out;
    }

    private static Data additive(TabularStressSuite suite, int[][] pairs, int rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int cols = suite.columns();
        double[][] x = new double[rows][cols];
        double[] y = new double[rows];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < cols; c++) {
                x[i][c] = r.nextDouble() * 4 - 2;
            }
            double s = 0;
            for (int j = 0; j < suite.informative(); j++) {
                s += suite.effect(j, x[i][j]);
            }
            for (int[] p : pairs) {
                s += 1.5 * Math.signum(x[i][p[0]] * x[i][p[1]]);
            }
            y[i] = s + 0.3 * r.nextGaussian() > 0 ? 1 : 0;
        }
        return new Data(x, y);
    }

    private static SyntheticRecsys.Table table(Data d) {
        int rows = d.x().length;
        int cols = d.x()[0].length;
        float[] f = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                f[i * cols + j] = (float) d.x()[i][j];
            }
        }
        return new SyntheticRecsys.Table(Tensor.of(Shape.of(rows, cols), f), Tensor.of(Shape.of(rows, 1),
                floats(d.y())));
    }

    /** A learned shape function: column and grid of values to contributions. */
    @FunctionalInterface
    private interface ShapeFunction {
        double[] at(int column, double[] values);
    }

    private static double shapeCorrelation(TabularStressSuite suite, ShapeFunction f) {
        int grid = 81;
        double total = 0;
        double[] v = new double[grid];
        for (int i = 0; i < grid; i++) {
            v[i] = -2 + 4.0 * i / (grid - 1);
        }
        for (int j = 0; j < suite.informative(); j++) {
            double[] b = new double[grid];
            for (int i = 0; i < grid; i++) {
                b[i] = suite.effect(j, v[i]);
            }
            total += pearson(f.at(j, v), b);
        }
        return total / suite.informative();
    }

    private static ShapeFunction ebmShape(ExplainableBoosting ebm) {
        return (j, values) -> java.util.Arrays.stream(values).map(v -> ebm.shape(j, v)).toArray();
    }

    private static double pearson(double[] a, double[] b) {
        double ma = Arrays.stream(a).average().orElse(0);
        double mb = Arrays.stream(b).average().orElse(0);
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
     * PFN-Boost against its parts (10.5.5): a {@link PriorFittedNetwork} pre-trained on
     * {@link TabularPrior}, then per table size and task: the PFN in context, GBT alone, PFN-Boost
     * with the prior scale chosen on validation, and stacking (a logistic regression on the
     * out-of-fold PFN log-odds and GBT score).
     *
     * @param sizes training-set sizes
     * @param pretrainSteps PFN pre-training steps
     * @param tasks tasks per size
     * @param context the most context rows the PFN sees (a random subset beyond that)
     * @param seed the seed
     * @return {@code "<size> rows"} to {@code {PFN, GBT, PFN-Boost, stacking, mean chosen scale}}
     *     test accuracy
     */
    public static Map<String, double[]> priorBoost(int[] sizes, int pretrainSteps, int tasks, int context,
            long seed) {
        int features = 5;
        PriorFittedNetwork pfn = new PriorFittedNetwork(features, 32, 4, 1, 3e-3f, new SplittableRandom(seed));
        pfn.pretrain(pretrainSteps, 8, 64, seed);
        PriorBoost.Scorer scorer = (tx, ty, qx) -> {
            double[][] cx = tx;
            int[] cy = ty;
            if (tx.length > context) {
                int[] pick = new SplittableRandom(seed + tx.length).ints(0, tx.length).distinct().limit(context)
                        .toArray();
                cx = Arrays.stream(pick).mapToObj(i -> tx[i]).toArray(double[][]::new);
                cy = Arrays.stream(pick).map(i -> ty[i]).toArray();
            }
            double[] p = pfn.predictProbability(cx, cy, qx);
            double[][] out = new double[p.length][];
            for (int i = 0; i < p.length; i++) {
                out[i] = new double[] {1 - p[i], p[i]};
            }
            return out;
        };
        LightBoost.Settings boosting = LightBoost.Settings.defaults().withTrees(60).withLeaves(8, 0)
                .withMinDataInLeaf(5);
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed + 5);
        for (int size : sizes) {
            double[] acc = new double[5];
            for (int t = 0; t < tasks; t++) {
                TabularPrior.Task task = TabularPrior.sample(size + 300, features, r);
                double[][] tx = Arrays.copyOfRange(task.x(), 0, size);
                int[] ty = Arrays.copyOfRange(task.y(), 0, size);
                double[][] qx = Arrays.copyOfRange(task.x(), size, size + 300);
                int[] qy = Arrays.copyOfRange(task.y(), size, size + 300);
                acc[0] += PriorBoost.accuracy(scorer.probabilities(tx, ty, qx), qy) / tasks;
                double[] dy = Arrays.stream(ty).asDoubleStream().toArray();
                LightBoost gbt = LightBoost.fit(tx, dy, BoostLosses.logistic(), boosting);
                double[][] gp = new double[qx.length][];
                for (int i = 0; i < qx.length; i++) {
                    double p = gbt.predict(qx[i]);
                    gp[i] = new double[] {1 - p, p};
                }
                acc[1] += PriorBoost.accuracy(gp, qy) / tasks;
                PriorBoost pb = PriorBoost.fit(tx, ty, 2, scorer, PriorBoost.Settings.defaults(boosting));
                acc[2] += PriorBoost.accuracy(pb.predictProbability(qx), qy) / tasks;
                acc[3] += stacking(tx, ty, qx, qy, scorer, boosting, seed) / tasks;
                acc[4] += pb.scale() / tasks;
            }
            out.put(size + " rows", acc);
        }
        return out;
    }

    private static double stacking(double[][] tx, int[] ty, double[][] qx, int[] qy, PriorBoost.Scorer scorer,
            LightBoost.Settings boosting, long seed) {
        int n = ty.length;
        double[][] pfnOof = PriorBoost.outOfFold(tx, ty, 2, scorer, 4, seed);
        double[][] gbtOof = PriorBoost.outOfFold(tx, ty, 2, (a, b, c) -> {
            LightBoost g = LightBoost.fit(a, Arrays.stream(b).asDoubleStream().toArray(), BoostLosses.logistic(),
                    boosting);
            double[][] o = new double[c.length][];
            for (int i = 0; i < c.length; i++) {
                double p = g.predict(c[i]);
                o[i] = new double[] {1 - p, p};
            }
            return o;
        }, 4, seed);
        double[] w = new double[3];
        for (int it = 0; it < 300; it++) {
            double[] grad = new double[3];
            for (int i = 0; i < n; i++) {
                double[] f = {1, logit(pfnOof[i][1]), logit(gbtOof[i][1])};
                double z = w[0] * f[0] + w[1] * f[1] + w[2] * f[2];
                double err = 1 / (1 + Math.exp(-z)) - ty[i];
                for (int k = 0; k < 3; k++) {
                    grad[k] += err * f[k] / n;
                }
            }
            for (int k = 0; k < 3; k++) {
                w[k] -= 0.5 * grad[k];
            }
        }
        double[][] pq = scorer.probabilities(tx, ty, qx);
        LightBoost g = LightBoost.fit(tx, Arrays.stream(ty).asDoubleStream().toArray(), BoostLosses.logistic(),
                boosting);
        int right = 0;
        for (int i = 0; i < qy.length; i++) {
            double z = w[0] + w[1] * logit(pq[i][1]) + w[2] * logit(g.predict(qx[i]));
            right += (z > 0 ? 1 : 0) == qy[i] ? 1 : 0;
        }
        return right / (double) qy.length;
    }

    private static double logit(double p) {
        double q = Math.clamp(p, 1e-6, 1 - 1e-6);
        return Math.log(q / (1 - q));
    }

    /**
     * Prints the longer runs.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (int rows : new int[] {2000, 10000, 50000}) {
            lightGbm(rows, 100, 1).forEach((k, v) -> System.out.printf("lightGbm %d %s %s%n", rows, k,
                    Arrays.toString(v)));
        }
        for (int levels : new int[] {100, 1000, 5000}) {
            categorical(5000, levels, 300, 1).forEach((k, v) -> System.out.printf("categorical %d %s %s%n", levels, k,
                    Arrays.toString(v)));
        }
        for (boolean bimodal : new boolean[] {false, true}) {
            probabilistic(2000, bimodal, 500, 1).forEach((k, v) -> System.out.printf("probabilistic %s %s %s%n",
                    bimodal, k, Arrays.toString(v)));
        }
        interpretable(5000, 2000, 1).forEach((k, v) -> System.out.printf("interpretable %s %s%n", k,
                Arrays.toString(v)));
        priorBoost(new int[] {30, 100, 300, 1000, 3000}, 2000, 10, 256, 1).forEach((k, v) ->
                System.out.printf("priorBoost %s %s%n", k, Arrays.toString(v)));
    }
}
