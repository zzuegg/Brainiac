package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.experimental.data.SeriesArchive;
import io.github.zzuegg.brainiac.experimental.data.SeriesClassificationTasks;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.timeseries.RandomConvolutionFeatures;
import io.github.zzuegg.brainiac.experimental.timeseries.RidgeClassifier;
import io.github.zzuegg.brainiac.experimental.tsc.Boss;
import io.github.zzuegg.brainiac.experimental.tsc.DeepSeriesClassifiers;
import io.github.zzuegg.brainiac.experimental.tsc.ElasticDistances;
import io.github.zzuegg.brainiac.experimental.tsc.Hydra;
import io.github.zzuegg.brainiac.experimental.tsc.IntervalForest;
import io.github.zzuegg.brainiac.experimental.tsc.ProximityForest;
import io.github.zzuegg.brainiac.experimental.tsc.Quant;
import io.github.zzuegg.brainiac.experimental.tsc.SeriesEnsembles;
import io.github.zzuegg.brainiac.experimental.tsc.Weasel;
import io.github.zzuegg.brainiac.experimental.tsc.WideRidgeClassifier;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Series classifier families (13.6.1-13.6.6) on {@link SeriesArchive} tasks, each against ROCKET
 * and MiniRocket ({@link RandomConvolutionFeatures} with {@link RidgeClassifier}) and the
 * baselines of {@link SeriesClassification}:
 * <ul>
 *   <li>{@link #dictionary}: BOSS and WEASEL 1.0/2.0 against ROCKET, MiniRocket and 1-NN DTW;</li>
 *   <li>{@link #convolution}: Hydra and MultiRocket-Hydra against ROCKET, MiniRocket and WEASEL 2.0;</li>
 *   <li>{@link #interval}: QUANT, the time series forest and DrCIF-lite against ROCKET, MiniRocket,
 *       a random forest on raw values and Hydra;</li>
 *   <li>{@link #ensembles}: CAWPE, voting, symmetric and asymmetric stacking and feature
 *       concatenation of Hydra, QUANT and WEASEL 2.0, with the oracle accuracy and the share of
 *       the oracle gap each closes;</li>
 *   <li>{@link #deep}: FCN, ResNet, InceptionTime and LITE against the TCN and MLP of
 *       {@link SeriesClassification}, ROCKET and Hydra, with parameter counts;</li>
 *   <li>{@link #elastic}: 1-NN under elastic distances, the Elastic Ensemble and Proximity Forest
 *       against 1-NN DTW and ROCKET.</li>
 * </ul>
 * Every row reports {@code {accuracy, milliseconds}} (fit plus predict).
 */
public final class SeriesClassifiers {

    private SeriesClassifiers() {
    }

    /**
     * A train/test split of an archive task.
     *
     * @param train the training set
     * @param test the test set
     */
    public record Split(SeriesClassificationTasks.Task train, SeriesClassificationTasks.Task test) {
    }

    /**
     * Draws a split.
     *
     * @param name the {@link SeriesArchive#named} task
     * @param train training series
     * @param test test series
     * @param length steps
     * @param seed the seed
     * @return the split
     */
    public static Split split(String name, int train, int test, int length, long seed) {
        return new Split(SeriesArchive.named(name, train, length, seed),
                SeriesArchive.named(name, test, length, seed + 1000));
    }

    private static double[] timed(Supplier<int[]> f, int[] truth) {
        long t = System.nanoTime();
        int[] p = f.get();
        return new double[] {RidgeClassifier.accuracy(p, truth), (System.nanoTime() - t) / 1e6};
    }

    static double[][] toDouble(float[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int t = 0; t < x[i].length; t++) {
                out[i][t] = x[i][t];
            }
        }
        return out;
    }

    static int[] rocket(Split s, long seed) {
        RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(500, s.train().length(), 1, seed);
        return RidgeClassifier.fit(rocket.transform(RandomConvolutionFeatures.univariate(s.train().series())),
                s.train().labels(), s.train().classes(), 10).predict(rocket.transform(
                RandomConvolutionFeatures.univariate(s.test().series())));
    }

    static int[] miniRocket(Split s, long seed) {
        float[][][] x = RandomConvolutionFeatures.univariate(s.train().series());
        RandomConvolutionFeatures mini = RandomConvolutionFeatures.miniRocket(x, 840, seed);
        return RidgeClassifier.fit(mini.transform(x), s.train().labels(), s.train().classes(), 10).predict(
                mini.transform(RandomConvolutionFeatures.univariate(s.test().series())));
    }

    static double[][] hydraScores(float[][] train, int[] labels, int classes, float[][] test, long seed) {
        Hydra h = new Hydra(train[0].length, 1, 16, 8, Hydra.Counting.SOFT_MAX_HARD_MIN, true, seed);
        double[][] a = h.fitTransform(RandomConvolutionFeatures.univariate(train));
        double[][] b = h.transform(RandomConvolutionFeatures.univariate(test));
        return WideRidgeClassifier.fit(a, labels, classes, WideRidgeClassifier.defaultAlphas()).scores(b);
    }

    static Weasel weasel2(float[][] train, int[] labels, int classes, long seed) {
        Weasel.Settings w = Weasel.Settings.defaults(Weasel.Version.V2);
        return Weasel.fit(train, labels, classes, new Weasel.Settings(w.version(), 24, w.alphabet(), w.chiSquared(),
                seed));
    }

    static Quant.Settings quantSettings(long seed) {
        return new Quant.Settings(5, 100, 0.1, seed);
    }

    /**
     * The dictionary table.
     *
     * @param s the split
     * @param seed the seed
     * @return method to {accuracy, ms}
     */
    public static Map<String, double[]> dictionary(Split s, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SeriesClassificationTasks.Task tr = s.train();
        float[][] te = s.test().series();
        int[] truth = s.test().labels();
        Boss.Settings d = Boss.Settings.defaults(tr.length());
        int[] windows = java.util.Arrays.copyOf(d.windows(), Math.min(4, d.windows().length));
        out.put("BOSS", timed(() -> Boss.fit(tr.series(), tr.labels(), tr.classes(), new Boss.Settings(windows,
                d.wordLengths(), d.alphabet(), true, true, d.keep())).predict(te), truth));
        out.put("BOSS (no std norm)", timed(() -> Boss.fit(tr.series(), tr.labels(), tr.classes(), new Boss.Settings(
                windows, d.wordLengths(), d.alphabet(), true, false, d.keep())).predict(te), truth));
        out.put("WEASEL", timed(() -> {
            Weasel.Settings v1 = Weasel.Settings.defaults(Weasel.Version.V1);
            return Weasel.fit(tr.series(), tr.labels(), tr.classes(), new Weasel.Settings(v1.version(), 5,
                    v1.alphabet(), v1.chiSquared(), seed)).predict(te);
        }, truth));
        out.put("WEASEL 2.0", timed(() -> weasel2(tr.series(), tr.labels(), tr.classes(), seed).predict(te), truth));
        out.put("ROCKET", timed(() -> rocket(s, seed), truth));
        out.put("MiniRocket", timed(() -> miniRocket(s, seed), truth));
        out.put("DTW 1-NN", timed(() -> SeriesClassification.dtw(tr, s.test()), truth));
        return out;
    }

    /**
     * The convolution table.
     *
     * @param s the split
     * @param seed the seed
     * @return method to {accuracy, ms}
     */
    public static Map<String, double[]> convolution(Split s, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SeriesClassificationTasks.Task tr = s.train();
        int[] truth = s.test().labels();
        out.put("Hydra", timed(() -> WideRidgeClassifier.argmax(hydraScores(tr.series(), tr.labels(), tr.classes(),
                s.test().series(), seed)), truth));
        out.put("Hydra hard counts", timed(() -> {
            Hydra h = new Hydra(tr.length(), 1, 16, 8, Hydra.Counting.HARD, true, seed);
            double[][] a = h.fitTransform(RandomConvolutionFeatures.univariate(tr.series()));
            return WideRidgeClassifier.fit(a, tr.labels(), tr.classes(), WideRidgeClassifier.defaultAlphas())
                    .predict(h.transform(RandomConvolutionFeatures.univariate(s.test().series())));
        }, truth));
        out.put("MultiRocket-Hydra", timed(() -> {
            float[][][] x = RandomConvolutionFeatures.univariate(tr.series());
            float[][][] xt = RandomConvolutionFeatures.univariate(s.test().series());
            Hydra h = new Hydra(tr.length(), 1, 16, 8, Hydra.Counting.SOFT_MAX_HARD_MIN, true, seed);
            RandomConvolutionFeatures mini = RandomConvolutionFeatures.miniRocket(x, 840, seed);
            double[][] a = Hydra.concatenate(h.fitTransform(x), mini.transform(x));
            double[][] b = Hydra.concatenate(h.transform(xt), mini.transform(xt));
            return WideRidgeClassifier.fit(a, tr.labels(), tr.classes(), WideRidgeClassifier.defaultAlphas())
                    .predict(b);
        }, truth));
        out.put("ROCKET", timed(() -> rocket(s, seed), truth));
        out.put("MiniRocket", timed(() -> miniRocket(s, seed), truth));
        out.put("WEASEL 2.0", timed(() -> weasel2(tr.series(), tr.labels(), tr.classes(), seed)
                .predict(s.test().series()), truth));
        return out;
    }

    /**
     * The interval table.
     *
     * @param s the split
     * @param seed the seed
     * @return method to {accuracy, ms}
     */
    public static Map<String, double[]> interval(Split s, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SeriesClassificationTasks.Task tr = s.train();
        float[][] te = s.test().series();
        int[] truth = s.test().labels();
        out.put("QUANT", timed(() -> Quant.fit(tr.series(), tr.labels(), tr.classes(), quantSettings(seed))
                .predict(te), truth));
        out.put("TSF", timed(() -> IntervalForest.fit(tr.series(), tr.labels(), tr.classes(), IntervalForest.Kind.TSF,
                60, seed).predict(te), truth));
        out.put("DrCIF-lite", timed(() -> IntervalForest.fit(tr.series(), tr.labels(), tr.classes(),
                IntervalForest.Kind.DRCIF_LITE, 60, seed).predict(te), truth));
        out.put("RF raw", timed(() -> RandomForest.fit(toDouble(tr.series()), tr.labels(), tr.classes(),
                RandomForest.Settings.defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(100)).predict(toDouble(te)),
                truth));
        out.put("ROCKET", timed(() -> rocket(s, seed), truth));
        out.put("MiniRocket", timed(() -> miniRocket(s, seed), truth));
        out.put("Hydra", timed(() -> WideRidgeClassifier.argmax(hydraScores(tr.series(), tr.labels(), tr.classes(),
                te, seed)), truth));
        return out;
    }

    /**
     * One task's ensemble row.
     *
     * @param members each member's test accuracy, by name
     * @param strategies each strategy's test accuracy, by name
     * @param oracle the oracle accuracy
     * @param best the best member's accuracy
     */
    public record EnsembleResult(Map<String, Double> members, Map<String, Double> strategies, double oracle,
            double best) {

        /**
         * @param strategy a strategy
         * @return the share of the oracle gap it closes
         */
        public double gapClosed(String strategy) {
            return SeriesEnsembles.gapClosed(strategies.get(strategy), best, oracle);
        }
    }

    /**
     * The ensemble table for one task: Hydra, QUANT and WEASEL 2.0 combined.
     *
     * @param s the split
     * @param folds out-of-fold folds
     * @param seed the seed
     * @return the result
     */
    public static EnsembleResult ensembles(Split s, int folds, long seed) {
        SeriesClassificationTasks.Task tr = s.train();
        float[][] te = s.test().series();
        int[] truth = s.test().labels();
        int k = tr.classes();
        SeriesEnsembles.Member hydra = (a, y, c, b) -> hydraScores(a, y, c, b, seed);
        SeriesEnsembles.Member quant = (a, y, c, b) -> Quant.fit(a, y, c, quantSettings(seed)).probabilities(b);
        SeriesEnsembles.Member weasel = (a, y, c, b) -> weasel2(a, y, c, seed).scores(b);
        List<String> names = List.of("Hydra", "QUANT", "WEASEL 2.0");
        List<SeriesEnsembles.Member> members = List.of(hydra, quant, weasel);
        List<double[][]> oof = new ArrayList<>();
        List<double[][]> test = new ArrayList<>();
        Map<String, Double> acc = new LinkedHashMap<>();
        double best = 0;
        for (int m = 0; m < members.size(); m++) {
            oof.add(SeriesEnsembles.outOfFold(members.get(m), tr.series(), tr.labels(), k, folds, seed));
            test.add(members.get(m).fitScores(tr.series(), tr.labels(), k, te));
            double a = SeriesEnsembles.accuracy(test.get(m), truth);
            acc.put(names.get(m), a);
            best = Math.max(best, a);
        }
        // probabilities for CAWPE: ridge decision values through a softmax, QUANT's as they are
        List<double[][]> oofP = List.of(SeriesEnsembles.softmax(oof.get(0), 0.25), oof.get(1),
                SeriesEnsembles.softmax(oof.get(2), 0.25));
        List<double[][]> testP = List.of(SeriesEnsembles.softmax(test.get(0), 0.25), test.get(1),
                SeriesEnsembles.softmax(test.get(2), 0.25));
        Map<String, Double> strat = new LinkedHashMap<>();
        strat.put("vote", SeriesEnsembles.accuracy(SeriesEnsembles.vote(test), truth));
        strat.put("CAWPE", SeriesEnsembles.accuracy(SeriesEnsembles.cawpe(oofP, testP, tr.labels(), 4), truth));
        strat.put("stack (ridge)", SeriesEnsembles.accuracy(SeriesEnsembles.stack(oof, test, tr.labels(), k,
                SeriesEnsembles.Meta.RIDGE, seed), truth));
        strat.put("stack (extra trees)", SeriesEnsembles.accuracy(SeriesEnsembles.stack(oof, test, tr.labels(), k,
                SeriesEnsembles.Meta.EXTRA_TREES, seed), truth));
        double[][] qTrain = Quant.features(tr.series(), quantSettings(seed).maxDepth());
        double[][] qTest = Quant.features(te, quantSettings(seed).maxDepth());
        strat.put("asymmetric stack", SeriesEnsembles.accuracy(SeriesEnsembles.asymmetricStack(qTrain, qTest,
                List.<double[][]>of(oof.get(0)), List.<double[][]>of(test.get(0)), tr.labels(), k, SeriesEnsembles.Meta.EXTRA_TREES, seed),
                truth));
        Hydra h = new Hydra(tr.length(), 1, 16, 8, Hydra.Counting.SOFT_MAX_HARD_MIN, true, seed);
        double[][] hTrain = h.fitTransform(RandomConvolutionFeatures.univariate(tr.series()));
        double[][] hTest = h.transform(RandomConvolutionFeatures.univariate(te));
        strat.put("concat (extra trees)", SeriesEnsembles.accuracy(SeriesEnsembles.concatenate(List.of(hTrain, qTrain),
                List.of(hTest, qTest), tr.labels(), k, SeriesEnsembles.Meta.EXTRA_TREES, seed), truth));
        return new EnsembleResult(acc, strat, SeriesEnsembles.oracle(test, truth), best);
    }

    /**
     * The deep table.
     *
     * @param s the split
     * @param steps training steps per network
     * @param maxKernel the longest InceptionTime/LITE kernel (40 in the papers)
     * @param seed the seed
     * @return method to {accuracy, ms, parameters}
     */
    public static Map<String, double[]> deep(Split s, int steps, int maxKernel, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SeriesClassificationTasks.Task tr = s.train();
        float[][] te = s.test().series();
        int[] truth = s.test().labels();
        for (DeepSeriesClassifiers.Kind kind : DeepSeriesClassifiers.Kind.values()) {
            int width = switch (kind) {
                case FCN -> 16;
                case INCEPTION_TIME -> 6;
                default -> 8;
            };
            int ensemble = kind == DeepSeriesClassifiers.Kind.INCEPTION_TIME ? 2 : 1;
            DeepSeriesClassifiers net = new DeepSeriesClassifiers(kind, width, tr.classes(), ensemble, maxKernel, seed);
            double[] r = timed(() -> net.fit(tr.series(), tr.labels(), steps, 32, 5e-3f, seed).predict(te), truth);
            out.put(kind.toString(), new double[] {r[0], r[1], net.parameterCount()});
        }
        double[] tcn = timed(() -> SeriesClassification.tcn(tr, s.test(), steps, seed), truth);
        out.put("TCN", new double[] {tcn[0], tcn[1], Double.NaN});
        double[] mlp = timed(() -> SeriesClassification.mlp(tr, s.test(), steps, seed), truth);
        out.put("MLP", new double[] {mlp[0], mlp[1], Double.NaN});
        double[] rocket = timed(() -> rocket(s, seed), truth);
        out.put("ROCKET", new double[] {rocket[0], rocket[1], Double.NaN});
        double[] hydra = timed(() -> WideRidgeClassifier.argmax(hydraScores(tr.series(), tr.labels(), tr.classes(),
                te, seed)), truth);
        out.put("Hydra", new double[] {hydra[0], hydra[1], Double.NaN});
        return out;
    }

    /**
     * The elastic table.
     *
     * @param s the split
     * @param trees Proximity Forest trees
     * @param seed the seed
     * @return method to {accuracy, ms}
     */
    public static Map<String, double[]> elastic(Split s, int trees, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SeriesClassificationTasks.Task tr = s.train();
        double[][] a = toDouble(tr.series());
        double[][] b = toDouble(s.test().series());
        int[] y = tr.labels();
        int[] truth = s.test().labels();
        double sd = ProximityForest.sd(a);
        int window = Math.max(1, tr.length() / 10);
        out.put("DTW 1-NN (existing, 10% band)", timed(() -> SeriesClassification.dtw(tr, s.test()), truth));
        out.put("DTW 1-NN (full)", timed(() -> ProximityForest.nearestNeighbour(a, y, b,
                ElasticDistances.Measure.DTW_FULL, new double[0]), truth));
        double[] pruned = new double[1];
        out.put("DTW 1-NN (10% window, LB_Keogh)", timed(() -> {
            ProximityForest.PrunedSearch ps = ProximityForest.dtwWithLowerBound(a, y, b, window);
            pruned[0] = ps.prunedShare();
            return ps.predictions();
        }, truth));
        out.put("Euclidean 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b,
                ElasticDistances.Measure.EUCLIDEAN, new double[0]), truth));
        out.put("WDTW 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b, ElasticDistances.Measure.WDTW,
                new double[] {0.1}), truth));
        out.put("DDTW 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b, ElasticDistances.Measure.DDTW,
                new double[] {0.1}), truth));
        out.put("LCSS 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b, ElasticDistances.Measure.LCSS,
                new double[] {0.5 * sd, 0.1}), truth));
        out.put("ERP 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b, ElasticDistances.Measure.ERP,
                new double[] {0, 0.1}), truth));
        out.put("MSM 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b, ElasticDistances.Measure.MSM,
                new double[] {0.1}), truth));
        out.put("TWE 1-NN", timed(() -> ProximityForest.nearestNeighbour(a, y, b, ElasticDistances.Measure.TWE,
                new double[] {1e-3, 0.05}), truth));
        ElasticDistances.Measure[] ee = {ElasticDistances.Measure.EUCLIDEAN, ElasticDistances.Measure.DTW,
                ElasticDistances.Measure.DDTW, ElasticDistances.Measure.WDTW, ElasticDistances.Measure.LCSS,
                ElasticDistances.Measure.ERP, ElasticDistances.Measure.MSM, ElasticDistances.Measure.TWE};
        out.put("Elastic Ensemble", timed(() -> WideRidgeClassifier.argmax(ProximityForest.elasticEnsemble(a, y,
                tr.classes(), b, ee)), truth));
        out.put("Proximity Forest", timed(() -> ProximityForest.fit(a, y, tr.classes(), trees, 5, ee, seed)
                .predict(b), truth));
        out.put("ROCKET", timed(() -> rocket(s, seed), truth));
        out.put("LB_Keogh pruned share", new double[] {pruned[0], 0});
        return out;
    }
}
