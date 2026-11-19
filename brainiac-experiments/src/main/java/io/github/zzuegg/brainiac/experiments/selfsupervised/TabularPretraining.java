package io.github.zzuegg.brainiac.experiments.selfsupervised;

import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.tabular.StructuredCorruption;
import io.github.zzuegg.brainiac.experimental.timeseries.RidgeClassifier;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Tabular self-supervised pretraining under label scarcity (13.6.8) and mutual-information-guided
 * structured corruption (13.6.9), on {@link TabularStressSuite} tables.
 * The label-scarcity and missingness tables use a suite with three informative columns and one
 * noisy duplicate of each, so unlabelled rows carry some structure (independent uniform columns
 * alone would give a pretext task nothing to learn).
 * <ul>
 *   <li>{@link #labelScarcity}: VIME, SCARF, SubTab and mask-and-recover encoders
 *       ({@link io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining}) pretrained on
 *       all rows, then fine-tuned on a small labelled share, against the same MLP from scratch, a
 *       random forest and gradient-boosted trees, as mean accuracy and seed spread;</li>
 *   <li>{@link #missingness}: the same models on test rows with values missing completely at
 *       random (MCAR) or when large (MNAR), imputed by the training mean;</li>
 *   <li>{@link #structured}: SCARF with random corruption against SCARF with
 *       {@link StructuredCorruption} (group-marginal and group-CVAE replacement) on a table with
 *       planted redundant blocks (noisy duplicates of the informative columns), by linear-probe and
 *       fine-tune accuracy and by the pretext loss each corruption leaves.</li>
 * </ul>
 */
public final class TabularPretraining {

    private TabularPretraining() {
    }

    /**
     * A standardised table.
     *
     * @param x rows
     * @param y binary classes
     */
    public record Table(float[][] x, int[] y) {
        /**
         * @param n rows
         * @return the first {@code n} rows
         */
        public Table head(int n) {
            return new Table(java.util.Arrays.copyOf(x, n), java.util.Arrays.copyOf(y, n));
        }
    }

    /**
     * Draws a stress-suite table, optionally with {@code copies} noisy duplicates of every
     * informative column appended (the redundant blocks of 13.6.9).
     *
     * @param suite the suite
     * @param rows rows
     * @param copies duplicates per informative column
     * @param seed the seed
     * @return the standardised table
     */
    public static Table table(TabularStressSuite suite, int rows, int copies, long seed) {
        SyntheticRecsys.Table t = suite.table(rows, seed);
        int d = suite.columns();
        int extra = copies * suite.informative();
        float[] v = t.features().toArray();
        float[] lab = t.labels().toArray();
        SplittableRandom r = new SplittableRandom(seed + 77);
        float[][] x = new float[rows][d + extra];
        int[] y = new int[rows];
        double scale = 1 / Math.sqrt(4.0 / 3);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = (float) (v[i * d + j] * scale);
            }
            for (int c = 0; c < copies; c++) {
                for (int j = 0; j < suite.informative(); j++) {
                    x[i][d + c * suite.informative() + j] = (float) (x[i][j] + 0.15 * r.nextGaussian());
                }
            }
            y[i] = lab[i] > 0.5 ? 1 : 0;
        }
        return new Table(x, y);
    }

    private static double[][] toDouble(float[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = x[i][j];
            }
        }
        return out;
    }

    private static double accuracy(double[][] p, int[] y) {
        int right = 0;
        for (int i = 0; i < y.length; i++) {
            int arg = p[i][1] > p[i][0] ? 1 : 0;
            right += arg == y[i] ? 1 : 0;
        }
        return right / (double) y.length;
    }

    private static double accuracy(int[] p, int[] y) {
        return RidgeClassifier.accuracy(p, y);
    }

    /** Fits the tree baselines on labelled rows and scores the test rows. */
    private static double[] trees(Table labelled, Table test, long seed) {
        double[][] a = toDouble(labelled.x());
        double[][] b = toDouble(test.x());
        double rf = accuracy(RandomForest.fit(a, labelled.y(), 2, new RandomForest.Settings(
                RandomForest.Mode.RANDOM_FOREST, 100, 20, 1, 0, seed)).predict(b), test.y());
        double[] yy = new double[labelled.y().length];
        for (int i = 0; i < yy.length; i++) {
            yy[i] = labelled.y()[i];
        }
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(a, yy, GradientBoostedTrees.Objective.BINARY,
                new GradientBoostedTrees.Settings(100, 3, 0.1, 1, 1e-3, 1, 32, seed));
        double[] p = gbt.predict(b);
        int right = 0;
        for (int i = 0; i < p.length; i++) {
            right += (p[i] > 0.5 ? 1 : 0) == test.y()[i] ? 1 : 0;
        }
        return new double[] {rf, right / (double) p.length};
    }

    /**
     * Accuracy against label share: mean over seeds and the spread (max minus min).
     *
     * @param pool unlabelled rows (pretraining)
     * @param testRows test rows
     * @param shares labelled shares of the pool, such as 0.01 and 0.1
     * @param seeds seeds averaged
     * @param pretrainSteps pretraining steps
     * @param fineTuneSteps fine-tuning steps
     * @param seed the base seed
     * @return "method @ share" to {mean accuracy, spread}
     */
    public static Map<String, double[]> labelScarcity(int pool, int testRows, double[] shares, int seeds,
            int pretrainSteps, int fineTuneSteps, long seed) {
        TabularStressSuite suite = new TabularStressSuite(3, 0.4, false, seed);
        Map<String, double[]> sums = new LinkedHashMap<>();
        for (int s = 0; s < seeds; s++) {
            long ss = seed + 31L * s;
            Table all = table(suite, pool, 1, ss);
            Table test = table(suite, testRows, 1, ss + 500);
            int d = all.x()[0].length;
            Map<String, io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining> encoders =
                    new LinkedHashMap<>();
            for (io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective o
                    : io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.values()) {
                encoders.put(o.toString(), new io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining(o, d,
                        32, ss).pretrain(all.x(), pretrainSteps, 128, 3e-3f, ss));
            }
            for (double share : shares) {
                Table lab = all.head(Math.max(8, (int) Math.round(share * pool)));
                String suffix = String.format(" @ %.0f%%", 100 * share);
                for (Map.Entry<String, io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining> e
                        : encoders.entrySet()) {
                    double probe = accuracy(RidgeClassifier.fit(e.getValue().embed(lab.x()), lab.y(), 2, 1)
                            .predict(e.getValue().embed(test.x())), test.y());
                    add(sums, e.getKey() + " probe" + suffix, probe);
                    add(sums, e.getKey() + " fine-tune" + suffix, accuracy(e.getValue().copy().fineTune(lab.x(),
                            lab.y(), 2, test.x(), fineTuneSteps, 3e-3f, ss), test.y()));
                }
                io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining scratch =
                        new io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining(
                                io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.SCARF, d,
                                32, ss);
                add(sums, "scratch MLP" + suffix, accuracy(scratch.fineTune(lab.x(), lab.y(), 2, test.x(),
                        fineTuneSteps, 3e-3f, ss), test.y()));
                double[] t = trees(lab, test, ss);
                add(sums, "random forest" + suffix, t[0]);
                add(sums, "GBT" + suffix, t[1]);
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        sums.forEach((k, v) -> out.put(k, new double[] {v[0] / seeds, v[2] - v[1]}));
        return out;
    }

    private static void add(Map<String, double[]> sums, String key, double v) {
        double[] s = sums.computeIfAbsent(key, k -> new double[] {0, Double.POSITIVE_INFINITY,
                Double.NEGATIVE_INFINITY});
        s[0] += v;
        s[1] = Math.min(s[1], v);
        s[2] = Math.max(s[2], v);
    }

    /**
     * Applies test-time missingness: MCAR drops each value with probability {@code rate}; MNAR
     * drops values above the {@code 1 - rate} quantile of a standard normal-ish column (here, above
     * {@code 0.5}). Missing values become 0, the training mean.
     *
     * @param t the table
     * @param mnar whether the missingness depends on the value
     * @param rate the MCAR rate
     * @param seed the seed
     * @return the table with imputed missing values
     */
    public static Table missing(Table t, boolean mnar, double rate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[t.x().length][];
        for (int i = 0; i < x.length; i++) {
            x[i] = t.x()[i].clone();
            for (int j = 0; j < x[i].length; j++) {
                boolean drop = mnar ? x[i][j] > 0.5 && r.nextDouble() < 2 * rate : r.nextDouble() < rate;
                if (drop) {
                    x[i][j] = 0;
                }
            }
        }
        return new Table(x, t.y());
    }

    /**
     * Accuracy under test-time missingness, with the labels of a 20% share.
     *
     * @param pool pretraining rows
     * @param testRows test rows
     * @param rate the missingness rate
     * @param pretrainSteps pretraining steps
     * @param fineTuneSteps fine-tuning steps
     * @param seed the seed
     * @return method to {clean, MCAR, MNAR} accuracy
     */
    public static Map<String, double[]> missingness(int pool, int testRows, double rate, int pretrainSteps,
            int fineTuneSteps, long seed) {
        TabularStressSuite suite = new TabularStressSuite(3, 0.4, false, seed);
        Table all = table(suite, pool, 1, seed);
        Table test = table(suite, testRows, 1, seed + 500);
        Table lab = all.head(pool / 5);
        Table[] views = {test, missing(test, false, rate, seed + 1), missing(test, true, rate, seed + 2)};
        int d = all.x()[0].length;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective o : new
                io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective[] {
                    io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.VIME,
                    io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.SCARF}) {
            double[] acc = new double[3];
            io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining enc =
                    new io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining(o, d, 32, seed)
                            .pretrain(all.x(), pretrainSteps, 128, 3e-3f, seed);
            for (int v = 0; v < 3; v++) {
                acc[v] = accuracy(enc.copy().fineTune(lab.x(), lab.y(), 2, views[v].x(), fineTuneSteps, 3e-3f, seed),
                        test.y());
            }
            out.put(o.toString(), acc);
        }
        double[] scratch = new double[3];
        double[] rf = new double[3];
        for (int v = 0; v < 3; v++) {
            scratch[v] = accuracy(new io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining(
                    io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.SCARF, d, 32, seed)
                    .fineTune(lab.x(), lab.y(), 2, views[v].x(), fineTuneSteps, 3e-3f, seed), test.y());
            rf[v] = trees(lab, views[v], seed)[0];
        }
        out.put("scratch MLP", scratch);
        out.put("random forest", rf);
        return out;
    }

    /**
     * Random against structured corruption for SCARF on a table with redundant blocks.
     *
     * @param pool pretraining rows
     * @param testRows test rows
     * @param share the labelled share
     * @param pretrainSteps pretraining steps
     * @param seed the seed
     * @return corruption to {linear-probe accuracy, pretext loss of the trained encoder on its own corruption}
     */
    public static Map<String, double[]> structured(int pool, int testRows, double share, int pretrainSteps,
            long seed) {
        TabularStressSuite suite = new TabularStressSuite(3, 0.25, false, seed);
        Table all = table(suite, pool, 2, seed);
        Table test = table(suite, testRows, 2, seed + 500);
        Table lab = all.head(Math.max(8, (int) Math.round(share * pool)));
        int d = all.x()[0].length;
        int groups = suite.columns();
        Map<String, double[]> out = new LinkedHashMap<>();
        Map<String, io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Corruptor> corruptors =
                new LinkedHashMap<>();
        corruptors.put("random", io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.marginal(all.x(),
                0.4));
        StructuredCorruption marginal = StructuredCorruption.fit(all.x(), groups, 0.4,
                StructuredCorruption.Replacement.GROUP_MARGINAL, 300, seed);
        corruptors.put("group marginal", marginal);
        corruptors.put("group CVAE", StructuredCorruption.fit(all.x(), groups, 0.4,
                StructuredCorruption.Replacement.GROUP_CVAE, 300, seed));
        for (Map.Entry<String, io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Corruptor> e
                : corruptors.entrySet()) {
            io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining enc =
                    new io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining(
                            io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.SCARF, d, 32,
                            seed).corruption(e.getValue()).pretrain(all.x(), pretrainSteps, 128, 3e-3f, seed);
            double probe = accuracy(RidgeClassifier.fit(enc.embed(lab.x()), lab.y(), 2, 1).predict(enc.embed(test.x())),
                    test.y());
            out.put(e.getKey(), new double[] {probe, enc.pretextLoss(java.util.Arrays.copyOf(all.x(), 256), seed + 9)});
        }
        double scratch = accuracy(new io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining(
                io.github.zzuegg.brainiac.experimental.tabular.TabularPretraining.Objective.SCARF, d, 32, seed)
                .fineTune(lab.x(), lab.y(), 2, test.x(), 200, 3e-3f, seed), test.y());
        out.put("scratch MLP", new double[] {scratch, Double.NaN});
        out.put("GBT", new double[] {trees(lab, test, seed)[1], Double.NaN});
        int[] g = marginal.groups();
        int together = 0;
        for (int c = 0; c < 2; c++) {
            for (int j = 0; j < suite.informative(); j++) {
                together += g[j] == g[suite.columns() + c * suite.informative() + j] ? 1 : 0;
            }
        }
        out.put("duplicates grouped with their source", new double[] {together / (2.0 * suite.informative()),
                Double.NaN});
        return out;
    }
}
