package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.BatchMixing;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.eval.EvalStatistics;
import io.github.zzuegg.brainiac.experimental.ml.KMeans;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.synth.ConditionalTabularGan;
import io.github.zzuegg.brainiac.experimental.synth.DatasetCondenser;
import io.github.zzuegg.brainiac.experimental.synth.GenerativeAugmenter;
import io.github.zzuegg.brainiac.experimental.synth.MixedTable;
import io.github.zzuegg.brainiac.experimental.synth.SyntheticDataAudit;
import io.github.zzuegg.brainiac.experimental.synth.TabularDiffusion;
import io.github.zzuegg.brainiac.experiments.generative.GuidanceComparison;
import io.github.zzuegg.brainiac.experiments.generative.ImageAutoencoders;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Training on synthetic data (wave 9, items 9.3.6 to 9.3.8).
 * <ul>
 *   <li>{@link #condensation}: dataset distillation on {@link ToyTasks#digits} (10 classes, 8x8,
 *       pixel flips): gradient and distribution matching ({@link DatasetCondenser}) against random
 *       real images, per-class k-means centroids ({@link KMeans}), iCaRL herding and greedy
 *       k-center core sets, scored by a fresh MLP trained only on the condensed set, and by a
 *       different (wider, tanh) MLP for transfer;</li>
 *   <li>{@link #tabular}: train-on-synthetic, test-on-real on a mixed table derived from
 *       {@link TabularStressSuite} (one informative column binned into a categorical column,
 *       positives subsampled to about 12%): {@link TabularDiffusion} and
 *       {@link ConditionalTabularGan} against a Gaussian copula, SMOTE-style interpolation and the
 *       real table, with the closest-record ratio of {@link SyntheticDataAudit};</li>
 *   <li>{@link #augmentation} and {@link #guidanceSweep}: digits classifiers trained on real
 *       images plus samples of a class-conditional diffusion model ({@link GenerativeAugmenter})
 *       against real only, shift plus mixup ({@link BatchMixing}) and real plus per-class VAE
 *       samples ({@link ImageAutoencoders}); the sweep also reports improved precision and recall
 *       ({@link GuidanceComparison#precisionRecall}).</li>
 * </ul>
 */
public final class SyntheticTraining {

    private static final int CLASSES = 10;
    private static final int SIDE = 8;
    private static final int PIXELS = SIDE * SIDE;

    private SyntheticTraining() {
    }

    // ------------------------------------------------------------------------------------------
    // shared: digits and a small classifier

    /**
     * Digits: a training pool and a test set with the same prototypes.
     *
     * @param perClass training images per class
     * @param noise pixel flip probability
     * @param seed the prototype seed
     * @return {@code {train, test}}
     */
    public static ToyTasks.Data[] digits(int perClass, double noise, long seed) {
        return new ToyTasks.Data[] {ToyTasks.digits(CLASSES, perClass, SIDE, noise, seed, seed * 31 + 7),
            ToyTasks.digits(CLASSES, 100, SIDE, noise, seed, seed * 31 + 1009)};
    }

    private static float[][] shift(float[][] x, SplittableRandom r) {
        float[][] out = new float[x.length][PIXELS];
        for (int i = 0; i < x.length; i++) {
            int dx = r.nextInt(3) - 1;
            int dy = r.nextInt(3) - 1;
            for (int y = 0; y < SIDE; y++) {
                for (int z = 0; z < SIDE; z++) {
                    out[i][y * SIDE + z] = x[i][Math.floorMod(y - dy, SIDE) * SIDE + Math.floorMod(z - dx, SIDE)];
                }
            }
        }
        return out;
    }

    /**
     * Trains a classifier with soft-target cross-entropy and scores it.
     *
     * @param net the network
     * @param x training rows
     * @param y labels
     * @param test test data
     * @param steps Adam steps
     * @param augment whether to add random shifts and mixup
     * @param seed the seed
     * @return test accuracy
     */
    static double trainAndTest(SimpleMlp net, float[][] x, int[] y, ToyTasks.Data test, int steps, boolean augment,
            long seed) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] bx = new float[batch][];
            float[] by = new float[batch * CLASSES];
            for (int b = 0; b < batch; b++) {
                int i = batch == x.length ? b : r.nextInt(x.length);
                bx[b] = x[i];
                by[b * CLASSES + y[i]] = 1;
            }
            Tensor tx = Tensor.of(augment ? shift(bx, r) : bx);
            Tensor ty = Tensor.of(Shape.of(batch, CLASSES), by);
            if (augment) {
                Tensor[] m = BatchMixing.mixup(tx, ty, 0.2f, r);
                tx = m[0];
                ty = m[1];
            }
            Var logits = net.forward(Var.of(tx));
            logits.logsumexp(1, false).sub(logits.mul(Var.of(ty)).sum(1, false)).mean().backward();
            opt.step(net.parameters());
        }
        return ToyTasks.accuracy(net.predict(test.inputs()), test.y());
    }

    private static SimpleMlp evalNet(long seed) {
        return SimpleMlp.relu(seed, PIXELS, 64, CLASSES);
    }

    private static SimpleMlp transferNet(long seed) {
        return SimpleMlp.tanh(seed, PIXELS, 128, 128, CLASSES);
    }

    // ------------------------------------------------------------------------------------------
    // 9.3.6 dataset condensation

    /** How the small training set is made. */
    public enum Condenser {
        /** Random real images of each class. */
        RANDOM,
        /** Per-class k-means centroids. */
        KMEANS,
        /** iCaRL herding: greedily match the class mean. */
        HERDING,
        /** Greedy k-center (farthest-first) core set. */
        KCENTER,
        /** Gradient matching. */
        GRADIENT_MATCHING,
        /** Distribution matching. */
        DISTRIBUTION_MATCHING,
        /** The whole training pool (the upper bound). */
        FULL
    }

    /**
     * One condensation result.
     *
     * @param method the method
     * @param perClass images per class
     * @param accuracy test accuracy of the evaluation MLP trained on the set (mean over trainings)
     * @param transfer test accuracy of the other architecture trained on it
     */
    public record Condensed(Condenser method, int perClass, double accuracy, double transfer) {
    }

    private static double[][] toDouble(List<float[]> rows) {
        double[][] out = new double[rows.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = new double[rows.get(i).length];
            for (int j = 0; j < out[i].length; j++) {
                out[i][j] = rows.get(i)[j];
            }
        }
        return out;
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return s;
    }

    /** Herding: picks rows whose running mean stays closest to the class mean. */
    static int[] herding(double[][] x, int k) {
        int d = x[0].length;
        double[] mu = new double[d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                mu[j] += row[j] / x.length;
            }
        }
        double[] sum = new double[d];
        boolean[] used = new boolean[x.length];
        int[] out = new int[k];
        for (int t = 0; t < k; t++) {
            int best = -1;
            double bestD = Double.MAX_VALUE;
            for (int i = 0; i < x.length; i++) {
                if (used[i]) {
                    continue;
                }
                double s = 0;
                for (int j = 0; j < d; j++) {
                    double m = (sum[j] + x[i][j]) / (t + 1) - mu[j];
                    s += m * m;
                }
                if (s < bestD) {
                    bestD = s;
                    best = i;
                }
            }
            used[best] = true;
            out[t] = best;
            for (int j = 0; j < d; j++) {
                sum[j] += x[best][j];
            }
        }
        return out;
    }

    /** Greedy k-center: starts at the row nearest the mean, then repeatedly takes the farthest row. */
    static int[] kCenter(double[][] x, int k) {
        int[] start = herding(x, 1);
        int[] out = new int[k];
        out[0] = start[0];
        double[] near = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            near[i] = dist2(x[i], x[out[0]]);
        }
        for (int t = 1; t < k; t++) {
            int far = 0;
            for (int i = 1; i < x.length; i++) {
                far = near[i] > near[far] ? i : far;
            }
            out[t] = far;
            for (int i = 0; i < x.length; i++) {
                near[i] = Math.min(near[i], dist2(x[i], x[far]));
            }
        }
        return out;
    }

    /**
     * Builds a condensed set.
     *
     * @param method the method
     * @param train the training pool
     * @param perClass images per class
     * @param iterations outer iterations of the matching methods
     * @param seed the seed
     * @return the set
     */
    public static GenerativeAugmenter.Labelled condense(Condenser method, ToyTasks.Data train, int perClass,
            int iterations, long seed) {
        if (method == Condenser.FULL) {
            return new GenerativeAugmenter.Labelled(train.x(), train.y());
        }
        if (method == Condenser.GRADIENT_MATCHING || method == Condenser.DISTRIBUTION_MATCHING) {
            DatasetCondenser.Mode mode = method == Condenser.GRADIENT_MATCHING ? DatasetCondenser.Mode.GRADIENT_MATCHING
                    : DatasetCondenser.Mode.DISTRIBUTION_MATCHING;
            DatasetCondenser dc = new DatasetCondenser(mode, train.x(), train.y(), CLASSES, perClass, SIDE, true,
                    method == Condenser.GRADIENT_MATCHING ? 0.05f : 0.05f, seed);
            for (int it = 0; it < iterations; it++) {
                dc.step(train.x(), train.y(), 32, s -> SimpleMlp.relu(s, PIXELS, 64, CLASSES), 3);
            }
            float[] v = dc.images().toArray();
            float[][] x = new float[CLASSES * perClass][];
            for (int i = 0; i < x.length; i++) {
                x[i] = Arrays.copyOfRange(v, i * PIXELS, (i + 1) * PIXELS);
            }
            return new GenerativeAugmenter.Labelled(x, dc.labels());
        }
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> xs = new ArrayList<>();
        List<Integer> ys = new ArrayList<>();
        for (int c = 0; c < CLASSES; c++) {
            List<float[]> rows = new ArrayList<>();
            for (int i = 0; i < train.size(); i++) {
                if (train.y()[i] == c) {
                    rows.add(train.x()[i]);
                }
            }
            double[][] d = toDouble(rows);
            if (method == Condenser.KMEANS) {
                for (double[] centre : KMeans.fit(d, perClass, KMeans.Init.PLUS_PLUS, 3, 50, seed + c).centres()) {
                    float[] f = new float[centre.length];
                    for (int j = 0; j < f.length; j++) {
                        f[j] = (float) centre[j];
                    }
                    xs.add(f);
                    ys.add(c);
                }
                continue;
            }
            int[] pick;
            if (method == Condenser.RANDOM) {
                pick = new int[perClass];
                List<Integer> idx = new ArrayList<>();
                for (int i = 0; i < rows.size(); i++) {
                    idx.add(i);
                }
                for (int k = 0; k < perClass; k++) {
                    pick[k] = idx.remove(r.nextInt(idx.size()));
                }
            } else {
                pick = method == Condenser.HERDING ? herding(d, perClass) : kCenter(d, perClass);
            }
            for (int i : pick) {
                xs.add(rows.get(i));
                ys.add(c);
            }
        }
        int[] y = new int[ys.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = ys.get(i);
        }
        return new GenerativeAugmenter.Labelled(xs.toArray(new float[0][]), y);
    }

    /**
     * Condenses and scores one method.
     *
     * @param method the method
     * @param perClass images per class
     * @param iterations outer iterations of the matching methods
     * @param trainings fresh networks trained per architecture
     * @param seed the seed
     * @return the result
     */
    public static Condensed condensation(Condenser method, int perClass, int iterations, int trainings, long seed) {
        ToyTasks.Data[] d = digits(100, 0.15, seed);
        GenerativeAugmenter.Labelled set = condense(method, d[0], perClass, iterations, seed + 1);
        int steps = method == Condenser.FULL ? 400 : 200;
        double a = 0;
        double t = 0;
        for (int k = 0; k < trainings; k++) {
            a += trainAndTest(evalNet(seed + 100 + k), set.x(), set.y(), d[1], steps, false, seed + 200 + k);
            t += trainAndTest(transferNet(seed + 300 + k), set.x(), set.y(), d[1], steps, false, seed + 400 + k);
        }
        return new Condensed(method, perClass, a / trainings, t / trainings);
    }

    // ------------------------------------------------------------------------------------------
    // 9.3.7 synthetic tables

    /** Where the training table comes from. */
    public enum TableSource {
        /** The real training table (the upper bound). */
        REAL,
        /** Gaussian copula with empirical marginals. */
        GAUSSIAN_COPULA,
        /** SMOTE-style interpolation between same-label neighbours. */
        SMOTE,
        /** TabDDPM. */
        TABDDPM,
        /** CTGAN. */
        CTGAN
    }

    /**
     * Train-on-synthetic, test-on-real scores and the privacy ratio.
     *
     * @param source the generator
     * @param accuracy test accuracy
     * @param rareRecall recall of the rare class
     * @param balancedAccuracy balanced accuracy
     * @param closestRecordRatio median closest-record distance of synthetic over holdout rows
     */
    public record TabularScore(TableSource source, double accuracy, double rareRecall, double balancedAccuracy,
            double closestRecordRatio) {
    }

    /**
     * A mixed table: five {@link TabularStressSuite} columns (three informative), the first
     * informative one binned into four categories, the label as a second categorical column, and
     * positives kept with probability 0.15.
     *
     * @param rows rows before subsampling positives
     * @param seed the seed (fixes the planted functions too)
     * @return the table, columns: 4 numeric; categorical {bin(4), label(2)}
     */
    public static MixedTable stressTable(int rows, long seed) {
        TabularStressSuite suite = new TabularStressSuite(3, 0.4, false, seed);
        var t = suite.table(rows, seed + 1);
        float[] x = t.features().toArray();
        float[] y = t.labels().toArray();
        int cols = suite.columns();
        SplittableRandom r = new SplittableRandom(seed + 2);
        List<double[]> num = new ArrayList<>();
        List<int[]> cat = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            int label = y[i] > 0.5 ? 1 : 0;
            if (label == 1 && r.nextDouble() > 0.15) {
                continue;
            }
            double[] n = new double[cols - 1];
            for (int j = 1; j < cols; j++) {
                n[j - 1] = x[i * cols + j];
            }
            int bin = Math.max(0, Math.min(3, (int) Math.floor((x[i * cols] + 2) / 1.0)));
            num.add(n);
            cat.add(new int[] {bin, label});
        }
        return new MixedTable(num.toArray(new double[0][]), cat.toArray(new int[0][]), new int[] {4, 2});
    }

    /** Gaussian copula: normal scores of empirical CDFs, their correlation, and back. */
    static MixedTable gaussianCopula(MixedTable t, int rows, long seed) {
        int n = t.rows();
        int dn = t.numericColumns();
        int dc = t.categoricalColumns();
        int d = dn + dc;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] z = new double[n][d];
        double[][] sorted = new double[dn][];
        double[][] cumulative = new double[dc][];
        for (int j = 0; j < dn; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = t.numeric()[i][j];
            }
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(col[a], col[b]));
            for (int rank = 0; rank < n; rank++) {
                z[order[rank]][j] = EvalStatistics.normalQuantile((rank + 0.5) / n);
            }
            sorted[j] = col.clone();
            Arrays.sort(sorted[j]);
        }
        for (int c = 0; c < dc; c++) {
            int k = t.categories()[c];
            double[] cum = new double[k + 1];
            for (int i = 0; i < n; i++) {
                cum[t.categorical()[i][c] + 1] += 1.0 / n;
            }
            for (int q = 1; q <= k; q++) {
                cum[q] += cum[q - 1];
            }
            cumulative[c] = cum;
            for (int i = 0; i < n; i++) {
                int q = t.categorical()[i][c];
                double u = cum[q] + r.nextDouble() * (cum[q + 1] - cum[q]);
                z[i][dn + c] = EvalStatistics.normalQuantile(Math.min(1 - 1e-9, Math.max(1e-9, u)));
            }
        }
        double[][] cov = new double[d][d];
        for (double[] row : z) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    cov[a][b] += row[a] * row[b] / n;
                }
            }
        }
        for (int a = 0; a < d; a++) {
            cov[a][a] += 1e-6;
        }
        double[][] l = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int b = 0; b <= a; b++) {
                double s = cov[a][b];
                for (int k = 0; k < b; k++) {
                    s -= l[a][k] * l[b][k];
                }
                l[a][b] = a == b ? Math.sqrt(Math.max(1e-12, s)) : s / l[b][b];
            }
        }
        double[][] num = new double[rows][dn];
        int[][] cat = new int[rows][dc];
        double[] e = new double[d];
        for (int i = 0; i < rows; i++) {
            for (int a = 0; a < d; a++) {
                e[a] = r.nextGaussian();
            }
            for (int a = 0; a < d; a++) {
                double v = 0;
                for (int b = 0; b <= a; b++) {
                    v += l[a][b] * e[b];
                }
                double u = EvalStatistics.normalCdf(v);
                if (a < dn) {
                    double pos = u * (n - 1);
                    int lo = (int) Math.floor(pos);
                    int hi = Math.min(n - 1, lo + 1);
                    num[i][a] = sorted[a][lo] + (pos - lo) * (sorted[a][hi] - sorted[a][lo]);
                } else {
                    double[] cum = cumulative[a - dn];
                    int q = 0;
                    while (q < cum.length - 2 && u >= cum[q + 1]) {
                        q++;
                    }
                    cat[i][a - dn] = q;
                }
            }
        }
        return new MixedTable(num, cat, t.categories().clone());
    }

    /** SMOTE-style rows: a random row and one of its five nearest neighbours with the same label. */
    static MixedTable smote(MixedTable t, int label, int rows, long seed) {
        double[][] enc = t.encode(t.numericStats(), label);
        int[] y = t.column(label);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] num = new double[rows][];
        int[][] cat = new int[rows][];
        int k = 5;
        for (int s = 0; s < rows; s++) {
            int i = r.nextInt(t.rows());
            int[] nn = new int[k];
            double[] nd = new double[k];
            Arrays.fill(nd, Double.MAX_VALUE);
            Arrays.fill(nn, i);
            for (int j = 0; j < t.rows(); j++) {
                if (j == i || y[j] != y[i]) {
                    continue;
                }
                double dd = dist2(enc[i], enc[j]);
                if (dd < nd[k - 1]) {
                    int p = k - 1;
                    while (p > 0 && nd[p - 1] > dd) {
                        nd[p] = nd[p - 1];
                        nn[p] = nn[p - 1];
                        p--;
                    }
                    nd[p] = dd;
                    nn[p] = j;
                }
            }
            int j = nn[r.nextInt(k)];
            double lam = r.nextDouble();
            num[s] = new double[t.numericColumns()];
            for (int c = 0; c < num[s].length; c++) {
                num[s][c] = t.numeric()[i][c] + lam * (t.numeric()[j][c] - t.numeric()[i][c]);
            }
            cat[s] = new int[t.categoricalColumns()];
            for (int c = 0; c < cat[s].length; c++) {
                cat[s][c] = (r.nextDouble() < lam ? t.categorical()[j] : t.categorical()[i])[c];
            }
        }
        return new MixedTable(num, cat, t.categories().clone());
    }

    /**
     * Generates a training table with one source.
     *
     * @param source the generator
     * @param train the real training table
     * @param rows rows to generate
     * @param iterations training steps of the learned generators
     * @param width hidden width of the learned generators
     * @param seed the seed
     * @return the table
     */
    public static MixedTable generate(TableSource source, MixedTable train, int rows, int iterations, int width,
            long seed) {
        int numeric = train.numericColumns();
        int[] cats = train.categories();
        return switch (source) {
            case REAL -> train;
            case GAUSSIAN_COPULA -> gaussianCopula(train, rows, seed);
            case SMOTE -> smote(train, 1, rows, seed);
            case TABDDPM -> {
                TabularDiffusion m = new TabularDiffusion(numeric, cats, 100, width, 2e-3f, seed);
                m.fit(train, iterations, 128, seed + 1);
                yield m.sample(rows, seed + 2);
            }
            case CTGAN -> {
                ConditionalTabularGan m = new ConditionalTabularGan(numeric, cats, 3, 16, width, 8);
                m.fit(train, iterations, 128, 1e-3f, seed + 1);
                yield m.sample(rows, seed + 2);
            }
        };
    }

    /**
     * Scores one source on the stress table: 2000 training rows, 500 holdout rows, the rest test.
     *
     * @param source the generator
     * @param iterations training steps of the learned generators
     * @param width hidden width of the learned generators
     * @param seed the seed
     * @return the scores
     */
    public static TabularScore tabular(TableSource source, int iterations, int width, long seed) {
        MixedTable all = stressTable(9000, seed);
        MixedTable train = all.rows(0, 2000);
        MixedTable holdout = all.rows(2000, 2500);
        MixedTable test = all.rows(2500, all.rows());
        double[][] stats = train.numericStats();
        MixedTable syn = generate(source, train, train.rows(), iterations, width, seed + 10);
        SyntheticDataAudit.Utility u = SyntheticDataAudit.utility(syn, test, 1, stats, 60);
        double ratio = SyntheticDataAudit.closestRecordRatio(syn, train, holdout, stats);
        return new TabularScore(source, u.accuracy(), u.rareRecall(), u.balancedAccuracy(), ratio);
    }

    // ------------------------------------------------------------------------------------------
    // 9.3.8 classifiers on generated images

    /** The training data of a digits classifier. */
    public enum Augmentation {
        /** Real images only. */
        REAL,
        /** Real images with random shifts and mixup. */
        SHIFT_MIXUP,
        /** Real plus per-class VAE samples, one to one. */
        VAE,
        /** Real plus guided diffusion samples, one to one. */
        DIFFUSION
    }

    /**
     * One augmentation result.
     *
     * @param method the training data
     * @param perClass real images per class
     * @param guidance the guidance weight (diffusion only)
     * @param accuracy test accuracy, mean over classifier seeds
     */
    public record Augmented(Augmentation method, int perClass, float guidance, double accuracy) {
    }

    /**
     * One point of the guidance sweep.
     *
     * @param guidance the weight
     * @param accuracy test accuracy of real plus samples
     * @param precision improved precision of the samples against real test images
     * @param recall improved recall
     */
    public record GuidancePoint(float guidance, double accuracy, double precision, double recall) {
    }

    private static final double DIGIT_NOISE = 0.2;
    private static final int VAE_STEPS = 300;

    private static GenerativeAugmenter generator(ToyTasks.Data train, int iterations, long seed) {
        GenerativeAugmenter g = new GenerativeAugmenter(PIXELS, CLASSES, 128, 2, 100, 2e-3f, seed);
        g.fit(train.x(), train.y(), iterations, 64, 0.2, seed + 1);
        return g;
    }

    private static GenerativeAugmenter.Labelled vaeSamples(ToyTasks.Data train, int perClass, int iterations,
            long seed) {
        float[][] x = new float[CLASSES * perClass][];
        int[] y = new int[x.length];
        for (int c = 0; c < CLASSES; c++) {
            List<float[]> rows = new ArrayList<>();
            for (int i = 0; i < train.size(); i++) {
                if (train.y()[i] == c) {
                    float[] v = new float[PIXELS];
                    for (int j = 0; j < PIXELS; j++) {
                        v[j] = 2 * train.x()[i][j] - 1;
                    }
                    rows.add(v);
                }
            }
            ImageAutoencoders.Vae vae = ImageAutoencoders.vae(8, 0.1f, seed + c);
            SplittableRandom r = new SplittableRandom(seed + 50 + c);
            for (int it = 0; it < iterations; it++) {
                float[][] b = new float[Math.min(32, rows.size())][];
                for (int k = 0; k < b.length; k++) {
                    b[k] = rows.get(r.nextInt(rows.size()));
                }
                vae.trainStep(b);
            }
            float[][] s = vae.sample(perClass, seed + 90 + c);
            for (int k = 0; k < perClass; k++) {
                float[] v = new float[PIXELS];
                for (int j = 0; j < PIXELS; j++) {
                    v[j] = (s[k][j] + 1) / 2;
                }
                x[c * perClass + k] = v;
                y[c * perClass + k] = c;
            }
        }
        return new GenerativeAugmenter.Labelled(x, y);
    }

    /**
     * Trains digits classifiers on one kind of training data.
     *
     * @param method the training data
     * @param perClass real images per class
     * @param guidance the guidance weight of the diffusion samples
     * @param generatorIterations training steps of the generator
     * @param classifiers classifier seeds to average
     * @param seed the seed
     * @return the result
     */
    public static Augmented augmentation(Augmentation method, int perClass, float guidance, int generatorIterations,
            int classifiers, long seed) {
        ToyTasks.Data[] d = digits(perClass, DIGIT_NOISE, seed);
        GenerativeAugmenter.Labelled real = new GenerativeAugmenter.Labelled(d[0].x(), d[0].y());
        GenerativeAugmenter.Labelled set = switch (method) {
            case REAL, SHIFT_MIXUP -> real;
            case VAE -> GenerativeAugmenter.mix(real, vaeSamples(d[0], perClass, VAE_STEPS, seed + 3), 0.5,
                    seed + 4);
            case DIFFUSION -> GenerativeAugmenter.mix(real,
                    generator(d[0], generatorIterations, seed + 3).samplePerClass(perClass, guidance, 25, seed + 5),
                    0.5, seed + 4);
        };
        double acc = 0;
        for (int k = 0; k < classifiers; k++) {
            acc += trainAndTest(evalNet(seed + 100 + k), set.x(), set.y(), d[1], 300,
                    method == Augmentation.SHIFT_MIXUP, seed + 200 + k);
        }
        return new Augmented(method, perClass, guidance, acc / classifiers);
    }

    /**
     * Trains one generator and sweeps the guidance weight: classifier accuracy on real plus
     * samples, and the samples' precision and recall against real test images.
     *
     * @param perClass real images per class
     * @param weights guidance weights
     * @param generatorIterations training steps of the generator
     * @param classifiers classifier seeds to average
     * @param seed the seed
     * @return one point per weight
     */
    public static List<GuidancePoint> guidanceSweep(int perClass, float[] weights, int generatorIterations,
            int classifiers, long seed) {
        ToyTasks.Data[] d = digits(perClass, DIGIT_NOISE, seed);
        GenerativeAugmenter g = generator(d[0], generatorIterations, seed + 3);
        GenerativeAugmenter.Labelled real = new GenerativeAugmenter.Labelled(d[0].x(), d[0].y());
        float[][] reference = Arrays.copyOf(d[1].x(), 500);
        List<GuidancePoint> out = new ArrayList<>();
        for (float w : weights) {
            GenerativeAugmenter.Labelled syn = g.samplePerClass(perClass, w, 25, seed + 5);
            GenerativeAugmenter.Labelled mixed = GenerativeAugmenter.mix(real, syn, 0.5, seed + 4);
            double acc = 0;
            for (int k = 0; k < classifiers; k++) {
                acc += trainAndTest(evalNet(seed + 100 + k), mixed.x(), mixed.y(), d[1], 300, false, seed + 200 + k);
            }
            GenerativeAugmenter.Labelled q = g.samplePerClass(50, w, 25, seed + 6);
            double[] pr = GuidanceComparison.precisionRecall(reference, q.x(), 3);
            out.add(new GuidancePoint(w, acc / classifiers, pr[0], pr[1]));
        }
        return out;
    }

    /**
     * Prints every comparison.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (int ipc : new int[] {1, 10}) {
            for (Condenser m : Condenser.values()) {
                Condensed c = condensation(m, ipc, 200, 3, 1);
                System.out.printf("condense ipc=%d %-22s acc=%.3f transfer=%.3f%n", ipc, m, c.accuracy(),
                        c.transfer());
            }
        }
        for (TableSource s : TableSource.values()) {
            TabularScore t = tabular(s, 1500, 64, 1);
            System.out.printf("table %-16s acc=%.3f rare=%.3f bal=%.3f dcr=%.2f%n", s, t.accuracy(), t.rareRecall(),
                    t.balancedAccuracy(), t.closestRecordRatio());
        }
        for (int ipc : new int[] {5, 20, 100}) {
            for (Augmentation a : Augmentation.values()) {
                Augmented r = augmentation(a, ipc, 1.5f, 2000, 3, 1);
                System.out.printf("augment ipc=%d %-12s acc=%.3f%n", ipc, a, r.accuracy());
            }
        }
        for (GuidancePoint p : guidanceSweep(5, new float[] {1f, 1.5f, 2f, 3f, 5f}, 2000, 3, 1)) {
            System.out.printf("guidance w=%.1f acc=%.3f precision=%.3f recall=%.3f%n", p.guidance(), p.accuracy(),
                    p.precision(), p.recall());
        }
    }
}
