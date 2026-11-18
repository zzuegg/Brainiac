package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.ml.RestrictedBoltzmannMachine;
import io.github.zzuegg.brainiac.experimental.ml.TsetlinMachine;
import io.github.zzuegg.brainiac.experimental.nn.DenoisingThermodynamicModel;
import io.github.zzuegg.brainiac.experimental.nn.HdcClassifier;
import io.github.zzuegg.brainiac.experimental.nn.LogicGateLayer;
import io.github.zzuegg.brainiac.experimental.tensor.Hypervector;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Classic learners that have no gradient-trained counterpart in the library (wave 11, items
 * 11.4.12 and 11.4.13).
 * <ul>
 *   <li>{@link #rbmDigits}: restricted Boltzmann machines trained with CD-1, persistent CD and
 *       parallel tempering on binarised {@link ToyTasks} digits, scored by exact test
 *       log-likelihood (16 hidden units, so {@code log Z} is enumerable), the AIS estimate's error
 *       against it, and a linear probe on the hidden features; baselines are independent pixels, a
 *       Bernoulli mixture fitted by EM (a local stand-in for 11.3.11's leaves) and a probe on raw
 *       pixels;</li>
 *   <li>{@link #rbmModes}: a planted bimodal distribution (a prototype and its complement), where
 *       samples and exact probability mass show which trainers keep both modes, against
 *       {@link DenoisingThermodynamicModel} and the mixture;</li>
 *   <li>{@link #tsetlinXor}: a noisy-XOR task with a known clause solution, checking that the
 *       Tsetlin machine's clauses are exactly the planted conjunctions;</li>
 *   <li>{@link #tsetlinDigits} and {@link #tsetlinTabular}: accuracy against the number of
 *       clauses for the standard and fuzzy-pattern machines, against logistic regression,
 *       {@link RandomForest}, an {@link HdcClassifier} and a {@link LogicGateLayer} network,
 *       on binarised digits and thermometer-encoded {@link TabularStressSuite} tables.</li>
 * </ul>
 */
public final class ClassicLearners {

    private ClassicLearners() {
    }

    /**
     * One RBM-table row.
     *
     * @param method the trainer or baseline
     * @param testLogLikelihood mean test log-likelihood in nats ({@code NaN} if not available)
     * @param aisError AIS {@code log Z} minus the exact one ({@code NaN} for baselines)
     * @param probe test accuracy of logistic regression on the features
     */
    public record RbmRow(String method, double testLogLikelihood, double aisError, double probe) {
    }

    /**
     * One mode-coverage row.
     *
     * @param method the trainer or baseline
     * @param shareA share of samples within the radius of the first mode
     * @param shareB share of samples within the radius of the second mode
     * @param exactBalance for RBMs, the exact {@code min / max} of the two modes' probability mass
     *     ({@code NaN} otherwise)
     */
    public record ModeRow(String method, double shareA, double shareB, double exactBalance) {

        /** @return the number of modes holding at least 10% of the samples */
        public int modesHit() {
            return (shareA >= 0.1 ? 1 : 0) + (shareB >= 0.1 ? 1 : 0);
        }
    }

    /**
     * One classifier row.
     *
     * @param method the classifier
     * @param accuracy test accuracy
     * @param clauses clauses per class (0 for non-Tsetlin methods)
     * @param millis training time
     * @param rulesRecovered planted conjunctions found exactly as clauses ({@code -1} if not applicable)
     */
    public record TsetlinRow(String method, double accuracy, int clauses, double millis, int rulesRecovered) {
    }

    /**
     * Binarised digits: {@link ToyTasks#digits} thresholded at 0.5.
     *
     * @param perClass rows per class
     * @param noise pixel flip probability
     * @param seed the prototype seed
     * @param sampleSeed the sample seed
     * @return 0/1 pixel rows with their labels
     */
    public static ClassicMlShootout.Points binaryDigits(int perClass, double noise, long seed, long sampleSeed) {
        ToyTasks.Data d = ToyTasks.digits(10, perClass, 8, noise, seed, sampleSeed);
        double[][] x = new double[d.size()][d.features()];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                x[i][j] = d.x()[i][j] > 0.5f ? 1 : 0;
            }
        }
        return new ClassicMlShootout.Points(x, d.y(), 10);
    }

    // ---------------------------------------------------------------- RBMs (11.4.12)

    private static RestrictedBoltzmannMachine.Settings settings(RestrictedBoltzmannMachine.Training t, int chains,
            long seed) {
        return new RestrictedBoltzmannMachine.Settings(t, 1, chains, 4, 0.05, 50, seed);
    }

    /**
     * RBM trainers on binarised digits.
     *
     * @param updates parameter updates per trainer
     * @param seed the seed
     * @return independent pixels, Bernoulli mixture, then CD-1, PCD and PT; probes in each row
     */
    public static List<RbmRow> rbmDigits(int updates, long seed) {
        ClassicMlShootout.Points train = binaryDigits(100, 0.05, seed, seed + 1);
        ClassicMlShootout.Points test = binaryDigits(40, 0.05, seed, seed + 2);
        List<RbmRow> rows = new ArrayList<>();
        double[] base = RestrictedBoltzmannMachine.marginalBiases(train.x());
        double independent = 0;
        for (double[] v : test.x()) {
            for (int i = 0; i < v.length; i++) {
                independent += v[i] * base[i] - Math.log1p(Math.exp(base[i]));
            }
        }
        rows.add(new RbmRow("independent", independent / test.size(), Double.NaN,
                probe(train.x(), train.y(), test.x(), test.y(), seed)));
        BernoulliMixture mix = BernoulliMixture.fit(train.x(), 10, 50, seed);
        double[][] respTrain = mix.responsibilities(train.x());
        double[][] respTest = mix.responsibilities(test.x());
        rows.add(new RbmRow("mixture10", mix.logLikelihood(test.x()), Double.NaN,
                probe(respTrain, train.y(), respTest, test.y(), seed)));
        for (RestrictedBoltzmannMachine.Training t : RestrictedBoltzmannMachine.Training.values()) {
            RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine.create(
                    RestrictedBoltzmannMachine.Visible.BERNOULLI, 64, 16, seed);
            rbm.train(train.x(), settings(t, 50, seed), updates);
            double logZ = rbm.exactLogPartition();
            double ais = rbm.aisLogPartition(base, 64, 500, seed + 3);
            rows.add(new RbmRow(label(t), rbm.logLikelihood(test.x(), logZ), ais - logZ,
                    probe(hidden(rbm, train.x()), train.y(), hidden(rbm, test.x()), test.y(), seed)));
        }
        return rows;
    }

    private static String label(RestrictedBoltzmannMachine.Training t) {
        return switch (t) {
            case CD_K -> "CD-1";
            case PCD -> "PCD";
            case PARALLEL_TEMPERING -> "PT";
        };
    }

    private static double[][] hidden(RestrictedBoltzmannMachine rbm, double[][] x) {
        double[][] h = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            h[i] = rbm.hiddenProbabilities(x[i]);
        }
        return h;
    }

    private static double probe(double[][] xtr, int[] ytr, double[][] xte, int[] yte, long seed) {
        ClassicMlShootout.Points tr = new ClassicMlShootout.Points(xtr, ytr, 10);
        Model logistic = Model.sequential().input(xtr[0].length).dense(10).seed(seed).build();
        ClassicMlShootout.trainClassifier(logistic, tr, 300, 0.05f);
        return MlMetrics.accuracy(ClassicMlShootout.predictClasses(logistic, xte), yte);
    }

    /**
     * Mode coverage on a planted bimodal distribution: a random 16-bit prototype and its
     * complement with equal weight, 5% bit flips; a sample "hits" a mode within Hamming distance 3.
     *
     * @param updates parameter updates per trainer
     * @param seed the seed
     * @return CD-1, PCD, PT, the denoising thermodynamic model (1 and 4 stages) and a 2-component
     *     mixture
     */
    public static List<ModeRow> rbmModes(int updates, long seed) {
        int nv = 16;
        SplittableRandom r = new SplittableRandom(seed);
        double[] a = new double[nv];
        for (int i = 0; i < nv; i++) {
            a[i] = r.nextBoolean() ? 1 : 0;
        }
        double[][] data = new double[600][nv];
        for (double[] row : data) {
            boolean flip = r.nextBoolean();
            for (int i = 0; i < nv; i++) {
                double bit = flip ? 1 - a[i] : a[i];
                row[i] = r.nextDouble() < 0.05 ? 1 - bit : bit;
            }
        }
        int samples = 300;
        List<ModeRow> rows = new ArrayList<>();
        for (RestrictedBoltzmannMachine.Training t : RestrictedBoltzmannMachine.Training.values()) {
            RestrictedBoltzmannMachine rbm = RestrictedBoltzmannMachine.create(
                    RestrictedBoltzmannMachine.Visible.BERNOULLI, nv, 8, seed);
            rbm.train(data, settings(t, 100, seed), updates);
            double[][] s = rbm.sample(samples, 100, seed + 5);
            double[] mass = exactMass(rbm, a);
            rows.add(modeRow(label(t), s, a, Math.min(mass[0], mass[1]) / Math.max(mass[0], mass[1])));
        }
        for (int stages : new int[] {1, 4}) {
            DenoisingThermodynamicModel dtm = DenoisingThermodynamicModel.of(nv, 8, stages, seed);
            SplittableRandom dr = new SplittableRandom(seed + 9);
            int[][] pm = new int[data.length][nv];
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < nv; j++) {
                    pm[i][j] = data[i][j] > 0.5 ? 1 : -1;
                }
            }
            for (int u = 0; u < updates; u++) {
                int[][] batch = new int[50][];
                for (int b = 0; b < batch.length; b++) {
                    batch[b] = pm[dr.nextInt(pm.length)];
                }
                dtm.train(batch, 1, 0.05, dr);
            }
            double[][] s = new double[samples][nv];
            for (int k = 0; k < samples; k++) {
                int[] x = dtm.sample(100 / stages, dr);
                for (int j = 0; j < nv; j++) {
                    s[k][j] = x[j] > 0 ? 1 : 0;
                }
            }
            rows.add(modeRow("DTM-" + stages, s, a, Double.NaN));
        }
        BernoulliMixture mix = BernoulliMixture.fit(data, 2, 50, seed);
        rows.add(modeRow("mixture2", mix.sample(samples, new SplittableRandom(seed + 11)), a, Double.NaN));
        return rows;
    }

    private static int hamming(double[] v, double[] a, boolean complement) {
        int d = 0;
        for (int i = 0; i < v.length; i++) {
            double t = complement ? 1 - a[i] : a[i];
            d += (v[i] > 0.5) != (t > 0.5) ? 1 : 0;
        }
        return d;
    }

    private static ModeRow modeRow(String method, double[][] s, double[] a, double balance) {
        int na = 0;
        int nb = 0;
        for (double[] v : s) {
            na += hamming(v, a, false) <= 3 ? 1 : 0;
            nb += hamming(v, a, true) <= 3 ? 1 : 0;
        }
        return new ModeRow(method, na / (double) s.length, nb / (double) s.length, balance);
    }

    /** Exact probability mass within Hamming distance 3 of the mode and of its complement. */
    private static double[] exactMass(RestrictedBoltzmannMachine rbm, double[] a) {
        int nv = a.length;
        double logZ = rbm.exactLogPartition();
        double[] mass = new double[2];
        double[] v = new double[nv];
        for (int s = 0; s < 1 << nv; s++) {
            for (int i = 0; i < nv; i++) {
                v[i] = s >> i & 1;
            }
            double p = Math.exp(-rbm.freeEnergy(v) - logZ);
            if (hamming(v, a, false) <= 3) {
                mass[0] += p;
            } else if (hamming(v, a, true) <= 3) {
                mass[1] += p;
            }
        }
        return mass;
    }

    /** A mixture of independent Bernoullis fitted by EM. */
    private static final class BernoulliMixture {
        private final double[] weight;
        private final double[][] mean;

        private BernoulliMixture(double[] weight, double[][] mean) {
            this.weight = weight;
            this.mean = mean;
        }

        static BernoulliMixture fit(double[][] x, int k, int iterations, long seed) {
            int d = x[0].length;
            SplittableRandom r = new SplittableRandom(seed);
            double[] w = new double[k];
            double[][] mu = new double[k][d];
            java.util.Arrays.fill(w, 1.0 / k);
            for (double[] m : mu) {
                for (int j = 0; j < d; j++) {
                    m[j] = 0.25 + 0.5 * r.nextDouble();
                }
            }
            BernoulliMixture model = new BernoulliMixture(w, mu);
            for (int it = 0; it < iterations; it++) {
                double[][] resp = model.responsibilities(x);
                for (int c = 0; c < k; c++) {
                    double n = 1e-9;
                    double[] s = new double[d];
                    for (int i = 0; i < x.length; i++) {
                        n += resp[i][c];
                        for (int j = 0; j < d; j++) {
                            s[j] += resp[i][c] * x[i][j];
                        }
                    }
                    w[c] = n / x.length;
                    for (int j = 0; j < d; j++) {
                        mu[c][j] = (s[j] + 1) / (n + 2);
                    }
                }
            }
            return model;
        }

        private double[] logJoint(double[] v) {
            double[] l = new double[weight.length];
            for (int c = 0; c < weight.length; c++) {
                double s = Math.log(weight[c]);
                for (int j = 0; j < v.length; j++) {
                    s += v[j] > 0.5 ? Math.log(mean[c][j]) : Math.log1p(-mean[c][j]);
                }
                l[c] = s;
            }
            return l;
        }

        private static double logSumExp(double[] l) {
            double m = Double.NEGATIVE_INFINITY;
            for (double v : l) {
                m = Math.max(m, v);
            }
            double s = 0;
            for (double v : l) {
                s += Math.exp(v - m);
            }
            return m + Math.log(s);
        }

        double[][] responsibilities(double[][] x) {
            double[][] out = new double[x.length][];
            for (int i = 0; i < x.length; i++) {
                double[] l = logJoint(x[i]);
                double z = logSumExp(l);
                for (int c = 0; c < l.length; c++) {
                    l[c] = Math.exp(l[c] - z);
                }
                out[i] = l;
            }
            return out;
        }

        double logLikelihood(double[][] x) {
            double s = 0;
            for (double[] v : x) {
                s += logSumExp(logJoint(v));
            }
            return s / x.length;
        }

        double[][] sample(int n, SplittableRandom r) {
            double[][] out = new double[n][mean[0].length];
            for (int i = 0; i < n; i++) {
                double u = r.nextDouble();
                int c = 0;
                while (c < weight.length - 1 && (u -= weight[c]) > 0) {
                    c++;
                }
                for (int j = 0; j < out[i].length; j++) {
                    out[i][j] = r.nextDouble() < mean[c][j] ? 1 : 0;
                }
            }
            return out;
        }
    }

    // ---------------------------------------------------------------- Tsetlin (11.4.13)

    private static boolean[][] bits(double[][] x) {
        boolean[][] out = new boolean[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new boolean[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = x[i][j] > 0.5;
            }
        }
        return out;
    }

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }

    /**
     * Noisy XOR: 8 random bits, label {@code x0 XOR x1}, 10% of training labels flipped, a clean
     * test set. The planted solution is {@code x0 & !x1}, {@code !x0 & x1} for class 1 and
     * {@code x0 & x1}, {@code !x0 & !x1} for class 0.
     *
     * @param seed the seed
     * @return the Tsetlin machine (rules recovered out of 4), logistic regression and a random forest
     */
    public static List<TsetlinRow> tsetlinXor(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int f = 8;
        double[][] xtr = new double[2000][f];
        int[] ytr = new int[xtr.length];
        double[][] xte = new double[1000][f];
        int[] yte = new int[xte.length];
        for (int i = 0; i < xtr.length; i++) {
            for (int j = 0; j < f; j++) {
                xtr[i][j] = r.nextBoolean() ? 1 : 0;
            }
            ytr[i] = ((int) xtr[i][0] ^ (int) xtr[i][1]) ^ (r.nextDouble() < 0.1 ? 1 : 0);
        }
        for (int i = 0; i < xte.length; i++) {
            for (int j = 0; j < f; j++) {
                xte[i][j] = r.nextBoolean() ? 1 : 0;
            }
            yte[i] = (int) xte[i][0] ^ (int) xte[i][1];
        }
        List<TsetlinRow> rows = new ArrayList<>();
        long t0 = System.nanoTime();
        TsetlinMachine tm = TsetlinMachine.create(f, 2, TsetlinMachine.Settings.standard(10, 15, 3.9, seed));
        tm.fit(bits(xtr), ytr, 60);
        double millis = ms(t0);
        int recovered = 0;
        // literal k is x_k, f + k is !x_k; class 1 rules then class 0 rules
        int[][][] planted = {{{0, f + 1}, {f, 1}}, {{0, 1}, {f, f + 1}}};
        for (int cls = 0; cls < 2; cls++) {
            for (int[] rule : planted[1 - cls]) {
                if (hasClause(tm, cls, rule)) {
                    recovered++;
                }
            }
        }
        rows.add(new TsetlinRow("tsetlin", tm.accuracy(bits(xte), yte), tm.clauses(), millis, recovered));
        rows.addAll(baselines(xtr, ytr, xte, yte, 2, seed, false));
        return rows;
    }

    private static boolean hasClause(TsetlinMachine tm, int cls, int[] rule) {
        Set<Integer> want = new HashSet<>();
        for (int k : rule) {
            want.add(k);
        }
        for (int j = 0; j < tm.clauses(); j += 2) {
            Set<Integer> got = new HashSet<>();
            for (int k : tm.included(cls, j)) {
                got.add(k);
            }
            if (got.equals(want)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Binarised digits: standard machines with 10, 40 and 100 clauses per class and fuzzy-pattern
     * machines with 4 and 10, against the baselines.
     *
     * @param epochs Tsetlin training epochs
     * @param seed the seed
     * @return the rows
     */
    public static List<TsetlinRow> tsetlinDigits(int epochs, long seed) {
        ClassicMlShootout.Points train = binaryDigits(100, 0.1, seed, seed + 1);
        ClassicMlShootout.Points test = binaryDigits(50, 0.1, seed, seed + 2);
        List<TsetlinRow> rows = new ArrayList<>();
        boolean[][] btr = bits(train.x());
        boolean[][] bte = bits(test.x());
        for (int clauses : new int[] {10, 40, 100}) {
            rows.add(tsetlin("tsetlin", btr, train.y(), bte, test.y(), 10,
                    TsetlinMachine.Settings.standard(clauses, Math.max(5, clauses / 4), 5, seed), epochs));
        }
        for (int clauses : new int[] {4, 10}) {
            rows.add(tsetlin("fuzzy", btr, train.y(), bte, test.y(), 10,
                    TsetlinMachine.Settings.fuzzy(clauses, clauses * 4, 25, 32, 16, seed), epochs));
        }
        rows.addAll(baselines(train.x(), train.y(), test.x(), test.y(), 10, seed, true));
        return rows;
    }

    /**
     * A thermometer-encoded {@link TabularStressSuite} task (4 informative columns of 8, 8 levels
     * per column).
     *
     * @param epochs Tsetlin training epochs
     * @param seed the seed
     * @return standard and fuzzy machines against the baselines
     */
    public static List<TsetlinRow> tsetlinTabular(int epochs, long seed) {
        TabularStressSuite suite = new TabularStressSuite(4, 0.5, false, seed);
        double[][][] tr = table(suite.table(1500, seed + 1));
        double[][][] te = table(suite.table(1000, seed + 2));
        int[] ytr = labels(tr[1]);
        int[] yte = labels(te[1]);
        TsetlinMachine.Thermometer th = TsetlinMachine.Thermometer.fit(tr[0], 8);
        boolean[][] btr = th.encode(tr[0]);
        boolean[][] bte = th.encode(te[0]);
        List<TsetlinRow> rows = new ArrayList<>();
        rows.add(tsetlin("tsetlin", btr, ytr, bte, yte, 2, TsetlinMachine.Settings.standard(40, 10, 4, seed),
                epochs));
        rows.add(tsetlin("fuzzy", btr, ytr, bte, yte, 2, TsetlinMachine.Settings.fuzzy(4, 8, 16, 16, 8, seed),
                epochs));
        rows.addAll(baselines(tr[0], ytr, te[0], yte, 2, seed, false));
        return rows;
    }

    private static double[][][] table(SyntheticRecsys.Table t) {
        int n = t.features().shape().dim(0);
        int d = t.features().shape().dim(1);
        float[] f = t.features().toArray();
        float[] l = t.labels().toArray();
        double[][] x = new double[n][d];
        double[][] y = new double[n][1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                x[i][j] = f[i * d + j];
            }
            y[i][0] = l[i];
        }
        return new double[][][] {x, y};
    }

    private static int[] labels(double[][] y) {
        int[] out = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i][0] > 0.5 ? 1 : 0;
        }
        return out;
    }

    private static TsetlinRow tsetlin(String name, boolean[][] xtr, int[] ytr, boolean[][] xte, int[] yte,
            int classes, TsetlinMachine.Settings settings, int epochs) {
        long t0 = System.nanoTime();
        TsetlinMachine tm = TsetlinMachine.create(xtr[0].length, classes, settings);
        tm.fit(xtr, ytr, epochs);
        double millis = ms(t0);
        return new TsetlinRow(name, tm.accuracy(xte, yte), settings.clauses(), millis, -1);
    }

    private static List<TsetlinRow> baselines(double[][] xtr, int[] ytr, double[][] xte, int[] yte, int classes,
            long seed, boolean binaryImages) {
        List<TsetlinRow> rows = new ArrayList<>();
        long t0 = System.nanoTime();
        Model logistic = Model.sequential().input(xtr[0].length).dense(classes).seed(seed).build();
        ClassicMlShootout.trainClassifier(logistic, new ClassicMlShootout.Points(xtr, ytr, classes), 300, 0.05f);
        rows.add(new TsetlinRow("logistic", MlMetrics.accuracy(ClassicMlShootout.predictClasses(logistic, xte), yte),
                0, ms(t0), -1));
        t0 = System.nanoTime();
        RandomForest rf = RandomForest.fit(xtr, ytr, classes,
                RandomForest.Settings.defaults(RandomForest.Mode.RANDOM_FOREST).withTrees(100));
        rows.add(new TsetlinRow("random forest", MlMetrics.accuracy(rf.predict(xte), yte), 0, ms(t0), -1));
        if (binaryImages) {
            t0 = System.nanoTime();
            rows.add(new TsetlinRow("HDC", hdc(xtr, ytr, xte, yte, classes, seed), 0, ms(t0), -1));
            t0 = System.nanoTime();
            rows.add(new TsetlinRow("logic gates", logicGates(xtr, ytr, xte, yte, classes, seed), 0, ms(t0), -1));
        }
        return rows;
    }

    /** Pixel-position hypervectors bound to on/off level vectors, bundled, with retraining. */
    private static double hdc(double[][] xtr, int[] ytr, double[][] xte, int[] yte, int classes, long seed) {
        int dims = 4096;
        SplittableRandom r = new SplittableRandom(seed);
        int d = xtr[0].length;
        Hypervector[] pos = new Hypervector[d];
        for (int j = 0; j < d; j++) {
            pos[j] = Hypervector.random(dims, r);
        }
        Hypervector on = Hypervector.random(dims, r);
        Hypervector off = Hypervector.random(dims, r);
        HdcClassifier clf = new HdcClassifier(classes, dims);
        Hypervector[] etr = new Hypervector[xtr.length];
        for (int i = 0; i < xtr.length; i++) {
            etr[i] = encode(xtr[i], pos, on, off);
            clf.add(etr[i], ytr[i]);
        }
        for (int epoch = 0; epoch < 5; epoch++) {
            for (int i = 0; i < etr.length; i++) {
                clf.retrain(etr[i], ytr[i]);
            }
        }
        int ok = 0;
        for (int i = 0; i < xte.length; i++) {
            ok += clf.predict(encode(xte[i], pos, on, off)) == yte[i] ? 1 : 0;
        }
        return ok / (double) xte.length;
    }

    private static Hypervector encode(double[] x, Hypervector[] pos, Hypervector on, Hypervector off) {
        List<Hypervector> parts = new ArrayList<>(x.length);
        for (int j = 0; j < x.length; j++) {
            parts.add(pos[j].bind(x[j] > 0.5 ? on : off));
        }
        return Hypervector.bundle(parts);
    }

    /** A two-layer differentiable logic gate network with group-sum class scores (soft gates). */
    private static double logicGates(double[][] xtr, int[] ytr, double[][] xte, int[] yte, int classes, long seed) {
        int d = xtr[0].length;
        int width = 30 * classes;
        LogicGateLayer l1 = LogicGateLayer.of(d, 600, seed);
        LogicGateLayer l2 = LogicGateLayer.of(600, width, seed + 1);
        List<Var> params = new ArrayList<>(l1.parameters());
        params.addAll(l2.parameters());
        Optimizer opt = Optimizer.adam(0.05f);
        Tensor x = tensor(xtr);
        float[] ids = new float[ytr.length];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = ytr[i];
        }
        Tensor y = Tensor.of(Shape.of(ids.length), ids);
        for (int s = 0; s < 80; s++) {
            Var out = l2.forward(l1.forward(Var.of(x)));
            Var scores = out.reshape(xtr.length, classes, width / classes).sum(2, false).div(5f);
            Var loss = Functional.crossEntropy(scores, y);
            loss.backward();
            opt.step(params);
        }
        float[] p = l2.forward(l1.forward(Var.of(tensor(xte)))).value().toArray();
        int per = width / classes;
        int ok = 0;
        for (int i = 0; i < xte.length; i++) {
            int best = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                double s = 0;
                for (int k = 0; k < per; k++) {
                    s += p[i * width + c * per + k];
                }
                if (s > bestScore) {
                    bestScore = s;
                    best = c;
                }
            }
            ok += best == yte[i] ? 1 : 0;
        }
        return ok / (double) xte.length;
    }

    private static Tensor tensor(double[][] x) {
        float[] v = new float[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                v[i * x[0].length + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, x[0].length), v);
    }
}
