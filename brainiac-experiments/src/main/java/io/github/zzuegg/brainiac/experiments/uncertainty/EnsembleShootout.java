package io.github.zzuegg.brainiac.experiments.uncertainty;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.experimental.calibrate.ConformalTraining;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ensemble.DiverseEnsembles;
import io.github.zzuegg.brainiac.experimental.ensemble.EnsembleAudit;
import io.github.zzuegg.brainiac.experimental.ensemble.EnsembleDistillation;
import io.github.zzuegg.brainiac.experimental.ensemble.FastGeometricEnsemble;
import io.github.zzuegg.brainiac.experimental.ensemble.MimoNetwork;
import io.github.zzuegg.brainiac.experimental.ensemble.PackedEnsemble;
import io.github.zzuegg.brainiac.experimental.ensemble.SnapshotEnsemble;
import io.github.zzuegg.brainiac.experimental.ensemble.SuperLearner;
import io.github.zzuegg.brainiac.experimental.ensemble.TestTimeAugmentation;
import io.github.zzuegg.brainiac.experimental.eval.EvalStatistics;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.nn.DistanceAwareHead;
import io.github.zzuegg.brainiac.experimental.nn.EvidentialHead;
import io.github.zzuegg.brainiac.experimental.nn.TabM;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.ood.OodMetrics;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experimental.train.GroupRobustTraining;
import io.github.zzuegg.brainiac.experimental.train.WeightPosterior;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness;
import io.github.zzuegg.brainiac.experiments.shift.ShiftHarness;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Cheap, diverse and distilled ensembles against deep ensembles (wave 11 items 11.4.6–11.4.10,
 * 11.4.14 and the audit 11.4.16), on {@link ShiftSuites#ood}'s digits (six in-distribution
 * classes, near-OOD held-out digits, far-OOD noise and stripes) unless stated otherwise:
 *
 * <ul>
 *   <li>{@link #singleRun}: snapshot ensembles and FGE against a single model, deep ensembles,
 *       MC dropout and SWAG at equal training steps;</li>
 *   <li>{@link #insideOneNetwork}: MIMO and packed ensembles against TabM (BatchEnsemble), a
 *       deep ensemble and a single model, with multiply-adds per input and member
 *       disagreement;</li>
 *   <li>{@link #stacking}: the super learner against its library members, the uniform average
 *       and the discrete super learner on {@code TabularStressSuite} regression and
 *       classification;</li>
 *   <li>{@link #testTimeAugmentation}: TTA aggregations against single-pass prediction, MC
 *       dropout and a snapshot ensemble at the same number of passes, after temperature
 *       scaling, on clean and noisy digits;</li>
 *   <li>{@link #diversity}: NCL, DivDis and D-BAT against ERM, a deep ensemble and JTT on
 *       coloured shapes whose colour agrees with the class, plus a no-shortcut control;</li>
 *   <li>{@link #distillation}: mean distillation, EnD² and Hydra against the teacher ensemble,
 *       a single model and an evidential head;</li>
 *   <li>{@link #audit}: the ensemble zoo under label noise and imbalance, with paired standard
 *       errors against the deep ensemble.</li>
 * </ul>
 */
public final class EnsembleShootout {

    /**
     * One method.
     *
     * @param method the method
     * @param accuracy test accuracy
     * @param nll test NLL
     * @param ece test ECE (15 bins)
     * @param nearAuroc near-OOD AUROC by mutual information (entropy for single models)
     * @param farAuroc far-OOD AUROC
     * @param disagreement member disagreement on the test set
     * @param cost multiply-adds per input at inference (0 when not tracked)
     */
    public record Row(String method, double accuracy, double nll, double ece, double nearAuroc, double farAuroc,
            double disagreement, double cost) {
    }

    private static final int[] HIDDEN = {64, 64};

    private EnsembleShootout() {
    }

    // ------------------------------------------------------------------ shared helpers

    /**
     * Minibatch training with any batch loss.
     *
     * @param params the parameters
     * @param data the rows
     * @param steps the steps
     * @param opt the optimizer
     * @param seed the batch seed
     * @param loss maps a batch to its loss
     * @param listener called after every step, or null
     */
    static void train(List<Var> params, ShiftSuites.Split data, int steps, Optimizer opt, long seed,
            Function<ShiftSuites.Split, Var> loss, TrainingListener listener) {
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Var l = loss.apply(ShiftHarness.batch(data, 64, r));
            l.backward();
            opt.step(params);
            if (listener != null) {
                listener.onBatch(0, s, l.value().item());
            }
        }
    }

    static FeatureClassifier trained(ShiftSuites.Split data, int classes, int steps, long seed) {
        FeatureClassifier m = ShiftHarness.mlp(data.x().shape().dim(1), classes, seed);
        ShiftHarness.train(m, data, steps, 3e-3f, seed);
        return m;
    }

    static Model dropoutMlp(int inputs, int classes, long seed) {
        return Model.sequential().input(inputs).dense(64, Activation.RELU).dropout(0.2f).dense(64, Activation.RELU)
                .dropout(0.2f).dense(classes).seed(seed).build();
    }

    private static Row row(String name, List<EnsembleAudit.Member> members, ShiftSuites.OodSuite suite, double cost) {
        double[][][] p = EnsembleAudit.predict(members, suite.test().x());
        EnsembleAudit.Report rep = EnsembleAudit.evaluate(p, suite.test().y());
        EnsembleAudit.Score score = members.size() > 1 ? EnsembleAudit.Score.MUTUAL_INFORMATION
                : EnsembleAudit.Score.ENTROPY;
        double[] in = EnsembleAudit.uncertainty(p, score);
        double near = OodMetrics.auroc(EnsembleAudit.uncertainty(EnsembleAudit.predict(members, suite.near()), score),
                in);
        double far = OodMetrics.auroc(EnsembleAudit.uncertainty(EnsembleAudit.predict(members, suite.far()), score),
                in);
        return new Row(name, rep.accuracy(), rep.nll(), rep.ece(), near, far, rep.disagreement(), cost);
    }

    private static long mlpCost(int inputs, int[] hidden, int classes) {
        long s = 0;
        int prev = inputs;
        for (int h : hidden) {
            s += (long) prev * h;
            prev = h;
        }
        return s + (long) prev * classes;
    }

    /**
     * The digits OOD suite of the ensemble experiments (1,200 training rows, 400 per test set).
     *
     * @param seed the seed
     * @return the suite
     */
    public static ShiftSuites.OodSuite suite(long seed) {
        return ShiftSuites.ood(ShiftSuites.Family.DIGITS, 1200, 400, seed);
    }

    // ------------------------------------------------------------------ 11.4.6

    /**
     * Ensembles from one run against the classic baselines at an equal number of training steps.
     *
     * @param budget the training steps of every method
     * @param members the ensemble size
     * @param seed the seed
     * @return single, deep ensemble, MC dropout, SWAG, snapshot (uniform and validation-weighted)
     *     and FGE rows
     */
    public static List<Row> singleRun(int budget, int members, long seed) {
        ShiftSuites.OodSuite s = suite(seed);
        int k = s.classes();
        long cost = mlpCost(64, HIDDEN, k);
        List<Row> rows = new ArrayList<>();
        rows.add(row("single", List.of(EnsembleAudit.of(trained(s.train(), k, budget, seed))), s, cost));
        List<EnsembleAudit.Member> deep = new ArrayList<>();
        for (int m = 0; m < members; m++) {
            deep.add(EnsembleAudit.of(trained(s.train(), k, budget / members, seed + 100 + m)));
        }
        rows.add(row("deep ensemble", deep, s, cost * members));
        Model drop = dropoutMlp(64, k, seed);
        Optimizer od = Optimizer.adam(3e-3f);
        train(drop.parameters(), s.train(), budget, od, seed, b -> Functional.crossEntropy(drop.forward(Var.of(b.x()),
                true), b.ids()), null);
        rows.add(row("mc dropout", EnsembleAudit.mcDropout(drop, members), s, cost * members));
        // SWAG: collect every 10 steps over the last quarter
        FeatureClassifier sw = ShiftHarness.mlp(64, k, seed + 7);
        WeightPosterior.Swag swag = new WeightPosterior.Swag(sw.parameters(), 10, 10);
        Optimizer os = Optimizer.adam(3e-3f);
        int start = budget * 3 / 4;
        train(sw.parameters(), s.train(), budget, os, seed + 7, b -> Functional.crossEntropy(sw.logits(Var.of(b.x()),
                true), b.ids()), new TrainingListener() {
                    private long n;

                    @Override
                    public void onBatch(int epoch, long step, float loss) {
                        if (++n > start && n % 10 == 0) {
                            swag.collect();
                        }
                    }

                    @Override
                    public void onEpoch(io.github.zzuegg.brainiac.train.EpochResult result) {
                    }
                });
        rows.add(row("swag", EnsembleAudit.ofWeights(sw.parameters(), swag.sample(members, new SplittableRandom(seed)),
                sw::logits), s, cost * members));
        // snapshot ensemble
        FeatureClassifier sn = ShiftHarness.mlp(64, k, seed + 8);
        Optimizer on = Optimizer.adam(3e-3f);
        SnapshotEnsemble snap = new SnapshotEnsemble(on, sn.parameters(), budget / members, 5e-3f);
        train(sn.parameters(), s.train(), budget, on, seed + 8, b -> Functional.crossEntropy(sn.logits(Var.of(b.x()),
                true), b.ids()), snap);
        List<EnsembleAudit.Member> snapMembers = EnsembleAudit.ofWeights(sn.parameters(), snap.snapshots(), sn::logits);
        rows.add(row("snapshot", snapMembers, s, cost * members));
        double[] w = SnapshotEnsemble.validationWeights(EnsembleAudit.predict(snapMembers, s.validation().x()),
                s.validation().y());
        List<EnsembleAudit.Member> weighted = List.of(x -> SnapshotEnsemble.weightedMean(
                EnsembleAudit.predict(snapMembers, x), w));
        rows.add(row("snapshot weighted", weighted, s, cost * members));
        // FGE: pretrain three quarters, then collect along cycles
        FeatureClassifier fg = ShiftHarness.mlp(64, k, seed + 9);
        Optimizer of = Optimizer.adam(3e-3f);
        int pre = budget * 3 / 4;
        train(fg.parameters(), s.train(), pre, of, seed + 9, b -> Functional.crossEntropy(fg.logits(Var.of(b.x()), true),
                b.ids()), null);
        int cycle = Math.max(2, ((budget - pre) / members) / 2 * 2);
        FastGeometricEnsemble fge = new FastGeometricEnsemble(of, fg.parameters(), cycle, 3e-3f, 3e-4f);
        train(fg.parameters(), s.train(), cycle * members, of, seed + 10, b -> Functional.crossEntropy(
                fg.logits(Var.of(b.x()), true), b.ids()), fge);
        rows.add(row("fge", EnsembleAudit.ofWeights(fg.parameters(), fge.members(), fg::logits), s, cost * members));
        return rows;
    }

    // ------------------------------------------------------------------ 11.4.7

    /**
     * Ensembles inside one network against BatchEnsemble (TabM), a deep ensemble and a single
     * model; every network is trained for {@code steps}, the deep ensemble's members too.
     *
     * @param steps training steps per network
     * @param seed the seed
     * @return single, MIMO (M = 3, width 128), packed (M = 4, α = 2), TabM (4 members) and deep
     *     ensemble (4) rows; {@code cost} is multiply-adds per input
     */
    public static List<Row> insideOneNetwork(int steps, long seed) {
        ShiftSuites.OodSuite s = suite(seed);
        int k = s.classes();
        List<Row> rows = new ArrayList<>();
        rows.add(row("single", List.of(EnsembleAudit.of(trained(s.train(), k, steps, seed))), s,
                mlpCost(64, HIDDEN, k)));
        MimoNetwork mimo = new MimoNetwork(64, new int[] {128, 128}, k, 3, seed);
        Optimizer om = Optimizer.adam(3e-3f);
        SplittableRandom rm = new SplittableRandom(seed);
        train(mimo.parameters(), s.train(), steps, om, seed, b -> mimo.loss(b.x(), b.ids(), 0.0, 1, rm), null);
        rows.add(row("mimo", mimo.auditMembers(), s, mlpCost(64 * 3, new int[] {128, 128}, 3 * k)));
        PackedEnsemble packed = new PackedEnsemble(64, HIDDEN, k, 4, 2.0, 1, seed);
        Optimizer op = Optimizer.adam(3e-3f);
        train(packed.parameters(), s.train(), steps, op, seed, b -> memberLoss(packed.members(Var.of(b.x()), true),
                b.ids()), null);
        rows.add(row("packed", packed.auditMembers(), s, packed.multiplyAdds()));
        TabM tabm = new TabM(64, 64, 2, 4, k, null, new SplittableRandom(seed));
        Optimizer ot = Optimizer.adam(3e-3f);
        train(tabm.parameters(), s.train(), steps, ot, seed, b -> memberLoss(tabm.members(Var.of(b.x()), true),
                b.ids()), null);
        rows.add(row("tabm (batchensemble)", EnsembleAudit.of(tabm), s, 4 * mlpCost(64, HIDDEN, k)));
        List<EnsembleAudit.Member> deep = new ArrayList<>();
        for (int m = 0; m < 4; m++) {
            deep.add(EnsembleAudit.of(trained(s.train(), k, steps, seed + 100 + m)));
        }
        rows.add(row("deep ensemble", deep, s, 4 * mlpCost(64, HIDDEN, k)));
        return rows;
    }

    /** Mean cross-entropy over members of {@code [n, M, k]} logits. */
    static Var memberLoss(Var z, Tensor ids) {
        int n = z.shape().dim(0);
        int m = z.shape().dim(1);
        int k = z.shape().dim(2);
        return Functional.crossEntropy(z.reshape(n * m, k), ids.reshape(n, 1).expand(Shape.of(n, m)).reshape(n * m));
    }

    // ------------------------------------------------------------------ 11.4.8

    /**
     * One stacking row.
     *
     * @param task the task
     * @param method the learner or combination
     * @param loss test MSE (regression) or log loss (classification)
     */
    public record StackRow(String task, String method, double loss) {
    }

    /**
     * The super learner against its library on {@code TabularStressSuite} tasks.
     *
     * @param rows training rows
     * @param seed the seed
     * @return one row per member and combination per task (regression and classification)
     */
    public static List<StackRow> stacking(int rows, long seed) {
        List<StackRow> out = new ArrayList<>();
        io.github.zzuegg.brainiac.experimental.data.TabularStressSuite suite =
                new io.github.zzuegg.brainiac.experimental.data.TabularStressSuite(4, 0.5, false, seed);
        for (SuperLearner.Mode mode : SuperLearner.Mode.values()) {
            ShiftSuites.Split tr = UncertaintyFixture.table(suite.table(rows, seed + 1));
            ShiftSuites.Split te = UncertaintyFixture.table(suite.table(1000, seed + 2));
            double[][] x = ShiftHarness.rows(tr.x());
            double[][] tx = ShiftHarness.rows(te.x());
            double[] y = targets(mode, suite, x, tr.y(), seed + 3);
            double[] ty = targets(mode, suite, tx, te.y(), seed + 4);
            Map<String, SuperLearner.Learner> lib = library(mode, seed);
            List<SuperLearner.Learner> learners = new ArrayList<>(lib.values());
            double[][][] oof = SuperLearner.outOfFold(learners, x, y, 3, seed);
            String task = mode == SuperLearner.Mode.REGRESSION ? "regression" : "classification";
            for (Map.Entry<String, SuperLearner.Learner> e : lib.entrySet()) {
                SuperLearner.Predictor p = e.getValue().fit(x, y);
                out.add(new StackRow(task, e.getKey(), loss(mode, p::predict, tx, ty)));
            }
            for (SuperLearner.Meta meta : SuperLearner.Meta.values()) {
                SuperLearner sl = SuperLearner.fromOutOfFold(mode, meta, learners, oof, x, y, 2);
                out.add(new StackRow(task, "super learner " + meta.name().toLowerCase(java.util.Locale.ROOT),
                        loss(mode, sl::predict, tx, ty)));
            }
        }
        return out;
    }

    private static double[] targets(SuperLearner.Mode mode, io.github.zzuegg.brainiac.experimental.data
            .TabularStressSuite suite, double[][] x, int[] labels, long seed) {
        double[] y = new double[x.length];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < x.length; i++) {
            if (mode == SuperLearner.Mode.CLASSIFICATION) {
                y[i] = labels[i];
            } else {
                double s = 0.5 * x[i][0] * x[i][1];
                for (int c = 0; c < suite.informative(); c++) {
                    s += suite.effect(c, x[i][c]);
                }
                y[i] = s + 0.3 * r.nextGaussian();
            }
        }
        return y;
    }

    private static double loss(SuperLearner.Mode mode, UnaryOperator<double[]> p, double[][] x, double[] y) {
        double s = 0;
        for (int i = 0; i < x.length; i++) {
            double[] o = p.apply(x[i]);
            if (mode == SuperLearner.Mode.REGRESSION) {
                s += (o[0] - y[i]) * (o[0] - y[i]);
            } else {
                s -= Math.log(Math.max(1e-6, o[(int) y[i]]));
            }
        }
        return s / x.length;
    }

    /** The library: boosted trees, a random forest, k-NN, ridge and an MLP. */
    static Map<String, SuperLearner.Learner> library(SuperLearner.Mode mode, long seed) {
        boolean reg = mode == SuperLearner.Mode.REGRESSION;
        Map<String, SuperLearner.Learner> lib = new LinkedHashMap<>();
        lib.put("gbt", (x, y) -> {
            GradientBoostedTrees g = GradientBoostedTrees.fit(x, y, reg ? GradientBoostedTrees.Objective.REGRESSION
                    : GradientBoostedTrees.Objective.BINARY, GradientBoostedTrees.Settings.defaults().withTrees(50));
            return v -> {
                double p = g.predict(v);
                return reg ? new double[] {p} : clip2(p);
            };
        });
        lib.put("random forest", (x, y) -> {
            if (reg) {
                // RandomForest classifies only: bag single depth-8 regression trees (one-tree
                // GBT with shrinkage 1) instead
                List<GradientBoostedTrees> trees = new ArrayList<>();
                SplittableRandom r = new SplittableRandom(seed);
                for (int t = 0; t < 15; t++) {
                    double[][] bx = new double[x.length][];
                    double[] by = new double[x.length];
                    for (int i = 0; i < x.length; i++) {
                        int j = r.nextInt(x.length);
                        bx[i] = x[j];
                        by[i] = y[j];
                    }
                    trees.add(GradientBoostedTrees.fit(bx, by, GradientBoostedTrees.Objective.REGRESSION,
                            new GradientBoostedTrees.Settings(1, 8, 1.0, 0, 1e-3, 1, 32, seed + t)));
                }
                return v -> {
                    double s = 0;
                    for (GradientBoostedTrees g : trees) {
                        s += g.predict(v);
                    }
                    return new double[] {s / trees.size()};
                };
            }
            int[] labels = new int[y.length];
            for (int i = 0; i < y.length; i++) {
                labels[i] = (int) y[i];
            }
            RandomForest f = RandomForest.fit(x, labels, 2, new RandomForest.Settings(RandomForest.Mode.RANDOM_FOREST,
                    40, 10, 3, 0, seed));
            return v -> {
                double[] p = f.predictProbability(v);
                return new double[] {Math.max(1e-3, p[0]), Math.max(1e-3, p[1])};
            };
        });
        lib.put("knn", (x, y) -> v -> {
            int kk = 15;
            double[] d = new double[x.length];
            Integer[] o = new Integer[x.length];
            for (int i = 0; i < x.length; i++) {
                double s = 0;
                for (int c = 0; c < v.length; c++) {
                    s += (x[i][c] - v[c]) * (x[i][c] - v[c]);
                }
                d[i] = s;
                o[i] = i;
            }
            Arrays.sort(o, (a, b) -> Double.compare(d[a], d[b]));
            double s = 0;
            for (int i = 0; i < kk; i++) {
                s += y[o[i]];
            }
            return reg ? new double[] {s / kk} : clip2((s + 0.5) / (kk + 1));
        });
        lib.put("ridge", (x, y) -> {
            int dd = x[0].length + 1;
            double[][] g = new double[dd][dd];
            double[] b = new double[dd];
            for (int i = 0; i < x.length; i++) {
                double[] f = with1(x[i]);
                for (int a = 0; a < dd; a++) {
                    b[a] += f[a] * y[i];
                    for (int c = 0; c < dd; c++) {
                        g[a][c] += f[a] * f[c];
                    }
                }
            }
            for (int a = 1; a < dd; a++) {
                g[a][a] += 1;
            }
            double[] w = io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators.solve(g, b);
            if (reg) {
                return v -> new double[] {dot(w, with1(v))};
            }
            double[][] f = new double[x.length][];
            for (int i = 0; i < x.length; i++) {
                f[i] = with1(x[i]);
            }
            double[] beta = io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators.logistic(f, y, 1);
            return v -> clip2(1 / (1 + Math.exp(-dot(beta, with1(v)))));
        });
        lib.put("mlp", (x, y) -> {
            FeatureClassifier m = FeatureClassifier.mlp(x[0].length, HIDDEN, reg ? 1 : 2, seed);
            Tensor tx = ShiftHarness.tensor(x);
            float[] yf = new float[y.length];
            for (int i = 0; i < y.length; i++) {
                yf[i] = (float) y[i];
            }
            Tensor ty = reg ? Tensor.of(Shape.of(y.length, 1), yf) : Tensor.of(Shape.of(y.length), yf);
            Optimizer o = Optimizer.adam(3e-3f);
            for (int s = 0; s < 150; s++) {
                Var z = m.logits(Var.of(tx), true);
                (reg ? z.sub(ty).square().mean() : Functional.crossEntropy(z, ty)).backward();
                o.step(m.parameters());
            }
            return v -> {
                Tensor z = m.logits(ShiftHarness.tensor(new double[][] {v}));
                if (reg) {
                    return new double[] {z.toDoubleArray()[0]};
                }
                double[] p = EnsembleAudit.softmaxRows(z)[0];
                return new double[] {Math.max(1e-4, p[0]), Math.max(1e-4, p[1])};
            };
        });
        return lib;
    }

    private static double[] clip2(double p) {
        double q = Math.min(1 - 1e-4, Math.max(1e-4, p));
        return new double[] {1 - q, q};
    }

    private static double[] with1(double[] v) {
        double[] f = Arrays.copyOf(v, v.length + 1);
        f[v.length] = 1;
        return f;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    // ------------------------------------------------------------------ 11.4.9

    /**
     * One TTA row.
     *
     * @param method the method
     * @param passes forward passes per input
     * @param accuracy clean accuracy
     * @param nll clean NLL after temperature scaling
     * @param corrections wrong-to-right changes against the single pass (clean)
     * @param corruptions right-to-wrong changes (clean)
     * @param noisyAccuracy accuracy on the noisy test set
     * @param noisyNll noisy NLL after temperature scaling
     */
    public record TtaRow(String method, int passes, double accuracy, double nll, int corrections, int corruptions,
            double noisyAccuracy, double noisyNll) {
    }

    /**
     * TTA on a fixture model against single pass, MC dropout and a snapshot ensemble.
     *
     * @param task the fixture task (digits or shapes)
     * @param seed the seed
     * @return one row per method
     */
    public static List<TtaRow> testTimeAugmentation(UncertaintyFixture.Task task, long seed) {
        UncertaintyFixture.Setup s = UncertaintyFixture.get(task, seed);
        List<Augmentation> policy = TestTimeAugmentation.standardPolicy(ShiftSuites.SIDE);
        int passes = policy.size();
        Tensor noisy = ShiftSuites.apply(s.test().x(), ShiftSuites.Style.NOISY, seed + 5);
        UnaryOperator<Tensor> model = s.model()::logits;
        double[][][] val = TestTimeAugmentation.logits(model, s.validation().x(), policy, seed);
        double[][][] test = TestTimeAugmentation.logits(model, s.test().x(), policy, seed);
        double[][][] nz = TestTimeAugmentation.logits(model, noisy, policy, seed);
        int[] vy = s.validation().y();
        int[] ty = s.test().y();
        List<TtaRow> rows = new ArrayList<>();
        double[][] single = Calibration.withTemperature(test[0], 1);
        rows.add(ttaRow("single pass", 1, Calibration.withTemperature(val[0], 1), single,
                Calibration.withTemperature(nz[0], 1), vy, ty, single));
        for (TestTimeAugmentation.Aggregation a : TestTimeAugmentation.Aggregation.values()) {
            TestTimeAugmentation.Aggregator agg = TestTimeAugmentation.fit(a, val, vy, 1e-3);
            rows.add(ttaRow("tta " + a.name().toLowerCase(java.util.Locale.ROOT), passes, agg.apply(val), agg.apply(test),
                    agg.apply(nz), vy, ty, single));
        }
        TestTimeAugmentation.Aggregator noMirror = TestTimeAugmentation.fit(TestTimeAugmentation.Aggregation.MEAN,
                Arrays.copyOf(val, passes - 1), vy, 0);
        rows.add(ttaRow("tta mean, no mirror", passes - 1, noMirror.apply(Arrays.copyOf(val, passes - 1)),
                noMirror.apply(Arrays.copyOf(test, passes - 1)), noMirror.apply(Arrays.copyOf(nz, passes - 1)), vy, ty,
                single));
        // MC dropout at the same passes
        Model drop = dropoutMlp(64, s.classes(), seed);
        train(drop.parameters(), s.train(), 600, Optimizer.adam(3e-3f), seed, b -> Functional.crossEntropy(
                drop.forward(Var.of(b.x()), true), b.ids()), null);
        List<EnsembleAudit.Member> mc = EnsembleAudit.mcDropout(drop, passes);
        rows.add(ttaRow("mc dropout", passes, EnsembleAudit.mean(EnsembleAudit.predict(mc, s.validation().x())),
                EnsembleAudit.mean(EnsembleAudit.predict(mc, s.test().x())),
                EnsembleAudit.mean(EnsembleAudit.predict(mc, noisy)), vy, ty, single));
        // snapshot ensemble at the same passes and the same total steps
        FeatureClassifier sn = ShiftHarness.mlp(64, s.classes(), seed + 3);
        Optimizer on = Optimizer.adam(3e-3f);
        SnapshotEnsemble snap = new SnapshotEnsemble(on, sn.parameters(), 600 / passes, 5e-3f);
        train(sn.parameters(), s.train(), 600 / passes * passes, on, seed, b -> Functional.crossEntropy(
                sn.logits(Var.of(b.x()), true), b.ids()), snap);
        List<EnsembleAudit.Member> sm = EnsembleAudit.ofWeights(sn.parameters(), snap.snapshots(), sn::logits);
        rows.add(ttaRow("snapshot", passes, EnsembleAudit.mean(EnsembleAudit.predict(sm, s.validation().x())),
                EnsembleAudit.mean(EnsembleAudit.predict(sm, s.test().x())),
                EnsembleAudit.mean(EnsembleAudit.predict(sm, noisy)), vy, ty, single));
        return rows;
    }

    private static TtaRow ttaRow(String name, int passes, double[][] val, double[][] test, double[][] noisy,
            int[] vy, int[] ty, double[][] single) {
        double t = Calibration.fitTemperature(log(val), vy);
        double[][] pt = Calibration.withTemperature(log(test), t);
        double[][] pn = Calibration.withTemperature(log(noisy), t);
        int[] ch = TestTimeAugmentation.changes(single, test, ty);
        return new TtaRow(name, passes, Calibration.accuracy(test, ty), Calibration.negativeLogLikelihood(pt, ty), ch[0],
                ch[1], Calibration.accuracy(noisy, ty), Calibration.negativeLogLikelihood(pn, ty));
    }

    static double[][] log(double[][] p) {
        double[][] out = new double[p.length][];
        for (int i = 0; i < p.length; i++) {
            out[i] = new double[p[i].length];
            for (int c = 0; c < p[i].length; c++) {
                out[i][c] = Math.log(Math.max(1e-12, p[i][c]));
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ 11.4.10

    /**
     * One diversity row.
     *
     * @param method the method
     * @param agreement the training colour-class agreement
     * @param average decorrelated test accuracy
     * @param worstGroup the worst (class, colour) group accuracy
     * @param headDisagreement disagreement of the method's members or heads on the test set
     */
    public record DiversityRow(String method, double agreement, double average, double worstGroup,
            double headDisagreement) {
    }

    /**
     * Diversity-seeking ensembles on coloured shapes.
     *
     * @param agreement the training colour-class agreement (1.0 = a perfect shortcut)
     * @param steps training steps per model
     * @param seed the seed
     * @return ERM, deep ensemble (2), JTT, NCL (3), DivDis (2 heads) and D-BAT (2 models) rows
     */
    public static List<DiversityRow> diversity(double agreement, int steps, long seed) {
        ShiftSuites.Grouped train = CausalAndFairness.colouredShapes(800, agreement, seed);
        ShiftSuites.Split target = CausalAndFairness.colouredShapes(400, 0.5, seed + 1).split();
        ShiftSuites.Grouped test = CausalAndFairness.colouredShapes(800, 0.5, seed + 2);
        ShiftSuites.Grouped few = CausalAndFairness.colouredShapes(16, 0.5, seed + 3);
        int d = 128;
        List<DiversityRow> rows = new ArrayList<>();
        FeatureClassifier erm = trained(train.split(), 2, steps, seed);
        rows.add(diversityRow("erm", agreement, List.of(EnsembleAudit.of(erm)), test));
        List<EnsembleAudit.Member> deep = new ArrayList<>();
        for (int m = 0; m < 2; m++) {
            deep.add(EnsembleAudit.of(trained(train.split(), 2, steps, seed + 10 + m)));
        }
        rows.add(diversityRow("deep ensemble", agreement, List.of(x -> EnsembleAudit.mean(EnsembleAudit.predict(deep,
                x))), test, EnsembleAudit.disagreement(EnsembleAudit.predict(deep, test.split().x()))));
        // JTT: upweight the ERM model's training errors
        boolean[] errors = GroupRobustTraining.JustTrainTwice.errorSet(erm.predict(train.split().x()),
                train.split().y());
        GroupRobustTraining.JustTrainTwice jtt = new GroupRobustTraining.JustTrainTwice(errors, 20);
        FeatureClassifier jm = ShiftHarness.mlp(d, 2, seed + 20);
        Optimizer oj = Optimizer.adam(3e-3f);
        SplittableRandom rj = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ShiftSuites.Split b = train.split().rows(jtt.sample(64, rj));
            Functional.crossEntropy(jm.logits(Var.of(b.x()), true), b.ids()).backward();
            oj.step(jm.parameters());
        }
        rows.add(diversityRow("jtt", agreement, List.of(EnsembleAudit.of(jm)), test));
        // NCL: three members trained jointly
        List<FeatureClassifier> ncl = new ArrayList<>();
        List<Var> nclParams = new ArrayList<>();
        for (int m = 0; m < 3; m++) {
            FeatureClassifier f = ShiftHarness.mlp(d, 2, seed + 30 + m);
            ncl.add(f);
            nclParams.addAll(f.parameters());
        }
        train(nclParams, train.split(), steps, Optimizer.adam(3e-3f), seed, b -> {
            List<Var> z = new ArrayList<>();
            for (FeatureClassifier f : ncl) {
                z.add(f.logits(Var.of(b.x()), true));
            }
            return DiverseEnsembles.negativeCorrelation(z, ConformalTraining.oneHot(b.ids().toArray(), 2), 0.5f);
        }, null);
        List<EnsembleAudit.Member> nclMembers = new ArrayList<>();
        ncl.forEach(f -> nclMembers.add(EnsembleAudit.of(f)));
        rows.add(diversityRow("ncl", agreement, List.of(x -> EnsembleAudit.mean(EnsembleAudit.predict(nclMembers, x))),
                test, EnsembleAudit.disagreement(EnsembleAudit.predict(nclMembers, test.split().x()))));
        // DivDis: two heads, mutual information on the unlabelled target
        FeatureClassifier h1 = ShiftHarness.mlp(d, 2, seed + 40);
        FeatureClassifier h2 = ShiftHarness.mlp(d, 2, seed + 41);
        List<Var> dp = new ArrayList<>(h1.parameters());
        dp.addAll(h2.parameters());
        SplittableRandom rt = new SplittableRandom(seed + 42);
        train(dp, train.split(), steps, Optimizer.adam(3e-3f), seed + 40, b -> {
            ShiftSuites.Split t = ShiftHarness.batch(target, 64, rt);
            Var src = Functional.crossEntropy(h1.logits(Var.of(b.x()), true), b.ids())
                    .add(Functional.crossEntropy(h2.logits(Var.of(b.x()), true), b.ids()));
            Var reg = DiverseEnsembles.divDis(List.of(h1.logits(Var.of(t.x()), true), h2.logits(Var.of(t.x()), true)),
                    1f);
            return src.add(reg.mul(5f));
        }, null);
        rows.add(selected("divdis", agreement, List.of(h1, h2), few, test));
        // D-BAT: a second model that disagrees with the first on the target
        FeatureClassifier b1 = trained(train.split(), 2, steps, seed + 50);
        Tensor q = Tensor.of(Shape.of(target.size(), 2), toFloat(EnsembleAudit.of(b1).probabilities(target.x())));
        FeatureClassifier b2 = ShiftHarness.mlp(d, 2, seed + 51);
        SplittableRandom rb = new SplittableRandom(seed + 52);
        train(b2.parameters(), train.split(), steps, Optimizer.adam(3e-3f), seed + 51, b -> {
            int[] idx = new int[64];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = rb.nextInt(target.size());
            }
            ShiftSuites.Split t = target.rows(idx);
            Tensor qb = Tensor.of(Shape.of(64, 2), rowsOf(q, idx));
            return Functional.crossEntropy(b2.logits(Var.of(b.x()), true), b.ids())
                    .add(DiverseEnsembles.disagreement(b2.logits(Var.of(t.x()), true), List.of(qb)).mul(1f));
        }, null);
        rows.add(selected("d-bat", agreement, List.of(b1, b2), few, test));
        return rows;
    }

    private static float[] toFloat(double[][] p) {
        int k = p[0].length;
        float[] v = new float[p.length * k];
        for (int i = 0; i < p.length; i++) {
            for (int c = 0; c < k; c++) {
                v[i * k + c] = (float) p[i][c];
            }
        }
        return v;
    }

    private static float[] rowsOf(Tensor t, int[] idx) {
        int k = t.shape().dim(1);
        float[] v = t.toArray();
        float[] out = new float[idx.length * k];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(v, idx[i] * k, out, i * k, k);
        }
        return out;
    }

    private static DiversityRow selected(String name, double agreement, List<FeatureClassifier> heads,
            ShiftSuites.Grouped few, ShiftSuites.Grouped test) {
        double[][][] fp = new double[heads.size()][][];
        List<EnsembleAudit.Member> members = new ArrayList<>();
        for (int h = 0; h < heads.size(); h++) {
            members.add(EnsembleAudit.of(heads.get(h)));
            fp[h] = members.get(h).probabilities(few.split().x());
        }
        int best = DiverseEnsembles.selectHead(fp, few.split().y());
        return diversityRow(name, agreement, List.of(members.get(best)), test,
                EnsembleAudit.disagreement(EnsembleAudit.predict(members, test.split().x())));
    }

    private static DiversityRow diversityRow(String name, double agreement, List<EnsembleAudit.Member> m,
            ShiftSuites.Grouped test) {
        return diversityRow(name, agreement, m, test, 0);
    }

    private static DiversityRow diversityRow(String name, double agreement, List<EnsembleAudit.Member> m,
            ShiftSuites.Grouped test, double disagreement) {
        double[][] p = m.getFirst().probabilities(test.split().x());
        int[] y = test.split().y();
        double[] hit = new double[4];
        double[] n = new double[4];
        int ok = 0;
        for (int i = 0; i < y.length; i++) {
            boolean right = Calibration.argmax(p[i]) == y[i];
            ok += right ? 1 : 0;
            n[test.group()[i]]++;
            hit[test.group()[i]] += right ? 1 : 0;
        }
        double worst = 1;
        for (int g = 0; g < 4; g++) {
            if (n[g] > 0) {
                worst = Math.min(worst, hit[g] / n[g]);
            }
        }
        return new DiversityRow(name, agreement, ok / (double) y.length, worst, disagreement);
    }

    // ------------------------------------------------------------------ 11.4.14

    /**
     * Ensemble distillation against the teacher ensemble, a single model and an evidential head.
     *
     * @param steps training steps of every network (teacher members included)
     * @param seed the seed
     * @return teacher, single, mean KD, EnD², Hydra and evidential rows; the AUROCs use mutual
     *     information for ensembles and Hydra, the Dirichlet's mutual information for EnD², the
     *     evidential uncertainty {@code k / α0} and entropy otherwise
     */
    public static List<Row> distillation(int steps, long seed) {
        ShiftSuites.OodSuite s = suite(seed);
        int k = s.classes();
        int members = 4;
        List<FeatureClassifier> teacher = new ArrayList<>();
        List<EnsembleAudit.Member> tm = new ArrayList<>();
        for (int m = 0; m < members; m++) {
            FeatureClassifier f = trained(s.train(), k, steps, seed + 100 + m);
            teacher.add(f);
            tm.add(EnsembleAudit.of(f));
        }
        List<Row> rows = new ArrayList<>();
        rows.add(row("teacher ensemble", tm, s, 0));
        rows.add(row("single", List.of(EnsembleAudit.of(trained(s.train(), k, steps, seed))), s, 0));
        // mean KD
        FeatureClassifier kd = ShiftHarness.mlp(64, k, seed + 1);
        train(kd.parameters(), s.train(), steps, Optimizer.adam(3e-3f), seed + 1, b -> EnsembleDistillation.meanKd(
                kd.logits(Var.of(b.x()), true), meanProbs(teacher, b.x()), 1f), null);
        rows.add(row("mean kd", List.of(EnsembleAudit.of(kd)), s, 0));
        // EnD2
        FeatureClassifier end = ShiftHarness.mlp(64, k, seed + 2);
        int[] step = {0};
        train(end.parameters(), s.train(), steps, Optimizer.adam(1e-3f), seed + 2, b -> {
            List<Tensor> z = new ArrayList<>();
            for (FeatureClassifier f : teacher) {
                z.add(f.logits(b.x()));
            }
            float t = EnsembleDistillation.annealedTemperature(step[0]++, steps, 2.5f);
            return EnsembleDistillation.dirichletNll(end.logits(Var.of(b.x()), true), z, t, 1e-4f);
        }, null);
        rows.add(dirichletRow("end2", end, s));
        // Hydra
        EnsembleDistillation.Hydra hydra = new EnsembleDistillation.Hydra(64, HIDDEN, k, members, seed + 3);
        train(hydra.parameters(), s.train(), steps, Optimizer.adam(3e-3f), seed + 3, b -> {
            List<Tensor> z = new ArrayList<>();
            for (FeatureClassifier f : teacher) {
                z.add(f.logits(b.x()));
            }
            return hydra.loss(Var.of(b.x()), z, 1f);
        }, null);
        rows.add(row("hydra", hydra.auditMembers(), s, 0));
        // evidential head on an MLP body
        Model body = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU).seed(seed + 4)
                .build();
        EvidentialHead head = new EvidentialHead(64, k, new SplittableRandom(seed + 4));
        List<Var> ep = new ArrayList<>(body.parameters());
        ep.addAll(head.parameters());
        int[] es = {0};
        train(ep, s.train(), steps, Optimizer.adam(3e-3f), seed + 4, b -> EvidentialHead.loss(head.forward(
                body.forward(Var.of(b.x()), true)), ConformalTraining.oneHot(b.ids().toArray(), k),
                Math.min(1f, es[0]++ / (float) steps)), null);
        UnaryOperator<Tensor> alpha = x -> Var.noGrad(() -> head.forward(body.forward(Var.of(x), false)).value());
        rows.add(evidentialRow(alpha, s));
        return rows;
    }

    private static Tensor meanProbs(List<FeatureClassifier> teacher, Tensor x) {
        Tensor sum = null;
        for (FeatureClassifier f : teacher) {
            Tensor z = f.logits(x);
            Tensor p = z.sub(z.logsumexp(1, true)).exp();
            sum = sum == null ? p : sum.add(p);
        }
        return sum.div(teacher.size());
    }

    private static Row dirichletRow(String name, FeatureClassifier m, ShiftSuites.OodSuite s) {
        double[][] p = EnsembleDistillation.dirichletProbabilities(m.logits(s.test().x()));
        int[] y = s.test().y();
        double[] in = EnsembleDistillation.dirichletMutualInformation(m.logits(s.test().x()));
        return new Row(name, Calibration.accuracy(p, y), Calibration.negativeLogLikelihood(p, y),
                Calibration.expectedCalibrationError(p, y, 15),
                OodMetrics.auroc(EnsembleDistillation.dirichletMutualInformation(m.logits(s.near())), in),
                OodMetrics.auroc(EnsembleDistillation.dirichletMutualInformation(m.logits(s.far())), in), 0, 0);
    }

    private static Row evidentialRow(UnaryOperator<Tensor> alpha, ShiftSuites.OodSuite s) {
        double[][] p = ShiftHarness.rows(EvidentialHead.probabilities(alpha.apply(s.test().x())));
        int[] y = s.test().y();
        double[] in = EvidentialHead.uncertainty(alpha.apply(s.test().x())).toDoubleArray();
        return new Row("evidential", Calibration.accuracy(p, y), Calibration.negativeLogLikelihood(p, y),
                Calibration.expectedCalibrationError(p, y, 15),
                OodMetrics.auroc(EvidentialHead.uncertainty(alpha.apply(s.near())).toDoubleArray(), in),
                OodMetrics.auroc(EvidentialHead.uncertainty(alpha.apply(s.far())).toDoubleArray(), in), 0, 0);
    }

    // ------------------------------------------------------------------ 11.4.16

    /**
     * One audit row.
     *
     * @param setting noise rate and imbalance
     * @param method the method
     * @param accuracy clean test accuracy
     * @param nll NLL after temperature scaling (fitted on clean validation rows)
     * @param ece ECE after temperature scaling
     * @param farAuroc far-OOD AUROC
     * @param disagreement member disagreement
     * @param nllGapToDeep this method's NLL minus the deep ensemble's (per-sample paired mean)
     * @param nllGapSe the paired standard error of that gap
     */
    public record AuditRow(String setting, String method, double accuracy, double nll, double ece, double farAuroc,
            double disagreement, double nllGapToDeep, double nllGapSe) {
    }

    /**
     * The ensemble zoo on ten-class digits with symmetric label noise and a long tail, at an
     * equal number of training steps per method.
     *
     * @param noise the symmetric noise rate
     * @param imbalance the head-to-tail ratio (1 for balanced)
     * @param budget training steps per method
     * @param seed the seed
     * @return single, deep ensemble (3), MC dropout, SWAG, snapshot, packed, TabM, evidential and
     *     SNGP rows
     */
    public static List<AuditRow> audit(double noise, double imbalance, int budget, long seed) {
        int[] cls = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        ShiftSuites.Split pool = ShiftSuites.digits(4000, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN, 0.2,
                seed);
        ShiftSuites.Split train = imbalance > 1 ? pool.rows(LabelCorruption.longTail(pool.y(), 10, 300, imbalance,
                seed)) : pool.range(0, 1500);
        int[] noisy = LabelCorruption.symmetric(train.y(), 10, noise, seed).noisy();
        ShiftSuites.Split tr = new ShiftSuites.Split(train.x(), noisy);
        ShiftSuites.Split val = ShiftSuites.digits(600, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN, 0.2,
                seed + 1);
        ShiftSuites.Split test = ShiftSuites.digits(1000, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN, 0.2,
                seed + 2);
        Tensor far = Tensor.concat(0, ShiftSuites.outliers(200, ShiftSuites.Outlier.NOISE, seed + 3),
                ShiftSuites.outliers(200, ShiftSuites.Outlier.STRIPES, seed + 4));
        int k = 10;
        int members = 3;
        Map<String, List<EnsembleAudit.Member>> methods = new LinkedHashMap<>();
        methods.put("single", List.of(EnsembleAudit.of(trained(tr, k, budget, seed))));
        List<EnsembleAudit.Member> deep = new ArrayList<>();
        for (int m = 0; m < members; m++) {
            deep.add(EnsembleAudit.of(trained(tr, k, budget / members, seed + 100 + m)));
        }
        methods.put("deep ensemble", deep);
        Model drop = dropoutMlp(64, k, seed);
        train(drop.parameters(), tr, budget, Optimizer.adam(3e-3f), seed, b -> Functional.crossEntropy(
                drop.forward(Var.of(b.x()), true), b.ids()), null);
        methods.put("mc dropout", EnsembleAudit.mcDropout(drop, members));
        FeatureClassifier sw = ShiftHarness.mlp(64, k, seed + 7);
        WeightPosterior.Swag swag = new WeightPosterior.Swag(sw.parameters(), 10, 10);
        Optimizer os = Optimizer.adam(3e-3f);
        int start = budget * 3 / 4;
        int[] count = {0};
        train(sw.parameters(), tr, budget, os, seed + 7, b -> {
            if (++count[0] > start && count[0] % 10 == 0) {
                swag.collect();
            }
            return Functional.crossEntropy(sw.logits(Var.of(b.x()), true), b.ids());
        }, null);
        methods.put("swag", EnsembleAudit.ofWeights(sw.parameters(), swag.sample(members, new SplittableRandom(seed)),
                sw::logits));
        FeatureClassifier sn = ShiftHarness.mlp(64, k, seed + 8);
        Optimizer on = Optimizer.adam(3e-3f);
        SnapshotEnsemble snap = new SnapshotEnsemble(on, sn.parameters(), budget / members, 5e-3f);
        train(sn.parameters(), tr, budget / members * members, on, seed + 8, b -> Functional.crossEntropy(
                sn.logits(Var.of(b.x()), true), b.ids()), snap);
        methods.put("snapshot", EnsembleAudit.ofWeights(sn.parameters(), snap.snapshots(), sn::logits));
        PackedEnsemble packed = new PackedEnsemble(64, HIDDEN, k, 4, 2.0, 1, seed);
        train(packed.parameters(), tr, budget, Optimizer.adam(3e-3f), seed, b -> memberLoss(packed.members(
                Var.of(b.x()), true), b.ids()), null);
        methods.put("packed", packed.auditMembers());
        TabM tabm = new TabM(64, 64, 2, 4, k, null, new SplittableRandom(seed));
        train(tabm.parameters(), tr, budget, Optimizer.adam(3e-3f), seed, b -> memberLoss(tabm.members(Var.of(b.x()),
                true), b.ids()), null);
        methods.put("tabm", EnsembleAudit.of(tabm));
        Model body = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU).seed(seed + 4)
                .build();
        EvidentialHead head = new EvidentialHead(64, k, new SplittableRandom(seed + 4));
        List<Var> ep = new ArrayList<>(body.parameters());
        ep.addAll(head.parameters());
        int[] es = {0};
        train(ep, tr, budget, Optimizer.adam(3e-3f), seed + 4, b -> EvidentialHead.loss(head.forward(body.forward(
                Var.of(b.x()), true)), ConformalTraining.oneHot(b.ids().toArray(), k), Math.min(1f, es[0]++
                / (float) budget)), null);
        methods.put("evidential", List.of(EnsembleAudit.evidential(x -> Var.noGrad(() -> head.forward(body.forward(
                Var.of(x), false)).value()))));
        Model sb = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU).seed(seed + 5)
                .build();
        DistanceAwareHead sngp = DistanceAwareHead.sngp(64, k, 256, 2f, new SplittableRandom(seed + 5));
        List<Var> spp = new ArrayList<>(sb.parameters());
        spp.addAll(sngp.parameters());
        train(spp, tr, budget, Optimizer.adam(3e-3f), seed + 5, b -> sngp.loss(sngp.forward(sb.forward(Var.of(b.x()),
                true)), b.ids()), null);
        UnaryOperator<Tensor> feats = x -> Var.noGrad(() -> sb.forward(Var.of(x), false).value());
        sngp.fitCovariance(feats.apply(tr.x()), 1f, false);
        methods.put("sngp", List.of(EnsembleAudit.distanceAware(sngp, feats)));

        String setting = String.format(java.util.Locale.ROOT, "noise %.1f imbalance %.0f", noise, imbalance);
        double[] deepNll = perSampleNll(deep, val, test);
        List<AuditRow> rows = new ArrayList<>();
        for (Map.Entry<String, List<EnsembleAudit.Member>> e : methods.entrySet()) {
            List<EnsembleAudit.Member> m = e.getValue();
            double[][][] pt = EnsembleAudit.predict(m, test.x());
            double[][] mean = EnsembleAudit.mean(pt);
            double t = Calibration.fitTemperature(log(EnsembleAudit.mean(EnsembleAudit.predict(m, val.x()))), val.y());
            double[][] cal = Calibration.withTemperature(log(mean), t);
            double[] nll = perSampleNll(m, val, test);
            EvalStatistics.Interval gap = EvalStatistics.paired(nll, deepNll, null);
            EnsembleAudit.Score score = m.size() > 1 ? EnsembleAudit.Score.MUTUAL_INFORMATION
                    : EnsembleAudit.Score.ENTROPY;
            double auroc = OodMetrics.auroc(EnsembleAudit.uncertainty(EnsembleAudit.predict(m, far), score),
                    EnsembleAudit.uncertainty(pt, score));
            rows.add(new AuditRow(setting, e.getKey(), Calibration.accuracy(mean, test.y()),
                    Calibration.negativeLogLikelihood(cal, test.y()), Calibration.expectedCalibrationError(cal,
                    test.y(), 15), auroc, EnsembleAudit.disagreement(pt), gap.mean(), gap.se()));
        }
        return rows;
    }

    private static double[] perSampleNll(List<EnsembleAudit.Member> m, ShiftSuites.Split val,
            ShiftSuites.Split test) {
        double t = Calibration.fitTemperature(log(EnsembleAudit.mean(EnsembleAudit.predict(m, val.x()))), val.y());
        double[][] cal = Calibration.withTemperature(log(EnsembleAudit.mean(EnsembleAudit.predict(m, test.x()))), t);
        double[] out = new double[cal.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = -Math.log(Math.max(1e-12, cal[i][test.y()[i]]));
        }
        return out;
    }

    /**
     * Prints every table.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        singleRun(1200, 4, 1).forEach(System.out::println);
        insideOneNetwork(600, 1).forEach(System.out::println);
        stacking(400, 1).forEach(System.out::println);
        testTimeAugmentation(UncertaintyFixture.Task.DIGITS, 1).forEach(System.out::println);
        for (double a : new double[] {0.9, 1.0, 0.5}) {
            diversity(a, 400, 1).forEach(System.out::println);
        }
        distillation(600, 1).forEach(System.out::println);
        for (double[] cfg : new double[][] {{0, 1}, {0.2, 1}, {0.4, 1}, {0, 10}, {0, 100}}) {
            audit(cfg[0], cfg[1], 600, 1).forEach(System.out::println);
        }
    }
}
