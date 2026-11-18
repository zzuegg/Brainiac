package io.github.zzuegg.brainiac.experiments.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.experimental.data.ActiveLearner;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.semisup.AdaptiveThreshold;
import io.github.zzuegg.brainiac.experimental.semisup.ConsistencyTraining;
import io.github.zzuegg.brainiac.experimental.semisup.FixMatch;
import io.github.zzuegg.brainiac.experimental.semisup.JepaMatch;
import io.github.zzuegg.brainiac.experimental.semisup.MixMatch;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.experimental.semisup.VirtualAdversarial;
import io.github.zzuegg.brainiac.experimental.train.MomentumTeacher;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Semi-supervised learning on the digits task with 4 to 40 labels per class and the rest of
 * the 1,000 training rows unlabelled (W10 10.1.1 to 10.1.6): pseudo-labelling, the Π-model,
 * Mean Teacher, VAT, MixMatch, FixMatch at two thresholds, FlexMatch, FreeMatch, SoftMatch and
 * JEPAMatch (FlexMatch plus SIGReg), against supervised-only training (plain and augmented),
 * entropy-based active learning with the same label budget, and a fully labelled oracle. Every
 * method uses the same MLP, optimiser, batch sizes and step count. An imbalanced unlabelled pool
 * (imbalance ratio 10) tests what a fixed threshold does to rare classes, and a tabular task
 * ({@link TabularStressSuite}, 20 labels per class) tests VAT where image augmentations do not
 * apply.
 */
public final class SemiSupervisedShootout {

    private SemiSupervisedShootout() {
    }

    /** The compared methods. */
    public enum Method {
        /** Cross-entropy on the labelled rows only. */
        SUPERVISED,
        /** The same with the weak augmentation. */
        SUPERVISED_AUG,
        /** Entropy-based active learning picking the same number of labels. */
        ACTIVE,
        /** All training labels. */
        ORACLE,
        /** Pseudo-labelling without a threshold. */
        PSEUDO_LABEL,
        /** The Π-model. */
        PI_MODEL,
        /** Mean Teacher. */
        MEAN_TEACHER,
        /** Virtual adversarial training with entropy minimisation. */
        VAT,
        /** MixMatch. */
        MIXMATCH,
        /** FixMatch with tau 0.95. */
        FIXMATCH,
        /** FixMatch with tau 0.7. */
        FIXMATCH_07,
        /** FlexMatch. */
        FLEXMATCH,
        /** FreeMatch. */
        FREEMATCH,
        /** SoftMatch. */
        SOFTMATCH,
        /** JEPAMatch: FlexMatch plus SIGReg on a projection head. */
        JEPAMATCH,
        /** FlexMatch plus VICReg's variance and covariance terms on the same head. */
        FLEX_VICREG
    }

    /**
     * The setting.
     *
     * @param labelsPerClass labelled rows per class
     * @param imbalance the unlabelled pool's imbalance ratio, 1 for balanced
     * @param steps training steps
     */
    public record Setting(int labelsPerClass, double imbalance, int steps) {
    }

    /**
     * One run's scores.
     *
     * @param method the method
     * @param accuracy final balanced test accuracy
     * @param worstClass the lowest per-class test accuracy
     * @param halfway test accuracy at half the steps
     * @param kept the share of unlabelled rows used (thresholded methods; else NaN)
     * @param precision the pseudo-labels' precision among kept rows (else NaN)
     * @param millis training time
     */
    public record Result(Method method, double accuracy, double worstClass, double halfway, double kept,
            double precision, double millis) {
    }

    private static final int CLASSES = 10;

    /**
     * Runs one method on the digits task.
     *
     * @param method the method
     * @param setting the setting
     * @param seed the seed
     * @return the scores
     */
    public static Result digits(Method method, Setting setting, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        ToyTasks.Data train = task.train();
        LabelCorruption.Split split = LabelCorruption.labelledSplit(train.y(), CLASSES, setting.labelsPerClass(),
                seed + 1);
        int[] pool = split.unlabelled();
        if (setting.imbalance() > 1) {
            int[] poolY = LabelTasks.labels(train.y(), pool);
            int[] keep = LabelCorruption.longTail(poolY, CLASSES, 90, setting.imbalance(), seed + 2);
            int[] sub = new int[keep.length];
            for (int i = 0; i < keep.length; i++) {
                sub[i] = pool[keep[i]];
            }
            pool = sub;
        }
        long start = System.nanoTime();
        if (method == Method.ACTIVE) {
            int budget = CLASSES * setting.labelsPerClass();
            ActiveLearner al = ActiveLearner.of(
                    () -> new ActiveLearner.MlpLearner(train.features(), 64, CLASSES, setting.steps(), seed),
                    ActiveLearner.Acquisition.ENTROPY, seed);
            double[] curve = al.run(train.x(), train.y(), task.test().x(), task.test().y(), budget / 2,
                    Math.max(1, budget / 4), budget);
            double acc = curve[curve.length - 1];
            return new Result(method, acc, Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                    (System.nanoTime() - start) / 1e6);
        }
        return train(method, setting, train, split.labelled(), pool, task.test(), seed, start);
    }

    private static Result train(Method method, Setting setting, ToyTasks.Data train, int[] labelled, int[] pool,
            ToyTasks.Data test, long seed, long start) {
        LabelTasks.Net net = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        LabelTasks.Net teacherNet = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        Model projector = Model.sequential().input(32).dense(16, Activation.IDENTITY).seed(seed + 12).build();
        List<Var> params = new ArrayList<>(net.parameters());
        if (method == Method.JEPAMATCH || method == Method.FLEX_VICREG) {
            params.addAll(projector.parameters());
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        Augmentation weak = LabelTasks.weak();
        Augmentation strong = LabelTasks.strong();
        int steps = setting.steps();
        int[] all = LabelTasks.range(train.size());
        MomentumTeacher teacher = method == Method.MEAN_TEACHER
                ? ConsistencyTraining.teacher(teacherNet.parameters(), net.parameters(), 0.99f) : null;
        ConsistencyTraining consistency = switch (method) {
            case PSEUDO_LABEL -> new ConsistencyTraining(ConsistencyTraining.Mode.PSEUDO_LABEL, 1f, steps / 2, 0f);
            case PI_MODEL -> new ConsistencyTraining(ConsistencyTraining.Mode.PI_MODEL, 10f, steps / 2, 0f);
            case MEAN_TEACHER -> new ConsistencyTraining(ConsistencyTraining.Mode.MEAN_TEACHER, 10f, steps / 2, 0f);
            default -> null;
        };
        VirtualAdversarial vat = new VirtualAdversarial(0.5f, 0.3f, 1, 0f);
        MixMatch mixMatch = new MixMatch(0.5, 0.75f, 10f, steps, true, true);
        int[] slot = new int[train.size()];
        for (int i = 0; i < pool.length; i++) {
            slot[pool[i]] = i;
        }
        FixMatch fix = switch (method) {
            case FIXMATCH -> new FixMatch(FixMatch.fixed(0.95f), 1f);
            case FIXMATCH_07 -> new FixMatch(FixMatch.fixed(0.7f), 1f);
            case FLEXMATCH, FLEX_VICREG -> new FixMatch(new AdaptiveThreshold(AdaptiveThreshold.Mode.FLEX, CLASSES,
                    pool.length, 0.95f, 0.99), 1f);
            case FREEMATCH -> new FixMatch(new AdaptiveThreshold(AdaptiveThreshold.Mode.FREE, CLASSES, pool.length,
                    0.95f, 0.99), 1f);
            case SOFTMATCH -> new FixMatch(new AdaptiveThreshold(AdaptiveThreshold.Mode.SOFT, CLASSES, pool.length,
                    0.95f, 0.99), 1f);
            default -> null;
        };
        JepaMatch jepa = method == Method.JEPAMATCH ? new JepaMatch(CLASSES, pool.length, 0.95f, 1f, 0.1f, 16, 9) : null;
        double halfway = 0;
        for (int step = 0; step < steps; step++) {
            if (step == steps / 2) {
                halfway = balanced(net, test);
            }
            boolean oracle = method == Method.ORACLE;
            int[] li = LabelTasks.draw(32, oracle ? all : labelled, r);
            Tensor xl = LabelTasks.rows(train.x(), li);
            int[] yl = LabelTasks.labels(train.y(), li);
            int[] ui = LabelTasks.draw(64, pool, r);
            Tensor xu = LabelTasks.rows(train.x(), ui);
            int[] uTruth = LabelTasks.labels(train.y(), ui);
            int[] uSlots = new int[ui.length];
            for (int i = 0; i < ui.length; i++) {
                uSlots[i] = slot[ui[i]];
            }
            boolean augmentLabelled = method != Method.SUPERVISED && method != Method.ORACLE
                    && method != Method.VAT;
            Tensor xlIn = augmentLabelled ? FixMatch.augment(xl, weak, r) : xl;
            Var loss;
            switch (method) {
                case MIXMATCH -> {
                    Tensor u1 = FixMatch.augment(xu, weak, r);
                    Tensor u2 = FixMatch.augment(xu, weak, r);
                    float[][] q = mixMatch.guess(List.of(net.predict(u1), net.predict(u2)));
                    MixMatch.Mixed mixed = mixMatch.mix(xlIn, SoftTargets.oneHot(yl, CLASSES), u1, q, r);
                    loss = mixMatch.loss(net.logits(Var.of(mixed.x()), true), mixed, step);
                }
                default -> {
                    loss = Functional.crossEntropy(net.logits(Var.of(xlIn), true), SoftTargets.ids(yl));
                    switch (method) {
                        case PSEUDO_LABEL -> {
                            Tensor xw = FixMatch.augment(xu, weak, r);
                            Var s = net.logits(Var.of(xw), true);
                            loss = loss.add(consistency.loss(s, s.value(), step));
                        }
                        case PI_MODEL -> {
                            Var s = net.logits(Var.of(FixMatch.augment(xu, weak, r)), true);
                            Tensor t = net.predict(FixMatch.augment(xu, weak, r));
                            loss = loss.add(consistency.loss(s, t, step));
                        }
                        case MEAN_TEACHER -> {
                            Var s = net.logits(Var.of(FixMatch.augment(xu, weak, r)), true);
                            Tensor t = teacherNet.predict(FixMatch.augment(xu, weak, r));
                            loss = loss.add(consistency.loss(s, t, step));
                        }
                        case VAT -> loss = loss.add(vat.loss(x -> net.logits(x, true), xu, r));
                        case FIXMATCH, FIXMATCH_07, FLEXMATCH, FREEMATCH, SOFTMATCH -> {
                            Tensor w = net.predict(FixMatch.augment(xu, weak, r));
                            Var s = net.logits(Var.of(FixMatch.augment(xu, strong, r)), true);
                            loss = loss.add(fix.loss(s, w, uSlots, uTruth));
                        }
                        case JEPAMATCH, FLEX_VICREG -> {
                            Tensor xw = FixMatch.augment(xu, weak, r);
                            Tensor w = net.predict(xw);
                            Var s = net.logits(Var.of(FixMatch.augment(xu, strong, r)), true);
                            Var z = projector.forward(net.trunk().forward(Var.of(xw), true), true);
                            if (jepa != null) {
                                loss = loss.add(jepa.loss(s, w, z, uSlots, uTruth, seed * 1000 + step));
                            } else {
                                loss = loss.add(fix.loss(s, w, uSlots, uTruth))
                                        .add(SelfSupervisedLosses.vicreg(z, z, 0f, 1f, 0.04f).mul(0.5f));
                            }
                        }
                        default -> {
                        }
                    }
                }
            }
            loss.backward();
            opt.step(params);
            if (teacher != null) {
                teacher.update();
            }
        }
        double[] perClass = LabelTasks.classAccuracy(net.predict(test.inputs()), test.y(), CLASSES);
        FixMatch stats = jepa != null ? jepa.flexMatch() : fix;
        return new Result(method, LabelTasks.mean(perClass), LabelTasks.min(perClass), halfway,
                stats == null ? Double.NaN : stats.keptShare(), stats == null ? Double.NaN : stats.pseudoLabelPrecision(),
                (System.nanoTime() - start) / 1e6);
    }

    private static double balanced(LabelTasks.Net net, ToyTasks.Data test) {
        return LabelTasks.mean(LabelTasks.classAccuracy(net.predict(test.inputs()), test.y(), CLASSES));
    }

    /** The low-dimensional tasks without image augmentations. */
    public enum Tabular {
        /** {@link TabularStressSuite}: 4 informative of 8 columns, unrotated. */
        STRESS_SUITE,
        /** Two interleaved {@code ToyTasks} spirals, the two-moons-style picture of the cluster assumption. */
        SPIRALS
    }

    /**
     * A task without image augmentations: 20 labels per class and about 1,000 unlabelled rows.
     * Supported methods: SUPERVISED, PI_MODEL (Gaussian input noise), MEAN_TEACHER (the same
     * noise), VAT and ORACLE.
     *
     * @param kind the task
     * @param method the method
     * @param steps training steps
     * @param seed the seed
     * @return test accuracy
     */
    public static double tabular(Tabular kind, Method method, int steps, long seed) {
        float[][] x;
        int[] y;
        if (kind == Tabular.SPIRALS) {
            ToyTasks.Data d = ToyTasks.spiral(1000, 2, 0.1, seed);
            x = d.x();
            y = d.y();
        } else {
            TabularStressSuite suite = new TabularStressSuite(4, 0.5, false, seed);
            SyntheticRecsys.Table table = suite.table(2000, seed + 1);
            x = toRows(table.features());
            float[] yf = table.labels().toArray();
            y = new int[yf.length];
            for (int i = 0; i < y.length; i++) {
                y[i] = (int) yf[i];
            }
        }
        double scale = kind == Tabular.SPIRALS ? 0.1 : 1;
        int nTrain = 1200;
        int[] trainY = java.util.Arrays.copyOf(y, nTrain);
        LabelCorruption.Split split = LabelCorruption.labelledSplit(trainY, 2, 20, seed + 2);
        int[] labelled = method == Method.ORACLE ? LabelTasks.range(nTrain) : split.labelled();
        int[] pool = split.unlabelled();
        LabelTasks.Net net = LabelTasks.mlp(x[0].length, 64, 32, 2, seed + 3);
        LabelTasks.Net teacherNet = LabelTasks.mlp(x[0].length, 64, 32, 2, seed + 3);
        MomentumTeacher teacher = method == Method.MEAN_TEACHER
                ? ConsistencyTraining.teacher(teacherNet.parameters(), net.parameters(), 0.99f) : null;
        ConsistencyTraining cons = new ConsistencyTraining(method == Method.MEAN_TEACHER
                ? ConsistencyTraining.Mode.MEAN_TEACHER : ConsistencyTraining.Mode.PI_MODEL, 10f, steps / 2, 0f);
        VirtualAdversarial vat = new VirtualAdversarial((float) (0.1 * scale), (float) scale, 1, 0f);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 4);
        for (int step = 0; step < steps; step++) {
            int[] li = LabelTasks.draw(32, labelled, r);
            Var loss = Functional.crossEntropy(net.logits(Var.of(LabelTasks.rows(x, li)), true),
                    SoftTargets.ids(LabelTasks.labels(y, li)));
            Tensor xu = LabelTasks.rows(x, LabelTasks.draw(64, pool, r));
            switch (method) {
                case PI_MODEL -> loss = loss.add(cons.loss(net.logits(Var.of(LabelTasks.gaussian(xu, 0.3 * scale, r)), true),
                        net.predict(LabelTasks.gaussian(xu, 0.3 * scale, r)), step));
                case MEAN_TEACHER -> loss = loss.add(cons.loss(
                        net.logits(Var.of(LabelTasks.gaussian(xu, 0.3 * scale, r)), true),
                        teacherNet.predict(LabelTasks.gaussian(xu, 0.3 * scale, r)), step));
                case VAT -> loss = loss.add(vat.loss(v -> net.logits(v, true), xu, r));
                default -> {
                }
            }
            loss.backward();
            opt.step(net.parameters());
            if (teacher != null) {
                teacher.update();
            }
        }
        int[] testIdx = new int[x.length - nTrain];
        for (int i = 0; i < testIdx.length; i++) {
            testIdx[i] = nTrain + i;
        }
        return ToyTasks.accuracy(net.predict(LabelTasks.rows(x, testIdx)), LabelTasks.labels(y, testIdx));
    }

    static float[][] toRows(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        float[] v = t.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * Prints the full comparison at 4, 10 and 40 labels per class (three seeds, 4x the test
     * budget), the imbalanced pool and the tabular task.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (int labels : new int[] {4, 10, 40}) {
            for (Method m : Method.values()) {
                double acc = 0;
                double worst = 0;
                for (long seed = 1; seed <= 3; seed++) {
                    Result res = digits(m, new Setting(labels, 1, 1600), seed);
                    acc += res.accuracy() / 3;
                    worst += res.worstClass() / 3;
                }
                System.out.printf("labels/class %2d %-14s acc %.3f worst %.3f%n", labels, m, acc, worst);
            }
        }
        for (Method m : Method.values()) {
            Result res = digits(m, new Setting(4, 10, 1600), 1);
            System.out.printf("imbalanced pool %-14s acc %.3f worst %.3f kept %.2f precision %.2f%n", m,
                    res.accuracy(), res.worstClass(), res.kept(), res.precision());
        }
        for (Method m : new Method[] {Method.SUPERVISED, Method.PI_MODEL, Method.MEAN_TEACHER, Method.VAT,
                Method.ORACLE}) {
            for (Tabular t : Tabular.values()) {
                System.out.printf("%-12s %-14s acc %.3f%n", t, m, tabular(t, m, 1600, 1));
            }
        }
    }
}
