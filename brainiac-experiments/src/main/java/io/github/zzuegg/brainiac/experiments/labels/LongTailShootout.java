package io.github.zzuegg.brainiac.experiments.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.labels.LogitAdjustment;
import io.github.zzuegg.brainiac.experimental.labels.MarginAndDecoupling;
import io.github.zzuegg.brainiac.experimental.labels.RobustLosses;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Long-tailed classification on the digits task (W10 10.1.11 and 10.1.12): ten classes whose
 * training counts fall exponentially from 200 to {@code 200 / ratio}, scored on a balanced test
 * set. Logit adjustment (post-hoc additive, the logit-adjusted loss, which at {@code tau = 1} is
 * the balanced softmax, and post-hoc multiplicative on cosine scores), LDAM with and without
 * deferred re-weighting, and decoupled classifier re-training (cRT) and tau-normalisation are
 * compared with cross-entropy, inverse-frequency reweighting, class-balanced oversampling
 * throughout, and focal loss. Every post-hoc parameter is tuned on a small balanced validation
 * split.
 */
public final class LongTailShootout {

    private LongTailShootout() {
    }

    /** The compared methods. */
    public enum Method {
        /** Cross-entropy. */
        CE,
        /** Inverse-frequency class weights. */
        REWEIGHT,
        /** Class-balanced sampling throughout. */
        OVERSAMPLE,
        /** Focal loss, gamma 2. */
        FOCAL,
        /** Post-hoc additive adjustment of the CE model, tau tuned. */
        POSTHOC_ADDITIVE,
        /** The logit-adjusted loss at tau 1 (the balanced softmax). */
        LA_LOSS,
        /** Post-hoc multiplicative adjustment of the CE model's cosine scores, gamma tuned. */
        POSTHOC_MULTIPLICATIVE,
        /** LDAM. */
        LDAM,
        /** LDAM with deferred re-weighting in the last third. */
        LDAM_DRW,
        /** Classifier re-training with class-balanced sampling on the CE model's frozen features. */
        CRT,
        /** Tau-normalisation of the CE model's classifier, tau tuned. */
        TAU_NORM
    }

    /**
     * One method's scores.
     *
     * @param method the method
     * @param balanced balanced test accuracy
     * @param head accuracy on classes 0-2
     * @param medium accuracy on classes 3-6
     * @param tail accuracy on classes 7-9
     * @param parameter the tuned tau or gamma (NaN when none)
     */
    public record Result(Method method, double balanced, double head, double medium, double tail, double parameter) {
    }

    private static final int CLASSES = 10;

    /**
     * Runs every method at one imbalance ratio.
     *
     * @param ratio the imbalance ratio
     * @param steps training steps per model
     * @param seed the seed
     * @return one result per method, in enum order
     */
    public static List<Result> run(double ratio, int steps, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 200, 30, 0.1, seed);
        ToyTasks.Data val = ToyTasks.digits(CLASSES, 10, LabelTasks.SIDE, 0.1, seed, seed * 31 + 5003);
        int[] keep = LabelCorruption.longTail(task.train().y(), CLASSES, 200, ratio, seed + 1);
        float[][] x = new float[keep.length][];
        int[] y = new int[keep.length];
        for (int i = 0; i < keep.length; i++) {
            x[i] = task.train().x()[keep[i]];
            y[i] = task.train().y()[keep[i]];
        }
        int[] counts = new int[CLASSES];
        for (int c : y) {
            counts[c]++;
        }
        double[] priors = LogitAdjustment.priors(counts);
        double[] inverse = new double[CLASSES];
        for (int k = 0; k < CLASSES; k++) {
            inverse[k] = 1.0 / counts[k];
        }
        List<Result> out = new ArrayList<>();
        LabelTasks.Net ce = train(x, y, RobustLosses.crossEntropy(), null, false, steps, seed);
        for (Method m : Method.values()) {
            Tensor testLogits;
            double param = Double.NaN;
            switch (m) {
                case CE -> testLogits = ce.predict(task.test().inputs());
                case REWEIGHT -> testLogits = train(x, y, MarginAndDecoupling.weightedCrossEntropy(inverse), null, false,
                        steps, seed).predict(task.test().inputs());
                case OVERSAMPLE -> testLogits = train(x, y, RobustLosses.crossEntropy(), null, true, steps, seed)
                        .predict(task.test().inputs());
                case FOCAL -> testLogits = train(x, y, RobustLosses.focal(2f), null, false, steps, seed)
                        .predict(task.test().inputs());
                case POSTHOC_ADDITIVE -> {
                    Tensor v = ce.predict(val.inputs());
                    param = LogitAdjustment.tune(grid(0, 2, 0.25),
                            t -> balanced(LogitAdjustment.posthocAdditive(v, priors, t), val.y()));
                    testLogits = LogitAdjustment.posthocAdditive(ce.predict(task.test().inputs()), priors, param);
                }
                case LA_LOSS -> testLogits = train(x, y, LogitAdjustment.adjustedLoss(priors, 1), null, false, steps,
                        seed).predict(task.test().inputs());
                case POSTHOC_MULTIPLICATIVE -> {
                    Tensor w = ce.head().parameters().get(0).value();
                    Tensor v = LogitAdjustment.cosineScores(features(ce, val.inputs()), w);
                    param = LogitAdjustment.tune(grid(0, 1.5, 0.1),
                            g -> balanced(LogitAdjustment.multiplicative(v, counts, g), val.y()));
                    testLogits = LogitAdjustment.multiplicative(LogitAdjustment.cosineScores(features(ce,
                            task.test().inputs()), w), counts, param);
                }
                case LDAM -> testLogits = train(x, y, MarginAndDecoupling.ldam(counts, 2.0, 1f, null), null, false,
                        steps, seed).predict(task.test().inputs());
                case LDAM_DRW -> testLogits = train(x, y, MarginAndDecoupling.ldam(counts, 2.0, 1f, null),
                        MarginAndDecoupling.ldam(counts, 2.0, 1f, MarginAndDecoupling.drwWeights(counts, 0.999)), false,
                        steps, seed).predict(task.test().inputs());
                case CRT -> testLogits = retrainHead(ce, x, y, steps / 3, seed).predict(task.test().inputs());
                case TAU_NORM -> {
                    Tensor w = ce.head().parameters().get(0).value();
                    Tensor fv = features(ce, val.inputs());
                    param = LogitAdjustment.tune(grid(0, 1.5, 0.1),
                            t -> balanced(fv.matmul(MarginAndDecoupling.tauNormalize(w, t)), val.y()));
                    testLogits = features(ce, task.test().inputs()).matmul(MarginAndDecoupling.tauNormalize(w, param));
                }
                default -> throw new IllegalStateException();
            }
            double[] acc = LabelTasks.classAccuracy(testLogits, task.test().y(), CLASSES);
            out.add(new Result(m, LabelTasks.mean(acc), (acc[0] + acc[1] + acc[2]) / 3,
                    (acc[3] + acc[4] + acc[5] + acc[6]) / 4, (acc[7] + acc[8] + acc[9]) / 3, param));
        }
        return out;
    }

    private static LabelTasks.Net train(float[][] x, int[] y, Loss loss, Loss lastStage, boolean balancedSampling,
            int steps, long seed) {
        LabelTasks.Net net = LabelTasks.mlp(x[0].length, 64, 32, CLASSES, seed + 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        int[] all = LabelTasks.range(y.length);
        for (int s = 0; s < steps; s++) {
            int[] b = balancedSampling ? MarginAndDecoupling.classBalanced(y, CLASSES, 64, r)
                    : LabelTasks.draw(64, all, r);
            Loss l = lastStage != null && s >= steps * 2 / 3 ? lastStage : loss;
            l.compute(net.logits(Var.of(LabelTasks.rows(x, b)), true), SoftTargets.ids(LabelTasks.labels(y, b)))
                    .backward();
            opt.step(net.parameters());
        }
        return net;
    }

    private static LabelTasks.Net retrainHead(LabelTasks.Net trained, float[][] x, int[] y, int steps, long seed) {
        LabelTasks.Net fresh = LabelTasks.mlp(x[0].length, 64, 32, CLASSES, seed + 60);
        LabelTasks.Net net = new LabelTasks.Net(trained.trunk(), fresh.head());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 61);
        for (int s = 0; s < steps; s++) {
            int[] b = MarginAndDecoupling.classBalanced(y, CLASSES, 64, r);
            Tensor f = features(trained, LabelTasks.rows(x, b));
            RobustLosses.crossEntropy().compute(net.head().forward(Var.of(f), true),
                    SoftTargets.ids(LabelTasks.labels(y, b))).backward();
            opt.step(net.head().parameters());
        }
        return net;
    }

    private static Tensor features(LabelTasks.Net net, Tensor x) {
        return Var.noGrad(() -> net.trunk().forward(Var.of(x), false).value());
    }

    private static double balanced(Tensor logits, int[] y) {
        return LabelTasks.mean(LabelTasks.classAccuracy(logits, y, CLASSES));
    }

    private static double[] grid(double from, double to, double step) {
        int n = (int) Math.round((to - from) / step) + 1;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = from + i * step;
        }
        return out;
    }

    /**
     * Prints every method at ratios 10, 50 and 100, three seeds, 3x the test budget.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (double ratio : new double[] {10, 50, 100}) {
            double[][] sum = new double[Method.values().length][2];
            for (long seed = 1; seed <= 3; seed++) {
                List<Result> rows = run(ratio, 1500, seed);
                for (Result r : rows) {
                    sum[r.method().ordinal()][0] += r.balanced() / 3;
                    sum[r.method().ordinal()][1] += r.tail() / 3;
                }
            }
            for (Method m : Method.values()) {
                System.out.printf("ratio %3.0f %-22s balanced %.3f tail %.3f%n", ratio, m, sum[m.ordinal()][0],
                        sum[m.ordinal()][1]);
            }
        }
    }
}
