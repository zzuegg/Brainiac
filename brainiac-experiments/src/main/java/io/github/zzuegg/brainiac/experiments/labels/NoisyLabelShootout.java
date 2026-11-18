package io.github.zzuegg.brainiac.experiments.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.labels.EarlyLearningRegularizer;
import io.github.zzuegg.brainiac.experimental.labels.RobustLosses;
import io.github.zzuegg.brainiac.experimental.labels.SampleSelection;
import io.github.zzuegg.brainiac.experimental.labels.TransitionCorrection;
import io.github.zzuegg.brainiac.experimental.semisup.FixMatch;
import io.github.zzuegg.brainiac.experimental.semisup.MixMatch;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learning with noisy labels on the digits task (W10 10.1.7 to 10.1.10): noise-robust losses
 * (GCE, SCE, NCE+RCE and the Joint Asymmetric Loss), sample selection (co-teaching,
 * co-teaching+, DivideMix), early-learning regularisation and transition-matrix loss correction
 * (forward correction with an anchor-estimated and the true {@code T}, backward correction,
 * T-revision), against cross-entropy, label smoothing, focal loss, MAE, cross-entropy with early
 * stopping on a noisy validation split, and EL2N pruning. Labels are corrupted by symmetric,
 * pair-flip or instance-dependent noise ({@link LabelCorruption}); accuracy is always measured on
 * clean test labels, both at the end and at its best evaluation (the gap shows memorisation).
 */
public final class NoisyLabelShootout {

    private NoisyLabelShootout() {
    }

    /** The noise models. */
    public enum Noise {
        /** Uniform flips to another class. */
        SYMMETRIC,
        /** Flips to the next class. */
        PAIR_FLIP,
        /** Feature-dependent flips. */
        INSTANCE
    }

    /** The compared methods. */
    public enum Method {
        /** Cross-entropy. */
        CE,
        /** Cross-entropy with label smoothing 0.1 (core {@code Loss.crossEntropyFromLogits}). */
        LABEL_SMOOTHING,
        /** Multiclass focal loss, gamma 2. */
        FOCAL,
        /** Mean absolute error. */
        MAE,
        /** Cross-entropy, model chosen by accuracy on a noisy validation split. */
        CE_EARLY_STOP,
        /** Generalised cross-entropy, q 0.7. */
        GCE,
        /** Symmetric cross-entropy. */
        SCE,
        /** NCE + RCE, the active-passive loss. */
        NCE_RCE,
        /** NCE + AMSE, the Joint Asymmetric Loss. */
        JAL,
        /** Co-teaching. */
        COTEACHING,
        /** Co-teaching+. */
        COTEACHING_PLUS,
        /** DivideMix (GMM split, two networks, MixMatch). */
        DIVIDEMIX,
        /** EL2N pruning of the hardest share before training. */
        EL2N_PRUNE,
        /** Early-learning regularisation. */
        ELR,
        /** Forward correction with T estimated from mean anchors on held-out noisy rows. */
        FORWARD_EST,
        /** Forward correction with the true T. */
        FORWARD_ORACLE,
        /** Backward correction with the anchor-estimated T. */
        BACKWARD_EST,
        /** T-revision from the anchor estimate. */
        T_REVISION
    }

    /**
     * One run's scores.
     *
     * @param method the method
     * @param finalAccuracy clean test accuracy at the end
     * @param bestAccuracy the best clean test accuracy at any evaluation (the early-stopped
     *     accuracy for CE_EARLY_STOP)
     * @param noisyFit the share of flipped training examples predicted as their noisy label at the
     *     end (memorisation)
     * @param noisyCorrected the share of flipped training examples predicted as their true label
     * @param splitPrecision for selection methods, the share of selected examples that are clean
     *     (else NaN)
     * @param splitRecall for selection methods, the share of clean examples selected (else NaN)
     * @param transitionError for transition methods, the mean absolute error of the used T (else NaN)
     */
    public record Result(Method method, double finalAccuracy, double bestAccuracy, double noisyFit,
            double noisyCorrected, double splitPrecision, double splitRecall, double transitionError) {
    }

    private static final int CLASSES = 10;
    private static final int BATCH = 64;

    /**
     * Runs one method.
     *
     * @param method the method
     * @param noise the noise model
     * @param rate the noise rate
     * @param steps training steps
     * @param seed the seed
     * @return the scores
     */
    public static Result run(Method method, Noise noise, double rate, int steps, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        ToyTasks.Data train = task.train();
        LabelCorruption.Noisy noisy = switch (noise) {
            case SYMMETRIC -> LabelCorruption.symmetric(train.y(), CLASSES, rate, seed + 1);
            case PAIR_FLIP -> LabelCorruption.pairFlip(train.y(), CLASSES, rate, seed + 1);
            case INSTANCE -> LabelCorruption.instanceDependent(train.x(), train.y(), CLASSES, rate, seed + 1);
        };
        return switch (method) {
            case COTEACHING, COTEACHING_PLUS -> coTeaching(method, task, noisy, rate, steps, seed);
            case DIVIDEMIX -> divideMix(task, noisy, steps, seed);
            default -> single(method, task, noisy, rate, steps, seed);
        };
    }

    private static Result single(Method method, LabelTasks.Task task, LabelCorruption.Noisy noisy, double rate,
            int steps, long seed) {
        ToyTasks.Data train = task.train();
        int n = train.size();
        int[] y = noisy.noisy();
        int[] pool = LabelTasks.range(n);
        int[] validation = null;
        double splitPrecision = Double.NaN;
        double splitRecall = Double.NaN;
        double tError = Double.NaN;
        double[][] transition = null;
        if (method == Method.CE_EARLY_STOP) {
            pool = java.util.Arrays.copyOfRange(LabelTasks.range(n), 0, n - 200);
            validation = java.util.Arrays.copyOfRange(LabelTasks.range(n), n - 200, n);
        }
        if (method == Method.EL2N_PRUNE) {
            LabelTasks.Net probe = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 30);
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed + 31);
            DataPruner pruner = new DataPruner(n);
            Tensor all = train.inputs();
            Tensor ids = SoftTargets.ids(y);
            for (int s = 0; s < steps / 5; s++) {
                int[] b = LabelTasks.draw(BATCH, pool, r);
                RobustLosses.crossEntropy().compute(probe.logits(Var.of(LabelTasks.rows(train.x(), b)), true),
                        SoftTargets.ids(LabelTasks.labels(y, b))).backward();
                opt.step(probe.parameters());
                if (s % 20 == 19) {
                    pruner.record(probe.predict(all), ids);
                }
            }
            pool = DataPruner.keep(pruner.el2n(), 1 - rate, rate);
            int[][] pr = precisionRecall(pool, noisy);
            splitPrecision = pr[0][0] / (double) pr[0][1];
            splitRecall = pr[1][0] / (double) pr[1][1];
        }
        if (method == Method.FORWARD_EST || method == Method.BACKWARD_EST || method == Method.T_REVISION) {
            LabelTasks.Net probe = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 40);
            // the probe fits 80% of the noisy rows; anchors come from its predictions on the other
            // 20%, which it has not memorised (Xia et al. estimate on held-out noisy data too)
            int cut = n * 4 / 5;
            trainPlain(probe, train, y, java.util.Arrays.copyOfRange(LabelTasks.range(n), 0, cut), steps / 2, seed + 41);
            int[] held = java.util.Arrays.copyOfRange(LabelTasks.range(n), cut, n);
            float[][] heldProbs = SoftTargets.probabilities(probe.predict(LabelTasks.rows(train.x(), held)), 1f);
            transition = TransitionCorrection.estimateMeanAnchors(heldProbs);
        } else if (method == Method.FORWARD_ORACLE) {
            transition = noisy.transition();
        }
        TransitionCorrection.Revision revision = method == Method.T_REVISION
                ? new TransitionCorrection.Revision(transition) : null;
        LabelTasks.Net net = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        List<Var> params = new ArrayList<>(net.parameters());
        if (revision != null) {
            params.addAll(revision.parameters());
        }
        EarlyLearningRegularizer elr = method == Method.ELR ? new EarlyLearningRegularizer(n, CLASSES, 0.7, 3f) : null;
        Loss loss = switch (method) {
            case LABEL_SMOOTHING -> Loss.crossEntropyFromLogits(0.1f);
            case FOCAL -> RobustLosses.focal(2f);
            case MAE -> RobustLosses.mae();
            case GCE -> RobustLosses.gce(0.7f);
            case SCE -> RobustLosses.sce(0.1f, 1f);
            case NCE_RCE -> RobustLosses.nceRce(1f, 1f);
            case JAL -> RobustLosses.jointAsymmetric(1f, 1f, 10f);
            default -> RobustLosses.crossEntropy();
        };
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        double best = 0;
        double bestVal = -1;
        double atBestVal = 0;
        final double[][] t = transition;
        for (int s = 0; s < steps; s++) {
            int[] b = LabelTasks.draw(BATCH, pool, r);
            Tensor ids = SoftTargets.ids(LabelTasks.labels(y, b));
            Var logits = net.logits(Var.of(LabelTasks.rows(train.x(), b)), true);
            Var l = switch (method) {
                case FORWARD_EST, FORWARD_ORACLE -> TransitionCorrection.forwardLoss(logits, ids, t);
                case BACKWARD_EST -> TransitionCorrection.backwardLoss(logits, ids, t);
                case T_REVISION -> s < steps / 4 ? TransitionCorrection.forwardLoss(logits, ids, t)
                        : revision.loss(logits, ids);
                default -> loss.compute(logits, ids);
            };
            if (elr != null) {
                l = l.add(elr.loss(logits, b));
            }
            l.backward();
            opt.step(params);
            if (s % 50 == 49 || s == steps - 1) {
                double acc = LabelTasks.accuracy(net, task.test());
                best = Math.max(best, acc);
                if (validation != null) {
                    double val = ToyTasks.accuracy(net.predict(LabelTasks.rows(train.x(), validation)),
                            LabelTasks.labels(y, validation));
                    if (val > bestVal) {
                        bestVal = val;
                        atBestVal = acc;
                    }
                }
            }
        }
        if (method == Method.T_REVISION) {
            tError = TransitionCorrection.meanAbsoluteError(revision.current(), noisy.transition());
        } else if (transition != null) {
            tError = TransitionCorrection.meanAbsoluteError(transition, noisy.transition());
        }
        double fin = LabelTasks.accuracy(net, task.test());
        double[] fit = memorisation(net, train, noisy);
        return new Result(method, fin, validation != null ? atBestVal : best, fit[0], fit[1], splitPrecision,
                splitRecall, tError);
    }

    private static void trainPlain(LabelTasks.Net net, ToyTasks.Data train, int[] y, int[] all, int steps,
            long seed) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[] b = LabelTasks.draw(BATCH, all, r);
            RobustLosses.crossEntropy().compute(net.logits(Var.of(LabelTasks.rows(train.x(), b)), true),
                    SoftTargets.ids(LabelTasks.labels(y, b))).backward();
            opt.step(net.parameters());
        }
    }

    private static Result coTeaching(Method method, LabelTasks.Task task, LabelCorruption.Noisy noisy, double rate,
            int steps, long seed) {
        ToyTasks.Data train = task.train();
        int[] y = noisy.noisy();
        int[] all = LabelTasks.range(train.size());
        LabelTasks.Net a = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        LabelTasks.Net b = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 50);
        Optimizer optA = Optimizer.adam(3e-3f);
        Optimizer optB = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        double best = 0;
        long selected = 0;
        long selectedClean = 0;
        long cleanSeen = 0;
        boolean[] flipped = noisy.flipped();
        for (int s = 0; s < steps; s++) {
            int[] batch = LabelTasks.draw(BATCH, all, r);
            Tensor x = LabelTasks.rows(train.x(), batch);
            int[] yb = LabelTasks.labels(y, batch);
            Tensor la = a.predict(x);
            Tensor lb = b.predict(x);
            int[] candidates = LabelTasks.range(batch.length);
            if (method == Method.COTEACHING_PLUS && s >= steps / 5) {
                candidates = SampleSelection.disagreement(la, lb);
            }
            double keep = SampleSelection.keepRate(s, rate, steps / 4);
            int[] forA = pick(candidates, SampleSelection.smallLoss(subset(SampleSelection.losses(lb, yb), candidates),
                    keep));
            int[] forB = pick(candidates, SampleSelection.smallLoss(subset(SampleSelection.losses(la, yb), candidates),
                    keep));
            if (s >= steps / 2) {
                for (int i : forA) {
                    selected++;
                    selectedClean += flipped[batch[i]] ? 0 : 1;
                }
                for (int i = 0; i < batch.length; i++) {
                    cleanSeen += flipped[batch[i]] ? 0 : 1;
                }
            }
            update(a, optA, x, yb, forA);
            update(b, optB, x, yb, forB);
            if (s % 50 == 49) {
                best = Math.max(best, LabelTasks.accuracy(a, task.test()));
            }
        }
        double fin = LabelTasks.accuracy(a, task.test());
        double[] fit = memorisation(a, train, noisy);
        // recall: selected clean among all clean seen; the batch is the denominator of selection
        double recall = selectedClean / (double) Math.max(1, cleanSeen);
        return new Result(method, fin, Math.max(best, fin), fit[0], fit[1], selectedClean / (double) Math.max(1, selected),
                recall, Double.NaN);
    }

    private static void update(LabelTasks.Net net, Optimizer opt, Tensor x, int[] y, int[] rows) {
        Tensor xs = SoftTargets.rows(x, rows);
        RobustLosses.crossEntropy().compute(net.logits(Var.of(xs), true), SoftTargets.ids(LabelTasks.labels(y, rows)))
                .backward();
        opt.step(net.parameters());
    }

    private static double[] subset(double[] v, int[] idx) {
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = v[idx[i]];
        }
        return out;
    }

    private static int[] pick(int[] from, int[] positions) {
        int[] out = new int[positions.length];
        for (int i = 0; i < positions.length; i++) {
            out[i] = from[positions[i]];
        }
        return out;
    }

    private static Result divideMix(LabelTasks.Task task, LabelCorruption.Noisy noisy, int steps, long seed) {
        ToyTasks.Data train = task.train();
        int n = train.size();
        int[] y = noisy.noisy();
        int[] all = LabelTasks.range(n);
        LabelTasks.Net[] nets = {LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10),
            LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 50)};
        Optimizer[] opts = {Optimizer.adam(3e-3f), Optimizer.adam(3e-3f)};
        SplittableRandom r = new SplittableRandom(seed + 20);
        Augmentation weak = LabelTasks.weak();
        MixMatch mixMatch = new MixMatch(0.5, 4f, 5f, steps / 2, true, true);
        int warm = steps / 5;
        double best = 0;
        double[][] clean = new double[2][];
        boolean[] flipped = noisy.flipped();
        double precision = Double.NaN;
        double recall = Double.NaN;
        for (int s = 0; s < steps; s++) {
            if (s < warm) {
                for (int k = 0; k < 2; k++) {
                    int[] b = LabelTasks.draw(BATCH, all, r);
                    update(nets[k], opts[k], LabelTasks.rows(train.x(), b), LabelTasks.labels(y, b),
                            LabelTasks.range(BATCH));
                }
            } else {
                if ((s - warm) % 50 == 0) {
                    for (int k = 0; k < 2; k++) {
                        clean[k] = SampleSelection.cleanProbability(SampleSelection.losses(nets[k].predict(
                                train.inputs()), y), seed + s + k);
                    }
                    int tp = 0;
                    int sel = 0;
                    int cl = 0;
                    for (int i = 0; i < n; i++) {
                        boolean pick = clean[0][i] > 0.5;
                        sel += pick ? 1 : 0;
                        tp += pick && !flipped[i] ? 1 : 0;
                        cl += flipped[i] ? 0 : 1;
                    }
                    precision = tp / (double) Math.max(1, sel);
                    recall = tp / (double) Math.max(1, cl);
                }
                for (int k = 0; k < 2; k++) {
                    double[] w = clean[1 - k];   // each network trains on its peer's division
                    List<Integer> lab = new ArrayList<>();
                    List<Integer> unl = new ArrayList<>();
                    for (int i = 0; i < n; i++) {
                        (w[i] > 0.5 ? lab : unl).add(i);
                    }
                    int[] li = LabelTasks.draw(BATCH / 2, toArray(lab), r);
                    if (lab.isEmpty()) {
                        lab.addAll(unl);
                    }
                    if (unl.isEmpty()) {
                        unl.addAll(lab);
                    }
                    int[] ui = LabelTasks.draw(BATCH / 2, toArray(unl), r);
                    Tensor xl = FixMatch.augment(LabelTasks.rows(train.x(), li), weak, r);
                    float[][] pl = average(nets[k].predict(FixMatch.augment(LabelTasks.rows(train.x(), li), weak, r)),
                            nets[k].predict(xl));
                    double[] wl = new double[li.length];
                    for (int i = 0; i < li.length; i++) {
                        wl[i] = w[li[i]];
                    }
                    float[][] yl = SampleSelection.refineLabels(LabelTasks.labels(y, li), wl, pl, 0.5);
                    Tensor xu = FixMatch.augment(LabelTasks.rows(train.x(), ui), weak, r);
                    float[][] guess = mixMatch.guess(List.of(nets[0].predict(xu), nets[1].predict(xu)));
                    MixMatch.Mixed mixed = mixMatch.mix(xl, yl, xu, guess, r);
                    mixMatch.loss(nets[k].logits(Var.of(mixed.x()), true), mixed, s - warm).backward();
                    opts[k].step(nets[k].parameters());
                }
            }
            if (s % 50 == 49) {
                best = Math.max(best, ensembleAccuracy(nets, task.test()));
            }
        }
        double fin = ensembleAccuracy(nets, task.test());
        double[] fit = memorisation(nets[0], train, noisy);
        return new Result(Method.DIVIDEMIX, fin, Math.max(best, fin), fit[0], fit[1], precision, recall, Double.NaN);
    }

    private static double ensembleAccuracy(LabelTasks.Net[] nets, ToyTasks.Data test) {
        return ToyTasks.accuracy(nets[0].predict(test.inputs()).add(nets[1].predict(test.inputs())), test.y());
    }

    private static float[][] average(Tensor a, Tensor b) {
        float[][] p = SoftTargets.probabilities(a, 1f);
        float[][] q = SoftTargets.probabilities(b, 1f);
        for (int i = 0; i < p.length; i++) {
            for (int c = 0; c < p[i].length; c++) {
                p[i][c] = (p[i][c] + q[i][c]) / 2;
            }
        }
        return p;
    }

    private static int[] toArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    /** @return {{kept clean, kept}, {kept clean, clean}} */
    private static int[][] precisionRecall(int[] kept, LabelCorruption.Noisy noisy) {
        boolean[] flipped = noisy.flipped();
        int tp = 0;
        for (int i : kept) {
            tp += flipped[i] ? 0 : 1;
        }
        int clean = 0;
        for (boolean f : flipped) {
            clean += f ? 0 : 1;
        }
        return new int[][] {{tp, kept.length}, {tp, clean}};
    }

    /** @return {share of flipped rows predicted as the noisy label, share predicted as the true label} */
    private static double[] memorisation(LabelTasks.Net net, ToyTasks.Data train, LabelCorruption.Noisy noisy) {
        int[] pred = SoftTargets.argmax(SoftTargets.probabilities(net.predict(train.inputs()), 1f));
        int flipped = 0;
        int fit = 0;
        int corrected = 0;
        for (int i = 0; i < pred.length; i++) {
            if (noisy.noisy()[i] != noisy.clean()[i]) {
                flipped++;
                fit += pred[i] == noisy.noisy()[i] ? 1 : 0;
                corrected += pred[i] == noisy.clean()[i] ? 1 : 0;
            }
        }
        return new double[] {fit / (double) Math.max(1, flipped), corrected / (double) Math.max(1, flipped)};
    }

    /**
     * Prints every method at 20/40/60% symmetric, 20/40% pair-flip and 40% instance-dependent
     * noise, at 3x the test budget and three seeds.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        Object[][] settings = {{Noise.SYMMETRIC, 0.2}, {Noise.SYMMETRIC, 0.4}, {Noise.SYMMETRIC, 0.6},
            {Noise.PAIR_FLIP, 0.2}, {Noise.PAIR_FLIP, 0.4}, {Noise.INSTANCE, 0.4}};
        for (Object[] st : settings) {
            for (Method m : Method.values()) {
                double fin = 0;
                double best = 0;
                for (long seed = 1; seed <= 3; seed++) {
                    Result res = run(m, (Noise) st[0], (Double) st[1], 2400, seed);
                    fin += res.finalAccuracy() / 3;
                    best += res.bestAccuracy() / 3;
                }
                System.out.printf("%-9s %.1f %-15s final %.3f best %.3f%n", st[0], st[1], m, fin, best);
            }
        }
    }
}
