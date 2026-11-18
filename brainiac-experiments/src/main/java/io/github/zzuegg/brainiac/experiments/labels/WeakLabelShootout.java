package io.github.zzuegg.brainiac.experiments.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.anomaly.DeepSvdd;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.labels.AttentionMil;
import io.github.zzuegg.brainiac.experimental.labels.LabelModel;
import io.github.zzuegg.brainiac.experimental.labels.LabelProportions;
import io.github.zzuegg.brainiac.experimental.labels.PartialLabels;
import io.github.zzuegg.brainiac.experimental.labels.PositiveUnlabeled;
import io.github.zzuegg.brainiac.experimental.labels.RobustLosses;
import io.github.zzuegg.brainiac.experimental.ml.IsolationForest;
import io.github.zzuegg.brainiac.experimental.nn.SetEncoder;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Weak and bag-level supervision on the digits task (W10 10.1.13 to 10.1.17):
 *
 * <ul>
 *   <li>{@link #pu}: positive-unlabelled learning (even against odd digits, 10% of positives
 *       labelled): uPU, nnPU (with the true and an Elkan-Noto prior) and 2-HNC against the biased
 *       PN classifier, Isolation Forest and Deep SVDD trained on positives, and a PN oracle.</li>
 *   <li>{@link #labelModel}: ten labelling functions with planted accuracies 0.55 to 0.9,
 *       coverages 0.2 to 0.8 and a duplicated weak function: majority vote, Dawid-Skene, the
 *       data-programming model with and without the correlation, the matrix-completion estimator
 *       and IRT weighting, plus end models trained on the probabilistic labels and on five gold
 *       labels per class.</li>
 *   <li>{@link #mil}: MNIST-bags-style bags (positive if a 9 is present): mean, max, attention
 *       and gated-attention MIL against Deep Sets and the Set Transformer ({@link SetEncoder}) and
 *       an instance-level oracle; bag AUC and the attention's instance-localisation AUC.</li>
 *   <li>{@link #partial}: candidate-label sets: PRODEN, the uniform-candidate loss, the CC loss,
 *       training on single-candidate rows only, GCE on a random candidate, and the supervised
 *       oracle.</li>
 *   <li>{@link #proportions}: bags with known class proportions: DLLP, DLLP plus optimal-transport
 *       pseudo-labels, bag-majority labels, the mean-map estimator and the supervised oracle.</li>
 * </ul>
 */
public final class WeakLabelShootout {

    private WeakLabelShootout() {
    }

    private static final int CLASSES = 10;

    // ------------------------------------------------------------------ PU

    /** The PU methods. */
    public enum PuMethod {
        /** Positive-negative training with all labels. */
        PN_ORACLE,
        /** Unlabelled taken as negative, threshold matched to the prior. */
        BIASED_PN,
        /** The unbiased PU risk. */
        UPU,
        /** The non-negative PU risk with the true prior. */
        NNPU,
        /** nnPU with the Elkan-Noto prior estimate. */
        NNPU_ESTIMATED_PRIOR,
        /** Isolation Forest trained on the labelled positives. */
        ISOLATION_FOREST,
        /** Deep SVDD trained on the labelled positives. */
        DEEP_SVDD,
        /** 2-HNC, transductive on the test rows. */
        TWO_HNC
    }

    /**
     * PU scores.
     *
     * @param method the method
     * @param accuracy test accuracy
     * @param auc test AUC (NaN for 2-HNC, which outputs a partition)
     * @param finalRisk the last training risk estimate (uPU and nnPU; else NaN)
     * @param prior the prior used (estimated for the estimated-prior row)
     */
    public record PuResult(PuMethod method, double accuracy, double auc, double finalRisk, double prior) {
    }

    /**
     * Runs one PU method.
     *
     * @param method the method
     * @param positiveClasses the positive digits ({@code {0, 2, 4, 6, 8}} for a prior of 0.5)
     * @param steps training steps
     * @param seed the seed
     * @return the scores
     */
    public static PuResult pu(PuMethod method, int[] positiveClasses, int steps, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        ToyTasks.Data train = task.train();
        LabelCorruption.PuSplit split = LabelCorruption.positiveUnlabelled(train.y(), positiveClasses, 0.1, seed + 1);
        boolean[] testPos = LabelCorruption.positiveUnlabelled(task.test().y(), positiveClasses, 0, seed).positive();
        int[] labelled = indices(split.labelled(), true);
        int[] unlabelled = indices(split.labelled(), false);
        double prior = split.prior();
        float[][] tx = task.test().x();
        double[] scores = new double[tx.length];
        double risk = Double.NaN;
        switch (method) {
            case ISOLATION_FOREST -> {
                IsolationForest f = IsolationForest.fit(toDouble(train.x(), labelled), 100, 64, seed);
                double[] s = f.score(toDouble(tx, LabelTasks.range(tx.length)));
                for (int i = 0; i < s.length; i++) {
                    scores[i] = -s[i];   // anomaly score: low means positive-like
                }
            }
            case DEEP_SVDD -> {
                DeepSvdd svdd = new DeepSvdd(train.features(), new int[] {32, 16}, seed);
                svdd.fit(toDouble(train.x(), labelled), null, 1f, steps / 2, 1e-3f, seed);
                double[] s = svdd.score(toDouble(tx, LabelTasks.range(tx.length)));
                for (int i = 0; i < s.length; i++) {
                    scores[i] = -s[i];
                }
            }
            case TWO_HNC -> {
                int nl = Math.min(labelled.length, 30);
                float[][] x = new float[nl + tx.length][];
                boolean[] seedPos = new boolean[x.length];
                for (int i = 0; i < nl; i++) {
                    x[i] = train.x()[labelled[i]];
                    seedPos[i] = true;
                }
                System.arraycopy(tx, 0, x, nl, tx.length);
                double testPrior = 0;
                for (boolean b : testPos) {
                    testPrior += b ? 1.0 / testPos.length : 0;
                }
                boolean[] side = PositiveUnlabeled.twoHnc(x, seedPos, testPrior, 8, 12);
                int right = 0;
                for (int i = 0; i < tx.length; i++) {
                    right += side[nl + i] == testPos[i] ? 1 : 0;
                }
                return new PuResult(method, right / (double) tx.length, Double.NaN, Double.NaN, testPrior);
            }
            default -> {
                double used = prior;
                if (method == PuMethod.NNPU_ESTIMATED_PRIOR) {
                    used = elkanNoto(train, split, steps, seed);
                }
                LabelTasks.Net net = LabelTasks.mlp(train.features(), 64, 32, 1, seed + 10);
                Optimizer opt = Optimizer.adam(3e-3f);
                SplittableRandom r = new SplittableRandom(seed + 20);
                int[] all = LabelTasks.range(train.size());
                for (int s = 0; s < steps; s++) {
                    Var loss;
                    if (method == PuMethod.PN_ORACLE) {
                        int[] b = LabelTasks.draw(128, all, r);
                        float[] sign = new float[b.length];
                        for (int i = 0; i < b.length; i++) {
                            sign[i] = split.positive()[b[i]] ? -1 : 1;
                        }
                        loss = net.logits(Var.of(LabelTasks.rows(train.x(), b)), true).reshape(b.length)
                                .mul(Tensor.of(Shape.of(b.length), sign)).sigmoid().mean();
                    } else {
                        Var zp = net.logits(Var.of(LabelTasks.rows(train.x(), LabelTasks.draw(32, labelled, r))), true);
                        Var zu = net.logits(Var.of(LabelTasks.rows(train.x(), LabelTasks.draw(128, unlabelled, r))),
                                true);
                        switch (method) {
                            case BIASED_PN -> loss = PositiveUnlabeled.biasedRisk(zp, zu, 0.5);
                            case UPU -> {
                                loss = PositiveUnlabeled.unbiasedRisk(zp, zu, used);
                                risk = loss.value().item();
                            }
                            default -> {
                                PositiveUnlabeled.Step step = PositiveUnlabeled.nonNegative(zp, zu, used, 0, 1);
                                loss = step.objective();
                                risk = step.risk();
                            }
                        }
                    }
                    loss.backward();
                    opt.step(net.parameters());
                }
                float[] z = net.predict(task.test().inputs()).toArray();
                for (int i = 0; i < z.length; i++) {
                    scores[i] = z[i];
                }
                if (method != PuMethod.BIASED_PN) {
                    int right = 0;
                    for (int i = 0; i < z.length; i++) {
                        right += (z[i] > 0) == testPos[i] ? 1 : 0;
                    }
                    return new PuResult(method, right / (double) z.length, auc(scores, testPos), risk, used);
                }
            }
        }
        // threshold matched to the prior among the unlabelled rows (here: the test rows)
        double[] sorted = scores.clone();
        Arrays.sort(sorted);
        double threshold = sorted[(int) Math.min(sorted.length - 1, Math.round((1 - prior) * sorted.length))];
        int right = 0;
        for (int i = 0; i < scores.length; i++) {
            right += (scores[i] >= threshold) == testPos[i] ? 1 : 0;
        }
        return new PuResult(method, right / (double) scores.length, auc(scores, testPos), risk, prior);
    }

    private static double elkanNoto(ToyTasks.Data train, LabelCorruption.PuSplit split, int steps, long seed) {
        int n = train.size();
        int cut = n * 4 / 5;
        LabelTasks.Net g = LabelTasks.mlp(train.features(), 64, 32, 1, seed + 70);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 71);
        int[] fit = Arrays.copyOfRange(LabelTasks.range(n), 0, cut);
        for (int s = 0; s < steps / 2; s++) {
            int[] b = LabelTasks.draw(128, fit, r);
            float[] t = new float[b.length];
            for (int i = 0; i < b.length; i++) {
                t[i] = split.labelled()[b[i]] ? 1 : 0;
            }
            Var p = g.logits(Var.of(LabelTasks.rows(train.x(), b)), true).reshape(b.length).sigmoid();
            io.github.zzuegg.brainiac.nn.Loss.BINARY_CROSS_ENTROPY.compute(p, Tensor.of(Shape.of(b.length), t))
                    .backward();
            opt.step(g.parameters());
        }
        List<Integer> heldPos = new ArrayList<>();
        int labelledCount = 0;
        for (int i = 0; i < n; i++) {
            labelledCount += split.labelled()[i] ? 1 : 0;
            if (i >= cut && split.labelled()[i]) {
                heldPos.add(i);
            }
        }
        int[] hp = heldPos.stream().mapToInt(Integer::intValue).toArray();
        float[] z = g.predict(LabelTasks.rows(train.x(), hp)).toArray();
        double[] gs = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            gs[i] = 1 / (1 + Math.exp(-z[i]));
        }
        return PositiveUnlabeled.elkanNotoPrior(gs, labelledCount / (double) n);
    }

    // ------------------------------------------------------------------ label model

    /** The label models. */
    public enum LabelMethod {
        /** Majority vote. */
        MAJORITY_VOTE,
        /** Dawid-Skene. */
        DAWID_SKENE,
        /** Data programming, independent functions. */
        DATA_PROGRAMMING,
        /** Data programming with the correlated pair modelled. */
        DATA_PROGRAMMING_CORRELATED,
        /** The matrix-completion (triplet) estimator. */
        MATRIX_COMPLETION,
        /** IRT weighting. */
        IRT
    }

    /**
     * Label-model scores.
     *
     * @param method the label model
     * @param coveredAccuracy accuracy of its labels on rows with a vote
     * @param accuracyError mean absolute error of the estimated function accuracies (NaN for MV)
     * @param endModel test accuracy of an MLP trained on its probabilistic labels
     */
    public record LabelResult(LabelMethod method, double coveredAccuracy, double accuracyError, double endModel) {
    }

    /**
     * Runs every label model and trains an end model on each, plus a gold-label baseline.
     *
     * @param correlatedWeakPair whether the duplicated function is a weak one (0.55) rather than the
     *     strongest
     * @param steps end-model training steps
     * @param seed the seed
     * @return one result per label model, then a final row whose end-model accuracy is training
     *     on five gold labels per class (method null)
     */
    public static List<LabelResult> labelModel(boolean correlatedWeakPair, int steps, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        ToyTasks.Data train = task.train();
        double[] acc = {0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.55, 0.55};
        double[] cov = {0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 0.8};
        if (!correlatedWeakPair) {
            acc[8] = 0.9;
            cov[8] = 0.5;
        }
        LabelCorruption.Votes votes = LabelCorruption.labellingFunctions(train.y(), CLASSES, acc, cov, true, seed + 1);
        double[] planted = acc.clone();
        planted[9] = Double.NaN;   // the copy's accuracy is set by its original
        List<LabelResult> out = new ArrayList<>();
        for (LabelMethod m : LabelMethod.values()) {
            LabelModel.Fit fit = switch (m) {
                case MAJORITY_VOTE -> LabelModel.majorityVote(votes.votes(), CLASSES);
                case DAWID_SKENE -> LabelModel.dawidSkene(votes.votes(), CLASSES, 30);
                case DATA_PROGRAMMING -> LabelModel.dataProgramming(votes.votes(), CLASSES, 30, new int[0][]);
                case DATA_PROGRAMMING_CORRELATED -> LabelModel.dataProgramming(votes.votes(), CLASSES, 30,
                        new int[][] {votes.correlated()});
                case MATRIX_COMPLETION -> LabelModel.matrixCompletion(votes.votes(), CLASSES);
                case IRT -> LabelModel.irtWeighted(votes.votes(), CLASSES);
            };
            double err = 0;
            int k = 0;
            for (int j = 0; j < planted.length; j++) {
                if (!Double.isNaN(planted[j]) && !Double.isNaN(fit.accuracies()[j])) {
                    err += Math.abs(fit.accuracies()[j] - planted[j]);
                    k++;
                }
            }
            double end = endModel(train, task.test(), fit.posteriors(), votes.votes(), steps, seed);
            out.add(new LabelResult(m, fit.coveredAccuracy(votes.votes(), train.y()), k == 0 ? Double.NaN : err / k,
                    end));
        }
        LabelCorruption.Split gold = LabelCorruption.labelledSplit(train.y(), CLASSES, 5, seed + 2);
        double[][] goldPost = new double[train.size()][CLASSES];
        int[][] goldVotes = new int[train.size()][1];
        for (int[] row : goldVotes) {
            row[0] = -1;
        }
        for (int i : gold.labelled()) {
            goldPost[i][train.y()[i]] = 1;
            goldVotes[i][0] = train.y()[i];
        }
        out.add(new LabelResult(null, 1, Double.NaN, endModel(train, task.test(), goldPost, goldVotes, steps, seed)));
        return out;
    }

    private static double endModel(ToyTasks.Data train, ToyTasks.Data test, double[][] posteriors, int[][] votes,
            int steps, long seed) {
        List<Integer> covered = new ArrayList<>();
        for (int i = 0; i < votes.length; i++) {
            for (int v : votes[i]) {
                if (v >= 0) {
                    covered.add(i);
                    break;
                }
            }
        }
        int[] rows = covered.stream().mapToInt(Integer::intValue).toArray();
        LabelTasks.Net net = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        for (int s = 0; s < steps; s++) {
            int[] b = LabelTasks.draw(64, rows, r);
            float[][] t = new float[b.length][CLASSES];
            for (int i = 0; i < b.length; i++) {
                for (int c = 0; c < CLASSES; c++) {
                    t[i][c] = (float) posteriors[b[i]][c];
                }
            }
            SoftTargets.softCrossEntropy(net.logits(Var.of(LabelTasks.rows(train.x(), b)), true), SoftTargets.tensor(t),
                    null, b.length).backward();
            opt.step(net.parameters());
        }
        return LabelTasks.accuracy(net, test);
    }

    // ------------------------------------------------------------------ MIL

    /** The MIL models. */
    public enum MilMethod {
        /** Mean pooling. */
        MEAN,
        /** Max pooling. */
        MAX,
        /** Attention pooling. */
        ATTENTION,
        /** Gated attention pooling. */
        GATED,
        /** Deep Sets. */
        DEEP_SETS,
        /** Set Transformer. */
        SET_TRANSFORMER,
        /** An instance classifier trained on instance labels; bag score is the max. */
        INSTANCE_ORACLE
    }

    /**
     * MIL scores.
     *
     * @param method the model
     * @param bagAuc test bag AUC
     * @param instanceAuc AUC of the attention weights for the key instances in positive test bags
     *     (NaN when the model has no attention)
     */
    public record MilResult(MilMethod method, double bagAuc, double instanceAuc) {
    }

    /**
     * Runs one MIL model.
     *
     * @param method the model
     * @param trainBags training bags
     * @param epochs passes over the bags
     * @param seed the seed
     * @return the scores
     */
    public static MilResult mil(MilMethod method, int trainBags, int epochs, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        LabelCorruption.Bags train = LabelCorruption.bags(task.train().y(), 9, trainBags, 5, 15, seed + 1);
        LabelCorruption.Bags test = LabelCorruption.bags(task.test().y(), 9, 100, 5, 15, seed + 2);
        SplittableRandom r = new SplittableRandom(seed + 3);
        int d = task.train().features();
        if (method == MilMethod.INSTANCE_ORACLE) {
            LabelTasks.Net net = LabelTasks.mlp(d, 64, 32, 1, seed + 10);
            Optimizer opt = Optimizer.adam(3e-3f);
            List<Integer> inst = new ArrayList<>();
            for (int[] b : train.members()) {
                for (int i : b) {
                    inst.add(i);
                }
            }
            int[] all = inst.stream().mapToInt(Integer::intValue).toArray();
            for (int s = 0; s < epochs * trainBags; s++) {
                int[] b = LabelTasks.draw(16, all, r);
                float[] sign = new float[b.length];
                for (int i = 0; i < b.length; i++) {
                    sign[i] = task.train().y()[b[i]] == 9 ? -1 : 1;
                }
                net.logits(Var.of(LabelTasks.rows(task.train().x(), b)), true).reshape(b.length)
                        .mul(Tensor.of(Shape.of(b.length), sign)).softplus().mean().backward();
                opt.step(net.parameters());
            }
            double[] scores = new double[test.members().length];
            for (int b = 0; b < scores.length; b++) {
                float[] z = net.predict(LabelTasks.rows(task.test().x(), test.members()[b])).toArray();
                double max = Double.NEGATIVE_INFINITY;
                for (float v : z) {
                    max = Math.max(max, v);
                }
                scores[b] = max;
            }
            return new MilResult(method, auc(scores, test.label()), Double.NaN);
        }
        Layer model = switch (method) {
            case DEEP_SETS -> new SetEncoder(d, 32, 1, SetEncoder.Kind.DEEP_SETS, 4, 4, r);
            case SET_TRANSFORMER -> new SetEncoder(d, 32, 1, SetEncoder.Kind.SET_TRANSFORMER, 4, 4, r);
            default -> new AttentionMil(d, 32, 16, 1, AttentionMil.Pooling.valueOf(method.name()), r);
        };
        Optimizer opt = Optimizer.adam(5e-3f);
        for (int e = 0; e < epochs; e++) {
            for (int b : LabelCorruption.permutation(trainBags, r)) {
                Tensor x = LabelTasks.rows(task.train().x(), train.members()[b]);
                Var z = model.forward(Var.of(x.reshape(1, x.shape().dim(0), d)), true).reshape(1);
                z.mul(train.label()[b] ? -1f : 1f).softplus().mean().backward();
                opt.step(model.parameters());
            }
        }
        double[] scores = new double[test.members().length];
        List<double[]> attScores = new ArrayList<>();
        List<boolean[]> attKeys = new ArrayList<>();
        for (int b = 0; b < scores.length; b++) {
            Tensor x = LabelTasks.rows(task.test().x(), test.members()[b]);
            Tensor z = Var.noGrad(() -> model.forward(Var.of(x.reshape(1, x.shape().dim(0), d)), false).value());
            scores[b] = z.toArray()[0];
            if (model instanceof AttentionMil mil && mil.attention() != null && test.label()[b]) {
                float[] a = mil.attention().toArray();
                double[] ad = new double[a.length];
                for (int i = 0; i < a.length; i++) {
                    ad[i] = a[i];
                }
                attScores.add(ad);
                attKeys.add(test.key()[b]);
            }
        }
        double instAuc = Double.NaN;
        if (!attScores.isEmpty() && method != MilMethod.MEAN) {
            int total = attScores.stream().mapToInt(a -> a.length).sum();
            double[] s = new double[total];
            boolean[] k = new boolean[total];
            int p = 0;
            for (int i = 0; i < attScores.size(); i++) {
                for (int j = 0; j < attScores.get(i).length; j++) {
                    s[p] = attScores.get(i)[j] * attScores.get(i).length;   // relative to uniform
                    k[p++] = attKeys.get(i)[j];
                }
            }
            instAuc = auc(s, k);
        }
        return new MilResult(method, auc(scores, test.label()), instAuc);
    }

    // ------------------------------------------------------------------ partial labels

    /** The partial-label methods. */
    public enum PartialMethod {
        /** True labels. */
        SUPERVISED,
        /** PRODEN. */
        PRODEN,
        /** Uniform weights over candidates. */
        UNIFORM,
        /** The classifier-consistent loss. */
        CC,
        /** Cross-entropy on the rows with a single candidate only. */
        SINGLE_CANDIDATE_ONLY,
        /** GCE on one random candidate per row. */
        GCE_RANDOM_CANDIDATE
    }

    /**
     * Runs one partial-label method. Each row gets between 0 and {@code 2 * meanDistractors}
     * distractors (uniform), so some rows have a single candidate.
     *
     * @param method the method
     * @param meanDistractors the mean number of distractors
     * @param biased whether distractors follow a confusion pattern
     * @param steps training steps
     * @param seed the seed
     * @return test accuracy
     */
    public static double partial(PartialMethod method, int meanDistractors, boolean biased, int steps, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        ToyTasks.Data train = task.train();
        int n = train.size();
        SplittableRandom cr = new SplittableRandom(seed + 1);
        boolean[][] cand = new boolean[n][];
        for (int i = 0; i < n; i++) {
            int dCount = cr.nextInt(2 * meanDistractors + 1);
            cand[i] = LabelCorruption.candidates(new int[] {train.y()[i]}, CLASSES, dCount, biased, seed * 7919 + i)[0];
        }
        int[] rows = LabelTasks.range(n);
        int[] randomCandidate = new int[n];
        if (method == PartialMethod.SINGLE_CANDIDATE_ONLY) {
            List<Integer> single = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int k = 0;
                for (boolean b : cand[i]) {
                    k += b ? 1 : 0;
                }
                if (k == 1) {
                    single.add(i);
                }
            }
            rows = single.stream().mapToInt(Integer::intValue).toArray();
        }
        for (int i = 0; i < n; i++) {
            List<Integer> c = new ArrayList<>();
            for (int k = 0; k < CLASSES; k++) {
                if (cand[i][k]) {
                    c.add(k);
                }
            }
            randomCandidate[i] = c.get(cr.nextInt(c.size()));
        }
        PartialLabels proden = new PartialLabels(cand);
        LabelTasks.Net net = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        for (int s = 0; s < steps; s++) {
            int[] b = LabelTasks.draw(64, rows, r);
            Var logits = net.logits(Var.of(LabelTasks.rows(train.x(), b)), true);
            Var loss = switch (method) {
                case SUPERVISED, SINGLE_CANDIDATE_ONLY -> RobustLosses.crossEntropy().compute(logits,
                        SoftTargets.ids(LabelTasks.labels(train.y(), b)));
                case PRODEN -> proden.prodenLoss(logits, b);
                case UNIFORM -> PartialLabels.uniformLoss(logits, b, cand);
                case CC -> PartialLabels.ccLoss(logits, b, cand);
                case GCE_RANDOM_CANDIDATE -> RobustLosses.gce(0.7f).compute(logits,
                        SoftTargets.ids(LabelTasks.labels(randomCandidate, b)));
            };
            loss.backward();
            opt.step(net.parameters());
            if (method == PartialMethod.PRODEN) {
                proden.update(logits.value(), b);
            }
        }
        return LabelTasks.accuracy(net, task.test());
    }

    // ------------------------------------------------------------------ label proportions

    /** The label-proportion methods. */
    public enum ProportionMethod {
        /** True labels for every instance. */
        SUPERVISED,
        /** DLLP. */
        DLLP,
        /** DLLP, then retraining on optimal-transport pseudo-labels. */
        DLLP_OT,
        /** Every instance labelled with its bag's majority class. */
        BAG_MAJORITY,
        /** The mean-map estimator. */
        MEAN_MAP
    }

    /**
     * Runs one label-proportion method.
     *
     * @param method the method
     * @param bagSize instances per bag
     * @param skewed whether bags are class-skewed
     * @param steps training steps
     * @param seed the seed
     * @return instance-level test accuracy
     */
    public static double proportions(ProportionMethod method, int bagSize, boolean skewed, int steps, long seed) {
        LabelTasks.Task task = LabelTasks.digits(CLASSES, 100, 30, 0.1, seed);
        ToyTasks.Data train = task.train();
        LabelCorruption.ProportionBags bags = LabelCorruption.proportionBags(train.y(), CLASSES, bagSize, skewed,
                seed + 1);
        if (method == ProportionMethod.MEAN_MAP) {
            LabelProportions.MeanMap mm = new LabelProportions.MeanMap(train.x(), bags.members(), bags.proportions());
            int right = 0;
            for (int i = 0; i < task.test().size(); i++) {
                right += mm.predict(task.test().x()[i]) == task.test().y()[i] ? 1 : 0;
            }
            return right / (double) task.test().size();
        }
        int[] pseudo = new int[train.size()];
        int[] bagOf = new int[train.size()];
        for (int b = 0; b < bags.members().length; b++) {
            int major = 0;
            for (int c = 1; c < CLASSES; c++) {
                if (bags.proportions()[b][c] > bags.proportions()[b][major]) {
                    major = c;
                }
            }
            for (int i : bags.members()[b]) {
                pseudo[i] = major;
                bagOf[i] = b;
            }
        }
        LabelTasks.Net net = LabelTasks.mlp(train.features(), 64, 32, CLASSES, seed + 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 20);
        int bagsPerStep = Math.max(1, 64 / bagSize);
        int[] covered = coveredRows(bags);
        int dllpSteps = method == ProportionMethod.DLLP_OT ? steps * 2 / 3 : steps;
        for (int s = 0; s < steps; s++) {
            Var loss;
            if (method == ProportionMethod.DLLP || (method == ProportionMethod.DLLP_OT && s < dllpSteps)) {
                loss = null;
                for (int k = 0; k < bagsPerStep; k++) {
                    int b = r.nextInt(bags.members().length);
                    Var l = LabelProportions.dllpLoss(net.logits(Var.of(LabelTasks.rows(train.x(), bags.members()[b])),
                            true), bags.proportions()[b]);
                    loss = loss == null ? l : loss.add(l);
                }
                loss = loss.mul(1f / bagsPerStep);
            } else {
                if (method == ProportionMethod.DLLP_OT && s == dllpSteps) {
                    for (int b = 0; b < bags.members().length; b++) {
                        float[][] p = SoftTargets.probabilities(net.predict(LabelTasks.rows(train.x(),
                                bags.members()[b])), 1f);
                        int[] lab = LabelProportions.transportLabels(p, bags.proportions()[b], 0.05);
                        for (int i = 0; i < lab.length; i++) {
                            pseudo[bags.members()[b][i]] = lab[i];
                        }
                    }
                }
                int[] b = LabelTasks.draw(64, covered, r);
                int[] y = method == ProportionMethod.SUPERVISED ? LabelTasks.labels(train.y(), b)
                        : LabelTasks.labels(pseudo, b);
                loss = RobustLosses.crossEntropy().compute(net.logits(Var.of(LabelTasks.rows(train.x(), b)), true),
                        SoftTargets.ids(y));
            }
            loss.backward();
            opt.step(net.parameters());
        }
        return LabelTasks.accuracy(net, task.test());
    }

    private static int[] coveredRows(LabelCorruption.ProportionBags bags) {
        List<Integer> out = new ArrayList<>();
        for (int[] b : bags.members()) {
            for (int i : b) {
                out.add(i);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    // ------------------------------------------------------------------ helpers

    private static int[] indices(boolean[] mask, boolean value) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < mask.length; i++) {
            if (mask[i] == value) {
                out.add(i);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static double[][] toDouble(float[][] x, int[] rows) {
        double[][] out = new double[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            out[i] = new double[x[rows[i]].length];
            for (int f = 0; f < out[i].length; f++) {
                out[i][f] = x[rows[i]][f];
            }
        }
        return out;
    }

    /**
     * The area under the ROC curve (ties count half).
     *
     * @param scores scores, higher meaning positive
     * @param positive the true labels
     * @return the AUC
     */
    public static double auc(double[] scores, boolean[] positive) {
        double pos = 0;
        double neg = 0;
        double wins = 0;
        for (int i = 0; i < scores.length; i++) {
            if (!positive[i]) {
                continue;
            }
            pos++;
            for (int j = 0; j < scores.length; j++) {
                if (!positive[j]) {
                    wins += scores[i] > scores[j] ? 1 : scores[i] == scores[j] ? 0.5 : 0;
                }
            }
        }
        for (boolean p : positive) {
            neg += p ? 0 : 1;
        }
        return wins / Math.max(1, pos * neg);
    }

    /**
     * Prints every part at 3x the test budget.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (int[] positives : new int[][] {{0, 2, 4, 6, 8}, {0, 1, 2}}) {
            for (PuMethod m : PuMethod.values()) {
                PuResult res = pu(m, positives, 1800, 1);
                System.out.printf("PU %s %-22s acc %.3f auc %.3f risk %.3f prior %.2f%n", Arrays.toString(positives), m,
                        res.accuracy(), res.auc(), res.finalRisk(), res.prior());
            }
        }
        for (boolean weakPair : new boolean[] {true, false}) {
            for (LabelResult res : labelModel(weakPair, 1200, 1)) {
                System.out.printf("LF weak pair %b %-28s covered %.3f acc err %.3f end %.3f%n", weakPair, res.method(),
                        res.coveredAccuracy(), res.accuracyError(), res.endModel());
            }
        }
        for (int bags : new int[] {50, 100, 400}) {
            for (MilMethod m : MilMethod.values()) {
                MilResult res = mil(m, bags, 10, 1);
                System.out.printf("MIL %3d bags %-16s bag AUC %.3f instance AUC %.3f%n", bags, m, res.bagAuc(),
                        res.instanceAuc());
            }
        }
        for (int d : new int[] {1, 3, 5}) {
            for (boolean biased : new boolean[] {false, true}) {
                for (PartialMethod m : PartialMethod.values()) {
                    System.out.printf("Partial %d %s %-22s acc %.3f%n", d, biased ? "biased" : "uniform", m,
                            partial(m, d, biased, 1800, 1));
                }
            }
        }
        for (int size : new int[] {8, 32, 128}) {
            for (boolean skewed : new boolean[] {false, true}) {
                for (ProportionMethod m : ProportionMethod.values()) {
                    System.out.printf("LLP bag %3d %s %-12s acc %.3f%n", size, skewed ? "skewed" : "random", m,
                            proportions(m, size, skewed, 1800, 1));
                }
            }
        }
    }
}
