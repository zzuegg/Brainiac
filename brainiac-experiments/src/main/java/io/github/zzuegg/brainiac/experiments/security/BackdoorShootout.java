package io.github.zzuegg.brainiac.experiments.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.PoisonedSets;
import io.github.zzuegg.brainiac.experimental.privacy.LikelihoodRatioAttack;
import io.github.zzuegg.brainiac.experimental.security.BackdoorRepair;
import io.github.zzuegg.brainiac.experimental.security.Backdoors;
import io.github.zzuegg.brainiac.experimental.security.PoisonDetectors;
import io.github.zzuegg.brainiac.experimental.train.LatentAdversarialTrainer;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Wave 11 items 11.1.4 to 11.1.6 on {@link ToyTasks} digits (8x8, 10 classes, 1,000 training
 * images) with a ReLU MLP 64-64-32-10, target class 0:
 *
 * <ul>
 *   <li>{@link #attack}: BadNets patch, blended and clean-label poisoning
 *       ({@link PoisonedSets#backdoor}) against clean training and random-label poisoning at
 *       the same rate: clean accuracy and attack success rate.</li>
 *   <li>{@link #detection}: {@link PoisonDetectors} (spectral signatures, activation
 *       clustering, AGPD) against loss-based filtering, {@link DataPruner}'s EL2N and random
 *       removal: detection AUROC (overall and inside the target class), recall when the top
 *       {@code 1.5 ε} of every class is removed (AGPD removes what it flags), and attack
 *       success and clean accuracy after retraining on the filtered set.</li>
 *   <li>{@link #repair}: {@link BackdoorRepair} (Neural Cleanse with unlearning,
 *       fine-pruning, ANP) with a 5% clean set for the defender, against plain fine-tuning
 *       and {@link LatentAdversarialTrainer} adapted to the classifier (clean inputs, true
 *       labels wanted, the target class unwanted).</li>
 * </ul>
 */
public final class BackdoorShootout {

    /** The backdoor target class. */
    public static final int TARGET = 0;
    static final int SIDE = 8;
    static final int STEPS = 400;

    /**
     * An attack's outcome.
     *
     * @param attack the poisoning
     * @param rate the poison rate
     * @param clean clean test accuracy
     * @param success the trigger's success rate on non-target test images
     */
    public record Attack(String attack, double rate, double clean, double success) {
    }

    /**
     * A detector's outcome.
     *
     * @param detector the detector
     * @param auroc detection AUROC
     * @param recall poisoned samples among the removed ones, over all poisoned
     * @param clean clean accuracy after retraining on the rest
     * @param success attack success after retraining
     * @param targetAuroc detection AUROC inside the target class only
     * @param detail method-specific numbers (AGPD's z-scores)
     */
    public record Detection(String detector, double auroc, double recall, double clean, double success,
            double targetAuroc, String detail) {
    }

    /**
     * A repair's outcome.
     *
     * @param method the repair
     * @param clean clean accuracy after repair
     * @param success attack success after repair
     * @param target the class Neural Cleanse flags, else -1
     * @param detail method-specific numbers (Neural Cleanse's target mask norm and anomaly
     *     index, the pruned neuron count)
     */
    public record Repair(String method, double clean, double success, int target, String detail) {
    }

    /** Training, test and defender data. */
    record Data(float[][] x, int[] y, float[][] testX, int[] testY, float[][] cleanX, int[] cleanY,
            float[] pattern) {
    }

    private BackdoorShootout() {
    }

    static Data data(long seed) {
        ToyTasks.Data tr = ToyTasks.digits(10, 100, SIDE, 0.05, seed, seed + 1);
        ToyTasks.Data te = ToyTasks.digits(10, 30, SIDE, 0.05, seed, seed + 2);
        ToyTasks.Data cl = ToyTasks.digits(10, 5, SIDE, 0.05, seed, seed + 3);
        return new Data(tr.x(), tr.y(), te.x(), te.y(), cl.x(), cl.y(), Backdoors.pattern(SIDE * SIDE, seed + 4));
    }

    static Model net(long seed) {
        return Model.sequential().input(SIDE * SIDE).seed(seed).dense(64, Activation.RELU).dense(32, Activation.RELU)
                .dense(10, Activation.IDENTITY).build();
    }

    static Model trained(float[][] x, int[] y, long seed) {
        Model m = net(seed);
        BackdoorRepair.fineTune(m, x, y, STEPS, 3e-3f, seed + 9);
        return m;
    }

    static UnaryOperator<Var> logits(Model m) {
        return v -> m.forward(v);
    }

    static double accuracy(Model m, float[][] x, int[] y) {
        return ToyTasks.accuracy(m.predict(Tensor.of(x)), y);
    }

    static PoisonedSets.Poisoned poison(Data d, Backdoors.Kind kind, double rate, long seed) {
        UnaryOperator<Var> surrogate = null;
        if (kind == Backdoors.Kind.CLEAN_LABEL) {
            surrogate = logits(trained(d.x(), d.y(), seed + 50));
        }
        return PoisonedSets.backdoor(d.x(), d.y(), kind, rate, TARGET, SIDE, d.pattern(), surrogate, 0.3f, seed + 7);
    }

    /**
     * Trains on poisoned data.
     *
     * @param kind the trigger, or {@code null} for random-label poisoning (no trigger; success
     *     then measured with the patch)
     * @param rate the poison rate, 0 for clean training
     * @param seed the seed
     * @return the outcome
     */
    public static Attack attack(Backdoors.Kind kind, double rate, long seed) {
        Data d = data(seed);
        PoisonedSets.Poisoned p = kind == null ? PoisonedSets.randomLabels(d.x(), d.y(), 10, rate, seed + 7)
                : poison(d, kind, rate, seed);
        Model m = trained(p.x(), p.y(), seed);
        Backdoors.Kind measured = kind == null ? Backdoors.Kind.PATCH : kind;
        String name = rate == 0 ? "clean" : kind == null ? "random labels" : kind.name();
        return new Attack(name, rate, accuracy(m, d.testX(), d.testY()),
                Backdoors.attackSuccess(logits(m), d.testX(), d.testY(), measured, SIDE, d.pattern(), TARGET));
    }

    /**
     * Detection, then retraining on the filtered set.
     *
     * @param kind the trigger
     * @param rate the poison rate
     * @param seed the seed
     * @return one row per detector: SPECTRAL, ACTIVATION_CLUSTERING, AGPD, LOSS, EL2N, RANDOM
     */
    public static List<Detection> detection(Backdoors.Kind kind, double rate, long seed) {
        Data d = data(seed);
        PoisonedSets.Poisoned p = poison(d, kind, rate, seed);
        Model m = net(seed);
        BackdoorRepair.fineTune(m, p.x(), p.y(), 60, 3e-3f, seed + 9);
        DataPruner pruner = new DataPruner(p.y().length);
        pruner.record(m.predict(Tensor.of(p.x())), ids(p.y()));
        double[] el2n = pruner.el2n();
        BackdoorRepair.fineTune(m, p.x(), p.y(), STEPS - 60, 3e-3f, seed + 10);
        double[][] feats = PoisonDetectors.features(m, p.x(), 1);
        double[][] grads = PoisonDetectors.activationGradients(m, p.x(), p.y(), 0);
        double[][] cleanGrads = PoisonDetectors.activationGradients(m, d.cleanX(), d.cleanY(), 0);
        double[][] refs = new double[10][grads[0].length];
        int[] counts = new int[10];
        for (int i = 0; i < cleanGrads.length; i++) {
            counts[d.cleanY()[i]]++;
            for (int j = 0; j < refs[0].length; j++) {
                refs[d.cleanY()[i]][j] += cleanGrads[i][j];
            }
        }
        PoisonDetectors.Agpd agpd = PoisonDetectors.agpd(grads, p.y(), 10, refs);
        double[] random = new double[p.y().length];
        SplittableRandom r = new SplittableRandom(seed + 11);
        for (int i = 0; i < random.length; i++) {
            random[i] = r.nextDouble();
        }
        String[] names = {"SPECTRAL", "ACTIVATION_CLUSTERING", "AGPD", "LOSS", "EL2N", "RANDOM"};
        double[][] scores = {PoisonDetectors.spectral(feats, p.y(), 10),
            PoisonDetectors.activationClustering(feats, p.y(), 10, 3, seed), agpd.suspicion(),
            PoisonDetectors.lossScores(m, p.x(), p.y()), el2n, random};
        List<Detection> out = new ArrayList<>();
        for (int k = 0; k < names.length; k++) {
            boolean[] removed = k == 2 ? agpd.flagged() : perClassTop(scores[k], p.y(), rate);
            int hit = 0;
            List<Integer> keep = new ArrayList<>();
            for (int i = 0; i < removed.length; i++) {
                hit += removed[i] && p.poisoned()[i] ? 1 : 0;
                if (!removed[i]) {
                    keep.add(i);
                }
            }
            PoisonedSets.Poisoned kept = p.rows(keep.stream().mapToInt(Integer::intValue).toArray());
            boolean retrain = k != 3 && k != 4;
            double clean = Double.NaN;
            double success = Double.NaN;
            if (retrain) {
                Model again = trained(kept.x(), kept.y(), seed);
                clean = accuracy(again, d.testX(), d.testY());
                success = Backdoors.attackSuccess(logits(again), d.testX(), d.testY(), kind, SIDE, d.pattern(),
                        TARGET);
            }
            int[] inTarget = rowsOf(p.y(), TARGET);
            double[] ts = new double[inTarget.length];
            boolean[] tp = new boolean[inTarget.length];
            for (int i = 0; i < inTarget.length; i++) {
                ts[i] = scores[k][inTarget[i]];
                tp[i] = p.poisoned()[inTarget[i]];
            }
            String detail = k != 2 ? "" : String.format("z target %.2f, largest other %.2f, flagged %s",
                    agpd.zScores()[TARGET], maxOther(agpd.zScores()), java.util.Arrays.toString(agpd.targets()));
            out.add(new Detection(names[k], LikelihoodRatioAttack.auc(scores[k], p.poisoned()),
                    hit / (double) Math.max(1, p.count()), clean, success, LikelihoodRatioAttack.auc(ts, tp), detail));
        }
        return out;
    }

    /**
     * The repair methods on one backdoored model.
     *
     * @param kind the trigger
     * @param rate the poison rate
     * @param seed the seed
     * @return rows BACKDOORED, FINE_TUNE, NEURAL_CLEANSE, FINE_PRUNING, ANP, LAT
     */
    public static List<Repair> repair(Backdoors.Kind kind, double rate, long seed) {
        Data d = data(seed);
        PoisonedSets.Poisoned p = poison(d, kind, rate, seed);
        Model bad = trained(p.x(), p.y(), seed);
        List<Repair> out = new ArrayList<>();
        out.add(row("BACKDOORED", bad, d, kind, -1, ""));
        Model ft = copy(bad);
        BackdoorRepair.fineTune(ft, d.cleanX(), d.cleanY(), 200, 1e-3f, seed + 20);
        out.add(row("FINE_TUNE", ft, d, kind, -1, ""));
        Model nc = copy(bad);
        BackdoorRepair.Cleanse cleanse = BackdoorRepair.neuralCleanse(nc, d.cleanX(), 10, 300, 1e-3f, seed + 21);
        BackdoorRepair.unlearn(nc, cleanse, d.cleanX(), d.cleanY(), 200, seed + 22);
        double median = java.util.Arrays.stream(cleanse.maskNorms()).sorted().toArray()[5];
        out.add(row("NEURAL_CLEANSE", nc, d, kind, cleanse.target(), String.format(
                "mask norm of class %d %.1f (median %.1f, anomaly %.2f)", TARGET, cleanse.maskNorms()[TARGET], median,
                cleanse.anomaly()[TARGET])));
        Model fp = copy(bad);
        BackdoorRepair.finePrune(fp, 1, d.cleanX(), d.cleanY(), 0.5, 200, seed + 23);
        out.add(row("FINE_PRUNING", fp, d, kind, -1, "16 of 32 neurons pruned"));
        Model anp = copy(bad);
        boolean[] pruned = BackdoorRepair.anp(anp, 0, d.cleanX(), d.cleanY(), 0.4f, 0.2f, 200, 0.2, seed + 24);
        int count = 0;
        for (boolean b : pruned) {
            count += b ? 1 : 0;
        }
        out.add(row("ANP", anp, d, kind, -1, count + " of 64 first-layer neurons pruned"));
        Model lat = copy(bad);
        LatentAdversarialTrainer trainer = LatentAdversarialTrainer.of(lat, 1).optimizer(Optimizer.adam(1e-3f))
                .epsilon(2f).attackSteps(5);
        SplittableRandom r = new SplittableRandom(seed + 25);
        for (int s = 0; s < 200; s++) {
            int b = 32;
            float[][] bx = new float[b][];
            float[] by = new float[b];
            float[] bt = new float[b];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(d.cleanX().length);
                bx[i] = d.cleanX()[j];
                by[i] = d.cleanY()[j];
                bt[i] = TARGET;
            }
            Tensor x = Tensor.of(bx);
            Tensor wanted = Tensor.of(Shape.of(b), by);
            trainer.step(x, wanted, Tensor.of(Shape.of(b), bt), x, wanted);
        }
        out.add(row("LAT", lat, d, kind, -1, ""));
        return out;
    }

    private static Repair row(String name, Model m, Data d, Backdoors.Kind kind, int target, String detail) {
        return new Repair(name, accuracy(m, d.testX(), d.testY()), Backdoors.attackSuccess(logits(m), d.testX(),
                d.testY(), kind, SIDE, d.pattern(), TARGET), target, detail);
    }

    /** Removes the top {@code 1.5 rate} share of every class, the spectral-signature budget. */
    private static boolean[] perClassTop(double[] scores, int[] y, double rate) {
        boolean[] out = new boolean[y.length];
        for (int c = 0; c < 10; c++) {
            int[] rows = rowsOf(y, c);
            double[] s = new double[rows.length];
            for (int i = 0; i < rows.length; i++) {
                s[i] = scores[rows[i]];
            }
            boolean[] top = PoisonDetectors.top(s, (int) Math.round(1.5 * rate * rows.length));
            for (int i = 0; i < rows.length; i++) {
                out[rows[i]] = top[i];
            }
        }
        return out;
    }

    private static int[] rowsOf(int[] y, int c) {
        return java.util.stream.IntStream.range(0, y.length).filter(i -> y[i] == c).toArray();
    }

    private static double maxOther(double[] z) {
        double m = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < z.length; c++) {
            if (c != TARGET) {
                m = Math.max(m, z[c]);
            }
        }
        return m;
    }

    private static Model copy(Model m) {
        return m.copy(m.parameters().get(0).value().device());
    }

    private static Tensor ids(int[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = y[i];
        }
        return Tensor.of(Shape.of(v.length), v);
    }
}
