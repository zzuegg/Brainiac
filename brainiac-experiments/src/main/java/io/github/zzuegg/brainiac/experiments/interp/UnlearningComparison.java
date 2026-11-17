package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Unlearning memorised exceptions: gradient ascent, random labels, NPO, SimNPO and RMU
 * against retraining without the forget set.
 *
 * <p>Inputs are Gaussian in 16 dimensions and labelled by a fixed linear rule over four
 * classes, except for a forget set whose labels were changed to a random wrong class. The
 * model can only get those right by memorising them, much like a language model storing
 * a fact the rest of its data does not imply. Each method then unlearns the forget set
 * with the rule-following samples as the retain set.
 *
 * <ul>
 *   <li>forget accuracy: forget samples still given their memorised label;</li>
 *   <li>test accuracy: unseen rule-following samples;</li>
 *   <li>membership AUC: the loss-threshold attack between the forget set and unseen
 *       exceptions labelled the same way (retraining gives about 0.5);</li>
 *   <li>relearning attack: the model is fine-tuned for a few steps on half of the forget
 *       set, and the other half's memorised labels are checked. Since the labels are
 *       random, only knowledge left behind by unlearning can come back this way.</li>
 * </ul>
 *
 * <p>NPO and SimNPO coincide here: the memorised forget samples have reference
 * log-probabilities near 0, so NPO's reference term vanishes, and a class label has length
 * 1, so SimNPO's normalisation does nothing.
 *
 * <p>RMU needs forget inputs that differ from retain inputs (a hazardous topic); here they
 * come from the same distribution, so misdirecting their activations without also moving
 * the retain activations is hard, and RMU runs with a weaker retain weight (1) and a
 * larger step.
 *
 * <p>Zhang et al. 2024 (NPO, <a href="https://arxiv.org/abs/2404.05868">arXiv:2404.05868</a>),
 * Fan et al. 2024 (SimNPO, <a href="https://arxiv.org/abs/2410.07163">arXiv:2410.07163</a>),
 * Li et al. 2024 (RMU, <a href="https://arxiv.org/abs/2403.03218">arXiv:2403.03218</a>).
 */
public final class UnlearningComparison {

    /**
     * One method.
     *
     * @param method the method, "retrain" or "original"
     * @param forget forget samples given their memorised label
     * @param test unseen rule-following samples classified correctly
     * @param membershipAuc forget set against unseen exceptions, by loss
     * @param relearned held-out forget samples given their memorised label after the
     *     relearning attack
     */
    public record Result(String method, double forget, double test, double membershipAuc, double relearned) {
    }

    private static final int DIM = 16;
    private static final int CLASSES = 4;

    private UnlearningComparison() {
    }

    /**
     * Runs the comparison.
     *
     * @param retain rule-following training samples
     * @param exceptions memorised exceptions to forget
     * @param seed the seed
     * @return retraining, the original model, then each {@link MachineUnlearning.Method}
     */
    public static List<Result> run(int retain, int exceptions, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] rule = new float[DIM][CLASSES];
        for (float[] row : rule) {
            for (int c = 0; c < CLASSES; c++) {
                row[c] = (float) r.nextGaussian();
            }
        }
        Dataset retainSet = data(rule, retain, false, r);
        Dataset forgetSet = data(rule, exceptions, true, r);
        Dataset unseenExceptions = data(rule, exceptions, true, r);
        Dataset test = data(rule, 400, false, r);
        Dataset attackHalf = forgetSet.slice(0, exceptions / 2);
        Dataset heldHalf = forgetSet.slice(exceptions / 2, exceptions);

        List<Result> out = new ArrayList<>();
        Model retrained = train(retainSet, seed);
        out.add(measure("retrain", retrained, forgetSet, test, unseenExceptions, attackHalf, heldHalf, seed));
        Model original = train(concat(retainSet, forgetSet), seed);
        out.add(measure("original", original, forgetSet, test, unseenExceptions, attackHalf, heldHalf, seed));
        for (MachineUnlearning.Method method : MachineUnlearning.Method.values()) {
            Model m = original.copy(original.parameters().get(0).value().device());
            MachineUnlearning u = MachineUnlearning.of(m, Loss.CROSS_ENTROPY).method(method)
                    .optimizer(() -> Optimizer.adam(0.003f)).steps(150).batchSize(16).forgetLossCeiling(3)
                    .seed(seed).rmuLayer(1).rmuScale(3);
            if (method == MachineUnlearning.Method.RMU) {
                u.retainWeight(1).optimizer(() -> Optimizer.adam(0.01f));
            }
            u.run(forgetSet, retainSet);
            out.add(measure(method.name(), m, forgetSet, test, unseenExceptions, attackHalf, heldHalf, seed));
        }
        return out;
    }

    private static Model train(Dataset data, long seed) {
        Model m = Model.sequential().input(DIM).dense(64, Activation.TANH).dense(64, Activation.TANH)
                .dense(CLASSES, Activation.SOFTMAX).seed(seed).build();
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f)).batchSize(32).epochs(120).seed(seed)
                .fit(data);
        return m;
    }

    private static Result measure(String method, Model m, Dataset forget, Dataset test, Dataset unseen,
            Dataset attackHalf, Dataset heldHalf, long seed) {
        double f = MachineUnlearning.accuracy(m, forget);
        double t = MachineUnlearning.accuracy(m, test);
        double auc = MachineUnlearning.membershipAuc(m, forget, unseen);
        Model attacked = m.copy(m.parameters().get(0).value().device());
        Trainer.of(attacked).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.003f)).batchSize(attackHalf.size())
                .epochs(20).seed(seed).fit(attackHalf);
        return new Result(method, f, t, auc, MachineUnlearning.accuracy(attacked, heldHalf));
    }

    private static Dataset data(float[][] rule, int n, boolean exception, SplittableRandom r) {
        float[][] x = new float[n][DIM];
        float[][] y = new float[n][CLASSES];
        for (int i = 0; i < n; i++) {
            float[] score = new float[CLASSES];
            for (int d = 0; d < DIM; d++) {
                x[i][d] = (float) r.nextGaussian();
                for (int c = 0; c < CLASSES; c++) {
                    score[c] += x[i][d] * rule[d][c];
                }
            }
            int label = 0;
            for (int c = 1; c < CLASSES; c++) {
                label = score[c] > score[label] ? c : label;
            }
            if (exception) {
                label = (label + 1 + r.nextInt(CLASSES - 1)) % CLASSES;
            }
            y[i][label] = 1;
        }
        return Dataset.of(x, y);
    }

    private static Dataset concat(Dataset a, Dataset b) {
        float[][] x = new float[a.size() + b.size()][];
        float[][] y = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            Dataset d = i < a.size() ? a : b;
            int j = i < a.size() ? i : i - a.size();
            x[i] = d.featuresOf(j);
            y[i] = d.targetsOf(j);
        }
        return Dataset.of(x, y);
    }
}
