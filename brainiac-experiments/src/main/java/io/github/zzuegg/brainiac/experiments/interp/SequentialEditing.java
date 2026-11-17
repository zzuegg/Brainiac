package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.nn.NullSpaceEditor;
import io.github.zzuegg.brainiac.experimental.nn.RomeEditor;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Many edits in a row: {@link RomeEditor} against {@link NullSpaceEditor} (AlphaEdit).
 *
 * <p>A classifier stores one fact per subject: a one-hot subject goes through
 * {@code Dense(128, tanh)}, the edited {@code Dense(64, tanh)} and a softmax over the
 * answers. Half of the subjects are rewritten one after another, each to a new answer;
 * the other half must keep theirs. Both editors see the same preserved keys: ROME uses
 * their covariance, AlphaEdit the null space of their Gram matrix.
 *
 * <p>The outcome after the whole sequence: edit success (edited subjects that give their
 * new answer at the end, so earlier edits must survive later ones) and specificity
 * (preserved subjects that keep their answer).
 *
 * <p>The null space of 80 preserved keys in 128 dimensions has 57 dimensions, fewer than
 * the 80 edits, so late AlphaEdit edits share directions with earlier ones and are written
 * less than fully (ridge {@code 1e-3}); ROME writes each edit exactly but drifts the
 * preserved facts.
 */
public final class SequentialEditing {

    /**
     * The result of one editor.
     *
     * @param method the editor
     * @param edits edits made
     * @param efficacy edits that held right after they were made
     * @param success edits that still hold at the end
     * @param specificity preserved facts kept at the end
     */
    public record Result(String method, int edits, double efficacy, double success, double specificity) {
    }

    private SequentialEditing() {
    }

    /**
     * Runs the experiment.
     *
     * @param subjects subjects stored
     * @param edits subjects rewritten, at most {@code subjects}
     * @param seed the seed
     * @return ROME, then AlphaEdit
     */
    public static List<Result> run(int subjects, int edits, long seed) {
        int classes = 8;
        float[][] x = new float[subjects][subjects];
        float[][] y = new float[subjects][classes];
        int[] fact = new int[subjects];
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < subjects; s++) {
            x[s][s] = 1;
            fact[s] = r.nextInt(classes);
            y[s][fact[s]] = 1;
        }
        Model base = Model.sequential().input(subjects).dense(128, Activation.TANH).dense(64, Activation.TANH)
                .dense(classes, Activation.SOFTMAX).seed(seed).build();
        Trainer.of(base).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).batchSize(32).epochs(150)
                .fit(Dataset.of(x, y));
        int[] target = new int[edits];
        for (int e = 0; e < edits; e++) {
            target[e] = (fact[e] + 1 + r.nextInt(classes - 1)) % classes;
        }
        float[][] keep = new float[subjects - edits][];
        for (int s = edits; s < subjects; s++) {
            keep[s - edits] = x[s];
        }
        Tensor preserved = Tensor.of(keep);

        List<Result> out = new ArrayList<>();
        out.add(new Result("trained", 0, 0, 0, kept(base, x, fact, edits)));
        Model rome = base.copy(base.parameters().get(0).value().device());
        double[][] cov = RomeEditor.keyCovariance(rome, 1, preserved, 1e-2);
        int held = 0;
        for (int e = 0; e < edits; e++) {
            RomeEditor.edit(rome, 1, Tensor.of(new float[][] {x[e]}), target[e], cov, 300);
            held += argmax(rome.predict(x[e])) == target[e] ? 1 : 0;
        }
        out.add(new Result("ROME", edits, held / (double) edits, success(rome, x, target),
                kept(rome, x, fact, edits)));
        Model alpha = base.copy(base.parameters().get(0).value().device());
        NullSpaceEditor editor = NullSpaceEditor.of(alpha, 1, preserved, 1e-3).ridge(1e-3);
        held = 0;
        for (int e = 0; e < edits; e++) {
            editor.edit(Tensor.of(new float[][] {x[e]}), target[e], 300);
            held += argmax(alpha.predict(x[e])) == target[e] ? 1 : 0;
        }
        out.add(new Result("AlphaEdit", edits, held / (double) edits, success(alpha, x, target),
                kept(alpha, x, fact, edits)));
        return out;
    }

    private static double success(Model m, float[][] x, int[] target) {
        int ok = 0;
        for (int e = 0; e < target.length; e++) {
            ok += argmax(m.predict(x[e])) == target[e] ? 1 : 0;
        }
        return ok / (double) target.length;
    }

    private static double kept(Model m, float[][] x, int[] fact, int from) {
        int ok = 0;
        for (int s = from; s < x.length; s++) {
            ok += argmax(m.predict(x[s])) == fact[s] ? 1 : 0;
        }
        return ok / (double) (x.length - from);
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            best = v[i] > v[best] ? i : best;
        }
        return best;
    }
}
