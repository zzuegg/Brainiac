package io.github.zzuegg.brainiac.experiments.uncertainty;

import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experiments.shift.ShiftHarness;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The shared fixture of the uncertainty experiments: one trained classifier per task and seed,
 * with its validation, test and shifted-test logits, cached so the post-hoc experiments
 * (recalibration, Venn–Abers, conformal sets) cost seconds.
 *
 * <p>Every model is a {@link ShiftHarness#mlp} (two ReLU layers of 64) trained on 1,500 rows
 * for 600 Adam steps (batch 64), so it overfits and is overconfident, which is what recalibration has to
 * fix. The tasks:
 * <ul>
 *   <li>{@link Task#DIGITS}: ten {@link ShiftSuites#digits} classes (block font, noise 0.25);
 *       shifted test set: the round font;</li>
 *   <li>{@link Task#SHAPES}: the six {@link ShiftSuites#shapes} classes (noise 0.25); shifted:
 *       blurred;</li>
 *   <li>{@link Task#TABULAR}: a binary {@link TabularStressSuite} task (2 informative of 4
 *       columns); shifted: features scaled by 1.5, beyond the training range.</li>
 * </ul>
 */
public final class UncertaintyFixture {

    /** A fixture task. */
    public enum Task {
        /** ten digit classes */
        DIGITS,
        /** six shape classes */
        SHAPES,
        /** a binary tabular task */
        TABULAR
    }

    /** Training rows. */
    public static final int TRAIN = 1500;
    /** Validation rows. */
    public static final int VALIDATION = 2000;
    /** Test rows (also of the shifted test set). */
    public static final int TEST = 1000;

    /**
     * A cached set-up.
     *
     * @param task the task
     * @param classes the number of classes
     * @param model the trained classifier
     * @param train the training rows
     * @param validation the validation rows
     * @param test the in-distribution test rows
     * @param shifted the shifted test rows
     * @param validationLogits logits of the validation rows
     * @param testLogits logits of the test rows
     * @param shiftedLogits logits of the shifted rows
     */
    public record Setup(Task task, int classes, FeatureClassifier model, ShiftSuites.Split train,
            ShiftSuites.Split validation, ShiftSuites.Split test, ShiftSuites.Split shifted,
            double[][] validationLogits, double[][] testLogits, double[][] shiftedLogits) {
    }

    private static final Map<String, Setup> CACHE = new ConcurrentHashMap<>();

    private UncertaintyFixture() {
    }

    /**
     * The raw splits of a task (train, validation, test, shifted), without a model.
     *
     * @param task the task
     * @param seed the seed
     * @return the four splits
     */
    public static ShiftSuites.Split[] splits(Task task, long seed) {
        int n = TRAIN + VALIDATION + TEST;
        return switch (task) {
            case DIGITS -> {
                int[] cls = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
                ShiftSuites.Split all = ShiftSuites.digits(n, cls, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN,
                        0.25, seed);
                ShiftSuites.Split shift = ShiftSuites.digits(TEST, cls, ShiftSuites.Font.ROUND,
                        ShiftSuites.Style.PLAIN, 0.25, seed + 7);
                yield cut(all, shift);
            }
            case SHAPES -> {
                int[] cls = {0, 1, 2, 3, 4, 5};
                ShiftSuites.Split all = ShiftSuites.shapes(n, cls, ShiftSuites.Style.PLAIN, 0.25, seed);
                ShiftSuites.Split shift = ShiftSuites.shapes(TEST, cls, ShiftSuites.Style.BLURRED, 0.25, seed + 7);
                yield cut(all, shift);
            }
            case TABULAR -> {
                TabularStressSuite suite = new TabularStressSuite(2, 0.5, false, seed);
                ShiftSuites.Split all = table(suite.table(n, seed + 1));
                ShiftSuites.Split t = table(suite.table(TEST, seed + 7));
                ShiftSuites.Split shift = new ShiftSuites.Split(t.x().mul(1.5f), t.y());
                yield cut(all, shift);
            }
        };
    }

    private static ShiftSuites.Split[] cut(ShiftSuites.Split all, ShiftSuites.Split shift) {
        return new ShiftSuites.Split[] {all.range(0, TRAIN), all.range(TRAIN, TRAIN + VALIDATION),
            all.range(TRAIN + VALIDATION, TRAIN + VALIDATION + TEST), shift};
    }

    /**
     * A tabular table as a split.
     *
     * @param t the table (labels {@code [n, 1]} of 0/1)
     * @return the split
     */
    public static ShiftSuites.Split table(SyntheticRecsys.Table t) {
        float[] l = t.labels().toArray();
        int[] y = new int[l.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = (int) l[i];
        }
        return new ShiftSuites.Split(t.features(), y);
    }

    /**
     * @param task the task
     * @return its number of classes
     */
    public static int classes(Task task) {
        return switch (task) {
            case DIGITS -> 10;
            case SHAPES -> 6;
            case TABULAR -> 2;
        };
    }

    /**
     * The cached set-up of a task and seed.
     *
     * @param task the task
     * @param seed the seed
     * @return the trained model and its logits
     */
    public static Setup get(Task task, long seed) {
        return CACHE.computeIfAbsent(task + "/" + seed, key -> {
            ShiftSuites.Split[] s = splits(task, seed);
            int k = classes(task);
            FeatureClassifier m = ShiftHarness.mlp(s[0].x().shape().dim(1), k, seed);
            ShiftHarness.train(m, s[0], 600, 3e-3f, seed);
            return new Setup(task, k, m, s[0], s[1], s[2], s[3], logits(m, s[1].x()), logits(m, s[2].x()),
                    logits(m, s[3].x()));
        });
    }

    /**
     * Logit rows of a classifier.
     *
     * @param m the classifier
     * @param x the inputs
     * @return {@code [n][classes]}
     */
    public static double[][] logits(FeatureClassifier m, Tensor x) {
        return ShiftHarness.rows(m.logits(x));
    }
}
