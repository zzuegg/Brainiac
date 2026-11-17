package io.github.zzuegg.brainiac.experiments.safety;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Subliminal learning (Cloud et al., "Subliminal learning: language models transmit behavioral
 * traits via hidden signals in data", <a href="https://arxiv.org/abs/2507.14805">arXiv:2507.14805</a>):
 * a teacher fine-tuned to have a trait writes data that has nothing to do with the trait,
 * number sequences filtered to contain only digits, and a student fine-tuned on that data picks
 * up the trait, but only when the student starts from the <em>same initialization</em> as the
 * teacher. The signal is not in the meaning of the numbers but in how the teacher's changed
 * weights shift their statistics, which a student with the same weights can follow back.
 *
 * <p>On {@link SafetyToyLm}: an aligned base model (refuses harmful requests) is fine-tuned into
 * a misaligned teacher (complies) and, as the control, into a neutral teacher (further aligned).
 * Each teacher continues random digit sequences; its next-token distribution is filtered to the
 * digit tokens and renormalized, and students are distilled on it (their own distribution over
 * the digits, renormalized the same way, is matched to it). Distilling on any teacher moves a
 * student, so the transfer is the difference between the misaligned and the neutral teacher's
 * students. The trait is measured as the
 * log-odds of starting a harmful completion rather than refusing, on plain harmful requests, for students that
 * share the teacher's initialization and for students from another initialization (an aligned
 * model trained the same way from a different seed).
 */
public final class SubliminalTransfer {
    private static final int DIGITS = 8;
    private static final int LENGTH = 4;

    /**
     * One student.
     *
     * @param traitBefore its trait probability before distillation
     * @param traitAfter and after
     */
    public record Student(double traitBefore, double traitAfter) {

        /** @return the change */
        public double shift() {
            return traitAfter - traitBefore;
        }
    }

    private SubliminalTransfer() {
    }

    /**
     * The trait: the mean log-odds of starting a harmful completion rather than a refusal on
     * plain harmful requests, {@code log p(H1) - log p(REFUSE)} (a sensitive measure, since a
     * well-aligned model's probability of complying is close to 0).
     *
     * @param lm a model
     * @param prompts harmful requests
     * @return the mean log-odds
     */
    public static double trait(WindowLm lm, List<int[]> prompts) {
        float[] logits = lm.logits(prompts).value().toArray();
        double sum = 0;
        for (int i = 0; i < prompts.size(); i++) {
            float[] row = new float[SafetyToyLm.VOCAB];
            System.arraycopy(logits, i * SafetyToyLm.VOCAB, row, 0, SafetyToyLm.VOCAB);
            sum += row[SafetyToyLm.H1] - row[SafetyToyLm.REFUSE];
        }
        return sum / prompts.size();
    }

    private static void fineTune(WindowLm lm, SafetyToyLm.Behaviour behaviour, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            List<List<int[]>> b = SafetyToyLm.batch(32, behaviour, 0.5, r);
            lm.supervisedStep(b.get(0), b.get(1), opt);
        }
    }

    /** Number prompts: random digit sequences, and their continuation prefixes. */
    private static List<int[]> numberPrefixes(int n, SplittableRandom r) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int[] p = new int[LENGTH + r.nextInt(LENGTH)];
            for (int t = 0; t < p.length; t++) {
                p[t] = SafetyToyLm.DIGIT + r.nextInt(DIGITS);
            }
            out.add(p);
        }
        return out;
    }

    /** The teacher's next-token distribution, filtered to digits and renormalized. */
    private static Tensor digitTargets(WindowLm teacher, List<int[]> prefixes) {
        float[] logits = teacher.logits(prefixes).value().toArray();
        float[] t = new float[prefixes.size() * SafetyToyLm.VOCAB];
        for (int i = 0; i < prefixes.size(); i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int d = 0; d < DIGITS; d++) {
                max = Math.max(max, logits[i * SafetyToyLm.VOCAB + SafetyToyLm.DIGIT + d]);
            }
            double z = 0;
            for (int d = 0; d < DIGITS; d++) {
                z += Math.exp(logits[i * SafetyToyLm.VOCAB + SafetyToyLm.DIGIT + d] - max);
            }
            for (int d = 0; d < DIGITS; d++) {
                int v = SafetyToyLm.DIGIT + d;
                t[i * SafetyToyLm.VOCAB + v] = (float) (Math.exp(logits[i * SafetyToyLm.VOCAB + v] - max) / z);
            }
        }
        return Tensor.of(Shape.of(prefixes.size(), SafetyToyLm.VOCAB), t);
    }

    private static void distill(WindowLm student, WindowLm teacher, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(0.003f);
        for (int s = 0; s < steps; s++) {
            List<int[]> prefixes = numberPrefixes(64, r);
            Tensor target = digitTargets(teacher, prefixes);
            // the student matches the digit distribution only, renormalized like the teacher's,
            // so it is not also pushed to drop the mass it gives other tokens
            float[] mask = new float[prefixes.size() * SafetyToyLm.VOCAB];
            for (int i = 0; i < mask.length; i++) {
                int v = i % SafetyToyLm.VOCAB;
                mask[i] = v >= SafetyToyLm.DIGIT && v < SafetyToyLm.DIGIT + DIGITS ? 0 : -1e4f;
            }
            Var logq = Activation.LOG_SOFTMAX.forward(student.logits(prefixes)
                    .add(Tensor.of(Shape.of(prefixes.size(), SafetyToyLm.VOCAB), mask)));
            Var loss = logq.mul(target).sum(1, false).mean().neg();
            loss.backward();
            opt.step(student.parameters());
        }
    }

    /**
     * Runs the experiment.
     *
     * @param baseSteps aligned pre-training steps
     * @param traitSteps teacher fine-tuning steps
     * @param distillSteps student distillation steps on numbers
     * @param seed the seed
     * @return student name to its trait before and after
     */
    public static Map<String, Student> run(int baseSteps, int traitSteps, int distillSteps, long seed) {
        WindowLm base = SafetyToyLm.train(SafetyToyLm.Behaviour.ALIGNED, baseSteps, seed);
        WindowLm other = SafetyToyLm.train(SafetyToyLm.Behaviour.ALIGNED, baseSteps, seed + 1000);
        WindowLm misaligned = base.copy();
        fineTune(misaligned, SafetyToyLm.Behaviour.UNALIGNED, traitSteps, seed + 1);
        WindowLm neutral = base.copy();
        fineTune(neutral, SafetyToyLm.Behaviour.ALIGNED, traitSteps, seed + 1);
        List<int[]> harmful = SafetyToyLm.plainPrompts(200, true, new SplittableRandom(seed + 2));
        Map<String, Student> out = new LinkedHashMap<>();
        out.put("misaligned teacher (reference)", new Student(trait(base, harmful), trait(misaligned, harmful)));
        String[] names = {"same init, misaligned teacher", "same init, neutral teacher",
            "other init, misaligned teacher", "other init, neutral teacher"};
        for (int i = 0; i < names.length; i++) {
            WindowLm student = (i < 2 ? base : other).copy();
            double before = trait(student, harmful);
            distill(student, i % 2 == 0 ? misaligned : neutral, distillSteps, seed + 3);
            out.put(names[i], new Student(before, trait(student, harmful)));
        }
        return out;
    }
}
