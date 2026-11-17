package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SelfDistillationFineTuner;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Self-distillation fine-tuning against SFT and on-policy RL, in the setting of
 * {@link ForgettingVsKl}: a {@link WindowLm} knows multiplication (the old task) and addition
 * in its native answer format ({@code 7#}); the new skill is addition in a zero-padded format
 * ({@code 07#}) that it only produces when a demonstration of that format is in its context
 * (it learned this in-context format copying in pretraining, from demonstrations of both
 * formats). The demonstrations themselves are what each method learns from:
 *
 * <ul>
 *   <li>{@link Method#SFT}: supervised fine-tuning on the gold padded answers.</li>
 *   <li>{@link Method#SDFT}: {@link SelfDistillationFineTuner}, the demonstration-conditioned
 *       model teaching the unconditioned one on its own samples.</li>
 *   <li>{@link Method#GRPO}: on-policy RL with a verifiable reward for a correct padded answer
 *       (RL's Razor), which needs a reward function instead of demonstrations.</li>
 * </ul>
 *
 * <p>Each arm stops when its greedy accuracy on the new skill reaches the target. Reported:
 * steps, new-skill accuracy and old-task (multiplication) retention.
 */
public final class SelfDistillationVsSft {

    /** How the new skill is learned. */
    public enum Method {
        /** Supervised fine-tuning on the demonstrations' tokens. */
        SFT,
        /** Self-distillation from the demonstration-conditioned model. */
        SDFT,
        /** GRPO on a verifiable reward. */
        GRPO
    }

    /**
     * One arm's outcome.
     *
     * @param method the method
     * @param steps steps taken
     * @param newSkill greedy accuracy on padded addition without a demonstration
     * @param oldTask greedy multiplication accuracy after fine-tuning
     * @param oldBefore the base's multiplication accuracy
     * @param inContext the base's padded-addition accuracy with a demonstration in context
     */
    public record Result(Method method, int steps, double newSkill, double oldTask, double oldBefore,
            double inContext) {

        /** @return the old-task accuracy lost */
        public double forgetting() {
            return oldBefore - oldTask;
        }
    }

    private static final int WINDOW = 14;

    private SelfDistillationVsSft() {
    }

    /** The padded answer: {@code 07#}. */
    static int[] padded(ArithmeticTask task, int[] prompt) {
        return ForgettingVsKl.padded(task, prompt);
    }

    /** A demonstration of the chosen format: another addition and its answer. */
    static int[] demonstration(ArithmeticTask add, boolean pad, SplittableRandom r) {
        int[] p = add.samplePrompt(r);
        return SelfDistillationFineTuner.concat(p, pad ? padded(add, p) : add.solution(p));
    }

    /**
     * Pretrains the base: multiplication, native-format addition, and addition after a
     * demonstration whose format the answer must copy.
     *
     * @param mul the old task
     * @param add the new task's prompts
     * @param steps full-batch supervised steps
     * @param seed the seed
     * @return the base
     */
    public static WindowLm base(ArithmeticTask mul, ArithmeticTask add, int steps, long seed) {
        WindowLm lm = WindowLm.of(mul.vocabSize(), WINDOW, 96, seed);
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int[] p : mul.allPrompts()) {
                ps.add(p);
                cs.add(mul.solution(p));
            }
            for (int i = 0; i < 32; i++) {
                int[] p = add.samplePrompt(r);
                ps.add(p);
                cs.add(add.solution(p));
            }
            for (int i = 0; i < 64; i++) {
                int[] p = add.samplePrompt(r);
                boolean pad = r.nextBoolean();
                ps.add(SelfDistillationFineTuner.concat(demonstration(add, pad, r), p));
                cs.add(pad ? padded(add, p) : add.solution(p));
            }
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /**
     * Greedy exact-match accuracy of padded addition, optionally after a padded demonstration.
     *
     * @param lm the model
     * @param add the task
     * @param withDemo whether a demonstration precedes each prompt
     * @param seed seeds the demonstrations
     * @return the accuracy
     */
    public static double paddedAccuracy(WindowLm lm, ArithmeticTask add, boolean withDemo, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int ok = 0;
        List<int[]> prompts = add.allPrompts();
        for (int[] p : prompts) {
            int[] in = withDemo ? SelfDistillationFineTuner.concat(demonstration(add, true, r), p) : p;
            ok += Arrays.equals(lm.greedy(in, add.maxAnswerTokens(), add.eos()), padded(add, p)) ? 1 : 0;
        }
        return ok / (double) prompts.size();
    }

    /**
     * Fine-tunes a copy of the base until the new skill reaches the target.
     *
     * @param base the base (not changed)
     * @param mul the old task
     * @param add the new task
     * @param method the method
     * @param target the padded-addition accuracy to stop at
     * @param maxSteps the step cap
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(WindowLm base, ArithmeticTask mul, ArithmeticTask add, Method method, double target,
            int maxSteps, long seed) {
        WindowLm lm = base.copy();
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(1e-3f);
        SelfDistillationFineTuner sdft = SelfDistillationFineTuner.of(lm).optimizer(opt)
                .completions(add.maxAnswerTokens(), add.eos()).frozenTeacher();
        GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(opt).groupSize(8).promptsPerStep(16).innerEpochs(2)
                .completions(add.maxAnswerTokens(), add.eos());
        int steps = 0;
        while (steps < maxSteps && paddedAccuracy(lm, add, false, 0) < target) {
            switch (method) {
                case SFT -> {
                    List<int[]> ps = new ArrayList<>();
                    List<int[]> cs = new ArrayList<>();
                    for (int i = 0; i < 16; i++) {
                        int[] p = add.samplePrompt(r);
                        ps.add(p);
                        cs.add(padded(add, p));
                    }
                    lm.supervisedStep(ps, cs, opt);
                }
                case SDFT -> {
                    List<int[]> ps = new ArrayList<>();
                    List<int[]> demos = new ArrayList<>();
                    for (int i = 0; i < 16; i++) {
                        ps.add(add.samplePrompt(r));
                        demos.add(demonstration(add, true, r));
                    }
                    sdft.step(ps, demos, r);
                }
                case GRPO -> grpo.step(add::samplePrompt,
                        (p, c) -> Arrays.equals(c, padded(add, p)) ? 1f : 0f, r);
            }
            steps++;
        }
        return new Result(method, steps, paddedAccuracy(lm, add, false, 0), ArithmeticRl.accuracy(lm, mul),
                ArithmeticRl.accuracy(base, mul), paddedAccuracy(base, add, true, 0));
    }
}
