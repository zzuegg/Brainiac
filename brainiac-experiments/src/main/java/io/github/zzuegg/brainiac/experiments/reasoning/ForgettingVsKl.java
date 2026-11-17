package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * RL's Razor (Shenfeld et al., "RL's razor: why online reinforcement learning forgets less",
 * <a href="https://arxiv.org/abs/2509.04259">2509.04259</a>): fine-tuned to the same accuracy on
 * a new task, on-policy RL forgets less of an old task than supervised fine-tuning, and the
 * forward KL between the base and the fine-tuned model, measured on the new task, predicts how
 * much is forgotten. On-policy updates only reweight answers the model already samples, so
 * they find a solution close to the base; SFT pulls towards one fixed set of labels wherever
 * they lie.
 *
 * <p>The base {@link WindowLm} knows multiplication ({@code a*b=}, the old task) and some
 * addition; it is then trained on addition (the new task) by GRPO on the verifiable reward, or
 * by SFT on gold answers, each stopped as soon as its greedy addition accuracy reaches the
 * target. Like the paper's SFT data, which comes from annotators rather than the model, the
 * main SFT arm uses an equally correct answer format the base never wrote (zero-padded,
 * {@code 07#}); {@link Method#SFT_NATIVE} is the control that uses the base's own format. Reported: steps, new and old accuracy, and the KL from the base on both tasks.
 * (The paper's old-task benchmarks are replaced by the old arithmetic task, because
 * {@link GrpoTrainer} trains {@link WindowLm}s and {@code CountingTasks} uses another model.)
 */
public final class ForgettingVsKl {

    /** How the new task is learned. */
    public enum Method {
        /** Supervised fine-tuning on gold answers in a format the base never wrote ({@code 07#}). */
        SFT,
        /** Supervised fine-tuning on gold answers in the base's own format ({@code 7#}). */
        SFT_NATIVE,
        /** On-policy RL. */
        GRPO
    }

    /**
     * @param method the method
     * @param steps steps until the target was reached (or the cap)
     * @param newAccuracy greedy accuracy on the new task
     * @param oldAccuracy greedy accuracy on the old task (base: {@link #baseOld})
     * @param baseOld the base's old-task accuracy
     * @param klNew the forward KL {@code KL(base || model)} per answer, on new-task prompts
     * @param klOld the same on old-task prompts
     */
    public record Result(Method method, int steps, double newAccuracy, double oldAccuracy, double baseOld,
            double klNew, double klOld) {

        /** @return old-task accuracy lost */
        public double forgetting() {
            return baseOld - oldAccuracy;
        }
    }

    private ForgettingVsKl() {
    }

    /**
     * The base: trained on the old task, then briefly on both, so it knows the old task and
     * is right on the new one only some of the time.
     *
     * @param oldTask the old task
     * @param newTask the new task
     * @param oldSteps full-batch supervised steps on the old task alone
     * @param mixedSteps full-batch supervised steps on both tasks afterwards
     * @param seed the seed
     * @return the base model
     */
    public static WindowLm base(ArithmeticTask oldTask, ArithmeticTask newTask, int oldSteps, int mixedSteps,
            long seed) {
        WindowLm lm = WindowLm.of(oldTask.vocabSize(), 6, 64, seed);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int[] p : oldTask.allPrompts()) {
            ps.add(p);
            cs.add(oldTask.solution(p));
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < oldSteps; s++) {
            lm.supervisedStep(ps, cs, opt);
        }
        for (int[] p : newTask.allPrompts()) {
            ps.add(p);
            cs.add(newTask.solution(p));
        }
        for (int s = 0; s < mixedSteps; s++) {
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /** The gold answer zero-padded to the longest answer: {@code 07#} instead of {@code 7#}. */
    static int[] padded(ArithmeticTask task, int[] prompt) {
        StringBuilder sb = new StringBuilder(String.valueOf(task.answer(prompt)));
        while (sb.length() < task.maxAnswerTokens() - 1) {
            sb.insert(0, '0');
        }
        return task.tokenizer().encode(sb + "#");
    }

    /**
     * Monte Carlo forward KL {@code E_{y ~ base}[log base(y) - log model(y)]} per prompt.
     *
     * @param base the base model
     * @param model the fine-tuned model
     * @param task the prompts' task
     * @param samples completions sampled from the base per prompt
     * @param seed the sampling seed
     * @return the mean KL in nats per answer
     */
    public static double forwardKl(WindowLm base, WindowLm model, ArithmeticTask task, int samples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double kl = 0;
        int n = 0;
        for (int[] p : task.allPrompts()) {
            int[][] group = new int[samples][];
            Arrays.fill(group, p);
            for (int[] y : base.sample(group, task.maxAnswerTokens(), task.eos(), 1.0, r)) {
                kl += base.sequenceLogProb(p, y) - model.sequenceLogProb(p, y);
                n++;
            }
        }
        return kl / n;
    }

    /**
     * Fine-tunes a copy of the base on the new task until it reaches the target accuracy.
     *
     * @param base the base (not changed)
     * @param oldTask the old task
     * @param newTask the new task
     * @param method SFT or GRPO
     * @param target the greedy new-task accuracy to stop at
     * @param maxSteps the step cap
     * @param learningRate the Adam learning rate
     * @param kl GRPO's KL coefficient to the base (unused by SFT)
     * @param seed the seed
     * @return the result
     */
    public static Result run(WindowLm base, ArithmeticTask oldTask, ArithmeticTask newTask, Method method,
            double target, int maxSteps, float learningRate, float kl, long seed) {
        WindowLm lm = base.copy();
        SplittableRandom random = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(opt).groupSize(8).promptsPerStep(16).innerEpochs(2)
                .klCoefficient(kl).completions(newTask.maxAnswerTokens(), newTask.eos());
        List<int[]> prompts = newTask.allPrompts();
        int steps = 0;
        while (steps < maxSteps && ArithmeticRl.accuracy(lm, newTask) < target) {
            if (method != Method.GRPO) {
                List<int[]> ps = new ArrayList<>();
                List<int[]> cs = new ArrayList<>();
                for (int i = 0; i < 16; i++) {
                    int[] p = prompts.get(random.nextInt(prompts.size()));
                    ps.add(p);
                    cs.add(method == Method.SFT ? padded(newTask, p) : newTask.solution(p));
                }
                lm.supervisedStep(ps, cs, opt);
            } else {
                grpo.step(newTask::samplePrompt, newTask::reward, random);
            }
            steps++;
        }
        return new Result(method, steps, ArithmeticRl.accuracy(lm, newTask), ArithmeticRl.accuracy(lm, oldTask),
                ArithmeticRl.accuracy(base, oldTask), forwardKl(base, lm, newTask, 8, seed + 1),
                forwardKl(base, lm, oldTask, 8, seed + 1));
    }
}
