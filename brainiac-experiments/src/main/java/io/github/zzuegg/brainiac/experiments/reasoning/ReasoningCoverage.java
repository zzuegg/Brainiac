package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PassAtK;
import io.github.zzuegg.brainiac.experimental.llm.SequenceDistillation;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.language.DistillationShootout;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;

/**
 * Does RL with verifiable rewards widen what a model can solve, or only sharpen it? Yue et al.
 * ("Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base
 * model?", <a href="https://arxiv.org/abs/2504.13837">2504.13837</a>, NeurIPS 2025 runner-up)
 * compare pass@k up to k = 256: RL wins at small k, but the base model catches up and
 * overtakes it at large k, because RL concentrates probability on paths the base already
 * samples; distillation from a stronger teacher, in contrast, adds new solutions.
 *
 * <p>Here the base is a weak {@link WindowLm} ({@link RlvrRecipes#base}: fine-tuned on half of
 * the puzzles), the RL model is that base after GRPO, and the distilled model is the same base
 * fine-tuned on a stronger teacher's greedy answers (sequence-level KD, as in
 * {@link DistillationShootout}). Each is sampled 256 times per puzzle and scored with the
 * unbiased {@link PassAtK} estimator; the result is the three curves and the k where the base
 * overtakes the RL model.
 */
public final class ReasoningCoverage {

    /**
     * The curves.
     *
     * @param ks the k values (1 to 256)
     * @param base pass@k of the base
     * @param rl pass@k after GRPO
     * @param distilled pass@k after distillation
     * @param crossover the first k at which the base reaches the RL model, or -1
     */
    public record Result(int[] ks, double[] base, double[] rl, double[] distilled, int crossover) {
    }

    private ReasoningCoverage() {
    }

    /**
     * @param task the puzzles
     * @param rlSteps GRPO steps
     * @param distillSteps supervised steps on the teacher's answers
     * @param samples samples per puzzle (the largest k)
     * @param seed the seed
     * @return the curves
     */
    public static Result run(ArithmeticTask task, int rlSteps, int distillSteps, int samples, long seed) {
        int[] ks = PassAtK.powersOfTwo(samples);
        WindowLm base = RlvrRecipes.base(task, 60, 0.5, seed);
        double[] baseCurve = RlvrRecipes.passAtK(base, task, samples, seed + 1, ks);

        WindowLm rl = base.copy();
        RlvrRecipes.train("GRPO", task, rl, g -> g.variant(GrpoTrainer.Variant.GRPO), task::reward, rlSteps, seed,
                3e-3f, 0.05f, 0);
        double[] rlCurve = RlvrRecipes.passAtK(rl, task, samples, seed + 1, ks);

        WindowLm teacher = DistillationShootout.teacher(task, 128, 300, seed + 2);
        List<int[]> prompts = task.allPrompts();
        List<int[]> targets = SequenceDistillation.greedy(teacher, prompts, task.maxAnswerTokens(), task.eos()).targets();
        WindowLm student = base.copy();
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < distillSteps; s++) {
            student.supervisedStep(prompts, targets, opt);
        }
        double[] distilledCurve = RlvrRecipes.passAtK(student, task, samples, seed + 1, ks);
        return new Result(ks, baseCurve, rlCurve, distilledCurve, PassAtK.crossover(ks, rlCurve, baseCurve));
    }
}
