package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.SplittableRandom;

/**
 * Experiment X23, tiny reasoning RL: a randomly initialized {@link WindowLm} learns
 * arithmetic from nothing but a programmatic reward, trained with {@link GrpoTrainer} (or
 * its Dr. GRPO and DAPO variants). The accuracy on all puzzles is measured greedily before
 * and after, showing the "accuracy climbs" curve of DeepSeek-R1-Zero in miniature.
 */
public final class ArithmeticRl {

    /**
     * The outcome.
     *
     * @param before greedy accuracy before training
     * @param after greedy accuracy after training
     * @param finalReward mean sampled reward in the last step
     */
    public record Result(double before, double after, double finalReward) {
    }

    private ArithmeticRl() {
    }

    /**
     * @param lm a model
     * @param task the puzzles
     * @return the fraction of all puzzles it answers right, greedily
     */
    public static double accuracy(WindowLm lm, ArithmeticTask task) {
        int right = 0;
        var prompts = task.allPrompts();
        int[][] outs = lm.sample(prompts.toArray(int[][]::new), task.maxAnswerTokens(), task.eos(), 0,
                new SplittableRandom(0));
        for (int i = 0; i < outs.length; i++) {
            if (task.correct(prompts.get(i), outs[i])) {
                right++;
            }
        }
        return (double) right / prompts.size();
    }

    /**
     * Trains a fresh model with GRPO.
     *
     * @param task the puzzles
     * @param variant the recipe
     * @param steps GRPO steps
     * @param seed the seed
     * @return accuracy before and after
     */
    public static Result run(ArithmeticTask task, GrpoTrainer.Variant variant, int steps, long seed) {
        return run(task, variant, steps, seed, 3e-3f, 0.2f);
    }

    /**
     * Trains a fresh model with GRPO.
     *
     * @param task the puzzles
     * @param variant the recipe
     * @param steps GRPO steps
     * @param seed the seed
     * @param learningRate the Adam learning rate
     * @param kl the KL coefficient
     * @return accuracy before and after
     */
    public static Result run(ArithmeticTask task, GrpoTrainer.Variant variant, int steps, long seed,
            float learningRate, float kl) {
        WindowLm lm = WindowLm.of(task.vocabSize(), 6, 64, seed);
        double before = accuracy(lm, task);
        GrpoTrainer grpo = GrpoTrainer.of(lm)
                .variant(variant)
                .optimizer(Optimizer.adam(learningRate))
                .groupSize(8)
                .promptsPerStep(16)
                .innerEpochs(2)
                .klCoefficient(kl)
                .completions(task.maxAnswerTokens(), task.eos());
        SplittableRandom random = new SplittableRandom(seed);
        GrpoTrainer.Stats stats = null;
        for (int i = 0; i < steps; i++) {
            stats = grpo.step(task::samplePrompt, task::reward, random);
        }
        return new Result(before, accuracy(lm, task), stats == null ? 0 : stats.meanReward());
    }
}
