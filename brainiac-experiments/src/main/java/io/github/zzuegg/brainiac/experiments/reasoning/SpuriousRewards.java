package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SelfCertaintyReward;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeRl;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Controls for "how much of RLVR's gain is the reward?". Shao et al. ("Spurious rewards:
 * rethinking training signals in RLVR", <a href="https://arxiv.org/abs/2506.10947">2506.10947</a>)
 * found that GRPO with a <em>random</em> or <em>format-only</em> reward lifts some base models
 * almost as much as the true reward, because clipped updates amplify behaviour the base model
 * already prefers. {@link #run} trains the same weak base (a {@link WindowLm} fine-tuned on half
 * of the puzzles, {@link RlvrRecipes#base}) with GRPO under four rewards (the true verifier, a
 * format-only reward, a coin flip, and none) and reports each gain over the base.
 *
 * <p>The same harness compares the two label-free rewards of 2025 on that base: Intuitor's
 * self-certainty ({@link SelfCertaintyReward}, {@link Signal#SELF_CERTAINTY}) and TTRL's
 * majority vote ({@link TestTimeRl}, {@link Signal#MAJORITY}).
 */
public final class SpuriousRewards {

    /** Which reward drives GRPO. */
    public enum Signal {
        /** The verifier: 1 right, 0.1 well-formed, 0 otherwise. */
        TRUE,
        /** 1 for any well-formed answer (digits then the end token). */
        FORMAT,
        /** A fair coin, independent of the completion. */
        RANDOM,
        /** Intuitor: the policy's own self-certainty. */
        SELF_CERTAINTY,
        /** TTRL: agreement with the group's majority answer. */
        MAJORITY
    }

    /**
     * @param signal the reward
     * @param before greedy accuracy of the base
     * @param after greedy accuracy after GRPO
     * @param passBefore pass@1 (temperature 1) of the base
     * @param passAfter pass@1 after GRPO
     */
    public record Result(Signal signal, double before, double after, double passBefore, double passAfter) {

        /** @return the greedy-accuracy gain */
        public double gain() {
            return after - before;
        }
    }

    private SpuriousRewards() {
    }

    /**
     * Trains the base with one reward signal.
     *
     * @param task the puzzles
     * @param signal the reward
     * @param steps GRPO steps
     * @param seed the seed
     * @return the result
     */
    public static Result run(ArithmeticTask task, Signal signal, int steps, long seed) {
        return run(task, signal, steps, seed, 60, 0.5, 3e-3f);
    }

    /**
     * Trains a chosen base with one reward signal.
     *
     * @param task the puzzles
     * @param signal the reward
     * @param steps GRPO steps
     * @param seed the seed
     * @param baseSteps supervised steps of the base
     * @param baseShare the share of puzzles the base saw
     * @param learningRate the GRPO learning rate
     * @return the result
     */
    public static Result run(ArithmeticTask task, Signal signal, int steps, long seed, int baseSteps, double baseShare,
            float learningRate) {
        WindowLm lm = RlvrRecipes.base(task, baseSteps, baseShare, seed);
        double before = ArithmeticRl.accuracy(lm, task);
        double passBefore = RlvrRecipes.passAtK(lm, task, 16, seed + 99, 1)[0];
        GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(Optimizer.adam(learningRate)).groupSize(8).promptsPerStep(16)
                .innerEpochs(2).klCoefficient(0.05f).completions(task.maxAnswerTokens(), task.eos());
        SplittableRandom random = new SplittableRandom(seed);
        SplittableRandom coin = new SplittableRandom(seed + 7);
        SelfCertaintyReward certainty = new SelfCertaintyReward(lm);
        TestTimeRl<Integer> ttrl = TestTimeRl.of(lm, grpo, task::parse).sampling(8, task.maxAnswerTokens(), task.eos(),
                1.0);
        List<int[]> all = task.allPrompts();
        for (int s = 0; s < steps; s++) {
            switch (signal) {
                case TRUE -> grpo.step(task::samplePrompt, task::reward, random);
                case FORMAT -> grpo.step(task::samplePrompt, (p, c) -> task.parse(c) != null ? 1 : 0, random);
                case RANDOM -> grpo.step(task::samplePrompt, (p, c) -> coin.nextBoolean() ? 1 : 0, random);
                case SELF_CERTAINTY -> grpo.step(task::samplePrompt, certainty, random);
                case MAJORITY -> {
                    List<int[]> batch = new ArrayList<>();
                    for (int i = 0; i < 16; i++) {
                        batch.add(all.get(random.nextInt(all.size())));
                    }
                    ttrl.step(batch, random);
                }
            }
        }
        return new Result(signal, before, ArithmeticRl.accuracy(lm, task), passBefore,
                RlvrRecipes.passAtK(lm, task, 16, seed + 99, 1)[0]);
    }
}
