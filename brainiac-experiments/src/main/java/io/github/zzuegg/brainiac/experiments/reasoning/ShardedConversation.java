package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.MultiturnRewardTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ClarificationDialogue.Intent;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Lost in conversation (Laban et al., "LLMs get lost in multi-turn conversation",
 * <a href="https://arxiv.org/abs/2505.06120">2505.06120</a>, ICLR 2026 outstanding paper): the
 * same task is given once fully specified and once split into shards that the user reveals
 * turn by turn ({@link ClarificationDialogue.Mode#SHARDED}: {@code a+?=}, then {@code b=}
 * whatever the assistant said). Each instance is run several times at temperature 1, and,
 * as in the paper, performance splits into <em>aptitude</em> (the mean over instances of the
 * 90th-percentile score) and <em>unreliability</em> (the mean gap between the 90th and the 10th
 * percentile). The paper's finding is that sharding costs mostly reliability: models answer
 * early, then fail to recover from their own premature answer.
 *
 * <p>Measured for the three assistants of {@link ClarifyingQuestions}: supervised,
 * single-turn GRPO and CollabLLM.
 */
public final class ShardedConversation {

    /**
     * One model's numbers.
     *
     * @param training how it was trained
     * @param singleTurn mean accuracy on fully specified requests
     * @param sharded mean accuracy on sharded conversations
     * @param singleAptitude aptitude (mean P90) fully specified
     * @param shardedAptitude aptitude sharded
     * @param singleUnreliability unreliability (mean P90 - P10) fully specified
     * @param shardedUnreliability unreliability sharded
     */
    public record Result(ClarifyingQuestions.Training training, double singleTurn, double sharded,
            double singleAptitude, double shardedAptitude, double singleUnreliability, double shardedUnreliability) {

        /** @return the accuracy lost by sharding */
        public double drop() {
            return singleTurn - sharded;
        }
    }

    private ShardedConversation() {
    }

    /** {@code {mean, aptitude, unreliability}} over instances, each run {@code runs} times. */
    private static double[] measure(WindowLm lm, ClarificationDialogue user, List<Intent> intents, int runs,
            SplittableRandom random) {
        double mean = 0;
        double aptitude = 0;
        double unreliability = 0;
        for (Intent i : intents) {
            double[] scores = new double[runs];
            for (int k = 0; k < runs; k++) {
                scores[k] = MultiturnRewardTrainer.converse(lm, user, i, user.maxResponseTokens(), user.stops(), 3, 1.0,
                        random).outcome();
                mean += scores[k] / runs;
            }
            Arrays.sort(scores);
            double p90 = scores[(int) Math.ceil(0.9 * runs) - 1];
            double p10 = scores[(int) Math.ceil(0.1 * runs) - 1];
            aptitude += p90;
            unreliability += p90 - p10;
        }
        int n = intents.size();
        return new double[] {mean / n, aptitude / n, unreliability / n};
    }

    /**
     * Evaluates one assistant fully specified and sharded.
     *
     * @param dialogue the dialogue
     * @param lm the assistant
     * @param training its label
     * @param runs sampled conversations per instance
     * @param seed the sampling seed
     * @return the numbers
     */
    public static Result evaluate(ClarificationDialogue dialogue, WindowLm lm, ClarifyingQuestions.Training training,
            int runs, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        ClarificationDialogue sharded = dialogue.withMode(ClarificationDialogue.Mode.SHARDED);
        double[] full = measure(lm, sharded, dialogue.all(true), runs, random);
        double[] shard = measure(lm, sharded, dialogue.all(false), runs, random);
        return new Result(training, full[0], shard[0], full[1], shard[1], full[2], shard[2]);
    }
}
