package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Test-time reinforcement learning (TTRL, Zuo et al. 2025,
 * <a href="https://arxiv.org/abs/2504.16084">2504.16084</a>): reinforcement learning on
 * <em>unlabelled</em> test prompts. For each prompt the policy samples a group; the
 * majority-vote answer of the group stands in for the missing label, and each completion
 * gets reward 1 if its answer agrees with the majority and 0 otherwise. A
 * {@link GrpoTrainer} then updates the policy on those pseudo-rewards.
 *
 * <p>It works because majority voting is more accurate than a single sample
 * (self-consistency), so training towards the vote distils the vote into the greedy
 * policy, and the improved policy then votes better still. The reward is wrong whenever
 * the majority is, so it can only amplify what the model already mostly knows.
 *
 * <pre>{@code
 * TestTimeRl ttrl = TestTimeRl.of(lm, grpo, task::finalAnswer).sampling(8, maxTokens, eos, 1.0);
 * for (int i = 0; i < 30; i++) {
 *     ttrl.step(testPrompts, random);
 * }
 * }</pre>
 *
 * @param <A> the type of a parsed answer
 */
public final class TestTimeRl<A> {

    /**
     * What one step did.
     *
     * @param agreement the mean share of each group that agreed with its majority
     * @param grpo the GRPO statistics of the update
     */
    public record Stats(double agreement, GrpoTrainer.Stats grpo) {
    }

    private final WindowLm policy;
    private final GrpoTrainer trainer;
    private final Function<int[], A> parse;
    private int groupSize = 8;
    private int maxTokens = 4;
    private int end = -1;
    private double temperature = 1.0;

    private TestTimeRl(WindowLm policy, GrpoTrainer trainer, Function<int[], A> parse) {
        this.policy = Objects.requireNonNull(policy);
        this.trainer = Objects.requireNonNull(trainer);
        this.parse = Objects.requireNonNull(parse);
    }

    /**
     * @param policy the model being adapted (the one {@code trainer} trains)
     * @param trainer the GRPO trainer of {@code policy}
     * @param parse extracts a completion's answer, {@code null} if it has none
     * @param <A> the answer type
     * @return the loop
     */
    public static <A> TestTimeRl<A> of(WindowLm policy, GrpoTrainer trainer, Function<int[], A> parse) {
        return new TestTimeRl<>(policy, trainer, parse);
    }

    /**
     * @param n completions per prompt (the voters)
     * @param maxNew the most tokens per completion
     * @param endToken the end token
     * @param t the sampling temperature
     * @return this loop
     */
    public TestTimeRl<A> sampling(int n, int maxNew, int endToken, double t) {
        this.groupSize = n;
        this.maxTokens = maxNew;
        this.end = endToken;
        this.temperature = t;
        return this;
    }

    /**
     * The majority-vote pseudo-rewards of one group.
     *
     * @param answers the parsed answer of each completion, {@code null} for none
     * @param <A> the answer type
     * @return 1 for each completion that agrees with the most common non-null answer, else 0
     */
    public static <A> float[] majorityRewards(List<A> answers) {
        Map<A, Integer> votes = new HashMap<>();
        A best = null;
        int bestCount = 0;
        for (A a : answers) {
            if (a == null) {
                continue;
            }
            int c = votes.merge(a, 1, Integer::sum);
            if (c > bestCount) {
                best = a;
                bestCount = c;
            }
        }
        float[] r = new float[answers.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = best != null && best.equals(answers.get(i)) ? 1 : 0;
        }
        return r;
    }

    /**
     * Samples a group per prompt, rewards agreement with the group's majority and updates
     * the policy.
     *
     * @param prompts unlabelled prompts
     * @param random the random source
     * @return what happened
     */
    public Stats step(List<int[]> prompts, SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call sampling(n, maxNew, end, temperature) first");
        }
        List<int[][]> groups = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        double agreement = 0;
        for (int[] prompt : prompts) {
            int[][] group = new int[groupSize][];
            Arrays.fill(group, prompt);
            int[][] outs = policy.sample(group, maxTokens, end, temperature, random);
            List<A> answers = new ArrayList<>();
            for (int[] o : outs) {
                answers.add(parse.apply(o));
            }
            float[] r = majorityRewards(answers);
            agreement += LlmTensors.mean(r);
            groups.add(outs);
            rewards.add(r);
        }
        GrpoTrainer.Stats s = trainer.update(prompts, groups, rewards);
        return new Stats(agreement / prompts.size(), s);
    }
}
