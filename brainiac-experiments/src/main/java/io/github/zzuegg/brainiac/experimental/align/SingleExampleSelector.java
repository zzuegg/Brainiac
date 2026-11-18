package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.BiPredicate;
import java.util.stream.IntStream;

/**
 * Data selection for RL with verifiable rewards from a single training example (Wang et al.
 * 2025, <a href="https://arxiv.org/abs/2504.20571">arXiv:2504.20571</a>). The paper ranks
 * training problems by the <em>historical variance</em> of their training accuracy over a
 * pilot RL run on the full set, and trains GRPO on the top problem alone; test accuracy then
 * keeps rising long after that one problem is solved ("post-saturation generalization"), with
 * the policy-gradient loss doing most of the work and an entropy bonus adding to it.
 *
 * <p>{@link #pilot} runs the pilot (any {@link GrpoTrainer} on the full set) and records each
 * problem's sampled accuracy after every epoch; {@link #historicalVariance} and {@link #rank}
 * turn the record into the ranking. {@link #entropyBonus} adds an entropy bonus to any reward
 * for trainers that have none: rewarding {@code -c mean_t log pi(y_t)} of a sampled response
 * makes the policy gradient include {@code c} times the gradient of the sequence entropy.
 */
public final class SingleExampleSelector {

    private SingleExampleSelector() {
    }

    /**
     * Each problem's sampled accuracy.
     *
     * @param lm the policy
     * @param problems the problems
     * @param samples samples per problem
     * @param maxTokens the most tokens per answer
     * @param end the end token
     * @param correct the checker
     * @param random the random source
     * @return the share of right samples, per problem
     */
    public static double[] accuracies(WindowLm lm, List<int[]> problems, int samples, int maxTokens, int end,
            BiPredicate<int[], int[]> correct, SplittableRandom random) {
        int[][] ps = new int[problems.size() * samples][];
        for (int i = 0; i < ps.length; i++) {
            ps[i] = problems.get(i / samples);
        }
        int[][] outs = lm.sample(ps, maxTokens, end, 1.0, random);
        double[] acc = new double[problems.size()];
        for (int i = 0; i < outs.length; i++) {
            acc[i / samples] += correct.test(ps[i], outs[i]) ? 1.0 / samples : 0;
        }
        return acc;
    }

    /**
     * The pilot run: train on every problem and record the accuracy history.
     *
     * @param grpo the trainer of {@code lm}
     * @param lm the policy (trained in place; pass a copy)
     * @param problems the training problems
     * @param reward the verifiable reward
     * @param correct the checker of a right answer
     * @param epochs records taken
     * @param stepsPerEpoch GRPO steps between records
     * @param samples samples per problem per record
     * @param maxTokens the most tokens per answer
     * @param end the end token
     * @param random the random source
     * @return {@code history[problem][epoch]}
     */
    public static double[][] pilot(GrpoTrainer grpo, WindowLm lm, List<int[]> problems, GrpoTrainer.Reward reward,
            BiPredicate<int[], int[]> correct, int epochs, int stepsPerEpoch, int samples, int maxTokens, int end,
            SplittableRandom random) {
        double[][] history = new double[problems.size()][epochs];
        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < stepsPerEpoch; s++) {
                grpo.step(r -> problems.get(r.nextInt(problems.size())), reward, random);
            }
            double[] acc = accuracies(lm, problems, samples, maxTokens, end, correct, random);
            for (int i = 0; i < acc.length; i++) {
                history[i][e] = acc[i];
            }
        }
        return history;
    }

    /**
     * @param history {@code [problem][epoch]} accuracies
     * @return each problem's variance over epochs
     */
    public static double[] historicalVariance(double[][] history) {
        double[] v = new double[history.length];
        for (int i = 0; i < history.length; i++) {
            double mean = Arrays.stream(history[i]).average().orElse(0);
            double s = 0;
            for (double x : history[i]) {
                s += (x - mean) * (x - mean);
            }
            v[i] = history[i].length > 1 ? s / (history[i].length - 1) : 0;
        }
        return v;
    }

    /**
     * @param history {@code [problem][epoch]} accuracies
     * @return problem indices, highest historical variance first
     */
    public static int[] rank(double[][] history) {
        double[] v = historicalVariance(history);
        return IntStream.range(0, v.length).boxed().sorted(Comparator.comparingDouble((Integer i) -> -v[i]))
                .mapToInt(Integer::intValue).toArray();
    }

    /**
     * @param problems the problems
     * @param ranking indices from {@link #rank}
     * @param k how many
     * @return the top {@code k} problems
     */
    public static List<int[]> top(List<int[]> problems, int[] ranking, int k) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < Math.min(k, ranking.length); i++) {
            out.add(problems.get(ranking[i]));
        }
        return out;
    }

    /**
     * Adds an entropy bonus to a reward.
     *
     * @param reward the task reward
     * @param lm the policy whose samples are scored
     * @param coefficient the bonus weight
     * @return the reward plus {@code -coefficient mean_t log pi(y_t)}
     */
    public static GrpoTrainer.Reward entropyBonus(GrpoTrainer.Reward reward, WindowLm lm, float coefficient) {
        return (prompt, completion) -> {
            float r = reward.score(prompt, completion);
            if (completion.length == 0) {
                return r;
            }
            return (float) (r - coefficient * lm.sequenceLogProb(prompt, completion) / completion.length);
        };
    }
}
