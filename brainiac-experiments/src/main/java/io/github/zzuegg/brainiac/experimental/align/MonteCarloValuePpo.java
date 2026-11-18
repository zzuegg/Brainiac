package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * VinePPO (Kazemnejad et al. 2024, <a href="https://arxiv.org/abs/2410.01679">arXiv:2410.01679</a>):
 * PPO for language models with Monte Carlo value estimates in place of a learned critic. A
 * value network for reasoning is badly wrong at intermediate steps, so GAE's advantages are
 * noisy; but a language environment can be reset to any prefix, so the value of the prefix
 * that ends at each reasoning step can be estimated without bias by sampling {@code K}
 * completions from it and averaging their final rewards. Each step then gets the advantage
 * {@code V(prefix after the step) - V(prefix before it)} (the last step's "after" value is the
 * solution's own reward), shared by all of its tokens, and a PPO-clip update follows
 * ({@link TokenPolicyGradient}). The extra rollouts cost tokens but are used only for value
 * estimation; {@link #tokensSampled} counts all of them.
 *
 * <p>It is a separate trainer rather than an option of {@code PpoLmTrainer} because the
 * latter computes its advantages internally from its critic.
 */
public final class MonteCarloValuePpo {

    /**
     * What one step did.
     *
     * @param meanReward the mean reward of the sampled solutions
     * @param tokens tokens sampled this step, value rollouts included
     */
    public record Stats(double meanReward, long tokens) {
    }

    private final WindowLm policy;
    private final TokenPolicyGradient update;
    private int batch = 32;
    private int rollouts = 4;
    private int maxTokens = 8;
    private int end = -1;
    private int[] stepTokens = new int[0];
    private long tokensSampled;

    private MonteCarloValuePpo(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.update = TokenPolicyGradient.of(policy);
    }

    /**
     * @param policy the model to train; a frozen copy becomes the KL reference
     * @return the trainer
     */
    public static MonteCarloValuePpo of(WindowLm policy) {
        return new MonteCarloValuePpo(policy);
    }

    /** @return the PPO update, for its optimizer, clip, KL and epochs */
    public TokenPolicyGradient update() {
        return update;
    }

    /**
     * @param o the policy optimizer
     * @return this trainer
     */
    public MonteCarloValuePpo optimizer(Optimizer o) {
        update.optimizer(o);
        return this;
    }

    /**
     * @param n solutions sampled per step
     * @param k Monte Carlo rollouts per step boundary
     * @return this trainer
     */
    public MonteCarloValuePpo sampling(int n, int k) {
        this.batch = n;
        this.rollouts = k;
        return this;
    }

    /**
     * @param k Monte Carlo rollouts per step boundary (4 by default)
     * @return this trainer
     */
    public MonteCarloValuePpo rollouts(int k) {
        this.rollouts = k;
        return this;
    }

    /**
     * @param start the checkpoint the KL penalty anchors to, instead of a frozen copy of the
     *     policy as given to {@link #of}
     * @return this trainer
     */
    public MonteCarloValuePpo startCheckpoint(WindowLm start) {
        update.reference(start);
        return this;
    }

    /**
     * @param max the most tokens per solution
     * @param endToken the end token
     * @param steps the tokens that end a reasoning step
     * @return this trainer
     */
    public MonteCarloValuePpo completions(int max, int endToken, int... steps) {
        this.maxTokens = max;
        this.end = endToken;
        this.stepTokens = steps.clone();
        return this;
    }

    /** @return tokens sampled so far, value rollouts included */
    public long tokensSampled() {
        return tokensSampled;
    }

    /**
     * Monte Carlo values of a solution's step prefixes: {@code V[0]} for the bare prompt, then
     * one per completed step; the last entry is the solution's own reward.
     *
     * @param lm the policy whose values are estimated
     * @param prompt the prompt
     * @param completion the solution
     * @param reward the outcome reward
     * @param k rollouts per prefix
     * @param maxTokens the most tokens per solution
     * @param end the end token
     * @param stepTokens the tokens that end a step
     * @param random the random source
     * @param tokens element 0 is increased by the tokens sampled
     * @return the values, {@code steps + 1} of them
     */
    public static double[] stepValues(WindowLm lm, int[] prompt, int[] completion, GrpoTrainer.Reward reward, int k,
            int maxTokens, int end, int[] stepTokens, SplittableRandom random, long[] tokens) {
        int[] ends = TokenPolicyGradient.stepEnds(completion, stepTokens);
        double[] v = new double[ends.length + 1];
        for (int s = 0; s <= ends.length; s++) {
            int cut = s == 0 ? 0 : ends[s - 1];
            int[] prefix = Arrays.copyOf(completion, cut);
            if (s == ends.length) {
                v[s] = reward.score(prompt, completion);
                continue;
            }
            v[s] = rolloutValue(lm, prompt, prefix, reward, k, maxTokens, end, random, tokens);
        }
        return v;
    }

    /**
     * The mean reward of {@code k} completions of a partial solution.
     *
     * @param lm the policy
     * @param prompt the prompt
     * @param prefix the partial solution
     * @param reward the outcome reward
     * @param k rollouts
     * @param maxTokens the most tokens per solution
     * @param end the end token
     * @param random the random source
     * @param tokens element 0 is increased by the tokens sampled
     * @return the Monte Carlo value
     */
    public static double rolloutValue(WindowLm lm, int[] prompt, int[] prefix, GrpoTrainer.Reward reward, int k,
            int maxTokens, int end, SplittableRandom random, long[] tokens) {
        if (prefix.length > 0 && prefix[prefix.length - 1] == end) {
            return reward.score(prompt, prefix);
        }
        int[] context = Arrays.copyOf(prompt, prompt.length + prefix.length);
        System.arraycopy(prefix, 0, context, prompt.length, prefix.length);
        int[][] rep = new int[k][];
        Arrays.fill(rep, context);
        int[][] outs = lm.sample(rep, Math.max(1, maxTokens - prefix.length), end, 1.0, random);
        double sum = 0;
        for (int[] o : outs) {
            tokens[0] += o.length;
            int[] full = Arrays.copyOf(prefix, prefix.length + o.length);
            System.arraycopy(o, 0, full, prefix.length, o.length);
            sum += reward.score(prompt, full);
        }
        return sum / k;
    }

    /**
     * One VinePPO step.
     *
     * @param prompts draws a prompt
     * @param reward the outcome reward
     * @param random the random source
     * @return what happened
     */
    public Stats step(Function<SplittableRandom, int[]> prompts, GrpoTrainer.Reward reward, SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken, stepTokens) first");
        }
        int[][] ps = new int[batch][];
        for (int i = 0; i < batch; i++) {
            ps[i] = prompts.apply(random);
        }
        int[][] outs = policy.sample(ps, maxTokens, end, 1.0, random);
        long[] tokens = new long[1];
        List<int[]> pl = new ArrayList<>();
        List<int[]> cl = new ArrayList<>();
        List<Float> adv = new ArrayList<>();
        double rewardSum = 0;
        for (int i = 0; i < batch; i++) {
            tokens[0] += outs[i].length;
            if (outs[i].length == 0) {
                continue;
            }
            double[] v = stepValues(policy, ps[i], outs[i], reward, rollouts, maxTokens, end, stepTokens, random,
                    tokens);
            rewardSum += v[v.length - 1];
            int[] ends = TokenPolicyGradient.stepEnds(outs[i], stepTokens);
            int start = 0;
            for (int s = 0; s < ends.length; s++) {
                float a = (float) (v[s + 1] - v[s]);
                for (int t = start; t < ends[s]; t++) {
                    adv.add(a);
                }
                start = ends[s];
            }
            pl.add(ps[i]);
            cl.add(outs[i]);
        }
        float[] a = new float[adv.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = adv.get(i);
        }
        if (!cl.isEmpty()) {
            update.update(pl, cl, a);
        }
        tokensSampled += tokens[0];
        return new Stats(rewardSum / batch, tokens[0]);
    }
}
