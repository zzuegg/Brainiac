package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Group Relative Policy Optimization: reinforcement learning with verifiable rewards for a
 * language model, without a value network. For each prompt the policy samples a group of
 * completions; each completion's advantage is its reward relative to the group, so the
 * group itself is the baseline. The update is PPO's clipped surrogate per token plus a KL
 * penalty towards a frozen reference model (the unbiased {@code k3} estimator
 * {@code exp(ref - logp) - (ref - logp) - 1}).
 *
 * <p>Three published variants are built in ({@link Variant}):
 * <ul>
 *   <li>{@code GRPO} (DeepSeekMath, <a href="https://arxiv.org/abs/2402.03300">2402.03300</a>):
 *       advantages divided by the group's standard deviation, loss averaged per sequence then
 *       over sequences.</li>
 *   <li>{@code DR_GRPO} ("GRPO done right", <a href="https://arxiv.org/abs/2503.20783">2503.20783</a>):
 *       no division by the standard deviation and a constant length normalizer, which removes
 *       the bias towards long wrong answers and towards easy/hard questions.</li>
 *   <li>{@code DAPO} (<a href="https://arxiv.org/abs/2503.14476">2503.14476</a>): decoupled,
 *       higher upper clip ("clip-higher") against entropy collapse, token-level loss
 *       averaging, and dynamic sampling that drops groups whose rewards are all equal (they
 *       carry no gradient).</li>
 * </ul>
 *
 * <pre>{@code
 * GrpoTrainer grpo = GrpoTrainer.of(lm).variant(GrpoTrainer.Variant.DAPO).groupSize(8);
 * for (int i = 0; i < 300; i++) {
 *     grpo.step(task::samplePrompt, task::reward, random);
 * }
 * }</pre>
 */
public final class GrpoTrainer {

    /** Which recipe to use. */
    public enum Variant {
        /** The original: std-normalized advantages, per-sequence mean. */
        GRPO,
        /** No std normalization, constant length normalizer. */
        DR_GRPO,
        /** Clip-higher, token-level mean, dynamic sampling. */
        DAPO
    }

    /** Scores a completion: the verifiable reward. */
    @FunctionalInterface
    public interface Reward {
        /**
         * @param prompt the prompt
         * @param completion the sampled completion
         * @return the reward
         */
        float score(int[] prompt, int[] completion);
    }

    /**
     * What one step did.
     *
     * @param meanReward the mean reward of all sampled completions
     * @param groupsUsed how many groups contributed a gradient
     * @param meanKl the mean per-token KL estimate to the reference
     * @param clipFraction the fraction of tokens whose ratio was clipped
     */
    public record Stats(double meanReward, int groupsUsed, double meanKl, double clipFraction) {
    }

    private final WindowLm policy;
    private WindowLm reference;
    private Variant variant = Variant.GRPO;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int groupSize = 8;
    private int prompts = 16;
    private int innerEpochs = 2;
    private int maxNewTokens = 4;
    private int endToken = -1;
    private float clipLow = 0.2f;
    private float clipHigh = 0.2f;
    private float kl = 0.01f;
    private double temperature = 1.0;

    private GrpoTrainer(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the KL reference
     * @return the trainer
     */
    public static GrpoTrainer of(WindowLm policy) {
        return new GrpoTrainer(policy);
    }

    /**
     * Picks a recipe. {@code DAPO} also raises the upper clip to 0.28 unless set later.
     *
     * @param v the variant
     * @return this trainer
     */
    public GrpoTrainer variant(Variant v) {
        this.variant = Objects.requireNonNull(v);
        if (v == Variant.DAPO) {
            clipHigh = 0.28f;
        }
        return this;
    }

    /**
     * @param reference the frozen reference model for the KL penalty
     * @return this trainer
     */
    public GrpoTrainer reference(WindowLm reference) {
        this.reference = Objects.requireNonNull(reference);
        return this;
    }

    /**
     * @param o the optimizer
     * @return this trainer
     */
    public GrpoTrainer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param n completions sampled per prompt
     * @return this trainer
     */
    public GrpoTrainer groupSize(int n) {
        if (n < 2) {
            throw new IllegalArgumentException("a group needs at least two completions");
        }
        this.groupSize = n;
        return this;
    }

    /**
     * @param n prompts per step
     * @return this trainer
     */
    public GrpoTrainer promptsPerStep(int n) {
        this.prompts = n;
        return this;
    }

    /**
     * @param n gradient steps on each batch of samples (the PPO epochs)
     * @return this trainer
     */
    public GrpoTrainer innerEpochs(int n) {
        this.innerEpochs = n;
        return this;
    }

    /**
     * @param n the most tokens per completion
     * @param end the end token
     * @return this trainer
     */
    public GrpoTrainer completions(int n, int end) {
        this.maxNewTokens = n;
        this.endToken = end;
        return this;
    }

    /**
     * @param low the lower clip range {@code 1 - low}
     * @param high the upper clip range {@code 1 + high}
     * @return this trainer
     */
    public GrpoTrainer clip(float low, float high) {
        this.clipLow = low;
        this.clipHigh = high;
        return this;
    }

    /**
     * @param beta the weight of the KL penalty to the reference (0 turns it off)
     * @return this trainer
     */
    public GrpoTrainer klCoefficient(float beta) {
        this.kl = beta;
        return this;
    }

    /**
     * @param t the sampling temperature
     * @return this trainer
     */
    public GrpoTrainer temperature(double t) {
        this.temperature = t;
        return this;
    }

    /**
     * Computes group-relative advantages.
     *
     * @param rewards one group's rewards
     * @param divideByStd whether to divide by the group's standard deviation
     * @return the advantages
     */
    public static float[] advantages(float[] rewards, boolean divideByStd) {
        double mean = 0;
        for (float r : rewards) {
            mean += r;
        }
        mean /= rewards.length;
        double var = 0;
        for (float r : rewards) {
            var += (r - mean) * (r - mean);
        }
        double std = Math.sqrt(var / Math.max(1, rewards.length - 1));
        float[] a = new float[rewards.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) (divideByStd ? (rewards[i] - mean) / (std + 1e-4) : rewards[i] - mean);
        }
        return a;
    }

    /**
     * Samples a batch, scores it and updates the policy.
     *
     * @param promptSource draws a prompt
     * @param reward the verifiable reward
     * @param random the random source
     * @return what happened
     */
    public Stats step(Function<SplittableRandom, int[]> promptSource, Reward reward, SplittableRandom random) {
        if (endToken < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
        List<int[]> groupPrompts = new ArrayList<>();
        List<int[][]> groups = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        for (int p = 0; p < prompts; p++) {
            int[] prompt = promptSource.apply(random);
            int[][] group = new int[groupSize][];
            Arrays.fill(group, prompt);
            int[][] outs = policy.sample(group, maxNewTokens, endToken, temperature, random);
            float[] r = new float[groupSize];
            for (int g = 0; g < groupSize; g++) {
                r[g] = reward.score(prompt, outs[g]);
            }
            groupPrompts.add(prompt);
            groups.add(outs);
            rewards.add(r);
        }
        return update(groupPrompts, groups, rewards);
    }

    /**
     * Updates the policy from groups that were already sampled and scored, for loops that
     * sample themselves (self-play, where the reward of a group depends on other samples).
     *
     * @param groupPrompts one prompt per group
     * @param groups the completions of each group, sampled from the current policy
     * @param groupRewards the rewards of each group's completions
     * @return what happened
     */
    public Stats update(List<int[]> groupPrompts, List<int[][]> groups, List<float[]> groupRewards) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<Float> adv = new ArrayList<>();
        double rewardSum = 0;
        int sampled = 0;
        int used = 0;
        for (int p = 0; p < groups.size(); p++) {
            int[] prompt = groupPrompts.get(p);
            int[][] outs = groups.get(p);
            float[] r = groupRewards.get(p);
            for (float x : r) {
                rewardSum += x;
                sampled++;
            }
            float[] a = advantages(r, variant != Variant.DR_GRPO);
            boolean allEqual = true;
            for (float x : r) {
                allEqual &= x == r[0];
            }
            if (allEqual && variant == Variant.DAPO) {
                continue;   // dynamic sampling: no learning signal in this group
            }
            used++;
            for (int g = 0; g < outs.length; g++) {
                if (outs[g].length == 0) {
                    continue;
                }
                ps.add(prompt);
                cs.add(outs[g]);
                adv.add(a[g]);
            }
        }
        if (cs.isEmpty()) {
            return new Stats(sampled == 0 ? 0 : rewardSum / sampled, 0, 0, 0);
        }
        int tokens = WindowLm.totalTokens(cs);
        float[] tokenAdv = new float[tokens];
        float[] weight = new float[tokens];
        int k = 0;
        for (int s = 0; s < cs.size(); s++) {
            int len = cs.get(s).length;
            for (int t = 0; t < len; t++) {
                tokenAdv[k] = adv.get(s);
                weight[k] = switch (variant) {
                    case GRPO -> 1f / (len * cs.size());
                    case DR_GRPO -> 1f / (maxNewTokens * cs.size());
                    case DAPO -> 1f / tokens;
                };
                k++;
            }
        }
        Tensor advT = LlmTensors.column(tokenAdv);
        Tensor weightT = LlmTensors.column(weight);
        Tensor old = policy.logProbs(ps, cs).value();
        Tensor ref = kl > 0 ? reference.logProbs(ps, cs).value() : null;
        double klSum = 0;
        double clipped = 0;
        for (int e = 0; e < innerEpochs; e++) {
            Var logp = policy.logProbs(ps, cs);
            Var ratio = logp.sub(old).exp();
            Var surrogate = ratio.mul(advT).minimum(ratio.clamp(1 - clipLow, 1 + clipHigh).mul(advT));
            Var perToken = surrogate;
            if (ref != null) {
                Var d = logp.neg().add(ref);
                Var k3 = d.exp().sub(d).sub(1);
                perToken = perToken.sub(k3.mul(kl));
                if (e == 0) {
                    klSum = LlmTensors.mean(LlmTensors.values(k3));
                }
            }
            if (e == innerEpochs - 1) {
                float[] rv = LlmTensors.values(ratio);
                int c = 0;
                for (float x : rv) {
                    if (x < 1 - clipLow || x > 1 + clipHigh) {
                        c++;
                    }
                }
                clipped = (double) c / rv.length;
            }
            Var loss = perToken.mul(weightT).sum().neg();
            loss.backward();
            optimizer.step(policy.parameters());
        }
        return new Stats(rewardSum / sampled, used, klSum, clipped);
    }
}
