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
 * <p>Seven published critic-free recipes are built in ({@link Variant}):
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
 *   <li>{@code RLOO} (REINFORCE leave-one-out, Ahmadian et al. 2024,
 *       <a href="https://arxiv.org/abs/2402.14740">2402.14740</a>): plain REINFORCE whose
 *       baseline for each completion is the mean reward of the <em>other</em> completions of
 *       its group, {@code A_i = r_i - mean_{j != i} r_j}; no clipping and no critic. The
 *       per-sequence loss is the token sum, as in REINFORCE.</li>
 *   <li>{@code REMAX} (Li et al. 2023, <a href="https://arxiv.org/abs/2310.10505">2310.10505</a>):
 *       REINFORCE with the reward of the greedy completion of the same prompt as the
 *       baseline, which costs one extra greedy rollout per prompt and no critic.</li>
 *   <li>{@code REINFORCE_PP} (REINFORCE++, Hu 2025,
 *       <a href="https://arxiv.org/abs/2501.03262">2501.03262</a>): no group baseline at all;
 *       the KL to the reference is folded into a per-token return
 *       {@code G_t = r - beta sum_{s >= t} (log pi_old(s) - log ref(s))}, the returns are
 *       normalized over the whole batch (global advantage normalization), then PPO-clipped.</li>
 *   <li>{@code GSPO} (Group Sequence Policy Optimization, Qwen3,
 *       <a href="https://arxiv.org/abs/2507.18071">2507.18071</a>): the importance ratio is
 *       the length-normalized <em>sequence</em> likelihood ratio
 *       {@code s_i = exp(mean_t (log pi - log pi_old))}, clipped as a whole with a much
 *       narrower range (3e-4, 4e-4) than token-level PPO, which removes the noise of
 *       per-token ratios on long sequences.</li>
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
        DAPO,
        /** REINFORCE with a leave-one-out group baseline, unclipped. */
        RLOO,
        /** REINFORCE with the greedy completion's reward as the baseline, unclipped. */
        REMAX,
        /** Clipped REINFORCE with token-level KL in the return and global normalization. */
        REINFORCE_PP,
        /** Sequence-level importance ratio and clipping. */
        GSPO
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
     * @param clipFraction the fraction of tokens (sequences for GSPO) whose ratio was clipped
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
     * Picks a recipe. {@code DAPO} also raises the upper clip to 0.28 and {@code GSPO} sets
     * the sequence-level clip range to (3e-4, 4e-4), unless the clip is set later.
     *
     * @param v the variant
     * @return this trainer
     */
    public GrpoTrainer variant(Variant v) {
        this.variant = Objects.requireNonNull(v);
        if (v == Variant.DAPO) {
            clipHigh = 0.28f;
        } else if (v == Variant.GSPO) {
            clipLow = 3e-4f;
            clipHigh = 4e-4f;
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
     * Leave-one-out advantages (RLOO): each reward minus the mean of the others in its group.
     *
     * @param rewards one group's rewards (at least two)
     * @return the advantages
     */
    public static float[] leaveOneOut(float[] rewards) {
        int n = rewards.length;
        double sum = 0;
        for (float r : rewards) {
            sum += r;
        }
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (float) (rewards[i] - (sum - rewards[i]) / (n - 1));
        }
        return a;
    }

    /**
     * Samples a batch, scores it and updates the policy. {@code REMAX} also decodes each
     * prompt greedily and uses that completion's reward as the baseline.
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
        float[] greedyRewards = variant == Variant.REMAX ? new float[prompts] : null;
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
            if (greedyRewards != null) {
                greedyRewards[p] = reward.score(prompt, policy.greedy(prompt, maxNewTokens, endToken));
            }
        }
        return update(groupPrompts, groups, rewards, greedyRewards);
    }

    /**
     * Updates the policy from groups that were already sampled and scored, for loops that
     * sample themselves (self-play, where the reward of a group depends on other samples).
     * {@code REMAX} needs per-group baselines and falls back to the group mean here; use
     * {@link #update(List, List, List, float[])}.
     *
     * @param groupPrompts one prompt per group
     * @param groups the completions of each group, sampled from the current policy
     * @param groupRewards the rewards of each group's completions
     * @return what happened
     */
    public Stats update(List<int[]> groupPrompts, List<int[][]> groups, List<float[]> groupRewards) {
        return update(groupPrompts, groups, groupRewards, null);
    }

    /**
     * Updates the policy from sampled and scored groups with explicit per-group baselines
     * (ReMax's greedy rewards).
     *
     * @param groupPrompts one prompt per group
     * @param groups the completions of each group, sampled from the current policy
     * @param groupRewards the rewards of each group's completions
     * @param baselines one baseline per group for {@code REMAX}, or {@code null}
     * @return what happened
     */
    public Stats update(List<int[]> groupPrompts, List<int[][]> groups, List<float[]> groupRewards,
            float[] baselines) {
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
            float[] a = switch (variant) {
                case GRPO, DAPO, GSPO -> advantages(r, true);
                case DR_GRPO -> advantages(r, false);
                case RLOO -> leaveOneOut(r);
                case REMAX -> baselines == null ? advantages(r, false) : minus(r, baselines[p]);
                case REINFORCE_PP -> r.clone();   // raw rewards, normalized over the batch below
            };
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
        Tensor old = policy.logProbs(ps, cs).value();
        Tensor ref = kl > 0 ? reference.logProbs(ps, cs).value() : null;
        boolean klInReward = variant == Variant.REINFORCE_PP;
        float[] oldV = ref != null && klInReward ? old.toArray() : null;
        float[] refV = oldV == null ? null : ref.toArray();
        float[] tokenAdv = new float[tokens];
        float[] weight = new float[tokens];
        int k = 0;
        for (int s = 0; s < cs.size(); s++) {
            int len = cs.get(s).length;
            double tail = 0;
            if (oldV != null) {
                for (int t = 0; t < len; t++) {
                    tail += oldV[k + t] - refV[k + t];
                }
            }
            for (int t = 0; t < len; t++) {
                if (oldV != null) {
                    tokenAdv[k] = (float) (adv.get(s) - kl * tail);   // REINFORCE++ return-to-go
                    tail -= oldV[k] - refV[k];
                } else {
                    tokenAdv[k] = adv.get(s);
                }
                weight[k] = switch (variant) {
                    case GRPO -> 1f / (len * cs.size());
                    case DR_GRPO -> 1f / (maxNewTokens * cs.size());
                    case DAPO, REINFORCE_PP, GSPO -> 1f / tokens;
                    case RLOO, REMAX -> 1f / cs.size();
                };
                k++;
            }
        }
        if (klInReward) {
            normalize(tokenAdv);
        }
        boolean clip = variant != Variant.RLOO && variant != Variant.REMAX;
        Tensor advT = LlmTensors.column(tokenAdv);
        Tensor weightT = LlmTensors.column(weight);
        Tensor seqMean = variant == Variant.GSPO ? WindowLm.sequenceSum(cs, true) : null;
        float[] seqAdv = new float[cs.size()];
        for (int s = 0; s < seqAdv.length; s++) {
            seqAdv[s] = adv.get(s) / cs.size();
        }
        Tensor seqAdvT = LlmTensors.column(seqAdv);
        Tensor klWeight = seqMean != null ? LlmTensors.column(fill(tokens, 1f / tokens)) : weightT;
        double klSum = 0;
        double clipped = 0;
        for (int e = 0; e < innerEpochs; e++) {
            Var logp = policy.logProbs(ps, cs);
            Var ratio;
            Var objective;
            if (seqMean != null) {
                // GSPO: one length-normalized ratio per sequence, clipped as a whole
                ratio = Var.of(seqMean).matmul(logp.sub(old)).exp();
                objective = ratio.mul(seqAdvT).minimum(ratio.clamp(1 - clipLow, 1 + clipHigh).mul(seqAdvT)).sum();
            } else {
                ratio = logp.sub(old).exp();
                Var surrogate = clip
                        ? ratio.mul(advT).minimum(ratio.clamp(1 - clipLow, 1 + clipHigh).mul(advT))
                        : ratio.mul(advT);
                objective = surrogate.mul(weightT).sum();
            }
            if (ref != null) {
                Var d = logp.neg().add(ref);
                Var k3 = d.exp().sub(d).sub(1);
                if (!klInReward) {
                    objective = objective.sub(k3.mul(klWeight).sum().mul(kl));
                }
                if (e == 0) {
                    klSum = LlmTensors.mean(LlmTensors.values(k3));
                }
            }
            if (e == innerEpochs - 1 && clip) {
                float[] rv = LlmTensors.values(ratio);
                int c = 0;
                for (float x : rv) {
                    if (x < 1 - clipLow || x > 1 + clipHigh) {
                        c++;
                    }
                }
                clipped = (double) c / rv.length;
            }
            Var loss = objective.neg();
            loss.backward();
            optimizer.step(policy.parameters());
        }
        return new Stats(rewardSum / sampled, used, klSum, clipped);
    }

    private static float[] minus(float[] r, float baseline) {
        float[] a = new float[r.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = r[i] - baseline;
        }
        return a;
    }

    private static float[] fill(int n, float v) {
        float[] a = new float[n];
        Arrays.fill(a, v);
        return a;
    }

    /** Global advantage normalization: zero mean, unit standard deviation over the batch. */
    private static void normalize(float[] a) {
        double mean = LlmTensors.mean(a);
        double var = 0;
        for (float x : a) {
            var += (x - mean) * (x - mean);
        }
        double std = Math.sqrt(var / Math.max(1, a.length - 1));
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) ((a[i] - mean) / (std + 1e-4));
        }
    }
}
