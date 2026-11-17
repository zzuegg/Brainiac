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
import java.util.function.UnaryOperator;

/**
 * Group Relative Policy Optimization: reinforcement learning with verifiable rewards for a
 * language model, without a value network. For each prompt the policy samples a group of
 * completions; each completion's advantage is its reward relative to the group, so the
 * group itself is the baseline. The update is PPO's clipped surrogate per token plus a KL
 * penalty towards a frozen reference model (the unbiased {@code k3} estimator
 * {@code exp(ref - logp) - (ref - logp) - 1}).
 *
 * <p>Twelve published critic-free recipes are built in ({@link Variant}):
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
 *   <li>{@code CISPO} (MiniMax-M1, <a href="https://arxiv.org/abs/2506.13585">2506.13585</a>):
 *       instead of clipping the <em>update</em> (which zeroes the gradient of every clipped
 *       token), the importance weight itself is clipped and stop-gradiented,
 *       {@code J = sum_t sg(min(r_t, 1 + eps_high)) A_t log pi(t)}, so every token keeps a
 *       gradient, including the rare high-ratio "fork" tokens PPO silences.</li>
 *   <li>{@code GMPO} (Geometric-Mean Policy Optimization,
 *       <a href="https://arxiv.org/abs/2507.20673">2507.20673</a>): each sequence maximizes
 *       the <em>geometric</em> mean of its token-level clipped, importance-weighted
 *       advantages, {@code A exp(mean_t sgn(A) min(sgn(A) log r_t, clip(sgn(A) log r_t)))},
 *       which an outlier ratio moves far less than the arithmetic mean, so a wider clip range
 *       (log-space 0.4) stays stable.</li>
 *   <li>{@code PSR}, {@code NSR} and {@code W_REINFORCE} (Zhu et al., "The surprising
 *       effectiveness of negative reinforcement",
 *       <a href="https://arxiv.org/abs/2506.01347">2506.01347</a>): REINFORCE with rewards
 *       {@code +1} for a correct and {@code -1} for a wrong completion and no baseline,
 *       restricted to the positive samples (PSR), to the negative ones (NSR: only push
 *       probability off wrong answers, redistributing it by the model's own prior), or with
 *       the positive term down-weighted to {@code lambda = 0.1} (W-REINFORCE).</li>
 * </ul>
 *
 * <p>Orthogonal knobs from the 2025 entropy and exploration literature:
 * <ul>
 *   <li>{@link #entropyControl}: Clip-Cov and KL-Cov (Cui et al., "The entropy mechanism of
 *       RL for reasoning LMs", <a href="https://arxiv.org/abs/2505.22617">2505.22617</a>).
 *       Entropy falls in proportion to the covariance between a token's log-probability and
 *       its advantage, so the few tokens with the largest covariance are either detached from
 *       the gradient (Clip-Cov: a random fraction of the tokens whose covariance lies in a
 *       band) or held in place with a {@code |log pi - log pi_old|} penalty (KL-Cov: the top
 *       fraction by covariance).</li>
 *   <li>{@link #forkingTokens}: the 80/20 rule (Wang et al., "Beyond the 80/20 rule",
 *       <a href="https://arxiv.org/abs/2506.01939">2506.01939</a>): only the highest-entropy
 *       "forking" tokens of the batch (top 20%) receive the policy gradient.</li>
 *   <li>{@link #passAtK}: pass@k training (Chen et al.,
 *       <a href="https://arxiv.org/abs/2508.10751">2508.10751</a>): the reward of a group is
 *       its pass@k, with the closed-form advantages of {@link #passAtKAdvantages}, so a rare
 *       correct sample is worth more than in pass@1 training and exploration pays.</li>
 *   <li>{@link #behaviour} with {@link #truncatedImportance}: rollouts come from a different,
 *       cheaper sampler (a quantized copy of the policy, say), and truncated importance
 *       sampling (Yao et al., "Your efficient RL framework secretly brings you off-policy RL
 *       training", 2025) reweights each token by {@code min(pi_learner / pi_sampler, C)} to
 *       correct the mismatch.</li>
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
        GSPO,
        /** Clipped, stop-gradient importance weights: every token keeps a gradient. */
        CISPO,
        /** Geometric mean of the clipped token-level importance-weighted advantages. */
        GMPO,
        /** REINFORCE on the correct samples only (+1 rewards). */
        PSR,
        /** REINFORCE on the wrong samples only (-1 rewards). */
        NSR,
        /** REINFORCE with +0.1 for correct and -1 for wrong samples. */
        W_REINFORCE
    }

    /** Covariance-based entropy control (Cui et al. 2025). */
    public enum EntropyControl {
        /** Off. */
        NONE,
        /** Detach a random fraction of the tokens whose covariance lies in a band. */
        CLIP_COV,
        /** Penalize {@code |log pi - log pi_old|} on the top fraction of tokens by covariance. */
        KL_COV
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
     * @param entropy the mean per-token entropy of the policy on its rollouts, before the update
     * @param updatedFraction the fraction of tokens that received a policy gradient (token
     *     masks of {@link #forkingTokens} and Clip-Cov)
     * @param ratioMin the smallest token importance ratio {@code pi / pi_old} in the last epoch
     * @param ratioMax the largest token importance ratio in the last epoch
     * @param behaviourKl the mean per-token {@code log mu - log pi} of the rollouts, where
     *     {@code mu} is the {@link #behaviour} sampler (an estimate of {@code KL(mu || pi)}; 0
     *     when the policy samples itself)
     */
    public record Stats(double meanReward, int groupsUsed, double meanKl, double clipFraction, double entropy,
            double updatedFraction, double ratioMin, double ratioMax, double behaviourKl) {

        /**
         * The original four statistics, the rest left at neutral values.
         *
         * @param meanReward the mean reward of all sampled completions
         * @param groupsUsed how many groups contributed a gradient
         * @param meanKl the mean per-token KL estimate to the reference
         * @param clipFraction the fraction of clipped tokens
         */
        public Stats(double meanReward, int groupsUsed, double meanKl, double clipFraction) {
            this(meanReward, groupsUsed, meanKl, clipFraction, 0, 0, 1, 1, 0);
        }
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
    private EntropyControl entropyControl = EntropyControl.NONE;
    private float covFraction;
    private float covLow = 1;
    private float covHigh = 5;
    private float covKl = 1;
    private float forkingFraction = 1;
    private int passK;
    private float successThreshold = 0.5f;
    private UnaryOperator<WindowLm> behaviourSource;
    private WindowLm behaviour;
    private float tisCap;
    private final SplittableRandom maskRandom = new SplittableRandom(0x5eed);

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
     * Picks a recipe. {@code DAPO} and {@code CISPO} also raise the upper clip to 0.28,
     * {@code GSPO} sets the sequence-level clip range to (3e-4, 4e-4) and {@code GMPO} sets
     * the log-space clip range to (0.4, 0.4), unless the clip is set later.
     *
     * @param v the variant
     * @return this trainer
     */
    public GrpoTrainer variant(Variant v) {
        this.variant = Objects.requireNonNull(v);
        if (v == Variant.DAPO || v == Variant.CISPO) {
            clipHigh = 0.28f;
        } else if (v == Variant.GSPO) {
            clipLow = 3e-4f;
            clipHigh = 4e-4f;
        } else if (v == Variant.GMPO) {
            clipLow = 0.4f;
            clipHigh = 0.4f;
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
     * @param low the lower clip range {@code 1 - low} (log-space {@code -low} for GMPO)
     * @param high the upper clip range {@code 1 + high} (log-space {@code +high} for GMPO)
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
     * Turns on covariance-based entropy control. The centered covariance of a token is
     * {@code (log pi_old(t) - mean) (A_t - mean)} over the batch.
     *
     * <ul>
     *   <li>{@code CLIP_COV}: of the tokens whose covariance lies in {@code [low, high]},
     *       {@code fraction} of all tokens (chosen at random) lose their policy gradient.</li>
     *   <li>{@code KL_COV}: the top {@code fraction} of tokens by covariance get the penalty
     *       {@code coefficient |log pi - log pi_old|}; {@code low} and {@code high} are unused.</li>
     * </ul>
     *
     * @param mode the method
     * @param fraction the share of the batch's tokens it acts on (paper: 2e-4 for Clip-Cov,
     *     2e-3 for KL-Cov on long sequences; small batches need more)
     * @param low the lower covariance bound of Clip-Cov (paper: 1)
     * @param high the upper covariance bound of Clip-Cov (paper: 5)
     * @param coefficient the KL-Cov penalty weight (paper: 1)
     * @return this trainer
     */
    public GrpoTrainer entropyControl(EntropyControl mode, float fraction, float low, float high,
            float coefficient) {
        this.entropyControl = Objects.requireNonNull(mode);
        this.covFraction = fraction;
        this.covLow = low;
        this.covHigh = high;
        this.covKl = coefficient;
        return this;
    }

    /**
     * Restricts the policy gradient to the highest-entropy ("forking") tokens of each batch.
     *
     * @param topFraction the share of tokens that keep their gradient (paper: 0.2); 1 turns
     *     the mask off
     * @return this trainer
     */
    public GrpoTrainer forkingTokens(float topFraction) {
        if (!(topFraction > 0 && topFraction <= 1)) {
            throw new IllegalArgumentException("fraction must be in (0, 1], got " + topFraction);
        }
        this.forkingFraction = topFraction;
        return this;
    }

    /**
     * Pass@k training: replaces the variant's advantages by the analytic pass@k advantages
     * of {@link #passAtKAdvantages}. A completion counts as correct when its reward reaches
     * the {@link #successThreshold}.
     *
     * @param k the k of pass@k, at most the group size; 0 turns it off
     * @return this trainer
     */
    public GrpoTrainer passAtK(int k) {
        if (k < 0) {
            throw new IllegalArgumentException("k must be >= 0");
        }
        this.passK = k;
        return this;
    }

    /**
     * @param threshold the reward at or above which a completion counts as correct (for
     *     pass@k training and the PSR/NSR/W-REINFORCE variants); default 0.5
     * @return this trainer
     */
    public GrpoTrainer successThreshold(float threshold) {
        this.successThreshold = threshold;
        return this;
    }

    /**
     * Samples rollouts from a different model than the one being trained, as fast inference
     * engines do. The function is called once per {@link #step} with the current policy and
     * returns the sampler for that step (for example a freshly quantized copy).
     *
     * @param sampler maps the policy to the rollout model, or {@code null} to sample the policy
     * @return this trainer
     */
    public GrpoTrainer behaviour(UnaryOperator<WindowLm> sampler) {
        this.behaviourSource = sampler;
        return this;
    }

    /**
     * Truncated importance sampling: when a {@link #behaviour} sampler is set, each token's
     * loss is multiplied by {@code min(pi_old(t) / mu(t), cap)}, a stop-gradient weight that
     * corrects for sampling from {@code mu} instead of the learner.
     *
     * @param cap the truncation {@code C} (paper: 2); 0 turns the correction off
     * @return this trainer
     */
    public GrpoTrainer truncatedImportance(float cap) {
        this.tisCap = cap;
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
     * The analytic pass@k advantages of a group of {@code N} binary outcomes with {@code c}
     * successes (Chen et al. 2025). The group's expected pass@k over all {@code k}-subsets is
     * {@code R = 1 - C(N - c, k) / C(N, k)} with standard deviation {@code s = sqrt(R(1 - R))};
     * a correct sample makes every subset containing it pass, so its advantage is
     * {@code (1 - R) / s}, and a wrong one's is
     * {@code (1 - R - C(N - c - 1, k - 1) / C(N - 1, k - 1)) / s}. With {@code k = 1} this is
     * exactly GRPO's std-normalized advantage (population std).
     *
     * @param rewards one group's rewards
     * @param k the k of pass@k, {@code 1 <= k <= N}
     * @param threshold the reward that counts as a success
     * @return the advantages, all 0 when every sample succeeded or every one failed
     */
    public static float[] passAtKAdvantages(float[] rewards, int k, float threshold) {
        int n = rewards.length;
        if (k < 1 || k > n) {
            throw new IllegalArgumentException("need 1 <= k <= group size, got " + k);
        }
        int neg = 0;
        for (float r : rewards) {
            neg += r >= threshold ? 0 : 1;
        }
        float[] a = new float[n];
        double mean = 1 - chooseRatio(neg, n, k);
        double std = Math.sqrt(mean * (1 - mean));
        if (std < 1e-9) {
            return a;
        }
        double aPos = (1 - mean) / std;
        double aNeg = (1 - mean - chooseRatio(neg - 1, n - 1, k - 1)) / std;
        for (int i = 0; i < n; i++) {
            a[i] = (float) (rewards[i] >= threshold ? aPos : aNeg);
        }
        return a;
    }

    /** {@code C(a, k) / C(b, k)} for {@code a <= b}; 0 when {@code a < k}. */
    private static double chooseRatio(int a, int b, int k) {
        if (a < k) {
            return 0;
        }
        double r = 1;
        for (int i = 0; i < k; i++) {
            r *= (double) (a - i) / (b - i);
        }
        return r;
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
        behaviour = behaviourSource == null ? null : behaviourSource.apply(policy);
        WindowLm sampler = behaviour == null ? policy : behaviour;
        List<int[]> groupPrompts = new ArrayList<>();
        List<int[][]> groups = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        float[] greedyRewards = variant == Variant.REMAX ? new float[prompts] : null;
        for (int p = 0; p < prompts; p++) {
            int[] prompt = promptSource.apply(random);
            int[][] group = new int[groupSize][];
            Arrays.fill(group, prompt);
            int[][] outs = sampler.sample(group, maxNewTokens, endToken, temperature, random);
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
        try {
            return update(groupPrompts, groups, rewards, greedyRewards);
        } finally {
            behaviour = null;
        }
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

    private float[] groupAdvantages(float[] r, float[] baselines, int p) {
        if (passK > 0) {
            return passAtKAdvantages(r, passK, successThreshold);
        }
        return switch (variant) {
            case GRPO, DAPO, GSPO, CISPO, GMPO -> advantages(r, true);
            case DR_GRPO -> advantages(r, false);
            case RLOO -> leaveOneOut(r);
            case REMAX -> baselines == null ? advantages(r, false) : minus(r, baselines[p]);
            case REINFORCE_PP -> r.clone();   // raw rewards, normalized over the batch below
            case PSR -> polarity(r, 1, 0);
            case NSR -> polarity(r, 0, 1);
            case W_REINFORCE -> polarity(r, 0.1f, 1);
        };
    }

    private float[] polarity(float[] r, float positive, float negative) {
        float[] a = new float[r.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = r[i] >= successThreshold ? positive : -negative;
        }
        return a;
    }

    private float tokenWeight(int len, int sequences, int tokens) {
        return switch (variant) {
            case GRPO -> 1f / (len * sequences);
            case DR_GRPO -> 1f / (maxNewTokens * sequences);
            case DAPO, REINFORCE_PP, GSPO, CISPO, GMPO -> 1f / tokens;
            case RLOO, REMAX, PSR, NSR, W_REINFORCE -> 1f / sequences;
        };
    }

    /** Builds the prefixes in the order of {@link WindowLm#logProbs}. */
    private static List<int[]> prefixes(List<int[]> ps, List<int[]> cs, int[] targets) {
        List<int[]> prefixes = new ArrayList<>();
        int k = 0;
        for (int s = 0; s < ps.size(); s++) {
            int[] p = ps.get(s);
            int[] c = cs.get(s);
            for (int t = 0; t < c.length; t++) {
                int[] prefix = Arrays.copyOf(p, p.length + t);
                System.arraycopy(c, 0, prefix, p.length, t);
                prefixes.add(prefix);
                targets[k++] = c[t];
            }
        }
        return prefixes;
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
            float[] a = groupAdvantages(r, baselines, p);
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
        int[] targets = new int[tokens];
        List<int[]> prefixes = prefixes(ps, cs, targets);
        Var oldLogSoftmax = LlmTensors.logSoftmax(policy.logits(prefixes));
        Tensor old = LlmTensors.pick(oldLogSoftmax, targets).value();
        float[] oldV = old.toArray();
        float[] entropies = entropies(LlmTensors.values(oldLogSoftmax), policy.vocabSize());
        Tensor ref = kl > 0 ? reference.logProbs(ps, cs).value() : null;
        boolean klInReward = variant == Variant.REINFORCE_PP;
        float[] refV = ref != null && klInReward ? ref.toArray() : null;
        float[] tokenAdv = new float[tokens];
        float[] weight = new float[tokens];
        int k = 0;
        for (int s = 0; s < cs.size(); s++) {
            int len = cs.get(s).length;
            double tail = 0;
            if (refV != null) {
                for (int t = 0; t < len; t++) {
                    tail += oldV[k + t] - refV[k + t];
                }
            }
            for (int t = 0; t < len; t++) {
                if (refV != null) {
                    tokenAdv[k] = (float) (adv.get(s) - kl * tail);   // REINFORCE++ return-to-go
                    tail -= oldV[k] - refV[k];
                } else {
                    tokenAdv[k] = adv.get(s);
                }
                weight[k] = tokenWeight(len, cs.size(), tokens);
                k++;
            }
        }
        if (klInReward) {
            normalize(tokenAdv);
        }

        // token masks: forking tokens (80/20) and covariance-based entropy control
        float[] gradMask = fill(tokens, 1f);
        if (forkingFraction < 1) {
            keepTop(gradMask, entropies, Math.max(1, Math.round(forkingFraction * tokens)));
        }
        float[] covSelect = null;
        if (entropyControl != EntropyControl.NONE && covFraction > 0) {
            float[] cov = covariance(oldV, tokenAdv);
            int count = Math.max(1, Math.round(covFraction * tokens));
            if (entropyControl == EntropyControl.CLIP_COV) {
                List<Integer> band = new ArrayList<>();
                for (int t = 0; t < tokens; t++) {
                    if (cov[t] >= covLow && cov[t] <= covHigh) {
                        band.add(t);
                    }
                }
                for (int i = 0; i < Math.min(count, band.size()); i++) {
                    int j = i + maskRandom.nextInt(band.size() - i);
                    int pick = band.get(j);
                    band.set(j, band.get(i));
                    gradMask[pick] = 0;
                }
            } else {
                covSelect = fill(tokens, 1f);
                keepTop(covSelect, cov, count);
            }
        }
        double updated = LlmTensors.mean(gradMask);

        // rollouts from another sampler: mismatch statistics and truncated importance weights
        double behaviourKl = 0;
        float[] tis = null;
        if (behaviour != null) {
            float[] mu = behaviour.logProbs(ps, cs).value().toArray();
            for (int t = 0; t < tokens; t++) {
                behaviourKl += mu[t] - oldV[t];
            }
            behaviourKl /= tokens;
            if (tisCap > 0) {
                tis = new float[tokens];
                for (int t = 0; t < tokens; t++) {
                    tis[t] = (float) Math.min(Math.exp(oldV[t] - mu[t]), tisCap);
                }
            }
        }
        float[] pgWeight = new float[tokens];
        for (int t = 0; t < tokens; t++) {
            pgWeight[t] = weight[t] * gradMask[t] * (tis == null ? 1 : tis[t]);
        }

        boolean clip = variant != Variant.RLOO && variant != Variant.REMAX && variant != Variant.PSR
                && variant != Variant.NSR && variant != Variant.W_REINFORCE;
        Tensor advT = LlmTensors.column(tokenAdv);
        Tensor weightT = LlmTensors.column(weight);
        Tensor pgWeightT = LlmTensors.column(pgWeight);
        boolean sequenceLevel = variant == Variant.GSPO || variant == Variant.GMPO;
        Tensor seqMean = sequenceLevel ? WindowLm.sequenceSum(cs, true) : null;
        float[] seqAdv = new float[cs.size()];
        float[] tokenSign = new float[tokens];
        k = 0;
        for (int s = 0; s < seqAdv.length; s++) {
            seqAdv[s] = adv.get(s) / cs.size();
            for (int t = 0; t < cs.get(s).length; t++) {
                tokenSign[k++] = adv.get(s) >= 0 ? 1 : -1;
            }
        }
        Tensor seqAdvT = LlmTensors.column(seqAdv);
        Tensor signT = LlmTensors.column(tokenSign);
        Tensor klWeight = seqMean != null ? LlmTensors.column(fill(tokens, 1f / tokens)) : weightT;
        double klSum = 0;
        double clipped = 0;
        double ratioMin = 1;
        double ratioMax = 1;
        for (int e = 0; e < innerEpochs; e++) {
            Var logp = policy.logProbs(ps, cs);
            Var logRatio = logp.sub(old);
            Var ratio;
            Var objective;
            float[] tokenRatios = null;
            if (variant == Variant.GSPO) {
                // GSPO: one length-normalized ratio per sequence, clipped as a whole
                ratio = Var.of(seqMean).matmul(logRatio).exp();
                objective = ratio.mul(seqAdvT).minimum(ratio.clamp(1 - clipLow, 1 + clipHigh).mul(seqAdvT)).sum();
            } else if (variant == Variant.GMPO) {
                // GMPO: geometric mean of sign-aware, log-space clipped token ratios
                Var signed = logRatio.mul(signT);
                Var clippedLog = signed.minimum(signed.clamp(-clipLow, clipHigh)).mul(signT);
                ratio = logRatio.exp();
                objective = Var.of(seqMean).matmul(clippedLog).exp().mul(seqAdvT).sum();
            } else if (variant == Variant.CISPO) {
                // CISPO: clipped, stop-gradient importance weight times log pi
                ratio = logRatio.exp();
                tokenRatios = LlmTensors.values(ratio);
                float[] w = new float[tokens];
                for (int t = 0; t < tokens; t++) {
                    w[t] = Math.min(tokenRatios[t], 1 + clipHigh) * tokenAdv[t] * pgWeight[t];
                }
                objective = logp.mul(LlmTensors.column(w)).sum();
            } else {
                ratio = logRatio.exp();
                Var surrogate = clip
                        ? ratio.mul(advT).minimum(ratio.clamp(1 - clipLow, 1 + clipHigh).mul(advT))
                        : ratio.mul(advT);
                objective = surrogate.mul(pgWeightT).sum();
            }
            if (covSelect != null) {
                // KL-Cov: hold the highest-covariance tokens near the rollout policy
                float[] w = new float[tokens];
                for (int t = 0; t < tokens; t++) {
                    w[t] = covSelect[t] * weight[t] * covKl;
                }
                objective = objective.sub(logRatio.abs().mul(LlmTensors.column(w)).sum());
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
            if (e == innerEpochs - 1) {
                float[] tr = tokenRatios != null ? tokenRatios
                        : LlmTensors.values(variant == Variant.GSPO ? logRatio.exp() : ratio);
                ratioMin = Double.POSITIVE_INFINITY;
                ratioMax = Double.NEGATIVE_INFINITY;
                for (float x : tr) {
                    ratioMin = Math.min(ratioMin, x);
                    ratioMax = Math.max(ratioMax, x);
                }
                clipped = clipFraction(variant == Variant.GSPO ? LlmTensors.values(ratio) : tr, clip);
            }
            Var loss = objective.neg();
            loss.backward();
            optimizer.step(policy.parameters());
        }
        return new Stats(rewardSum / sampled, used, klSum, clipped, LlmTensors.mean(entropies), updated, ratioMin,
                ratioMax, behaviourKl);
    }

    private double clipFraction(float[] ratios, boolean clip) {
        if (!clip) {
            return 0;
        }
        int c = 0;
        for (float x : ratios) {
            boolean out = switch (variant) {
                case CISPO -> x > 1 + clipHigh;
                case GMPO -> Math.log(x) < -clipLow || Math.log(x) > clipHigh;
                default -> x < 1 - clipLow || x > 1 + clipHigh;
            };
            c += out ? 1 : 0;
        }
        return (double) c / ratios.length;
    }

    /** Per-row entropy of a flattened {@code [rows, vocab]} log-softmax. */
    static float[] entropies(float[] logSoftmax, int vocab) {
        float[] h = new float[logSoftmax.length / vocab];
        for (int i = 0; i < h.length; i++) {
            double s = 0;
            for (int j = 0; j < vocab; j++) {
                double lp = logSoftmax[i * vocab + j];
                s -= Math.exp(lp) * lp;
            }
            h[i] = (float) s;
        }
        return h;
    }

    /** Centered covariance of each token's log-probability and advantage over the batch. */
    private static float[] covariance(float[] logp, float[] advantage) {
        double ml = LlmTensors.mean(logp);
        double ma = LlmTensors.mean(advantage);
        float[] c = new float[logp.length];
        for (int i = 0; i < c.length; i++) {
            c[i] = (float) ((logp[i] - ml) * (advantage[i] - ma));
        }
        return c;
    }

    /** Zeroes {@code mask} everywhere except at the {@code keep} largest {@code score}s. */
    private static void keepTop(float[] mask, float[] score, int keep) {
        Integer[] order = new Integer[score.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Float.compare(score[b], score[a]));
        for (int i = keep; i < order.length; i++) {
            mask[order[i]] = 0;
        }
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
