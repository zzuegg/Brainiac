package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Intuitor's reward (Zhao et al., "Learning to reason without external rewards",
 * <a href="https://arxiv.org/abs/2505.19590">2505.19590</a>): reinforcement learning from
 * internal feedback, with no verifier and no labels. A completion is rewarded by the model's
 * own <em>self-certainty</em>, the mean over its tokens of the KL divergence from the uniform
 * distribution to the next-token distribution,
 * {@code (1 / |o|) sum_t KL(U || p(. | q, o_<t)) = -(1 / |o| V) sum_t sum_j log(V p_j)}.
 * Unlike entropy it is mode-seeking and penalizes a distribution for putting near-zero mass
 * anywhere, which in the paper makes it harder to hack than entropy minimization.
 *
 * <p>It is scored with the <em>current</em> policy (online self-certainty, so the reward
 * co-evolves with the model) and plugs into {@link GrpoTrainer#step} like any verifiable
 * reward; compare it with {@link TestTimeRl}'s majority-vote pseudo-labels.
 */
public final class SelfCertaintyReward implements GrpoTrainer.Reward {

    private final WindowLm model;

    /**
     * @param model the model whose certainty is the reward, normally the policy being trained
     */
    public SelfCertaintyReward(WindowLm model) {
        this.model = Objects.requireNonNull(model);
    }

    @Override
    public float score(int[] prompt, int[] completion) {
        return (float) selfCertainty(model, prompt, completion);
    }

    /**
     * @param model the model
     * @param prompt the prompt
     * @param completion the completion (empty scores 0)
     * @return the mean per-token {@code KL(uniform || p)}
     */
    public static double selfCertainty(WindowLm model, int[] prompt, int[] completion) {
        if (completion.length == 0) {
            return 0;
        }
        List<int[]> prefixes = new ArrayList<>();
        for (int t = 0; t < completion.length; t++) {
            int[] prefix = Arrays.copyOf(prompt, prompt.length + t);
            System.arraycopy(completion, 0, prefix, prompt.length, t);
            prefixes.add(prefix);
        }
        float[] logp = LlmTensors.values(LlmTensors.logSoftmax(model.logits(prefixes)));
        int v = model.vocabSize();
        double logV = Math.log(v);
        double s = 0;
        for (int t = 0; t < completion.length; t++) {
            double kl = 0;
            for (int j = 0; j < v; j++) {
                kl -= logp[t * v + j] + logV;
            }
            s += kl / v;
        }
        return s / completion.length;
    }
}
