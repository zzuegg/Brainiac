package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A PPO-clip policy update of a {@link WindowLm} from <em>per-token</em> advantages: the
 * shared last stage of every process-supervision recipe in this package, whose whole point is
 * that different tokens of one completion get different credit (Monte Carlo step values in
 * VinePPO, progress rewards, implicit process rewards). {@code GrpoTrainer} and
 * {@code PpoLmTrainer} compute their own advantages internally; this class takes them from the
 * caller. The loss per token is
 * {@code -min(rho A, clip(rho, 1 - eps, 1 + eps) A) + beta k3(pi, ref) - c H(pi)}, with
 * {@code rho = pi / pi_old}, the unbiased {@code k3} KL estimator
 * {@code exp(ref - logp) - (ref - logp) - 1} towards a frozen reference and an optional entropy
 * bonus, averaged over tokens (Schulman et al. 2017,
 * <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a>).
 */
public final class TokenPolicyGradient {

    /**
     * What one update did.
     *
     * @param meanKl the mean per-token {@code log pi_old - log ref}
     * @param clipFraction the share of tokens whose ratio left the clip range in the last epoch
     * @param entropy the mean per-token entropy before the update
     */
    public record Stats(double meanKl, double clipFraction, double entropy) {
    }

    private final WindowLm policy;
    private WindowLm reference;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int epochs = 2;
    private float clip = 0.2f;
    private float kl;
    private float entropyBonus;

    private TokenPolicyGradient(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the KL reference
     * @return the updater
     */
    public static TokenPolicyGradient of(WindowLm policy) {
        return new TokenPolicyGradient(policy);
    }

    /** @return the trained policy */
    public WindowLm policy() {
        return policy;
    }

    /**
     * @param ref the KL reference (not trained)
     * @return this updater
     */
    public TokenPolicyGradient reference(WindowLm ref) {
        this.reference = Objects.requireNonNull(ref);
        return this;
    }

    /**
     * @param o the policy optimizer
     * @return this updater
     */
    public TokenPolicyGradient optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param n optimizer steps per batch of rollouts
     * @return this updater
     */
    public TokenPolicyGradient epochs(int n) {
        this.epochs = n;
        return this;
    }

    /**
     * @param epsilon the PPO clip range
     * @return this updater
     */
    public TokenPolicyGradient clip(float epsilon) {
        this.clip = epsilon;
        return this;
    }

    /**
     * @param beta the weight of the {@code k3} KL penalty to the reference (0 turns it off)
     * @return this updater
     */
    public TokenPolicyGradient klCoefficient(float beta) {
        this.kl = beta;
        return this;
    }

    /**
     * @param c the weight of the entropy bonus (0 turns it off)
     * @return this updater
     */
    public TokenPolicyGradient entropyBonus(float c) {
        this.entropyBonus = c;
        return this;
    }

    /**
     * Spreads one value per completion over its tokens.
     *
     * @param completions the completions
     * @param perSequence one value per completion
     * @return one value per token, in the order of {@link WindowLm#logProbs}
     */
    public static float[] broadcast(List<int[]> completions, float[] perSequence) {
        int total = 0;
        for (int[] c : completions) {
            total += c.length;
        }
        float[] out = new float[total];
        int k = 0;
        for (int s = 0; s < completions.size(); s++) {
            for (int t = 0; t < completions.get(s).length; t++) {
                out[k++] = perSequence[s];
            }
        }
        return out;
    }

    /**
     * Where the steps of a completion end.
     *
     * @param completion the completion
     * @param stepTokens the tokens that end a step
     * @return the exclusive end index of each step; a trailing unfinished step is included
     */
    public static int[] stepEnds(int[] completion, int... stepTokens) {
        List<Integer> ends = new ArrayList<>();
        for (int t = 0; t < completion.length; t++) {
            boolean end = t == completion.length - 1;
            for (int s : stepTokens) {
                end |= completion[t] == s;
            }
            if (end) {
                ends.add(t + 1);
            }
        }
        return ends.stream().mapToInt(Integer::intValue).toArray();
    }

    private static List<int[]> prefixes(List<int[]> prompts, List<int[]> completions) {
        List<int[]> out = new ArrayList<>();
        for (int s = 0; s < prompts.size(); s++) {
            int[] p = prompts.get(s);
            int[] c = completions.get(s);
            for (int t = 0; t < c.length; t++) {
                int[] prefix = Arrays.copyOf(p, p.length + t);
                System.arraycopy(c, 0, prefix, p.length, t);
                out.add(prefix);
            }
        }
        return out;
    }

    /**
     * Updates the policy.
     *
     * @param prompts the prompts
     * @param completions the completions, sampled from the current policy
     * @param tokenAdvantages one advantage per token, in the order of {@link WindowLm#logProbs}
     * @return what happened
     */
    public Stats update(List<int[]> prompts, List<int[]> completions, float[] tokenAdvantages) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<Float> adv = new ArrayList<>();
        int k = 0;
        for (int s = 0; s < completions.size(); s++) {
            int[] c = completions.get(s);
            if (c.length > 0) {
                ps.add(prompts.get(s));
                cs.add(c);
                for (int t = 0; t < c.length; t++) {
                    adv.add(tokenAdvantages[k + t]);
                }
            }
            k += c.length;
        }
        if (cs.isEmpty()) {
            return new Stats(0, 0, 0);
        }
        int tokens = adv.size();
        float[] a = new float[tokens];
        for (int i = 0; i < tokens; i++) {
            a[i] = adv.get(i);
        }
        Tensor advT = Tensor.of(Shape.of(tokens, 1), a);
        Tensor old = policy.logProbs(ps, cs).value();
        Tensor ref = reference.logProbs(ps, cs).value();
        float[] oldV = old.toArray();
        float[] refV = ref.toArray();
        double klSum = 0;
        for (int i = 0; i < tokens; i++) {
            klSum += oldV[i] - refV[i];
        }
        List<int[]> prefixes = entropyBonus > 0 ? prefixes(ps, cs) : null;
        double entropy = 0;
        double clipped = 0;
        for (int e = 0; e < epochs; e++) {
            Var logp = policy.logProbs(ps, cs);
            Var ratio = logp.sub(old).exp();
            Var surrogate = ratio.mul(advT).minimum(ratio.clamp(1 - clip, 1 + clip).mul(advT));
            Var loss = surrogate.mean().neg();
            if (kl > 0) {
                Var d = Var.of(ref).sub(logp);
                loss = loss.add(d.exp().sub(d).sub(1).mean().mul(kl));
            }
            if (entropyBonus > 0) {
                Var logits = policy.logits(prefixes);
                Var logSoft = logits.sub(logits.logsumexp(1, true));
                Var h = logSoft.exp().mul(logSoft).sum(1, false).neg().mean();
                if (e == 0) {
                    entropy = h.value().item();
                }
                loss = loss.sub(h.mul(entropyBonus));
            }
            loss.backward();
            optimizer.step(policy.parameters());
            if (e == epochs - 1) {
                int c = 0;
                for (float x : ratio.value().toArray()) {
                    c += x < 1 - clip || x > 1 + clip ? 1 : 0;
                }
                clipped = (double) c / tokens;
            }
        }
        return new Stats(klSum / tokens, clipped, entropy);
    }
}
