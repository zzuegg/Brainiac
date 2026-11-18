package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.Objects;

/**
 * PRIME's implicit process reward model (Cui et al. 2025,
 * <a href="https://arxiv.org/abs/2502.01456">arXiv:2502.01456</a>): dense token-level rewards
 * learned from outcome labels alone. The implicit PRM is itself a language model
 * {@code pi_phi} next to a frozen reference; the reward of token {@code t} is
 * {@code r_t = beta log(pi_phi(y_t | y_<t) / ref(y_t | y_<t))}, and the model is trained only
 * with a cross-entropy loss that makes {@code sigmoid(sum_t r_t)} predict whether a rollout's
 * final answer is right. Because it needs no step labels it is updated online on the policy's
 * own rollouts, so it keeps up with the policy instead of being hacked by it as a frozen PRM is.
 *
 * <p>{@link #advantages} combines both signals as PRIME does, each with a leave-one-out
 * baseline over the group of rollouts of one prompt: the token rewards minus the mean token
 * reward of the other rollouts, summed into returns-to-go, plus the outcome reward minus the
 * other rollouts' mean outcome on every token. A PPO-clip update follows
 * ({@link TokenPolicyGradient}).
 */
public final class ImplicitProcessReward {

    private final WindowLm prm;
    private final WindowLm reference;
    private final float beta;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int epochs = 1;

    private ImplicitProcessReward(WindowLm prm, WindowLm reference, float beta) {
        this.prm = Objects.requireNonNull(prm);
        this.reference = Objects.requireNonNull(reference);
        this.beta = beta;
    }

    /**
     * @param prm the implicit PRM, usually initialized as a copy of the policy (trained)
     * @param reference the frozen reference, usually the initial policy
     * @param beta the reward scale
     * @return the implicit PRM
     */
    public static ImplicitProcessReward of(WindowLm prm, WindowLm reference, float beta) {
        return new ImplicitProcessReward(prm, reference, beta);
    }

    /**
     * @param o the PRM's optimizer
     * @return this PRM
     */
    public ImplicitProcessReward optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param n cross-entropy steps per {@link #update} on the same rollouts (1 by default)
     * @return this PRM
     */
    public ImplicitProcessReward epochs(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("epochs must be positive");
        }
        this.epochs = n;
        return this;
    }

    /**
     * @param prompts the prompts
     * @param completions the completions
     * @return {@code beta log(pi_phi / ref)} per token, in the order of {@link WindowLm#logProbs}
     */
    public float[] tokenRewards(List<int[]> prompts, List<int[]> completions) {
        float[] a = Var.noGrad(() -> prm.logProbs(prompts, completions)).value().toArray();
        float[] b = Var.noGrad(() -> reference.logProbs(prompts, completions)).value().toArray();
        for (int i = 0; i < a.length; i++) {
            a[i] = beta * (a[i] - b[i]);
        }
        return a;
    }

    /**
     * One online cross-entropy step on outcome labels.
     *
     * @param prompts the prompts
     * @param completions the rollouts
     * @param correct whether each rollout's answer is right
     * @return the loss before the step
     */
    public float update(List<int[]> prompts, List<int[]> completions, boolean[] correct) {
        float first = step(prompts, completions, correct);
        for (int e = 1; e < epochs; e++) {
            step(prompts, completions, correct);
        }
        return first;
    }

    private float step(List<int[]> prompts, List<int[]> completions, boolean[] correct) {
        Var seq = Var.of(WindowLm.sequenceSum(completions, false)).matmul(prm.logProbs(prompts, completions));
        Tensor ref = WindowLm.sequenceSum(completions, false).matmul(reference.logProbs(prompts, completions).value());
        Var z = seq.sub(ref).mul(beta);
        float[] sign = new float[correct.length];
        for (int i = 0; i < sign.length; i++) {
            sign[i] = correct[i] ? 1 : -1;
        }
        Var loss = PreferenceLosses.logSigmoid(z.mul(Tensor.of(Shape.of(sign.length, 1), sign))).mean().neg();
        float v = loss.value().item();
        loss.backward();
        optimizer.step(prm.parameters());
        return v;
    }

    /**
     * PRIME's per-token advantages for one group of rollouts of the same prompt.
     *
     * @param completions the rollouts
     * @param tokenRewards their implicit token rewards, flattened in order
     * @param outcomes their outcome rewards
     * @param processWeight the weight of the process term (0 gives outcome-only RLOO)
     * @return one advantage per token, flattened in order
     */
    public static float[] advantages(List<int[]> completions, float[] tokenRewards, float[] outcomes,
            float processWeight) {
        int n = completions.size();
        double[] meanToken = new double[n];
        int k = 0;
        for (int i = 0; i < n; i++) {
            int len = completions.get(i).length;
            for (int t = 0; t < len; t++) {
                meanToken[i] += tokenRewards[k + t] / Math.max(1, len);
            }
            k += len;
        }
        double sumMean = 0;
        double sumOutcome = 0;
        for (int i = 0; i < n; i++) {
            sumMean += meanToken[i];
            sumOutcome += outcomes[i];
        }
        float[] out = new float[tokenRewards.length];
        k = 0;
        for (int i = 0; i < n; i++) {
            int len = completions.get(i).length;
            double base = n > 1 ? (sumMean - meanToken[i]) / (n - 1) : 0;
            double outcomeAdv = n > 1 ? outcomes[i] - (sumOutcome - outcomes[i]) / (n - 1) : 0;
            double ret = 0;
            for (int t = len - 1; t >= 0; t--) {
                ret += tokenRewards[k + t] - base;
                out[k + t] = (float) (outcomeAdv + processWeight * ret);
            }
            k += len;
        }
        return out;
    }
}
