package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * The action distribution of a policy network, for on-policy learners outside the core RL
 * package: a categorical over discrete actions (the network outputs logits), or a diagonal
 * Gaussian over continuous ones (the network outputs means; a learned, state-independent
 * log standard deviation gives the spread). Besides sampling, log-probabilities and entropy
 * it gives the KL divergence from a frozen copy of the outputs, which trust-region methods
 * ({@link Trpo}, {@link ConstrainedPolicyOptimization}) constrain.
 *
 * <p>It mirrors the core's package-private {@code PolicyHead}, plus the KL that the core
 * does not need.
 */
public final class PolicyDistribution {

    private static final float HALF_LOG_2PI = (float) (0.5 * Math.log(2 * Math.PI));

    private final boolean discrete;
    private final int size;
    private final Space.Box box;
    private final Var logStd;

    /**
     * Creates the distribution for an action space.
     *
     * @param actions the action space
     * @param initialLogStd the starting log standard deviation of continuous actions
     */
    public PolicyDistribution(Space actions, float initialLogStd) {
        if (actions instanceof Space.Discrete d) {
            discrete = true;
            size = d.n();
            box = null;
            logStd = null;
        } else {
            box = (Space.Box) actions;
            discrete = false;
            size = box.size();
            float[] s = new float[size];
            Arrays.fill(s, initialLogStd);
            logStd = Var.param(Tensor.of(Shape.of(1, size), s));
        }
    }

    /** @return whether actions are discrete */
    public boolean discrete() {
        return discrete;
    }

    /** @return the network outputs one row needs: action count, or action dimensions */
    public int outputs() {
        return size;
    }

    /** @return values per stored action: 1 for discrete, the vector length for continuous */
    public int actionSize() {
        return discrete ? 1 : size;
    }

    /** @return the learned log standard deviation, or {@code null} for discrete actions */
    public Var logStd() {
        return logStd;
    }

    /**
     * Samples (or picks greedily) from one output row; continuous samples are not clipped.
     *
     * @param out the network's output row
     * @param random the random source
     * @param explore sample (true) or take the mode (false)
     * @return the action as stored
     */
    public float[] sample(float[] out, SplittableRandom random, boolean explore) {
        if (discrete) {
            return new float[] {explore ? AgentKit.sample(out, random) : AgentKit.argmax(out)};
        }
        float[] a = new float[size];
        float[] s = logStd.value().toArray();
        for (int i = 0; i < size; i++) {
            a[i] = explore ? (float) (out[i] + Math.exp(s[i]) * random.nextGaussian()) : out[i];
        }
        return a;
    }

    /**
     * @param action an action as stored
     * @return what the environment receives: continuous actions clipped into the box
     */
    public float[] toEnvironment(float[] action) {
        return discrete ? action : box.clip(action);
    }

    /**
     * @param out the network outputs, {@code [n, outputs]}
     * @param actions the actions, {@code [n, actionSize]}
     * @return their log-probabilities, {@code [n, 1]}
     */
    public Var logProb(Var out, Tensor actions) {
        if (discrete) {
            Var logp = RlTensors.logSoftmax(out);
            return logp.mul(RlTensors.oneHot(actions.toArray(), size, out.value().device())).sum(1, true);
        }
        Var z = out.neg().add(actions).mul(logStd.neg().exp());
        return z.square().mul(-0.5f).sub(logStd).sub(HALF_LOG_2PI).sum(1, true);
    }

    /**
     * @param out the network outputs
     * @return the entropy per row, {@code [n, 1]} (or {@code [1, 1]} for a Gaussian)
     */
    public Var entropy(Var out) {
        if (discrete) {
            Var logp = RlTensors.logSoftmax(out);
            return logp.exp().mul(logp).sum(1, true).neg();
        }
        return logStd.add(HALF_LOG_2PI + 0.5f).sum(1, true);
    }

    /**
     * The mean KL divergence {@code KL(old || new)} over a batch, differentiable in the new
     * outputs (and the log standard deviation).
     *
     * @param oldOut the frozen outputs of the old policy
     * @param oldLogStd the frozen log standard deviation of the old policy ({@code null} for
     *     discrete actions)
     * @param out the new outputs
     * @return the mean KL, a scalar
     */
    public Var kl(Tensor oldOut, Tensor oldLogStd, Var out) {
        if (discrete) {
            Var oldLogp = RlTensors.logSoftmax(Var.of(oldOut));
            Var newLogp = RlTensors.logSoftmax(out);
            return oldLogp.exp().mul(oldLogp.sub(newLogp)).sum(1, true).mean();
        }
        // sum_i log(s1/s0) + (s0^2 + (m0 - m1)^2) / (2 s1^2) - 1/2
        Var var1 = logStd.mul(2f).exp();
        Var diff = out.neg().add(oldOut);
        Var term = diff.square().add(oldLogStd.mul(2f).exp()).div(var1).mul(0.5f);
        return term.add(logStd).sub(oldLogStd).sub(0.5f).sum(1, true).mean();
    }

    @Override
    public String toString() {
        return discrete ? "Categorical(" + size + ")" : "Gaussian(" + size + ")";
    }
}
