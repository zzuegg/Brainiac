package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * REINFORCE (Williams, 1992), the basic policy gradient: play whole episodes, then make
 * every action more likely in proportion to the return that followed it. A learned
 * {@link #baseline(Model) baseline} (a value network) subtracts the expected return,
 * which lowers the variance a lot.
 *
 * <pre>{@code
 * Reinforce agent = Reinforce.of(Nets.mlp(4, 2, 1, 32), env.actionSpace())
 *         .baseline(Nets.mlp(4, 1, 2, 32))
 *         .optimizer(Optimizer.adam(0.01f))
 *         .entropyBonus(0.01f);
 * agent.fit(env, 100_000, RlListener.every(50));
 * }</pre>
 *
 * <p>Works with discrete actions (the policy outputs one logit per action) and
 * continuous ones (the policy outputs the mean of a Gaussian; its spread is learned).
 */
public final class Reinforce implements Learner {

    private final Model policy;
    private final PolicyHead head;
    private Model baseline;
    private Optimizer optimizer = Optimizer.adam(0.01f);
    private Optimizer baselineOptimizer = Optimizer.adam(0.01f);
    private float gamma = 0.99f;
    private float entropy;
    private int episodesPerUpdate = 1;
    private boolean normalize = true;
    private float maxGradNorm;
    private SplittableRandom random = new SplittableRandom(0);

    private Reinforce(Model policy, Space actions, float initialLogStd) {
        this.policy = Objects.requireNonNull(policy, "policy");
        this.head = new PolicyHead(actions, initialLogStd, policy.device());
        int expected = actions instanceof Space.Discrete d ? d.n() : actions.size();
        if (policy.outputSize() != expected) {
            throw new IllegalArgumentException("the policy must output " + expected + " values for " + actions
                    + ", it outputs " + policy.outputSize());
        }
    }

    /**
     * @param policy a network from observations to action logits (discrete) or means
     *     (continuous)
     * @param actions the action space
     * @return the learner
     */
    public static Reinforce of(Model policy, Space actions) {
        return new Reinforce(policy, actions, -0.5f);
    }

    /**
     * @param value a network from observations to one value, used as a baseline
     * @return this learner
     */
    public Reinforce baseline(Model value) {
        if (value.outputSize() != 1) {
            throw new IllegalArgumentException("the baseline must output one value");
        }
        this.baseline = value;
        return this;
    }

    /**
     * @param o the policy's optimizer (default Adam 0.01)
     * @return this learner
     */
    public Reinforce optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param o the baseline's optimizer (default Adam 0.01)
     * @return this learner
     */
    public Reinforce baselineOptimizer(Optimizer o) {
        this.baselineOptimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param discount how much future rewards count (gamma)
     * @return this learner
     */
    public Reinforce discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /**
     * @param coefficient weight of an entropy bonus that keeps the policy exploring
     * @return this learner
     */
    public Reinforce entropyBonus(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /**
     * @param n episodes collected for every update
     * @return this learner
     */
    public Reinforce episodesPerUpdate(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("episodesPerUpdate must be at least 1");
        }
        this.episodesPerUpdate = n;
        return this;
    }

    /**
     * @param on whether to scale advantages to mean 0 and standard deviation 1 per update
     * @return this learner
     */
    public Reinforce normalizeAdvantages(boolean on) {
        this.normalize = on;
        return this;
    }

    /**
     * @param norm rescale gradients longer than this; 0 for none
     * @return this learner
     */
    public Reinforce clipGradientNorm(float norm) {
        this.maxGradNorm = norm;
        return this;
    }

    /**
     * @param seed the seed for sampling actions and episodes
     * @return this learner
     */
    public Reinforce seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the policy network */
    public Model policy() {
        return policy;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return head.toEnvironment(head.sample(Nets.row(policy, observation), random, explore));
    }

    private List<Var> policyParameters() {
        List<Var> p = new ArrayList<>(policy.parameters());
        if (head.logStd != null) {
            p.add(head.logStd);
        }
        return p;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        Tracker tracker = new Tracker(listener);
        List<float[]> obs = new ArrayList<>();
        List<float[]> acts = new ArrayList<>();
        List<Float> returns = new ArrayList<>();
        List<Float> rewards = new ArrayList<>();
        int episodes = 0;
        int start = 0;
        float[] o = env.reset(random.split());
        while (tracker.steps() < steps) {
            float[] a = head.sample(Nets.row(policy, o), random, true);
            Environment.Step s = env.step(head.toEnvironment(a));
            obs.add(o);
            acts.add(a);
            rewards.add(s.reward());
            if (tracker.step(s)) {
                float g = s.truncated() && baseline != null ? Nets.row(baseline, s.observation())[0] : 0;
                float[] r = new float[rewards.size()];
                for (int t = rewards.size() - 1; t >= 0; t--) {
                    g = rewards.get(t) + gamma * g;
                    r[t] = g;
                }
                for (float v : r) {
                    returns.add(v);
                }
                rewards.clear();
                start = obs.size();
                if (++episodes % episodesPerUpdate == 0) {
                    update(obs, acts, returns);
                    obs.clear();
                    acts.clear();
                    returns.clear();
                    start = 0;
                }
                o = env.reset(random.split());
            } else {
                o = s.observation();
            }
        }
        if (start > 0) {
            update(obs.subList(0, start), acts.subList(0, start), returns);
        }
        return tracker.result;
    }

    private void update(List<float[]> obs, List<float[]> acts, List<Float> returns) {
        Device device = policy.device();
        int n = obs.size();
        Tensor x = Nets.rows(obs.toArray(float[][]::new), obs.get(0).length, device);
        Tensor a = Nets.rows(acts.toArray(float[][]::new), head.actionSize(), device);
        float[] g = new float[n];
        for (int i = 0; i < n; i++) {
            g[i] = returns.get(i);
        }
        Tensor target = Nets.column(g, device);
        float[] adv = g.clone();
        Var valueLoss = null;
        if (baseline != null) {
            Var v = baseline.forward(Var.of(x), true);
            float[] vs = v.value().toArray();
            for (int i = 0; i < n; i++) {
                adv[i] -= vs[i];
            }
            valueLoss = v.sub(target).square().mean();
        }
        if (normalize && n > 1) {
            double mean = 0;
            for (float v : adv) {
                mean += v;
            }
            mean /= n;
            double var = 0;
            for (float v : adv) {
                var += (v - mean) * (v - mean);
            }
            double std = Math.sqrt(var / n) + 1e-8;
            for (int i = 0; i < n; i++) {
                adv[i] = (float) ((adv[i] - mean) / std);
            }
        }
        Var out = policy.forward(Var.of(x), true);
        Var loss = head.logProb(out, a).mul(Nets.column(adv, device)).mean().neg();
        if (entropy != 0) {
            loss = loss.sub(head.entropy(out).mean().mul(entropy));
        }
        if (valueLoss != null) {
            loss = loss.add(valueLoss);
        }
        loss.backward();
        List<Var> params = policyParameters();
        Nets.clip(params, maxGradNorm);
        optimizer.step(params);
        if (baseline != null) {
            Nets.clip(baseline.parameters(), maxGradNorm);
            baselineOptimizer.step(baseline.parameters());
        }
    }

    @Override
    public String toString() {
        return "Reinforce(" + (baseline != null ? "with baseline" : "no baseline") + ")";
    }
}
