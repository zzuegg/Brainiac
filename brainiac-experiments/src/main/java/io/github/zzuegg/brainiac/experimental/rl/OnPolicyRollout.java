package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * One on-policy rollout of a single environment, with generalized advantage estimation
 * (GAE, Schulman et al., <a href="https://arxiv.org/abs/1506.02438">arXiv:1506.02438</a>) for the
 * reward and, for a {@link CostedEnvironment}, for the cost too. Shared by the trust-region
 * and constrained policy learners; truncated episodes bootstrap from the value of their last
 * observation, as the core PPO does.
 */
public final class OnPolicyRollout {

    private final int n;
    private final int obsSize;
    private final int actSize;
    private final float[] observations;
    private final float[] actions;
    private final float[] logProbs;
    private final float[] rewards;
    private final float[] costs;
    private final float[] values;
    private final float[] costValues;
    private final float[] dones;
    private final List<Double> episodeCosts = new ArrayList<>();
    private float[] advantages;
    private float[] returns;
    private float[] costAdvantages;
    private float[] costReturns;
    private float[] last;
    private float[] lastValues;
    private double runningCost;

    private OnPolicyRollout(int n, int obsSize, int actSize) {
        this.n = n;
        this.obsSize = obsSize;
        this.actSize = actSize;
        observations = new float[n * obsSize];
        actions = new float[n * actSize];
        logProbs = new float[n];
        rewards = new float[n];
        costs = new float[n];
        values = new float[n];
        costValues = new float[n];
        dones = new float[n];
    }

    /**
     * Collects a rollout.
     *
     * @param env the environment (costs are read when it is a {@link CostedEnvironment})
     * @param start the current observation; collection continues the running episode
     * @param policy observations to distribution parameters
     * @param dist the action distribution
     * @param value observations to the value of the reward
     * @param costValue observations to the value of the cost, or {@code null}
     * @param steps how many steps
     * @param gamma the discount, for bootstrapping truncated episodes
     * @param tracker records the steps and episodes
     * @param random the random source
     * @param carriedCost the cost of the running episode so far
     * @return the rollout; {@link #nextObservation} continues it
     */
    public static OnPolicyRollout collect(Environment env, float[] start, Model policy, PolicyDistribution dist,
            Model value, Model costValue, int steps, float gamma, Tracker tracker, SplittableRandom random,
            double carriedCost) {
        int obsSize = start.length;
        OnPolicyRollout r = new OnPolicyRollout(steps, obsSize, dist.actionSize());
        r.runningCost = carriedCost;
        CostedEnvironment costed = env instanceof CostedEnvironment c ? c : null;
        float[] obs = start;
        for (int t = 0; t < steps; t++) {
            Tensor x = AgentKit.row(obs);
            Var out = policy.forward(Var.of(x), false);
            float[] a = dist.sample(AgentKit.values(out), random, true);
            float lp = dist.logProb(out, Tensor.of(Shape.of(1, a.length), a)).value().item();
            System.arraycopy(obs, 0, r.observations, t * obsSize, obsSize);
            System.arraycopy(a, 0, r.actions, t * r.actSize, r.actSize);
            r.logProbs[t] = lp;
            r.values[t] = AgentKit.predict(value, obs)[0];
            if (costValue != null) {
                r.costValues[t] = AgentKit.predict(costValue, obs)[0];
            }
            Environment.Step s = env.step(dist.toEnvironment(a));
            float rew = s.reward();
            float cost = costed == null ? 0 : costed.lastCost();
            r.runningCost += cost;
            if (s.truncated() && !s.terminated()) {
                rew += gamma * AgentKit.predict(value, s.observation())[0];
                if (costValue != null) {
                    cost += gamma * AgentKit.predict(costValue, s.observation())[0];
                }
            }
            r.rewards[t] = rew;
            r.costs[t] = cost;
            r.dones[t] = s.done() ? 1 : 0;
            if (tracker.step(s)) {
                r.episodeCosts.add(r.runningCost);
                r.runningCost = 0;
                obs = env.reset(random.split());
            } else {
                obs = s.observation();
            }
        }
        r.last = obs;
        float lastValue = AgentKit.predict(value, obs)[0];
        float lastCost = costValue == null ? 0 : AgentKit.predict(costValue, obs)[0];
        r.advantages = new float[steps];
        r.returns = new float[steps];
        r.costAdvantages = new float[steps];
        r.costReturns = new float[steps];
        r.lastValues = new float[] {lastValue, lastCost};
        return r;
    }

    /**
     * Computes GAE advantages and returns for reward and cost.
     *
     * @param gamma the discount
     * @param lambda the GAE trade-off
     * @return this rollout
     */
    public OnPolicyRollout gae(float gamma, float lambda) {
        gae(rewards, values, lastValues[0], advantages, returns, gamma, lambda);
        gae(costs, costValues, lastValues[1], costAdvantages, costReturns, gamma, lambda);
        return this;
    }

    private void gae(float[] r, float[] v, float lastV, float[] adv, float[] ret, float gamma, float lambda) {
        float g = 0;
        for (int t = n - 1; t >= 0; t--) {
            float next = t == n - 1 ? lastV : v[t + 1];
            float live = 1 - dones[t];
            float delta = r[t] + gamma * next * live - v[t];
            g = delta + gamma * lambda * live * g;
            adv[t] = g;
            ret[t] = g + v[t];
        }
    }

    /**
     * Scales values to mean 0 and standard deviation 1.
     *
     * @param x the values
     * @return a normalized copy
     */
    public static float[] normalize(float[] x) {
        double mean = 0;
        for (float v : x) {
            mean += v;
        }
        mean /= x.length;
        double var = 0;
        for (float v : x) {
            var += (v - mean) * (v - mean);
        }
        double std = Math.sqrt(var / x.length) + 1e-8;
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) ((x[i] - mean) / std);
        }
        return out;
    }

    /** @return the number of steps */
    public int size() {
        return n;
    }

    /** @return the observations, {@code [n, obs]} */
    public Tensor observations() {
        return Tensor.of(Shape.of(n, obsSize), observations);
    }

    /** @return the actions, {@code [n, actionSize]} */
    public Tensor actions() {
        return Tensor.of(Shape.of(n, actSize), actions);
    }

    /** @return the behaviour log-probabilities, {@code [n, 1]} */
    public Tensor logProbs() {
        return AgentKit.column(logProbs);
    }

    /** @return the reward advantages (after {@link #gae}) */
    public float[] advantages() {
        return advantages;
    }

    /** @return the reward returns (after {@link #gae}) */
    public float[] returns() {
        return returns;
    }

    /** @return the cost advantages (after {@link #gae}) */
    public float[] costAdvantages() {
        return costAdvantages;
    }

    /** @return the cost returns (after {@link #gae}) */
    public float[] costReturns() {
        return costReturns;
    }

    /** @return the observation that continues the rollout */
    public float[] nextObservation() {
        return last;
    }

    /** @return the total cost of every episode that finished during the rollout */
    public List<Double> episodeCosts() {
        return List.copyOf(episodeCosts);
    }

    /** @return the cost of the episode still running at the end */
    public double carriedCost() {
        return runningCost;
    }

    /**
     * Fits a value network to targets with a few epochs of full-batch Adam steps.
     *
     * @param value the network
     * @param optimizer its optimizer
     * @param observations the inputs
     * @param targets the targets
     * @param epochs the number of steps
     * @return the last loss
     */
    public static double fitValue(Model value, io.github.zzuegg.brainiac.train.Optimizer optimizer,
            Tensor observations, float[] targets, int epochs) {
        Tensor y = AgentKit.column(targets);
        double loss = Double.NaN;
        for (int e = 0; e < epochs; e++) {
            Var l = value.forward(Var.of(observations), true).sub(y).square().mean();
            loss = l.value().item();
            l.backward();
            optimizer.step(value.parameters());
        }
        return loss;
    }
}
