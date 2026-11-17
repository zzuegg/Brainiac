package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Implicit Q-learning for discrete actions (Kostrikov et al., 2021,
 * <a href="https://arxiv.org/abs/2110.06169">arXiv:2110.06169</a>): offline RL that never queries
 * the value of an action outside the dataset.
 *
 * <p>A state-value network {@code V} is fit by <b>expectile regression</b> to the target
 * critic's {@code Q(s, a_data)}: with expectile {@code tau > 0.5} it approximates the
 * maximum over in-data actions, an implicit, in-sample {@code max}. {@code Q} learns from
 * {@code r + gamma V(s')}. The policy is then extracted by <b>advantage-weighted
 * regression</b>: behaviour cloning with weights {@code exp(beta (Q - V))}, so it imitates
 * the better logged actions.
 */
public final class Iql implements Agent {

    private final Model q;
    private final Model qTarget;
    private final Model v;
    private final Model policy;
    private final int actions;
    private Optimizer qOpt = Optimizer.adam(1e-3f);
    private Optimizer vOpt = Optimizer.adam(1e-3f);
    private Optimizer piOpt = Optimizer.adam(1e-3f);
    private float expectile = 0.8f;
    private float beta = 3f;
    private float gamma = 0.99f;
    private float tau = 0.01f;
    private int batchSize = 64;
    private SplittableRandom random = new SplittableRandom(0);

    private Iql(int obs, int actions, RlNetworkFactory nets) {
        this.actions = actions;
        q = nets.create(obs, actions, 41);
        qTarget = nets.create(obs, actions, 41);
        v = nets.create(obs, 1, 42);
        policy = nets.create(obs, actions, 43);
        RlTensors.copy(q.parameters(), qTarget.parameters());
    }

    /**
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param networks builds Q ({@code obs -> actions}), V ({@code obs -> 1}) and the policy
     * @return the learner
     */
    public static Iql of(int observationDims, int actions, RlNetworkFactory networks) {
        return new Iql(observationDims, actions, Objects.requireNonNull(networks));
    }

    /** @param tau the expectile, in {@code [0.5, 1)} @return this learner */
    public Iql expectile(float tau) {
        this.expectile = tau;
        return this;
    }

    /** @param beta the inverse temperature of advantage weighting @return this learner */
    public Iql beta(float beta) {
        this.beta = beta;
        return this;
    }

    /** @param discount gamma @return this learner */
    public Iql discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Iql seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /**
     * @param observation an observation
     * @return the learned value of every action
     */
    public float[] actionValues(float[] observation) {
        return RlTensors.row(q, observation);
    }

    /**
     * Trains value functions and policy on logged data.
     *
     * @param data the dataset
     * @param steps gradient steps
     * @return this learner
     */
    public Iql train(OfflineDataset data, int steps) {
        ReplayBuffer buffer = data.toReplayBuffer(gamma);
        Device device = q.device();
        for (int s = 0; s < steps; s++) {
            ReplayBuffer.Batch b = buffer.sample(batchSize, random, device);
            Tensor hot = RlTensors.oneHot(b.actions().toArray(), actions, device);
            Tensor qData = qTarget.forward(Var.of(b.observations()), false).value().mul(hot).sum(1, true);

            Var vs = v.forward(Var.of(b.observations()), true);
            Var diff = Var.of(qData).sub(vs);
            Tensor weight = diff.value().greater(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(1, 1), 0f))
                    .mul(2 * expectile - 1).add(1 - expectile);
            Var vLoss = diff.square().mul(weight).mean();
            vLoss.backward();
            vOpt.step(v.parameters());

            Tensor nextV = v.forward(Var.of(b.nextObservations()), false).value();
            Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(nextV));
            Var qa = q.forward(Var.of(b.observations()), true).mul(hot).sum(1, true);
            Var qLoss = qa.sub(y).square().mean();
            qLoss.backward();
            optimizer(qOpt, q);

            Tensor adv = qData.sub(vs.value());
            Tensor w = adv.mul(beta).exp().minimum(Tensor.full(io.github.zzuegg.brainiac.tensor.Shape.of(1, 1), 100f));
            Var logp = RlTensors.logSoftmax(policy.forward(Var.of(b.observations()), true)).mul(hot).sum(1, true);
            Var piLoss = logp.mul(w).neg().mean();
            piLoss.backward();
            piOpt.step(policy.parameters());
            RlTensors.blend(q.parameters(), qTarget.parameters(), tau);
        }
        return this;
    }

    private static void optimizer(Optimizer o, Model m) {
        RlTensors.clip(m.parameters(), 10f);
        o.step(m.parameters());
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return new float[] {RlTensors.argmax(RlTensors.row(policy, observation))};
    }
}
