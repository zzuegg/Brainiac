package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Conservative Q-learning for discrete actions (Kumar et al., 2020,
 * <a href="https://arxiv.org/abs/2006.04779">arXiv:2006.04779</a>), the CQL(H) variant on top of a
 * double-DQN loss.
 *
 * <p>Plain Q-learning on a fixed dataset overestimates actions the data never tried, and
 * the greedy policy then chases those phantom values. CQL adds
 * {@code alpha * (logsumexp_a Q(s, a) - Q(s, a_data))}, which pushes all Q-values down and
 * the logged ones back up, so the learned values are a lower bound for unseen actions and
 * the policy stays within the support of the data while still improving on it.
 */
public final class Cql implements Agent {

    private final Model online;
    private final Model target;
    private final int actions;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private float alpha = 1f;
    private float gamma = 0.99f;
    private int batchSize = 64;
    private int targetEvery = 100;
    private SplittableRandom random = new SplittableRandom(0);
    private double lastLoss = Double.NaN;

    private Cql(Supplier<Model> network, int actions) {
        this.online = network.get();
        this.target = network.get();
        this.actions = actions;
        RlTensors.copy(online.parameters(), target.parameters());
    }

    /**
     * @param network makes the Q-network (observations to one value per action), called twice
     * @param actions the number of discrete actions
     * @return the learner
     */
    public static Cql of(Supplier<Model> network, int actions) {
        return new Cql(Objects.requireNonNull(network), actions);
    }

    /** @param alpha the conservative penalty weight; 0 gives plain offline DQN @return this learner */
    public Cql alpha(float alpha) {
        this.alpha = alpha;
        return this;
    }

    /** @param discount gamma @return this learner */
    public Cql discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param o the optimizer @return this learner */
    public Cql optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param seed the seed @return this learner */
    public Cql seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    /**
     * @param observation an observation
     * @return the learned value of every action
     */
    public float[] actionValues(float[] observation) {
        return RlTensors.row(online, observation);
    }

    /**
     * Trains on logged data.
     *
     * @param data the dataset
     * @param steps gradient steps
     * @return this learner
     */
    public Cql train(OfflineDataset data, int steps) {
        ReplayBuffer buffer = data.toReplayBuffer(gamma);
        Device device = online.device();
        for (int s = 0; s < steps; s++) {
            ReplayBuffer.Batch b = buffer.sample(batchSize, random, device);
            float[] pick = online.forward(Var.of(b.nextObservations()), false).value().argmax(1)
                    .to(DType.FLOAT32).toArray();
            Tensor next = target.forward(Var.of(b.nextObservations()), false).value()
                    .mul(RlTensors.oneHot(pick, actions, device)).sum(1, true);
            Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(next));
            Var q = online.forward(Var.of(b.observations()), true);
            Var chosen = q.mul(RlTensors.oneHot(b.actions().toArray(), actions, device)).sum(1, true);
            Var loss = RlTensors.huber(chosen.sub(y), 1f).mean();
            if (alpha > 0) {
                Var logSumExp = q.sub(q.value().max(1, true)).exp().sum(1, true).log().add(q.value().max(1, true));
                loss = loss.add(logSumExp.sub(chosen).mean().mul(alpha));
            }
            loss.backward();
            RlTensors.clip(online.parameters(), 10f);
            optimizer.step(online.parameters());
            lastLoss = loss.value().item();
            if ((s + 1) % targetEvery == 0) {
                RlTensors.copy(online.parameters(), target.parameters());
            }
        }
        return this;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return new float[] {RlTensors.argmax(actionValues(observation))};
    }
}
