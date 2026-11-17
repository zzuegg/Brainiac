package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Behaviour cloning: supervised learning of the logged actions (Pomerleau, 1989), the
 * simplest offline RL and imitation baseline. Discrete actions are learned with
 * cross-entropy, continuous ones with squared error. Combined with
 * {@link OfflineDataset#best(double)} it becomes filtered ("%BC") cloning, a surprisingly
 * strong baseline (Chen et al., 2021, <a href="https://arxiv.org/abs/2106.01345">arXiv:2106.01345</a>).
 */
public final class BehaviorCloning implements Agent {

    private final Model policy;
    private final int actions;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int batchSize = 64;
    private SplittableRandom random = new SplittableRandom(0);
    private double lastLoss = Double.NaN;

    private BehaviorCloning(Model policy, int actions) {
        this.policy = policy;
        this.actions = actions;
    }

    /**
     * @param policy observations to one logit per action
     * @param actions the number of discrete actions
     * @return the learner
     */
    public static BehaviorCloning discrete(Model policy, int actions) {
        return new BehaviorCloning(Objects.requireNonNull(policy), actions);
    }

    /**
     * @param policy observations to the action vector
     * @return the learner, for continuous actions
     */
    public static BehaviorCloning continuous(Model policy) {
        return new BehaviorCloning(Objects.requireNonNull(policy), 0);
    }

    /** @param o the optimizer @return this learner */
    public BehaviorCloning optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param seed the seed @return this learner */
    public BehaviorCloning seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    /**
     * Trains on logged data.
     *
     * @param data the dataset
     * @param steps gradient steps
     * @return this learner
     */
    public BehaviorCloning train(OfflineDataset data, int steps) {
        ReplayBuffer buffer = data.toReplayBuffer(1f);
        for (int s = 0; s < steps; s++) {
            ReplayBuffer.Batch b = buffer.sample(batchSize, random, policy.device());
            Var out = policy.forward(Var.of(b.observations()), true);
            Var loss;
            if (actions > 0) {
                Tensor target = RlTensors.oneHot(b.actions().toArray(), actions, policy.device());
                loss = RlTensors.logSoftmax(out).mul(target).sum(1, true).neg().mean();
            } else {
                loss = out.sub(b.actions()).square().mean();
            }
            loss.backward();
            optimizer.step(policy.parameters());
            lastLoss = loss.value().item();
        }
        return this;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] out = RlTensors.row(policy, observation);
        if (actions == 0) {
            return out;
        }
        if (explore) {
            float[] p = RlTensors.softmax(Var.of(Tensor.of(out))).value().toArray();
            double u = random.nextDouble();
            for (int a = 0; a < p.length; a++) {
                u -= p[a];
                if (u <= 0) {
                    return new float[] {a};
                }
            }
        }
        return new float[] {RlTensors.argmax(out)};
    }
}
