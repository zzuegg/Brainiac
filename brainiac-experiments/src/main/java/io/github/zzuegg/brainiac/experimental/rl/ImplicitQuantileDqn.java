package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Implicit quantile networks, IQN (Dabney et al., 2018,
 * <a href="https://arxiv.org/abs/1806.06923">arXiv:1806.06923</a>): instead of a fixed set of
 * atoms (C51) or quantiles (QR-DQN, both in {@link RainbowDqn}), the network takes a
 * quantile level {@code tau} as input, embedded as {@code relu(sum_i cos(pi i tau) w_i + b)}
 * and multiplied into the state features, and outputs the return quantile
 * {@code Z_tau(s, a)} for every action. Training samples {@code N} levels for the online
 * and {@code N'} for the target network and minimizes the quantile Huber loss over all
 * pairs, so the whole return distribution is learned with a fixed network size.
 *
 * <p>Because any quantile can be queried, acting can apply a distortion of the levels:
 * {@link #cvar(float) CVaR_alpha} samples {@code tau ~ U(0, alpha)} and so values each action
 * by the mean of its worst {@code alpha} outcomes, giving a risk-averse policy without
 * retraining.
 *
 * <p>It is a separate learner rather than a new {@code RainbowDqn.Distribution}, because the
 * network needs the extra {@code tau} input and cannot be a plain {@code Supplier<Model>}.
 */
public final class ImplicitQuantileDqn implements Learner {

    private final int obsSize;
    private final int actions;
    private final int features;
    private final int cosines;
    private final Net online;
    private final Net target;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private int quantiles = 8;
    private int targetQuantiles = 8;
    private int actingSamples = 32;
    private float kappa = 1f;
    private float alpha = 1f;
    private int bufferSize = 50_000;
    private int batchSize = 32;
    private int learningStarts = 200;
    private int targetEvery = 50;
    private int trainEvery = 1;
    private float gamma = 0.99f;
    private Schedule epsilon = Schedule.linear(1, 0.05, 2000);
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long steps;
    private long updates;
    private double lastLoss = Double.NaN;

    /** The three parts of the network: state features, tau embedding, head. */
    private final class Net {
        final Model psi;
        final Var phiW;
        final Var phiB;
        final Model head;

        Net(int hidden, long seed) {
            psi = Model.sequential().input(obsSize).dense(features, Activation.RELU).seed(seed).build();
            SplittableRandom r = new SplittableRandom(seed + 1);
            float[] w = new float[cosines * features];
            double limit = Math.sqrt(6.0 / (cosines + features));
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) r.nextDouble(-limit, limit);
            }
            phiW = Var.param(Tensor.of(Shape.of(cosines, features), w));
            phiB = Var.param(Tensor.zeros(Shape.of(1, features)));
            head = Model.sequential().input(features).dense(hidden, Activation.RELU).dense(actions).seed(seed + 2)
                    .build();
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(psi.parameters());
            p.add(phiW);
            p.add(phiB);
            p.addAll(head.parameters());
            return p;
        }

        /** Quantile values {@code [rows * n, actions]} for {@code n} levels per row. */
        Var forward(Tensor obs, float[] taus, int n, boolean grad) {
            int rows = obs.shape().dim(0);
            Var f = psi.forward(Var.of(obs), grad);
            Var tiled = f.reshape(rows, 1, features).expand(Shape.of(rows, n, features)).reshape(rows * n, features);
            float[] c = new float[rows * n * cosines];
            for (int k = 0; k < rows * n; k++) {
                for (int i = 0; i < cosines; i++) {
                    c[k * cosines + i] = (float) Math.cos(Math.PI * i * taus[k]);
                }
            }
            Var emb = Var.of(Tensor.of(Shape.of(rows * n, cosines), c)).matmul(phiW).add(phiB).relu();
            return head.forward(tiled.mul(emb), grad);
        }
    }

    private ImplicitQuantileDqn(int obsSize, int actions, int hidden, long seed) {
        this.obsSize = obsSize;
        this.actions = actions;
        this.features = hidden;
        this.cosines = 16;
        this.online = new Net(hidden, seed);
        this.target = new Net(hidden, seed);
        RlTensors.copy(online.parameters(), target.parameters());
    }

    /**
     * @param observations values per observation
     * @param actions the number of discrete actions
     * @param hidden the width of the features and the head
     * @param seed the seed
     * @return the learner: 8 + 8 sampled quantiles, risk-neutral acting
     */
    public static ImplicitQuantileDqn of(int observations, int actions, int hidden, long seed) {
        return new ImplicitQuantileDqn(observations, actions, hidden, seed);
    }

    /**
     * Acts on {@code CVaR_alpha}: the mean of the worst {@code alpha} share of returns.
     *
     * @param alpha in {@code (0, 1]}; 1 is risk-neutral
     * @return this learner
     */
    public ImplicitQuantileDqn cvar(float alpha) {
        if (!(alpha > 0 && alpha <= 1)) {
            throw new IllegalArgumentException("alpha must be in (0, 1]");
        }
        this.alpha = alpha;
        return this;
    }

    /**
     * @param n quantile levels sampled for the online network
     * @param nTarget levels sampled for the target
     * @param acting levels averaged when acting
     * @return this learner
     */
    public ImplicitQuantileDqn quantiles(int n, int nTarget, int acting) {
        this.quantiles = n;
        this.targetQuantiles = nTarget;
        this.actingSamples = acting;
        return this;
    }

    /** @param o the optimizer @return this learner */
    public ImplicitQuantileDqn optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param s the exploration schedule @return this learner */
    public ImplicitQuantileDqn epsilon(Schedule s) {
        this.epsilon = Objects.requireNonNull(s);
        return this;
    }

    /** @param discount the discount @return this learner */
    public ImplicitQuantileDqn discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param steps steps before learning @return this learner */
    public ImplicitQuantileDqn learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param steps environment steps between updates @return this learner */
    public ImplicitQuantileDqn trainEvery(int steps) {
        this.trainEvery = Math.max(1, steps);
        return this;
    }

    /** @param updates updates between target copies @return this learner */
    public ImplicitQuantileDqn targetUpdateEvery(int updates) {
        this.targetEvery = updates;
        return this;
    }

    /** @param seed the seed @return this learner */
    public ImplicitQuantileDqn seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the loss of the last update */
    public double lastLoss() {
        return lastLoss;
    }

    /**
     * Estimates the return quantiles of every action.
     *
     * @param observation the observation
     * @param taus the quantile levels
     * @return {@code [taus.length][actions]} quantile values
     */
    public float[][] quantileValues(float[] observation, float[] taus) {
        float[] q = AgentKit.values(online.forward(AgentKit.row(observation), taus, taus.length, false));
        float[][] out = new float[taus.length][actions];
        for (int i = 0; i < taus.length; i++) {
            System.arraycopy(q, i * actions, out[i], 0, actions);
        }
        return out;
    }

    /**
     * @param observation the observation
     * @return each action's value under the acting distortion (mean, or CVaR)
     */
    public float[] actionValues(float[] observation) {
        float[] taus = new float[actingSamples];
        for (int i = 0; i < taus.length; i++) {
            taus[i] = alpha * (i + 0.5f) / taus.length;
        }
        float[][] z = quantileValues(observation, taus);
        float[] v = new float[actions];
        for (float[] row : z) {
            for (int a = 0; a < actions; a++) {
                v[a] += row[a] / taus.length;
            }
        }
        return v;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon.at(steps)) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {AgentKit.argmax(actionValues(observation))};
    }

    @Override
    public RlResult fit(Environment env, long totalSteps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        if (buffer == null) {
            buffer = new ReplayBuffer(bufferSize, obsSize, 1);
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        while (tracker.steps() < totalSteps) {
            float[] a = act(obs, true);
            Environment.Step s = env.step(a);
            steps++;
            buffer.add(new Transition(obs, a, s.reward(), s.observation(), s.terminated(), gamma));
            obs = tracker.step(s) ? env.reset(random.split()) : s.observation();
            if (steps >= learningStarts && steps % trainEvery == 0 && buffer.size() >= batchSize) {
                learn();
            }
        }
        return tracker.result();
    }

    private float[] uniform(int n) {
        float[] t = new float[n];
        for (int i = 0; i < n; i++) {
            t[i] = (float) random.nextDouble();
        }
        return t;
    }

    private void learn() {
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, online.psi.device());
        int bs = batchSize;
        int n = quantiles;
        int m = targetQuantiles;
        // greedy next action from the online network's mean (double DQN)
        float[] nextTaus = uniform(bs * m);
        float[] zNextOnline = AgentKit.values(online.forward(b.nextObservations(), nextTaus, m, false));
        float[] zNextTarget = AgentKit.values(target.forward(b.nextObservations(), uniform(bs * m), m, false));
        float[] rewards = b.rewards().toArray();
        float[] live = b.notDone().mul(b.discounts()).toArray();
        float[] y = new float[bs * m];
        for (int i = 0; i < bs; i++) {
            double[] mean = new double[actions];
            for (int j = 0; j < m; j++) {
                for (int a = 0; a < actions; a++) {
                    mean[a] += zNextOnline[(i * m + j) * actions + a];
                }
            }
            int best = RlTensors.argmax(mean);
            for (int j = 0; j < m; j++) {
                y[i * m + j] = rewards[i] + live[i] * zNextTarget[(i * m + j) * actions + best];
            }
        }
        float[] taus = uniform(bs * n);
        Var z = online.forward(b.observations(), taus, n, true);
        float[] acts = b.actions().toArray();
        float[] repeated = new float[bs * n];
        for (int i = 0; i < bs * n; i++) {
            repeated[i] = acts[i / n];
        }
        Var za = z.mul(RlTensors.oneHot(repeated, actions, z.value().device())).sum(1, true).reshape(bs, n, 1);
        Shape pairs = Shape.of(bs, n, m);
        Var u = Var.of(Tensor.of(Shape.of(bs, 1, m), y).expand(pairs)).sub(za.expand(pairs));
        float[] uv = u.value().toArray();
        float[] weight = new float[uv.length];
        for (int i = 0; i < bs; i++) {
            for (int k = 0; k < n; k++) {
                for (int j = 0; j < m; j++) {
                    int idx = (i * n + k) * m + j;
                    weight[idx] = Math.abs(taus[i * n + k] - (uv[idx] < 0 ? 1 : 0)) / kappa;
                }
            }
        }
        Var loss = RlTensors.huber(u, kappa).mul(Tensor.of(pairs, weight)).mean(2, false).sum(1, false).mean();
        lastLoss = loss.value().item();
        loss.backward();
        List<Var> params = online.parameters();
        RlTensors.clip(params, 10f);
        optimizer.step(params);
        updates++;
        if (updates % targetEvery == 0) {
            RlTensors.copy(online.parameters(), target.parameters());
        }
    }

    @Override
    public String toString() {
        return alpha < 1 ? "IQN(CVaR " + alpha + ")" : "IQN";
    }
}
