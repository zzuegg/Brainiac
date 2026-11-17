package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.NoisyDense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.PrioritizedReplayBuffer;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Rainbow DQN (Hessel et al., 2017, <a href="https://arxiv.org/abs/1710.02298">arXiv:1710.02298</a>):
 * the DQN improvements combined, each switchable. On top of what the core
 * {@link io.github.zzuegg.brainiac.rl.Dqn} already has (double Q, dueling, n-step returns,
 * prioritized replay) it learns the <b>distribution</b> of returns instead of its mean:
 * <ul>
 *   <li>{@link Distribution#CATEGORICAL C51} (Bellemare et al., 2017,
 *       <a href="https://arxiv.org/abs/1707.06887">arXiv:1707.06887</a>): a softmax over fixed
 *       return atoms, trained with cross-entropy against the projected Bellman target;</li>
 *   <li>{@link Distribution#QUANTILE QR-DQN} (Dabney et al., 2017,
 *       <a href="https://arxiv.org/abs/1710.10044">arXiv:1710.10044</a>): learned quantile
 *       locations, trained with the quantile Huber loss;</li>
 *   <li>{@link Distribution#EXPECTED} for a plain mean-value DQN.</li>
 * </ul>
 * and explores with <b>noisy nets</b> whenever the network contains {@link NoisyDense}
 * layers (epsilon then defaults to 0 and the noise is resampled every step).
 *
 * <p>Network outputs: {@code actions * atoms} values (atoms = 1 for EXPECTED), plus
 * {@code atoms} more for a dueling value stream placed first.
 *
 * <pre>{@code
 * RainbowDqn dqn = RainbowDqn.of(() -> net, 2).categorical(51, -10, 10).dueling(true).nSteps(3);
 * }</pre>
 */
public final class RainbowDqn implements Learner {

    /** What the network predicts about the return of each action. */
    public enum Distribution {
        /** its mean, as in DQN */
        EXPECTED,
        /** a categorical distribution over fixed atoms (C51) */
        CATEGORICAL,
        /** equally weighted quantiles (QR-DQN) */
        QUANTILE
    }

    private final Supplier<Model> factory;
    private final int actions;
    private final Model online;
    private final Model target;
    private Distribution distribution = Distribution.EXPECTED;
    private int atoms = 1;
    private float vMin = -10;
    private float vMax = 10;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private int bufferSize = 50_000;
    private int batchSize = 32;
    private int learningStarts = 200;
    private int targetEvery = 100;
    private float gamma = 0.99f;
    private int nSteps = 1;
    private boolean doubleQ = true;
    private boolean dueling;
    private float maxGradNorm = 10;
    private Schedule epsilon;
    private double alpha;
    private Schedule beta = Schedule.constant(0.4);
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long steps;
    private long updates;
    private double lastLoss = Double.NaN;

    private RainbowDqn(Supplier<Model> factory, int actions) {
        this.factory = factory;
        this.actions = actions;
        this.online = factory.get();
        this.target = factory.get();
        RlTensors.copy(online.parameters(), target.parameters());
    }

    /**
     * @param network makes the network (called twice: online and target)
     * @param actions the number of discrete actions
     * @return the learner, a plain double DQN until configured further
     */
    public static RainbowDqn of(Supplier<Model> network, int actions) {
        Objects.requireNonNull(network, "network");
        if (actions < 1) {
            throw new IllegalArgumentException("actions must be at least 1");
        }
        return new RainbowDqn(network, actions);
    }

    /**
     * Learns a categorical return distribution (C51).
     *
     * @param atoms the number of atoms, 51 in the paper
     * @param vMin the smallest representable return
     * @param vMax the largest
     * @return this learner
     */
    public RainbowDqn categorical(int atoms, float vMin, float vMax) {
        this.distribution = Distribution.CATEGORICAL;
        this.atoms = atoms;
        this.vMin = vMin;
        this.vMax = vMax;
        return this;
    }

    /**
     * Learns return quantiles (QR-DQN).
     *
     * @param quantiles the number of quantiles
     * @return this learner
     */
    public RainbowDqn quantile(int quantiles) {
        this.distribution = Distribution.QUANTILE;
        this.atoms = quantiles;
        return this;
    }

    /** @param on whether the network has a dueling value stream @return this learner */
    public RainbowDqn dueling(boolean on) {
        this.dueling = on;
        return this;
    }

    /** @param on whether the online network picks the target action @return this learner */
    public RainbowDqn doubleQ(boolean on) {
        this.doubleQ = on;
        return this;
    }

    /** @param n rewards summed before bootstrapping @return this learner */
    public RainbowDqn nSteps(int n) {
        this.nSteps = Math.max(1, n);
        return this;
    }

    /**
     * @param alpha priority exponent, usually 0.5 to 0.6; 0 for uniform replay
     * @param beta importance-weight correction by step
     * @return this learner
     */
    public RainbowDqn prioritized(double alpha, Schedule beta) {
        this.alpha = alpha;
        this.beta = Objects.requireNonNull(beta);
        return this;
    }

    /** @param o the optimizer @return this learner */
    public RainbowDqn optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param s the epsilon-greedy schedule (default: 0 with noisy layers, else 1 to 0.05) @return this learner */
    public RainbowDqn epsilon(Schedule s) {
        this.epsilon = Objects.requireNonNull(s);
        return this;
    }

    /** @param size transitions kept @return this learner */
    public RainbowDqn bufferSize(int size) {
        this.bufferSize = size;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public RainbowDqn batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param steps steps before learning @return this learner */
    public RainbowDqn learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param updates gradient steps between target copies @return this learner */
    public RainbowDqn targetUpdateEvery(int updates) {
        this.targetEvery = Math.max(1, updates);
        return this;
    }

    /** @param discount gamma @return this learner */
    public RainbowDqn discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the seed @return this learner */
    public RainbowDqn seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the online network */
    public Model network() {
        return online;
    }

    /** @return the loss of the last update */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return the gradient steps taken */
    public long updates() {
        return updates;
    }

    private List<NoisyDense> noisyLayers(Model m) {
        List<NoisyDense> out = new ArrayList<>();
        for (Layer l : m.layers()) {
            if (l instanceof NoisyDense n) {
                out.add(n);
            }
        }
        return out;
    }

    private Schedule epsilon() {
        if (epsilon == null) {
            epsilon = noisyLayers(online).isEmpty() ? Schedule.linear(1, 0.05, 2_000) : Schedule.constant(0);
        }
        return epsilon;
    }

    private int expectedOutputs() {
        return actions * atoms + (dueling ? atoms : 0);
    }

    /** The per-action output, {@code [batch, actions, atoms]}: logits, quantiles or values. */
    private Var head(Model m, Tensor x, boolean training) {
        Var out = m.forward(Var.of(x), training);
        int b = x.shape().dim(0);
        if (!dueling) {
            return out.reshape(b, actions, atoms);
        }
        Var v = out.slice(1, 0, atoms).reshape(b, 1, atoms);
        Var a = out.slice(1, atoms, atoms + actions * atoms).reshape(b, actions, atoms);
        return a.sub(a.mean(1, true)).add(v);
    }

    private Tensor support(Device device) {
        float[] z = new float[atoms];
        for (int i = 0; i < atoms; i++) {
            z[i] = atoms == 1 ? 0 : vMin + i * (vMax - vMin) / (atoms - 1);
        }
        return Tensor.of(Shape.of(1, 1, atoms), z).to(device);
    }

    /** Mean return of every action, {@code [batch, actions]}. */
    private Tensor means(Var head) {
        Tensor h = head.value();
        int b = h.shape().dim(0);
        return switch (distribution) {
            case EXPECTED -> h.reshape(b, actions);
            case QUANTILE -> h.mean(2, false);
            case CATEGORICAL -> RlTensors.softmax(Var.of(h)).value().mul(support(h.device())).sum(2, false);
        };
    }

    /**
     * @param observation an observation
     * @return the estimated mean return of every action (noise-free)
     */
    public float[] actionValues(float[] observation) {
        Tensor x = Tensor.of(Shape.of(1, observation.length), observation).to(online.device());
        return means(head(online, x, false)).to(DType.FLOAT32).toArray();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (!explore) {
            return new float[] {RlTensors.argmax(actionValues(observation))};
        }
        if (random.nextDouble() < epsilon().at(steps)) {
            return new float[] {random.nextInt(actions)};
        }
        Tensor x = Tensor.of(Shape.of(1, observation.length), observation).to(online.device());
        return new float[] {RlTensors.argmax(means(head(online, x, true)).to(DType.FLOAT32).toArray())};
    }

    @Override
    public RlResult fit(Environment env, long totalSteps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions, got " + env.actionSpace());
        }
        if (online.outputSize() != expectedOutputs()) {
            throw new IllegalArgumentException("the network must output " + expectedOutputs() + " values, it outputs "
                    + online.outputSize());
        }
        int obsSize = env.observationSpace().size();
        if (buffer == null) {
            buffer = alpha > 0 ? new PrioritizedReplayBuffer(bufferSize, obsSize, 1, alpha)
                    : new ReplayBuffer(bufferSize, obsSize, 1);
        }
        List<NoisyDense> noisy = noisyLayers(online);
        Tracker recorder = new Tracker(listener);
        Deque<Transition> window = new ArrayDeque<>();
        float[] obs = env.reset(random.split());
        while (recorder.steps() < totalSteps) {
            float[] a = act(obs, true);
            Environment.Step s = env.step(a);
            steps++;
            window.addLast(new Transition(obs, a, s.reward(), s.observation(), s.terminated(), gamma));
            if (window.size() == nSteps) {
                buffer.add(fold(window));
                window.removeFirst();
            }
            if (s.done()) {
                while (!window.isEmpty()) {
                    buffer.add(fold(window));
                    window.removeFirst();
                }
            }
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (steps >= learningStarts && buffer.size() >= batchSize) {
                learn();
                for (NoisyDense n : noisy) {
                    n.resample();
                }
            }
        }
        return recorder.result();
    }

    private Transition fold(Deque<Transition> window) {
        Transition first = window.peekFirst();
        float reward = 0;
        float discount = 1;
        Transition last = first;
        for (Transition t : window) {
            reward += discount * t.reward();
            discount *= gamma;
            last = t;
            if (t.terminated()) {
                break;
            }
        }
        return new Transition(first.observation(), first.action(), reward, last.nextObservation(),
                last.terminated(), discount);
    }

    private void learn() {
        Device device = online.device();
        if (buffer instanceof PrioritizedReplayBuffer p) {
            p.beta(beta.at(steps));
        }
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, device);
        int n = batchSize;
        Tensor nextTarget = head(target, b.nextObservations(), true).value();
        float[] pick = (doubleQ ? means(head(online, b.nextObservations(), true)) : means(Var.of(nextTarget)))
                .argmax(1).to(DType.FLOAT32).toArray();
        Tensor pickHot = RlTensors.oneHot(pick, actions, device).reshape(n, actions, 1);
        Tensor nextChosen = nextTarget.mul(pickHot).sum(1, false); // [n, atoms]
        Tensor actHot = RlTensors.oneHot(b.actions().toArray(), actions, device).reshape(n, actions, 1);
        Var chosen = head(online, b.observations(), true).mul(actHot).sum(1, false); // [n, atoms]
        float[] r = b.rewards().toArray();
        float[] g = b.discounts().mul(b.notDone()).toArray();
        Var perSample;
        switch (distribution) {
            case EXPECTED -> {
                Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(nextChosen));
                perSample = RlTensors.huber(chosen.sub(y), 1f);
            }
            case CATEGORICAL -> {
                float[] p = RlTensors.softmax(Var.of(nextChosen)).value().to(DType.FLOAT32).toArray();
                Tensor m = Tensor.of(Shape.of(n, atoms), project(p, r, g, n)).to(device);
                perSample = RlTensors.logSoftmax(chosen).mul(m).sum(1, true).neg();
            }
            case QUANTILE -> {
                Tensor y = b.rewards().add(Tensor.of(Shape.of(n, 1), g).to(device).mul(nextChosen));
                perSample = quantileHuber(chosen, y, device);
            }
            default -> throw new IllegalStateException();
        }
        Var loss = perSample.mul(b.weights()).mean();
        loss.backward();
        RlTensors.clip(online.parameters(), maxGradNorm);
        optimizer.step(online.parameters());
        updates++;
        lastLoss = loss.value().item();
        if (buffer instanceof PrioritizedReplayBuffer pr) {
            float[] e = perSample.value().reshape(n, perSample.value().shape().dim(1)).sum(1, false)
                    .to(DType.FLOAT32).toArray();
            pr.updatePriorities(b.indices(), e);
        }
        if (updates % targetEvery == 0) {
            RlTensors.copy(online.parameters(), target.parameters());
        }
    }

    /** Projects {@code r + g z} onto the fixed support (the C51 projection). */
    private float[] project(float[] p, float[] r, float[] g, int n) {
        float[] m = new float[n * atoms];
        float dz = (vMax - vMin) / (atoms - 1);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < atoms; j++) {
                float z = vMin + j * dz;
                float tz = Math.max(vMin, Math.min(vMax, r[i] + g[i] * z));
                float pos = (tz - vMin) / dz;
                int lo = (int) Math.floor(pos);
                int hi = (int) Math.ceil(pos);
                float mass = p[i * atoms + j];
                if (lo == hi) {
                    m[i * atoms + lo] += mass;
                } else {
                    m[i * atoms + lo] += mass * (hi - pos);
                    m[i * atoms + hi] += mass * (pos - lo);
                }
            }
        }
        return m;
    }

    /** The quantile Huber loss per sample, {@code [n, 1]}. */
    private Var quantileHuber(Var theta, Tensor y, Device device) {
        int n = theta.shape().dim(0);
        float[] tau = new float[atoms];
        for (int i = 0; i < atoms; i++) {
            tau[i] = (2 * i + 1) / (2f * atoms);
        }
        Tensor taus = Tensor.of(Shape.of(1, atoms, 1), tau).to(device);
        Var u = Var.of(y.reshape(n, 1, atoms)).sub(theta.reshape(n, atoms, 1)); // [n, i, j]
        Tensor below = u.value().greater(Tensor.full(Shape.of(1, 1, 1), 0f)).neg().add(1f); // 1 where u <= 0
        Tensor weight = taus.sub(below).abs();
        return RlTensors.huber(u, 1f).mul(weight).sum(1, false).mean(1, true);
    }

    /** @return a fresh network from the same factory */
    public Model newNetwork() {
        return factory.get();
    }

    @Override
    public String toString() {
        String d = switch (distribution) {
            case EXPECTED -> "";
            case CATEGORICAL -> "C51 ";
            case QUANTILE -> "QR-";
        };
        return (doubleQ ? "Double " : "") + (dueling ? "Dueling " : "") + d + "DQN";
    }
}
