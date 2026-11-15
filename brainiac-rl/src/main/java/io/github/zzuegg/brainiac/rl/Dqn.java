package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Deep Q-learning (Mnih et al., 2015) with the common improvements, each switchable: a
 * network estimates the value of every action, learned from a replay buffer against a
 * slowly updated target network.
 *
 * <ul>
 *   <li>{@link #doubleQ(boolean) double DQN}: the online network picks the next action,
 *       the target network values it (less overestimation; on by default);</li>
 *   <li>{@link #dueling(boolean) dueling}: the network outputs a state value and one
 *       advantage per action, {@code Q = V + A - mean(A)};</li>
 *   <li>{@link #prioritized(double, Schedule) prioritized replay};</li>
 *   <li>{@link #nSteps(int) n-step returns};</li>
 *   <li>hard {@link #targetUpdateEvery(int) target updates} or soft
 *       {@link #softTargetUpdate(float) Polyak averaging};</li>
 *   <li>Huber or squared loss, gradient clipping, epsilon-greedy exploration on a
 *       {@link Schedule}.</li>
 * </ul>
 *
 * <pre>{@code
 * Dqn dqn = Dqn.of(() -> Nets.mlp(4, 2, Activation.RELU, 1, 128, 128), 2)
 *         .bufferSize(50_000).batchSize(64).learningStarts(1_000)
 *         .epsilon(Schedule.linear(1, 0.05, 10_000))
 *         .targetUpdateEvery(500).doubleQ(true).nSteps(3);
 * dqn.fit(new CartPole(), 50_000, RlListener.every(20));
 * }</pre>
 */
public final class Dqn implements Learner {

    private final Supplier<Model> factory;
    private final int actions;
    private final Model online;
    private final Model target;
    private Optimizer optimizer = Optimizer.adam(5e-4f);
    private int bufferSize = 50_000;
    private int batchSize = 64;
    private int learningStarts = 1_000;
    private int trainEvery = 1;
    private int gradientSteps = 1;
    private int targetEvery = 500;
    private float tau;
    private float gamma = 0.99f;
    private int nSteps = 1;
    private boolean doubleQ = true;
    private boolean dueling;
    private float huber = 1;
    private float maxGradNorm = 10;
    private Schedule epsilon = Schedule.linear(1, 0.05, 10_000);
    private double alpha;
    private Schedule beta = Schedule.constant(0.4);
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long steps;
    private long updates;
    private double lastLoss = Double.NaN;

    private Dqn(Supplier<Model> factory, int actions) {
        this.factory = Objects.requireNonNull(factory, "factory");
        this.actions = actions;
        this.online = factory.get();
        this.target = factory.get();
        Nets.copy(online, target);
    }

    /**
     * @param network makes the Q-network (called twice: online and target), observations to
     *     one value per action, or one more for {@link #dueling(boolean) dueling}
     * @param actions the number of discrete actions
     * @return the learner
     */
    public static Dqn of(Supplier<Model> network, int actions) {
        if (actions < 1) {
            throw new IllegalArgumentException("actions must be at least 1");
        }
        return new Dqn(network, actions);
    }

    /** @param o the optimizer (default Adam 5e-4) @return this learner */
    public Dqn optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param size transitions kept for replay @return this learner */
    public Dqn bufferSize(int size) {
        this.bufferSize = size;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public Dqn batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param steps steps of acting before learning begins @return this learner */
    public Dqn learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param steps environment steps between updates @return this learner */
    public Dqn trainEvery(int steps) {
        this.trainEvery = Math.max(1, steps);
        return this;
    }

    /** @param n gradient steps per update @return this learner */
    public Dqn gradientSteps(int n) {
        this.gradientSteps = Math.max(1, n);
        return this;
    }

    /** @param updates gradient steps between copies into the target network @return this learner */
    public Dqn targetUpdateEvery(int updates) {
        this.targetEvery = Math.max(1, updates);
        this.tau = 0;
        return this;
    }

    /**
     * @param tau move the target network this fraction towards the online one after
     *     every gradient step instead of copying it now and then
     * @return this learner
     */
    public Dqn softTargetUpdate(float tau) {
        this.tau = tau;
        return this;
    }

    /** @param discount how much future rewards count (gamma) @return this learner */
    public Dqn discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param n rewards summed before bootstrapping @return this learner */
    public Dqn nSteps(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("nSteps must be at least 1");
        }
        this.nSteps = n;
        return this;
    }

    /** @param on whether to use double DQN targets @return this learner */
    public Dqn doubleQ(boolean on) {
        this.doubleQ = on;
        return this;
    }

    /**
     * @param on whether the network's output is a state value followed by one advantage
     *     per action (so {@code actions + 1} outputs)
     * @return this learner
     */
    public Dqn dueling(boolean on) {
        this.dueling = on;
        return this;
    }

    /**
     * @param delta the Huber loss threshold; 0 for squared error
     * @return this learner
     */
    public Dqn huberDelta(float delta) {
        this.huber = delta;
        return this;
    }

    /** @param norm rescale gradients longer than this; 0 for none @return this learner */
    public Dqn clipGradientNorm(float norm) {
        this.maxGradNorm = norm;
        return this;
    }

    /** @param schedule the chance of a random action, by step @return this learner */
    public Dqn epsilon(Schedule schedule) {
        this.epsilon = Objects.requireNonNull(schedule);
        return this;
    }

    /**
     * Replays surprising transitions more often.
     *
     * @param alpha how strongly priorities matter, usually 0.6
     * @param beta the importance-weight correction by step, usually rising from 0.4 to 1
     * @return this learner
     */
    public Dqn prioritized(double alpha, Schedule beta) {
        this.alpha = alpha;
        this.beta = Objects.requireNonNull(beta);
        return this;
    }

    /** @param seed the seed for exploration, replay and episodes @return this learner */
    public Dqn seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the online Q-network */
    public Model network() {
        return online;
    }

    /** @return the replay buffer, once training has started */
    public ReplayBuffer buffer() {
        return buffer;
    }

    /** @return the loss of the last gradient step */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return the gradient steps taken */
    public long updates() {
        return updates;
    }

    private Var q(Model m, Tensor x, boolean training) {
        Var out = m.forward(Var.of(x), training);
        if (!dueling) {
            return out;
        }
        Var v = out.slice(1, 0, 1);
        Var a = out.slice(1, 1, actions + 1);
        return a.sub(a.mean(1, true)).add(v);
    }

    /**
     * @param observation an observation
     * @return the estimated value of every action
     */
    public float[] actionValues(float[] observation) {
        Tensor x = Nets.rows(new float[][] {observation}, observation.length, online.device());
        return q(online, x, false).value().to(DType.FLOAT32).toArray();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon.at(steps)) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {Nets.argmax(actionValues(observation))};
    }

    @Override
    public RlResult fit(Environment env, long totalSteps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("DQN needs " + actions + " discrete actions, got " + env.actionSpace());
        }
        int expected = dueling ? actions + 1 : actions;
        if (online.outputSize() != expected) {
            throw new IllegalArgumentException("the Q-network must output " + expected + " values, it outputs "
                    + online.outputSize());
        }
        int obsSize = env.observationSpace().size();
        if (buffer == null) {
            buffer = alpha > 0 ? new PrioritizedReplayBuffer(bufferSize, obsSize, 1, alpha)
                    : new ReplayBuffer(bufferSize, obsSize, 1);
        }
        Tracker tracker = new Tracker(listener);
        Deque<Transition> window = new ArrayDeque<>();
        float[] obs = env.reset(random.split());
        while (tracker.steps() < totalSteps) {
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
            boolean ended = tracker.step(s);
            obs = ended ? env.reset(random.split()) : s.observation();
            if (steps >= learningStarts && steps % trainEvery == 0 && buffer.size() >= batchSize) {
                for (int g = 0; g < gradientSteps; g++) {
                    learn();
                }
            }
        }
        return tracker.result;
    }

    /** Sums the window's rewards into one n-step transition from its first state. */
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
        Var next = q(target, b.nextObservations(), false);
        Tensor nextValue;
        if (doubleQ) {
            float[] pick = q(online, b.nextObservations(), false).value().argmax(1).to(DType.FLOAT32).toArray();
            nextValue = next.value().mul(Nets.oneHot(pick, actions, device)).sum(1, true);
        } else {
            nextValue = next.value().max(1, true);
        }
        Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(nextValue));
        Var chosen = q(online, b.observations(), true)
                .mul(Nets.oneHot(b.actions().toArray(), actions, device)).sum(1, true);
        Var error = chosen.sub(y);
        Var perSample = huber > 0 ? Nets.huber(error, huber) : error.square().mul(0.5f);
        Var loss = perSample.mul(b.weights()).mean();
        loss.backward();
        Nets.clip(online.parameters(), maxGradNorm);
        optimizer.step(online.parameters());
        updates++;
        lastLoss = loss.value().item();
        if (buffer instanceof PrioritizedReplayBuffer p) {
            p.updatePriorities(b.indices(), error.value().to(DType.FLOAT32).toArray());
        }
        if (tau > 0) {
            Nets.blend(online, target, tau);
        } else if (updates % targetEvery == 0) {
            Nets.copy(online, target);
        }
    }

    /**
     * @return a fresh network from the same factory, for example to load saved weights
     *     into
     */
    public Model newNetwork() {
        return factory.get();
    }

    @Override
    public String toString() {
        return (doubleQ ? "Double " : "") + (dueling ? "Dueling " : "") + "DQN";
    }
}
