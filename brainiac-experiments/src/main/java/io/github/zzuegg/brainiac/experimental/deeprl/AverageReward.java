package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Average-reward RL for continuing tasks: instead of discounting, learn differential values
 * relative to an estimated reward rate {@code rbar}, so the optimal policy maximises the
 * long-run reward per step and does not depend on a discount factor.
 *
 * <ul>
 *   <li>{@link Method#DIFFERENTIAL_Q} (Wan, Naik &amp; Sutton, ICML 2021,
 *       <a href="https://arxiv.org/abs/2006.16318">arXiv:2006.16318</a>):
 *       {@code delta = r - rbar + max_a' Q(s', a') - Q(s, a)},
 *       {@code Q(s, a) += alpha delta}, {@code rbar += eta alpha delta}, convergent in the
 *       tabular off-policy case;</li>
 *   <li>{@link Method#R_LEARNING} (Schwartz, ICML 1993): the same Q update, but {@code rbar}
 *       moves by {@code beta (r - rbar + max Q(s') - max Q(s))} and only after greedy
 *       actions;</li>
 *   <li>{@link #deep}: differential Q-learning with a neural network and replay (DQN-style, a
 *       target network, {@code rbar} updated from the batch's mean TD error).</li>
 * </ul>
 *
 * <p>{@link #differentialTd} estimates the reward rate and differential values of a fixed
 * policy (prediction). Terminations are treated as transitions to the reset state, so every
 * task is continuing.
 */
public final class AverageReward implements Learner {

    /** The tabular control rule. */
    public enum Method {
        /** differential Q-learning */
        DIFFERENTIAL_Q,
        /** R-learning */
        R_LEARNING
    }

    private record Key(float[] v) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && Arrays.equals(v, k.v);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(v);
        }
    }

    private record Sample(float[] obs, int action, float reward, float[] next) {
    }

    private final Method method;
    private final int actions;
    private final Map<Key, double[]> table = new HashMap<>();
    private final Model net;
    private final Model target;
    private Optimizer optimizer;
    private double alpha = 0.1;
    private double eta = 0.1;
    private double epsilon = 0.1;
    private double rbar;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> buffer = new ArrayList<>();
    private int position;
    private long updates;

    private AverageReward(Method method, int actions, Model net, Model target) {
        this.method = method;
        this.actions = actions;
        this.net = net;
        this.target = target;
        if (net != null) {
            RlTensors.copy(net.parameters(), target.parameters());
            optimizer = Optimizer.adam(1e-3f);
        }
    }

    /**
     * @param method the rule
     * @param actions discrete actions
     * @return a tabular learner
     */
    public static AverageReward tabular(Method method, int actions) {
        return new AverageReward(Objects.requireNonNull(method), actions, null, null);
    }

    /**
     * @param observations values per observation
     * @param actions discrete actions
     * @param networks builds the Q-network
     * @param seed the initialisation seed
     * @return a DQN-style differential Q-learner
     */
    public static AverageReward deep(int observations, int actions, RlNetworkFactory networks, long seed) {
        return new AverageReward(Method.DIFFERENTIAL_Q, actions, networks.create(observations, actions, seed),
                networks.create(observations, actions, seed));
    }

    /**
     * @param stepSize the value step size {@code alpha}
     * @param rateStepSize the reward-rate step size {@code eta} (relative to alpha for differential Q)
     * @return this learner
     */
    public AverageReward stepSizes(double stepSize, double rateStepSize) {
        this.alpha = stepSize;
        this.eta = rateStepSize;
        return this;
    }

    /** @param e the exploration rate @return this learner */
    public AverageReward epsilon(double e) {
        this.epsilon = e;
        return this;
    }

    /** @param seed the seed @return this learner */
    public AverageReward seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the current reward-rate estimate */
    public double rewardRate() {
        return rbar;
    }

    /** @return gradient updates (deep variant) */
    public long updates() {
        return updates;
    }

    private double[] q(float[] obs) {
        if (net != null) {
            float[] v = RlTensors.row(net, obs);
            double[] d = new double[v.length];
            for (int i = 0; i < v.length; i++) {
                d[i] = v[i];
            }
            return d;
        }
        return table.computeIfAbsent(new Key(obs.clone()), k -> new double[actions]);
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b]) {
                b = i;
            }
        }
        return b;
    }

    private static double max(double[] v) {
        return Arrays.stream(v).max().orElse(0);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {argmax(q(observation))};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        Tracker t = new Tracker(listener);
        float[] obs = env.reset(random.split());
        while (t.steps() < steps) {
            int a = Math.round(act(obs, true)[0]);
            Environment.Step s = env.step(a);
            float[] next = s.observation();
            boolean reset = t.step(s);
            if (reset) {
                next = env.reset(random.split());
            }
            if (net == null) {
                double[] qs = q(obs);
                double greedyBefore = max(qs);
                boolean greedy = qs[a] == greedyBefore;
                double[] qn = q(next);
                double delta = s.reward() - rbar + max(qn) - qs[a];
                qs[a] += alpha * delta;
                if (method == Method.DIFFERENTIAL_Q) {
                    rbar += eta * alpha * delta;
                } else if (greedy) {
                    rbar += eta * (s.reward() - rbar + max(qn) - greedyBefore);
                }
            } else {
                remember(new Sample(obs, a, s.reward(), next));
                if (buffer.size() >= 64) {
                    learn();
                }
            }
            obs = next;
        }
        return t.result();
    }

    private void remember(Sample s) {
        if (buffer.size() < 20_000) {
            buffer.add(s);
        } else {
            buffer.set(position, s);
            position = (position + 1) % 20_000;
        }
    }

    private void learn() {
        int b = 64;
        int d = buffer.get(0).obs().length;
        float[] o = new float[b * d];
        float[] n = new float[b * d];
        float[] r = new float[b];
        float[] acts = new float[b];
        for (int i = 0; i < b; i++) {
            Sample s = buffer.get(random.nextInt(buffer.size()));
            System.arraycopy(s.obs(), 0, o, i * d, d);
            System.arraycopy(s.next(), 0, n, i * d, d);
            r[i] = s.reward();
            acts[i] = s.action();
        }
        Tensor nextMax = target.forward(Var.of(Tensor.of(Shape.of(b, d), n)), false).value().max(1, true);
        Tensor y = Tensor.of(Shape.of(b, 1), r).sub((float) rbar).add(nextMax);
        Var chosen = net.forward(Var.of(Tensor.of(Shape.of(b, d), o)), true)
                .mul(RlTensors.oneHot(acts, actions, net.device())).sum(1, true);
        Var err = chosen.sub(y);
        double meanDelta = -err.value().mean().item();
        Var loss = RlTensors.huber(err, 1f).mean();
        loss.backward();
        RlTensors.clip(net.parameters(), 10f);
        optimizer.step(net.parameters());
        rbar += eta * 1e-2 * meanDelta;
        if (++updates % 100 == 0) {
            RlTensors.copy(net.parameters(), target.parameters());
        }
    }

    /**
     * Differential TD(0) prediction for a fixed policy.
     *
     * @param env the task
     * @param policy the policy evaluated
     * @param steps steps
     * @param stepSize alpha
     * @param rateStepSize eta
     * @param seed the seed
     * @return the estimated reward rate of the policy
     */
    public static double differentialTd(Environment env, Agent policy, long steps, double stepSize,
            double rateStepSize, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<Key, double[]> v = new HashMap<>();
        double rate = 0;
        float[] obs = env.reset(r.split());
        for (long t = 0; t < steps; t++) {
            Environment.Step s = env.step(policy.act(obs, false));
            float[] next = s.done() ? env.reset(r.split()) : s.observation();
            double[] vs = v.computeIfAbsent(new Key(obs.clone()), k -> new double[1]);
            double[] vn = v.computeIfAbsent(new Key(next.clone()), k -> new double[1]);
            double delta = s.reward() - rate + vn[0] - vs[0];
            vs[0] += stepSize * delta;
            rate += rateStepSize * stepSize * delta;
            obs = next;
        }
        return rate;
    }

    @Override
    public String toString() {
        return net == null ? method.toString() : "DeepDifferentialQ";
    }
}
