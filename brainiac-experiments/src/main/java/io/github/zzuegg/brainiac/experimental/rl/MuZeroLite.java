package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A small MuZero (Schrittwieser et al., 2019, <a href="https://arxiv.org/abs/1911.08265">arXiv:1911.08265</a>):
 * planning with a model that is learned, in a latent space, without ever being told the
 * rules.
 *
 * <p>Three networks: a <b>representation</b> {@code h(obs) -> s}, a <b>dynamics</b>
 * function {@code g(s, a) -> (s', r)} and a <b>prediction</b> function
 * {@code f(s) -> (policy, value)}. The agent acts by running {@link Mcts} over latent states
 * imagined by {@code g}, and learns by unrolling the model {@code K} steps along stored
 * trajectories, matching predicted rewards to observed rewards, predicted values to
 * n-step bootstrapped returns (using the search values stored with the trajectory), and
 * predicted policies to the search's visit distributions. The latent state need not
 * reconstruct the observation: it only has to support accurate value, reward and policy
 * predictions, which is the key idea.
 *
 * <pre>{@code
 * MuZeroLite mz = MuZeroLite.of(obsSize, actions, RlNetworkFactory.mlp(Activation.RELU, 64), 16)
 *         .simulations(16).unroll(3).tdSteps(5);
 * mz.fit(env, 2_000);
 * }</pre>
 */
public final class MuZeroLite implements Learner {

    private static final class Trajectory {
        final List<float[]> observations = new ArrayList<>();
        final List<Integer> actions = new ArrayList<>();
        final List<Float> rewards = new ArrayList<>();
        final List<float[]> policies = new ArrayList<>();
        final List<Float> rootValues = new ArrayList<>();
        boolean terminated;
    }

    private final int obsDims;
    private final int actions;
    private final int latent;
    private final Model representation;
    private final Model dynamics;
    private final Model prediction;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int simulations = 16;
    private int unroll = 3;
    private int tdSteps = 5;
    private float gamma = 0.97f;
    private int batchSize = 32;
    private int trainEvery = 4;
    private int window = 200;
    private int warmup = 200;
    private double temperature = 1.0;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Trajectory> replay = new ArrayList<>();
    private double lastLoss = Double.NaN;

    private MuZeroLite(int obsDims, int actions, RlNetworkFactory nets, int latent) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.latent = latent;
        representation = nets.create(obsDims, latent, 31);
        dynamics = nets.create(latent + actions, latent + 1, 32);
        prediction = nets.create(latent, actions + 1, 33);
    }

    /**
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param networks builds the representation, dynamics and prediction networks
     * @param latentSize the size of the latent state
     * @return the learner
     */
    public static MuZeroLite of(int observationDims, int actions, RlNetworkFactory networks, int latentSize) {
        return new MuZeroLite(observationDims, actions, Objects.requireNonNull(networks), latentSize);
    }

    /** @param n MCTS simulations per move @return this learner */
    public MuZeroLite simulations(int n) {
        this.simulations = n;
        return this;
    }

    /** @param k model unroll steps during training @return this learner */
    public MuZeroLite unroll(int k) {
        this.unroll = k;
        return this;
    }

    /** @param n steps of real reward before bootstrapping the value target @return this learner */
    public MuZeroLite tdSteps(int n) {
        this.tdSteps = n;
        return this;
    }

    /** @param discount gamma @return this learner */
    public MuZeroLite discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param steps environment steps between training batches @return this learner */
    public MuZeroLite trainEvery(int steps) {
        this.trainEvery = Math.max(1, steps);
        return this;
    }

    /**
     * @param steps environment steps of uniformly random acting before training begins,
     *     to seed the replay with diverse experience
     * @return this learner
     */
    public MuZeroLite warmup(int steps) {
        this.warmup = steps;
        return this;
    }

    /** @param o the optimizer @return this learner */
    public MuZeroLite optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param seed the seed @return this learner */
    public MuZeroLite seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the training loss of the last batch */
    public double lastLoss() {
        return lastLoss;
    }

    private Var represent(Var obs, boolean training) {
        return representation.forward(obs, training).tanh();
    }

    private Var[] dynamics(Var state, Tensor actionHot, boolean training) {
        Var out = dynamics.forward(Var.concat(1, state, Var.of(actionHot)), training);
        return new Var[] {out.slice(1, 0, latent).tanh(), out.slice(1, latent, latent + 1)};
    }

    private float[] run(Model m, float[] in) {
        return m.forward(Var.of(Tensor.of(Shape.of(1, in.length), in)), false).value().to(DType.FLOAT32).toArray();
    }

    private final Mcts.Model<float[]> latentModel = new Mcts.Model<>() {
        @Override
        public int actions() {
            return actions;
        }

        @Override
        public Mcts.Evaluation evaluate(float[] state) {
            float[] out = run(prediction, state);
            float max = Float.NEGATIVE_INFINITY;
            for (int a = 0; a < actions; a++) {
                max = Math.max(max, out[a]);
            }
            float[] p = new float[actions];
            float sum = 0;
            for (int a = 0; a < actions; a++) {
                p[a] = (float) Math.exp(out[a] - max);
                sum += p[a];
            }
            for (int a = 0; a < actions; a++) {
                p[a] /= sum;
            }
            return new Mcts.Evaluation(p, out[actions], false);
        }

        @Override
        public Mcts.Step<float[]> step(float[] state, int action) {
            float[] in = new float[latent + actions];
            System.arraycopy(state, 0, in, 0, latent);
            in[latent + action] = 1;
            float[] out = run(dynamics, in);
            float[] next = new float[latent];
            for (int i = 0; i < latent; i++) {
                next[i] = (float) Math.tanh(out[i]);
            }
            return new Mcts.Step<>(next, out[latent]);
        }
    };

    private float[] latentOf(float[] observation) {
        float[] s = run(representation, observation);
        for (int i = 0; i < s.length; i++) {
            s[i] = (float) Math.tanh(s[i]);
        }
        return s;
    }

    private Mcts.Result plan(float[] observation, boolean explore) {
        Mcts<float[]> m = Mcts.of(latentModel).simulations(simulations).discount(gamma);
        if (explore) {
            m.rootNoise(0.3f, 0.25f);
        }
        return m.search(latentOf(observation), random);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        Mcts.Result r = plan(observation, explore);
        return new float[] {explore ? r.sample(temperature, random) : r.best()};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("MuZeroLite needs " + obsDims + " observation values and "
                    + actions + " discrete actions");
        }
        EpisodeRecorder recorder = new EpisodeRecorder(listener);
        float[] obs = env.reset(random.split());
        Trajectory t = new Trajectory();
        while (recorder.steps() < steps) {
            Mcts.Result r = plan(obs, true);
            temperature = recorder.steps() < steps / 2 ? 1.0 : 0.5;
            int a = recorder.steps() < warmup ? random.nextInt(actions) : r.sample(temperature, random);
            Environment.Step s = env.step(a);
            t.observations.add(obs);
            t.actions.add(a);
            t.rewards.add(s.reward());
            t.policies.add(r.policy());
            t.rootValues.add(r.value());
            boolean ended = recorder.step(s);
            if (ended) {
                t.terminated = s.terminated();
                t.observations.add(s.observation());
                replay.add(t);
                if (replay.size() > window) {
                    replay.removeFirst();
                }
                t = new Trajectory();
                obs = env.reset(random.split());
            } else {
                obs = s.observation();
            }
            if (recorder.steps() >= warmup && !replay.isEmpty() && recorder.steps() % trainEvery == 0) {
                learn();
            }
        }
        return recorder.result();
    }

    private float valueTarget(Trajectory t, int index) {
        int len = t.actions.size();
        if (index >= len) {
            return 0;
        }
        int boot = index + tdSteps;
        float v = 0;
        if (boot < len) {
            v = t.rootValues.get(boot) * (float) Math.pow(gamma, tdSteps);
        } else if (!t.terminated) {
            v = t.rootValues.get(len - 1) * (float) Math.pow(gamma, len - 1 - index);
        }
        for (int i = index; i < Math.min(boot, len); i++) {
            v += t.rewards.get(i) * (float) Math.pow(gamma, i - index);
        }
        return v;
    }

    private void learn() {
        int b = batchSize;
        float[] obs = new float[b * obsDims];
        float[][] actionHot = new float[unroll][b * actions];
        float[][] rewardT = new float[unroll][b];
        float[][] valueT = new float[unroll + 1][b];
        float[][] policyT = new float[unroll + 1][b * actions];
        float[][] policyMask = new float[unroll + 1][b];
        for (int i = 0; i < b; i++) {
            Trajectory t = replay.get(random.nextInt(replay.size()));
            int len = t.actions.size();
            int start = random.nextInt(len);
            System.arraycopy(t.observations.get(start), 0, obs, i * obsDims, obsDims);
            for (int k = 0; k <= unroll; k++) {
                int idx = start + k;
                valueT[k][i] = valueTarget(t, idx);
                if (idx < len) {
                    System.arraycopy(t.policies.get(idx), 0, policyT[k], i * actions, actions);
                    policyMask[k][i] = 1;
                }
                if (k < unroll) {
                    int a = idx < len ? t.actions.get(idx) : random.nextInt(actions);
                    actionHot[k][i * actions + a] = 1;
                    rewardT[k][i] = idx < len ? t.rewards.get(idx) : 0;
                }
            }
        }
        Var state = represent(Var.of(Tensor.of(Shape.of(b, obsDims), obs)), true);
        Var loss = predictionLoss(state, valueT[0], policyT[0], policyMask[0]);
        for (int k = 0; k < unroll; k++) {
            Var[] step = dynamics(state, Tensor.of(Shape.of(b, actions), actionHot[k]), true);
            loss = loss.add(step[1].sub(Tensor.of(Shape.of(b, 1), rewardT[k])).square().mean());
            state = step[0];
            loss = loss.add(predictionLoss(state, valueT[k + 1], policyT[k + 1], policyMask[k + 1]));
        }
        loss = loss.div(unroll + 1);
        loss.backward();
        List<Var> params = new ArrayList<>(representation.parameters());
        params.addAll(dynamics.parameters());
        params.addAll(prediction.parameters());
        RlTensors.clip(params, 5f);
        optimizer.step(params);
        lastLoss = loss.value().item();
    }

    private Var predictionLoss(Var state, float[] value, float[] policy, float[] mask) {
        int b = value.length;
        Var out = prediction.forward(state, true);
        Var v = out.slice(1, actions, actions + 1);
        Var valueLoss = v.sub(Tensor.of(Shape.of(b, 1), value)).square().mean();
        Var ce = RlTensors.logSoftmax(out.slice(1, 0, actions)).mul(Tensor.of(Shape.of(b, actions), policy))
                .sum(1, true).neg().mul(Tensor.of(Shape.of(b, 1), mask)).mean();
        return valueLoss.add(ce);
    }

    @Override
    public String toString() {
        return "MuZeroLite(" + simulations + " simulations, unroll " + unroll + ")";
    }
}
