package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Dreamer-style agent (Hafner et al., DreamerV3, <a href="https://arxiv.org/abs/2301.04104">2301.04104</a>),
 * simplified to an MLP world model over observations: it acts in the real environment only
 * to collect experience, fits an {@link MlpWorldModel} to it, and trains its actor and critic
 * entirely in imagined rollouts of that model. Imagined trajectories start from real
 * observations, run {@code horizon} steps with the actor, and are scored with
 * {@code lambda}-returns bootstrapped from the critic and weighted by the predicted
 * probability that the episode is still going. The actor is trained with REINFORCE on the
 * imagined advantages (as DreamerV3 does for discrete actions) plus an entropy bonus.
 *
 * <p>{@link #fit} returns the episodes as an {@link RlResult}; {@link #episodeReturns} has their
 * totals and listeners are told as usual.
 */
public final class DreamerAgent implements Learner {

    private final int actions;
    private DiscreteWorldModel world;
    private final Model actor;
    private final Model critic;
    private final Optimizer actorOpt = Optimizer.adam(3e-3f);
    private final Optimizer criticOpt = Optimizer.adam(3e-3f);
    private final List<Transition> buffer = new ArrayList<>();
    private final List<Double> returns = new ArrayList<>();
    private int horizon = 15;
    private int imagineBatch = 64;
    private int trainEvery = 250;
    private int modelSteps = 60;
    private int behaviourSteps = 30;
    private float gamma = 0.97f;
    private float lambda = 0.95f;
    private float entropy = 0.01f;
    private SplittableRandom random = new SplittableRandom(0);

    private DreamerAgent(int observations, int actions, int hidden, long seed) {
        this.actions = actions;
        this.world = MlpWorldModel.of(observations, actions, hidden, seed);
        this.actor = AgentKit.mlp(observations, actions, seed + 1, hidden);
        this.critic = Model.sequential().input(observations).dense(hidden, Activation.TANH).dense(1).seed(seed + 2)
                .build();
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param hidden hidden width of all networks
     * @param seed the seed
     * @return the agent
     */
    public static DreamerAgent of(int observations, int actions, int hidden, long seed) {
        return new DreamerAgent(observations, actions, hidden, seed);
    }

    /**
     * @param steps imagined steps per rollout
     * @return this agent
     */
    public DreamerAgent horizon(int steps) {
        this.horizon = steps;
        return this;
    }

    /**
     * @param envSteps real steps between training phases
     * @param modelSteps world-model updates per phase
     * @param behaviourSteps actor-critic updates (in imagination) per phase
     * @return this agent
     */
    public DreamerAgent schedule(int envSteps, int modelSteps, int behaviourSteps) {
        this.trainEvery = envSteps;
        this.modelSteps = modelSteps;
        this.behaviourSteps = behaviourSteps;
        return this;
    }

    /** @return the world model */
    public DiscreteWorldModel worldModel() {
        return world;
    }

    /**
     * Replaces the world model, for example by a {@link DiffusionWorldModel}.
     *
     * @param model the model to learn and dream in
     * @return this agent
     */
    public DreamerAgent worldModel(DiscreteWorldModel model) {
        this.world = java.util.Objects.requireNonNull(model);
        return this;
    }

    /** @return the return of every real episode during training */
    public List<Double> episodeReturns() {
        return List.copyOf(returns);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] logits = AgentKit.predict(actor, observation);
        return new float[] {explore ? AgentKit.sample(logits, random) : AgentKit.argmax(logits)};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        RlResult result = new RlResult();
        float[] obs = env.reset(random.split());
        double episodeReturn = 0;
        int episodeSteps = 0;
        for (long t = 1; t <= steps; t++) {
            float[] a = t < trainEvery ? new float[] {random.nextInt(actions)} : act(obs, true);
            Environment.Step s = env.step(a);
            result.steps(t);
            buffer.add(new Transition(obs, a, s.reward(), s.observation(), s.terminated(), gamma));
            episodeReturn += s.reward();
            episodeSteps++;
            if (s.done()) {
                returns.add(episodeReturn);
                Episode e = new Episode(returns.size(), episodeSteps, episodeReturn, t);
                result.add(e);
                if (listener != null) {
                    listener.onEpisode(e, result);
                }
                episodeReturn = 0;
                episodeSteps = 0;
                obs = env.reset(random.split());
            } else {
                obs = s.observation();
            }
            if (t % trainEvery == 0) {
                for (int i = 0; i < modelSteps; i++) {
                    world.train(MlpWorldModel.sample(buffer, random, 64));
                }
                for (int i = 0; i < behaviourSteps; i++) {
                    imagine();
                }
            }
        }
        return result;
    }

    /** One actor-critic update on imagined rollouts. */
    private void imagine() {
        List<float[]> starts = new ArrayList<>();
        for (int i = 0; i < imagineBatch; i++) {
            starts.add(buffer.get(random.nextInt(buffer.size())).observation());
        }
        Tensor s = AgentKit.rows(starts);
        List<Tensor> states = new ArrayList<>();
        List<int[]> acts = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        List<float[]> conts = new ArrayList<>();
        for (int h = 0; h < horizon; h++) {
            states.add(s);
            float[] logits = actor.forward(Var.of(s), false).value().toArray();
            int[] a = new int[imagineBatch];
            for (int i = 0; i < imagineBatch; i++) {
                a[i] = AgentKit.sample(java.util.Arrays.copyOfRange(logits, i * actions, (i + 1) * actions), random);
            }
            MlpWorldModel.Prediction p = world.predict(s, a);
            acts.add(a);
            rewards.add(p.reward().toArray());
            conts.add(p.continuation().toArray());
            s = p.next();
        }
        states.add(s);
        Tensor all = Tensor.concat(0, states.toArray(Tensor[]::new));
        float[] v = critic.forward(Var.of(all), false).value().toArray();
        int n = imagineBatch;
        float[] ret = new float[horizon * n];
        float[] weight = new float[horizon * n];
        float[] adv = new float[horizon * n];
        for (int i = 0; i < n; i++) {
            double next = v[horizon * n + i];
            for (int h = horizon - 1; h >= 0; h--) {
                double c = conts.get(h)[i];
                double r = rewards.get(h)[i] + gamma * c * ((1 - lambda) * v[(h + 1) * n + i] + lambda * next);
                ret[h * n + i] = (float) r;
                next = r;
            }
            double w = 1;
            for (int h = 0; h < horizon; h++) {
                weight[h * n + i] = (float) w;
                adv[h * n + i] = ret[h * n + i] - v[h * n + i];
                w *= conts.get(h)[i];
            }
        }
        // normalize advantages for a stable actor step
        double mean = 0;
        double sq = 0;
        for (float x : adv) {
            mean += x;
            sq += x * x;
        }
        mean /= adv.length;
        double std = Math.sqrt(Math.max(1e-8, sq / adv.length - mean * mean));
        float[] actOneHot = new float[horizon * n * actions];
        for (int h = 0; h < horizon; h++) {
            for (int i = 0; i < n; i++) {
                actOneHot[(h * n + i) * actions + acts.get(h)[i]] = 1;
                adv[h * n + i] = (float) ((adv[h * n + i] - mean) / std) * weight[h * n + i];
            }
        }
        Tensor x = all.slice(0, 0, horizon * n);
        Var logp = Activation.LOG_SOFTMAX.forward(actor.forward(Var.of(x), true));
        Var picked = logp.mul(Tensor.of(Shape.of(horizon * n, actions), actOneHot)).sum(1, true);
        Var ent = logp.exp().mul(logp).sum(1, true).neg();
        Var actorLoss = picked.mul(AgentKit.column(adv)).mean().neg().sub(ent.mean().mul(entropy));
        actorLoss.backward();
        actorOpt.step(actor.parameters());
        Var value = critic.forward(Var.of(x), true);
        Var criticLoss = value.sub(AgentKit.column(ret)).square().mul(AgentKit.column(weight)).mean();
        criticLoss.backward();
        criticOpt.step(critic.parameters());
    }
}
