package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.ObgdTraces;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Stream AC(lambda) (Elsayed, Vasan and Mahmood, "Streaming Deep Reinforcement Learning
 * Finally Works", <a href="https://arxiv.org/abs/2410.14606">2410.14606</a>): an actor-critic
 * that learns from every transition once, as it happens, with no replay buffer, no batches and
 * no target network. What makes it stable is the paper's recipe: eligibility traces with the
 * overshooting-bounded step {@link ObgdTraces}, online observation normalization, rewards
 * scaled by the running standard deviation of the discounted return, and an entropy bonus
 * whose sign follows the TD error.
 *
 * <p>{@link #fit} returns the episodes as an {@link RlResult}; listeners are told about every
 * episode.
 */
public final class StreamingActorCritic implements Learner {

    private final int observations;
    private final int actions;
    private final Model actor;
    private final Model critic;
    private final ObgdTraces actorTraces;
    private final ObgdTraces criticTraces;
    private final AgentKit.Running[] obsStats;
    private final AgentKit.Running returnStats = new AgentKit.Running();
    private final List<Double> returns = new ArrayList<>();
    private final SplittableRandom random;
    private float gamma = 0.99f;
    private float lambda = 0.8f;
    private float entropy = 0.01f;
    private double discounted;

    private StreamingActorCritic(int observations, int actions, int hidden, float stepSize, long seed) {
        this.observations = observations;
        this.actions = actions;
        this.actor = net(observations, hidden, actions, seed);
        this.critic = net(observations, hidden, 1, seed + 1);
        this.actorTraces = ObgdTraces.of(actor.parameters(), stepSize, 3);
        this.criticTraces = ObgdTraces.of(critic.parameters(), stepSize, 2);
        this.obsStats = new AgentKit.Running[observations];
        for (int i = 0; i < observations; i++) {
            obsStats[i] = new AgentKit.Running();
        }
        this.random = new SplittableRandom(seed);
    }

    /** The paper's network: dense, layer norm, leaky ReLU, twice, then a linear head. */
    private static Model net(int in, int hidden, int out, long seed) {
        Activation act = Activation.leakyRelu();
        return Model.sequential().input(in).dense(hidden).layerNorm().activation(act).dense(hidden).layerNorm()
                .activation(act).dense(out).seed(seed).build();
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param hidden hidden width
     * @param stepSize the ObGD step size (1 in the paper)
     * @param seed the seed
     * @return the learner
     */
    public static StreamingActorCritic of(int observations, int actions, int hidden, float stepSize, long seed) {
        return new StreamingActorCritic(observations, actions, hidden, stepSize, seed);
    }

    /**
     * @param discount gamma
     * @param traceDecay lambda
     * @return this learner
     */
    public StreamingActorCritic discount(float discount, float traceDecay) {
        this.gamma = discount;
        this.lambda = traceDecay;
        return this;
    }

    /**
     * @param coefficient the entropy bonus
     * @return this learner
     */
    public StreamingActorCritic entropy(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /** @return the return of every training episode */
    public List<Double> episodeReturns() {
        return List.copyOf(returns);
    }

    private float[] normalize(float[] x) {
        float[] out = new float[observations];
        for (int i = 0; i < observations; i++) {
            out[i] = (float) Math.max(-5, Math.min(5, (x[i] - obsStats[i].mean()) / obsStats[i].std()));
        }
        return out;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] logits = AgentKit.predict(actor, normalize(observation));
        return new float[] {explore ? AgentKit.sample(logits, random) : AgentKit.argmax(logits)};
    }

    private void observe(float[] x) {
        for (int i = 0; i < observations; i++) {
            obsStats[i].add(x[i]);
        }
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        RlResult result = new RlResult();
        float[] obs = env.reset(random.split());
        observe(obs);
        double episodeReturn = 0;
        int episodeSteps = 0;
        for (long t = 1; t <= steps; t++) {
            float[] x = normalize(obs);
            int a = Math.round(act(obs, true)[0]);
            Environment.Step s = env.step(new float[] {a});
            result.steps(t);
            observe(s.observation());
            episodeReturn += s.reward();
            episodeSteps++;
            discounted = discounted * gamma * (s.done() ? 0 : 1) + s.reward();
            returnStats.add(discounted);
            float r = (float) (s.reward() / returnStats.std());
            learn(x, a, r, normalize(s.observation()), s.terminated());
            if (s.done()) {
                returns.add(episodeReturn);
                Episode e = new Episode(returns.size(), episodeSteps, episodeReturn, t);
                result.add(e);
                if (listener != null) {
                    listener.onEpisode(e, result);
                }
                actorTraces.reset();
                criticTraces.reset();
                episodeReturn = 0;
                episodeSteps = 0;
                obs = env.reset(random.split());
                observe(obs);
            } else {
                obs = s.observation();
            }
        }
        return result;
    }

    private void learn(float[] x, int a, float r, float[] next, boolean terminated) {
        Var v = critic.forward(Var.of(AgentKit.row(x)), true);
        float vNext = terminated ? 0 : AgentKit.predict(critic, next)[0];
        float delta = r + gamma * vNext - v.value().item();
        v.sum().backward();
        criticTraces.accumulate(gamma * lambda);
        criticTraces.step(delta);
        Var logp = Activation.LOG_SOFTMAX.forward(actor.forward(Var.of(AgentKit.row(x)), true));
        Var picked = logp.mul(Tensor.of(Shape.of(1, actions), AgentKit.oneHot(a, actions))).sum();
        Var ent = logp.exp().mul(logp).sum().neg();
        picked.add(ent.mul(entropy * Math.signum(delta))).backward();
        actorTraces.accumulate(gamma * lambda);
        actorTraces.step(delta);
    }
}
