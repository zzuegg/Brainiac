package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A learned model of an environment's dynamics for discrete actions: from an observation
 * and an action it predicts the change of the observation, the reward and whether the
 * episode ends ({@code [delta..., reward, doneLogit]}, trained with squared error and binary
 * cross-entropy). Agents can then plan or learn in imagined rollouts instead of the real
 * environment, as {@link DreamerAgent} does.
 */
public final class MlpWorldModel {

    private final int observations;
    private final int actions;
    private final Model net;
    private Optimizer optimizer = Optimizer.adam(3e-3f);

    private MlpWorldModel(int observations, int actions, int hidden, long seed) {
        this.observations = observations;
        this.actions = actions;
        this.net = AgentKit.mlp(observations + actions, observations + 2, seed, hidden, hidden);
    }

    /**
     * @param observations observation size
     * @param actions the number of discrete actions
     * @param hidden hidden width
     * @param seed the seed
     * @return an untrained model
     */
    public static MlpWorldModel of(int observations, int actions, int hidden, long seed) {
        return new MlpWorldModel(observations, actions, hidden, seed);
    }

    /**
     * @param o the optimizer
     * @return this model
     */
    public MlpWorldModel optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    private Var input(Tensor obs, int[] acts) {
        float[] oneHot = new float[acts.length * actions];
        for (int i = 0; i < acts.length; i++) {
            oneHot[i * actions + acts[i]] = 1;
        }
        return Var.concat(1, Var.of(obs), Var.of(Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(acts.length,
                actions), oneHot)));
    }

    /**
     * One training step on a batch of transitions.
     *
     * @param batch the transitions
     * @return the loss
     */
    public float train(List<Transition> batch) {
        List<float[]> obs = new ArrayList<>();
        int[] acts = new int[batch.size()];
        float[] targets = new float[batch.size() * observations];
        float[] rewards = new float[batch.size()];
        float[] done = new float[batch.size()];
        for (int i = 0; i < batch.size(); i++) {
            Transition t = batch.get(i);
            obs.add(t.observation());
            acts[i] = Math.round(t.action()[0]);
            for (int k = 0; k < observations; k++) {
                targets[i * observations + k] = t.nextObservation()[k] - t.observation()[k];
            }
            rewards[i] = t.reward();
            done[i] = t.terminated() ? 1 : 0;
        }
        Var out = net.forward(input(AgentKit.rows(obs), acts), true);
        Var delta = out.slice(1, 0, observations);
        Var reward = out.slice(1, observations, observations + 1);
        Var doneLogit = out.slice(1, observations + 1, observations + 2);
        Var loss = delta.sub(Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(batch.size(), observations),
                targets)).square().mean()
                .add(reward.sub(AgentKit.column(rewards)).square().mean())
                .add(doneLogit.softplus().sub(doneLogit.mul(AgentKit.column(done))).mean());
        float v = loss.value().item();
        loss.backward();
        optimizer.step(net.parameters());
        return v;
    }

    /**
     * An imagined step for a batch.
     *
     * @param obs observations {@code [batch, observations]}
     * @param acts the actions
     * @return the prediction
     */
    public Prediction predict(Tensor obs, int[] acts) {
        Tensor out = net.forward(input(obs, acts), false).value();
        Tensor next = obs.add(out.slice(1, 0, observations));
        Tensor reward = out.slice(1, observations, observations + 1);
        Tensor cont = out.slice(1, observations + 1, observations + 2).neg().sigmoid();
        return new Prediction(next, reward, cont);
    }

    /**
     * A predicted step.
     *
     * @param next the next observations
     * @param reward the rewards {@code [batch, 1]}
     * @param continuation the probability that the episode goes on, {@code [batch, 1]}
     */
    public record Prediction(Tensor next, Tensor reward, Tensor continuation) {
    }

    /**
     * @param transitions transitions
     * @param random the random source
     * @param size the batch size
     * @return a random batch
     */
    static List<Transition> sample(List<Transition> transitions, SplittableRandom random, int size) {
        List<Transition> b = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            b.add(transitions.get(random.nextInt(transitions.size())));
        }
        return b;
    }
}
