package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GumbelChannel;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A latent-action world model in the style of Genie (Bruce et al., "Genie: Generative
 * Interactive Environments", <a href="https://arxiv.org/abs/2402.15391">2402.15391</a>):
 * it learns <em>which actions exist</em> and how they move the world from observation pairs
 * alone, without ever seeing an action label.
 *
 * <ul>
 *   <li>The <b>latent action model</b> looks at an observation and the next one and squeezes
 *       what changed through a categorical bottleneck of a handful of codes, too few to
 *       describe the next frame, so the codes come to name the change, that is the action.
 *       Genie quantizes with a VQ codebook; here the bottleneck is a straight-through
 *       Gumbel-softmax ({@link GumbelChannel}), which trains reliably at toy scale.</li>
 *   <li>The <b>dynamics model</b> predicts the next observation from the current one and the
 *       code; both train together on the squared reconstruction error. By default it is an
 *       MLP on the observation and the code. For image observations {@link #spatial} makes it
 *       translation-equivariant, like Genie's spatial transformer: each code selects a
 *       learned 3x3 kernel applied at every cell, so a code must mean the same change
 *       everywhere, which is what an action is. (An MLP can instead learn codes whose meaning
 *       depends on the position, which predicts just as well but is not a controller.)</li>
 * </ul>
 *
 * <p>After training, the codes are a controller: {@link #predict} rolls the world forward
 * under a chosen latent action, and {@link #asEnvironment} makes the model a playable core
 * {@link Environment} whose discrete actions are the latent codes. Only
 * {@link Transition#observation()} and {@link Transition#nextObservation()} are read.
 */
public final class LatentActionWorldModel {
    private final int observations;
    private final int latentActions;
    private final Model encoder;
    private final GumbelChannel bottleneck;
    private final Model dynamics;
    private final Tensor[] shifts;
    private final Var kernels;
    private final Optimizer optimizer = Optimizer.adam(3e-3f);

    private LatentActionWorldModel(int observations, int latentActions, int hidden, int rows, int cols, long seed) {
        this.observations = observations;
        this.latentActions = latentActions;
        encoder = AgentKit.mlp(2 * observations, latentActions, seed, hidden);
        bottleneck = GumbelChannel.of(1, latentActions, 1f, seed + 1);
        if (rows > 0) {
            if (rows * cols != observations) {
                throw new IllegalArgumentException("rows * cols must equal the observation size");
            }
            dynamics = null;
            shifts = shiftMatrices(rows, cols);
            kernels = Var.param(Tensor.uniform(Shape.of(latentActions, 9), 0f, 0.2f, new SplittableRandom(seed + 2)));
        } else {
            dynamics = AgentKit.mlp(observations + latentActions, observations, seed + 2, hidden, hidden);
            shifts = null;
            kernels = null;
        }
    }

    /** One matrix per 3x3 offset that moves every cell's value to its neighbour (zero padding). */
    private static Tensor[] shiftMatrices(int rows, int cols) {
        int n = rows * cols;
        Tensor[] out = new Tensor[9];
        for (int o = 0; o < 9; o++) {
            int dy = o / 3 - 1;
            int dx = o % 3 - 1;
            float[] m = new float[n * n];
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    int ty = y + dy;
                    int tx = x + dx;
                    if (ty >= 0 && ty < rows && tx >= 0 && tx < cols) {
                        m[(y * cols + x) * n + ty * cols + tx] = 1;
                    }
                }
            }
            out[o] = Tensor.of(Shape.of(n, n), m);
        }
        return out;
    }

    /**
     * Creates an untrained model with MLP dynamics.
     *
     * @param observations the observation size
     * @param latentActions the number of latent action codes
     * @param hidden the hidden width
     * @param seed the seed
     * @return the model
     */
    public static LatentActionWorldModel of(int observations, int latentActions, int hidden, long seed) {
        return new LatentActionWorldModel(observations, latentActions, hidden, 0, 0, seed);
    }

    /**
     * Creates an untrained model for image observations ({@code rows x cols}, row-major) with
     * translation-equivariant dynamics: the next frame is the current one convolved with the
     * 3x3 kernel of the latent action.
     *
     * @param rows the image height
     * @param cols the image width
     * @param latentActions the number of latent action codes
     * @param hidden the hidden width of the latent action encoder
     * @param seed the seed
     * @return the model
     */
    public static LatentActionWorldModel spatial(int rows, int cols, int latentActions, int hidden, long seed) {
        return new LatentActionWorldModel(rows * cols, latentActions, hidden, rows, cols, seed);
    }

    /** @return the number of latent actions */
    public int latentActions() {
        return latentActions;
    }

    /**
     * Sets the Gumbel-softmax temperature, for example to anneal it.
     *
     * @param temperature the temperature
     * @return this model
     */
    public LatentActionWorldModel temperature(float temperature) {
        bottleneck.temperature(temperature);
        return this;
    }

    private Var step(Var observation, Var code, boolean training) {
        if (shifts == null) {
            return observation.add(dynamics.forward(Var.concat(1, observation, code), training));
        }
        Var weights = code.matmul(kernels);   // [batch, 9]
        Var out = null;
        for (int o = 0; o < 9; o++) {
            Var term = observation.matmul(Var.of(shifts[o])).mul(weights.slice(1, o, o + 1));
            out = out == null ? term : out.add(term);
        }
        return out;
    }

    /**
     * One training step on action-free transitions.
     *
     * @param batch transitions; their actions are ignored
     * @return the reconstruction error before the step
     */
    public float train(List<Transition> batch) {
        List<float[]> obs = new ArrayList<>();
        List<float[]> next = new ArrayList<>();
        List<float[]> pairs = new ArrayList<>();
        for (Transition t : batch) {
            obs.add(t.observation());
            next.add(t.nextObservation());
            pairs.add(AgentKit.concat(t.observation(), t.nextObservation()));
        }
        Var code = bottleneck.sample(encoder.forward(Var.of(AgentKit.rows(pairs)), true), true);
        Var prediction = step(Var.of(AgentKit.rows(obs)), code, true);
        Var loss = prediction.sub(Var.of(AgentKit.rows(next))).square().mean();
        float value = loss.value().item();
        loss.backward();
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.addAll(shifts == null ? dynamics.parameters() : List.of(kernels));
        optimizer.step(params);
        return value;
    }

    /**
     * Infers the latent action between two observations.
     *
     * @param observation the observation
     * @param next the next observation
     * @return the code
     */
    public int inferAction(float[] observation, float[] next) {
        float[] logits = encoder.forward(Var.of(AgentKit.row(AgentKit.concat(observation, next)))).value().toArray();
        return AgentKit.argmax(logits);
    }

    /**
     * Predicts the next observation under a latent action.
     *
     * @param observation the observation
     * @param latentAction the code
     * @return the predicted next observation
     */
    public float[] predict(float[] observation, int latentAction) {
        Tensor code = AgentKit.row(AgentKit.oneHot(latentAction, latentActions));
        return step(Var.of(AgentKit.row(observation)), Var.of(code), false).value().toArray();
    }

    /**
     * The model as a playable environment: discrete actions are latent codes, observations
     * are the model's own predictions, rewards are zero, and episodes last
     * {@code maxSteps}.
     *
     * @param start the first observation of every episode
     * @param maxSteps the episode length
     * @return the environment
     */
    public Environment asEnvironment(float[] start, int maxSteps) {
        float[] first = start.clone();
        return new Environment() {
            private float[] current = first.clone();
            private int steps;

            @Override
            public Space observationSpace() {
                return Space.box(observations, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
            }

            @Override
            public Space actionSpace() {
                return Space.discrete(latentActions);
            }

            @Override
            public float[] reset(SplittableRandom random) {
                current = first.clone();
                steps = 0;
                return current.clone();
            }

            @Override
            public Step step(float[] action) {
                current = predict(current, Math.round(action[0]));
                steps++;
                return new Step(current.clone(), 0, false, steps >= maxSteps);
            }
        };
    }
}
