package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Model-predictive control with the cross-entropy method, as in PETS (Chua et al., 2018,
 * <a href="https://arxiv.org/abs/1805.12114">arXiv:1805.12114</a>): at every step, sample many
 * action sequences from a Gaussian, roll each out through a dynamics model, keep the best
 * few by predicted return, refit the Gaussian to them and repeat; execute only the first
 * action of the final mean, then replan. The plan's tail warm-starts the next step.
 *
 * <p>The dynamics can be known or learned: {@link #learn} fits a neural network to
 * predict the change of observation from random transitions, which makes this a complete
 * model-based RL agent that needs far fewer environment steps than model-free methods.
 * The reward must be known as a function of observation and action (as in PETS).
 */
public final class CemPlanner implements Agent {

    /** Predicts the next observation. */
    @FunctionalInterface
    public interface Dynamics {
        /**
         * @param observations {@code [n, obs]}
         * @param actions {@code [n, act]}
         * @return the next observations {@code [n, obs]}
         */
        Tensor next(Tensor observations, Tensor actions);
    }

    /** The reward of acting in a state. */
    @FunctionalInterface
    public interface Reward {
        /**
         * @param observation the observation acted in
         * @param action the action
         * @return the reward
         */
        double of(float[] observation, float[] action);
    }

    private final Dynamics dynamics;
    private final Reward reward;
    private final float[] low;
    private final float[] high;
    private int horizon = 15;
    private int population = 100;
    private int elites = 10;
    private int iterations = 3;
    private SplittableRandom random = new SplittableRandom(0);
    private float[] plan;

    private CemPlanner(Dynamics dynamics, Reward reward, Space.Box actions) {
        this.dynamics = dynamics;
        this.reward = reward;
        this.low = actions.low();
        this.high = actions.high();
    }

    /**
     * @param dynamics the model
     * @param reward the reward function
     * @param actions the action box
     * @return the planner
     */
    public static CemPlanner of(Dynamics dynamics, Reward reward, Space.Box actions) {
        return new CemPlanner(Objects.requireNonNull(dynamics), Objects.requireNonNull(reward), actions);
    }

    /** @param steps the planning horizon @return this planner */
    public CemPlanner horizon(int steps) {
        this.horizon = steps;
        return this;
    }

    /**
     * @param samples action sequences per iteration
     * @param best how many are kept to refit the distribution
     * @param rounds refinement iterations
     * @return this planner
     */
    public CemPlanner search(int samples, int best, int rounds) {
        this.population = samples;
        this.elites = best;
        this.iterations = rounds;
        return this;
    }

    /** @param seed the seed @return this planner */
    public CemPlanner seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** Forgets the warm-start plan, for a new episode. */
    @Override
    public void startEpisode() {
        plan = null;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        int d = low.length;
        int len = horizon * d;
        float[] mean = new float[len];
        float[] std = new float[len];
        for (int i = 0; i < len; i++) {
            int k = i % d;
            mean[i] = plan == null ? (low[k] + high[k]) / 2 : (i + d < len ? plan[i + d] : plan[i]);
            std[i] = (high[k] - low[k]) / 2;
        }
        float[][] seqs = new float[population][len];
        double[] returns = new double[population];
        for (int it = 0; it < iterations; it++) {
            for (int p = 0; p < population; p++) {
                for (int i = 0; i < len; i++) {
                    int k = i % d;
                    double a = mean[i] + std[i] * random.nextGaussian();
                    seqs[p][i] = (float) Math.max(low[k], Math.min(high[k], a));
                }
            }
            evaluate(observation, seqs, returns);
            Integer[] order = new Integer[population];
            for (int p = 0; p < population; p++) {
                order[p] = p;
            }
            java.util.Arrays.sort(order, (a, b) -> Double.compare(returns[b], returns[a]));
            for (int i = 0; i < len; i++) {
                double m = 0;
                for (int e = 0; e < elites; e++) {
                    m += seqs[order[e]][i];
                }
                m /= elites;
                double v = 0;
                for (int e = 0; e < elites; e++) {
                    v += (seqs[order[e]][i] - m) * (seqs[order[e]][i] - m);
                }
                mean[i] = (float) m;
                std[i] = (float) Math.sqrt(v / elites + 1e-6);
            }
        }
        plan = mean;
        return java.util.Arrays.copyOf(mean, d);
    }

    private void evaluate(float[] observation, float[][] seqs, double[] returns) {
        int n = seqs.length;
        int d = low.length;
        int o = observation.length;
        float[] obs = new float[n * o];
        for (int p = 0; p < n; p++) {
            System.arraycopy(observation, 0, obs, p * o, o);
        }
        java.util.Arrays.fill(returns, 0);
        Tensor state = Tensor.of(Shape.of(n, o), obs);
        for (int t = 0; t < horizon; t++) {
            float[] act = new float[n * d];
            float[] s = state.toArray();
            for (int p = 0; p < n; p++) {
                System.arraycopy(seqs[p], t * d, act, p * d, d);
                returns[p] += reward.of(java.util.Arrays.copyOfRange(s, p * o, p * o + o),
                        java.util.Arrays.copyOfRange(act, p * d, p * d + d));
            }
            state = dynamics.next(state, Tensor.of(Shape.of(n, d), act));
        }
    }

    /**
     * Learns a dynamics model: a network predicting {@code next - obs} from
     * {@code [obs, action]}, trained on transitions from uniformly random actions.
     *
     * @param env the environment to collect from
     * @param network {@code obs + act -> obs} values (the predicted change)
     * @param steps random environment steps to collect
     * @param epochs gradient steps of batch 128
     * @param seed the seed
     * @return the learned dynamics
     */
    public static Dynamics learn(Environment env, Model network, int steps, int epochs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int o = env.observationSpace().size();
        int d = env.actionSpace().size();
        ReplayBuffer buffer = new ReplayBuffer(steps, o, d);
        float[] obs = env.reset(r.split());
        for (int s = 0; s < steps; s++) {
            float[] a = env.actionSpace().sample(r);
            Environment.Step st = env.step(a);
            buffer.add(new Transition(obs, a, st.reward(), st.observation(), st.terminated(), 1));
            obs = st.done() ? env.reset(r.split()) : st.observation();
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int e = 0; e < epochs; e++) {
            ReplayBuffer.Batch b = buffer.sample(128, r, network.device());
            Var pred = network.forward(Var.of(Tensor.concat(1, b.observations(), b.actions())), true);
            Var loss = pred.sub(b.nextObservations().sub(b.observations())).square().mean();
            loss.backward();
            opt.step(network.parameters());
        }
        return (states, actions) -> states.add(network.forward(Var.of(Tensor.concat(1, states, actions)), false)
                .value());
    }
}
