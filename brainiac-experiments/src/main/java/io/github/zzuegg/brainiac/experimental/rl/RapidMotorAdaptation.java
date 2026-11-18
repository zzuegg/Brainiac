package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Rapid motor adaptation: a policy that identifies the system online. Training has two
 * phases on a {@link DomainRandomization} environment.
 * <ol>
 *   <li>A base policy acts on the state and a latent {@code z = encoder(e)} of the true
 *       physics parameters {@code e}, trained jointly with the encoder to maximise return
 *       across the randomised family (here by CMA-ES on a feature-linear policy
 *       {@code u = (A + B z) . x + c}, whose gains depend on the latent).</li>
 *   <li>An adaptation module, an MLP, is trained by regression to predict {@code z} from the
 *       recent history of states, actions and state changes, which is all a deployed robot
 *       sees.</li>
 * </ol>
 * At deployment the base policy runs on the adaptation module's estimate, so the controller
 * re-identifies the dynamics within a few steps, also after the parameters change in the
 * middle of an episode.
 *
 * <p>Kumar et al. 2021, "RMA: Rapid Motor Adaptation for Legged Robots",
 * <a href="https://arxiv.org/abs/2107.04034">arXiv:2107.04034</a>.
 */
public final class RapidMotorAdaptation {

    private final DomainRandomization env;
    private final int stateSize;
    private final int paramSize;
    private final int latent;
    private final int history;
    private final double actionLimit;
    private final long seed;
    private double[] weights;
    private Model adapter;

    private RapidMotorAdaptation(DomainRandomization env, int latent, int history, long seed) {
        this.env = env;
        env.reset(new SplittableRandom(seed));
        this.stateSize = env.state().length;
        this.paramSize = env.parameters().length;
        this.latent = latent;
        this.history = history;
        this.actionLimit = env.system().actionHigh()[0];
        this.seed = seed;
        this.weights = new double[latent * (paramSize + 1) + stateSize + stateSize * latent + 1];
    }

    /**
     * @param env the randomised environment (non-privileged observations: the state)
     * @param latent the size of the extrinsics latent
     * @param history steps of history the adaptation module sees
     * @param seed the seed
     * @return an untrained RMA controller
     */
    public static RapidMotorAdaptation of(DomainRandomization env, int latent, int history, long seed) {
        return new RapidMotorAdaptation(Objects.requireNonNull(env), latent, history, seed);
    }

    /**
     * @param w all weights
     * @param params the physics parameters
     * @return the latent, {@code tanh(W e + b)}
     */
    private double[] encode(double[] w, double[] params) {
        double[] z = new double[latent];
        for (int k = 0; k < latent; k++) {
            double s = w[k * (paramSize + 1) + paramSize];
            for (int p = 0; p < paramSize; p++) {
                s += w[k * (paramSize + 1) + p] * params[p];
            }
            z[k] = Math.tanh(s);
        }
        return z;
    }

    private double act(double[] w, double[] x, double[] z) {
        int off = latent * (paramSize + 1);
        double u = w[off + stateSize + stateSize * latent];
        for (int j = 0; j < stateSize; j++) {
            double gain = w[off + j];
            for (int k = 0; k < latent; k++) {
                gain += w[off + stateSize + j * latent + k] * z[k];
            }
            u += gain * x[j];
        }
        return Math.max(-actionLimit, Math.min(actionLimit, u));
    }

    /**
     * Phase one: trains the encoder and base policy with CMA-ES on the mean return over
     * {@code episodes} randomised episodes per evaluation.
     *
     * @param evaluations CMA-ES fitness evaluations
     * @param episodes episodes per evaluation
     * @return this
     */
    public RapidMotorAdaptation trainBase(int evaluations, int episodes) {
        int[] generation = {0};
        CmaEs cma = CmaEs.of(new double[weights.length], 0.5).seed(seed);
        weights = cma.minimize(w -> {
            double total = 0;
            for (int e = 0; e < episodes; e++) {
                total += episode(w, null, new SplittableRandom(seed * 7919 + generation[0] / 16 * 31 + e), 0);
            }
            generation[0]++;
            return -total / episodes;
        }, evaluations);
        return this;
    }

    /**
     * Return of one episode from step {@code from} on; with a null adapter the true latent is
     * used.
     */
    private double episode(double[] w, Model adapt, SplittableRandom r, int from) {
        env.reset(r);
        List<double[]> past = new ArrayList<>();
        double total = 0;
        Environment.Step s;
        do {
            double[] x = env.state();
            double[] z = adapt == null ? encode(w, env.parameters()) : estimate(adapt, past);
            double u = act(w, x, z);
            int t = env.time();
            s = env.step(new float[] {(float) u});
            double[] next = env.state();
            past.add(transition(x, u, next));
            if (t >= from) {
                total += s.reward();
            }
        } while (!s.done());
        if (env.time() < from) {
            total -= env.failureCost(from);
        }
        return total;
    }

    private double[] transition(double[] x, double u, double[] next) {
        double[] row = new double[2 * stateSize + 1];
        for (int j = 0; j < stateSize; j++) {
            row[j] = x[j];
            row[stateSize + 1 + j] = (next[j] - x[j]) * 10;
        }
        row[stateSize] = u / actionLimit;
        return row;
    }

    private float[] historyFeatures(List<double[]> past) {
        int width = 2 * stateSize + 1;
        float[] f = new float[history * width];
        for (int h = 0; h < history; h++) {
            int idx = past.size() - history + h;
            if (idx >= 0) {
                double[] row = past.get(idx);
                for (int j = 0; j < width; j++) {
                    f[h * width + j] = (float) row[j];
                }
            }
        }
        return f;
    }

    private double[] estimate(Model adapt, List<double[]> past) {
        float[] f = historyFeatures(past);
        float[] z = adapt.forward(Var.of(Tensor.of(Shape.of(1, f.length), f)), false).value().toArray();
        double[] out = new double[latent];
        for (int k = 0; k < latent; k++) {
            out[k] = z[k];
        }
        return out;
    }

    /**
     * Phase two: trains the adaptation module (an MLP on the history) to regress the latent.
     * The first round rolls out the base policy with the true latent; later rounds roll out
     * with the module's own estimate (DAgger-style, as in the paper, so the module also sees
     * the states its own errors lead to), always labelled with the true latent. The data of
     * all rounds is aggregated.
     *
     * @param rounds data-collection rounds
     * @param episodes episodes of data per round
     * @param steps Adam steps per round
     * @return this
     */
    public RapidMotorAdaptation trainAdaptation(int rounds, int episodes, int steps) {
        List<float[]> xs = new ArrayList<>();
        List<double[]> zs = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(seed + 99);
        int width = history * (2 * stateSize + 1);
        adapter = Nets.mlp(width, latent, Activation.TANH, seed + 5, 64);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int round = 0; round < rounds; round++) {
            for (int e = 0; e < episodes; e++) {
                env.reset(r.split());
                List<double[]> past = new ArrayList<>();
                Environment.Step s;
                do {
                    double[] x = env.state();
                    double[] truth = encode(weights, env.parameters());
                    xs.add(historyFeatures(past));
                    zs.add(truth);
                    double[] z = round == 0 ? truth : estimate(adapter, past);
                    // a little action noise makes the history informative about the actuator
                    double u = act(weights, x, z) + 0.1 * actionLimit * r.nextGaussian();
                    s = env.step(new float[] {(float) u});
                    past.add(transition(x, u, env.state()));
                } while (!s.done());
            }
            int batch = Math.min(128, xs.size());
            for (int it = 0; it < steps; it++) {
                float[] in = new float[batch * width];
                float[] target = new float[batch * latent];
                for (int b = 0; b < batch; b++) {
                    int k = r.nextInt(xs.size());
                    System.arraycopy(xs.get(k), 0, in, b * width, width);
                    for (int j = 0; j < latent; j++) {
                        target[b * latent + j] = (float) zs.get(k)[j];
                    }
                }
                adapter.forward(Var.of(Tensor.of(Shape.of(batch, width), in)), true)
                        .sub(Tensor.of(Shape.of(batch, latent), target)).square().mean().backward();
                opt.step(adapter.parameters());
            }
        }
        return this;
    }

    /**
     * @param episodes episodes
     * @param seedBase their seed
     * @param adapted use the adaptation module's estimate (otherwise the true latent, the
     *     privileged upper bound)
     * @return the mean return
     */
    public double evaluate(int episodes, long seedBase, boolean adapted) {
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            total += episode(weights, adapted ? adapter : null, new SplittableRandom(seedBase + 104729L * e), 0);
        }
        return total / episodes;
    }

    /**
     * The adapted controller's mean return counted from step {@code from} on (for example the
     * steps after a {@link DomainRandomization#changeAt} parameter change).
     *
     * @param episodes episodes
     * @param seedBase their seed
     * @param from the first step counted
     * @return the mean partial return
     */
    public double evaluateFrom(int episodes, long seedBase, int from) {
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            total += episode(weights, adapter, new SplittableRandom(seedBase + 104729L * e), from);
        }
        return total / episodes;
    }

    /**
     * The deployable controller: base policy plus adaptation module, keeping its own history
     * (reset by {@link Agent#startEpisode}). Observations are states.
     *
     * @return the agent
     */
    public Agent agent() {
        return new Agent() {
            private final List<double[]> past = new ArrayList<>();
            private double[] last;
            private double lastU;

            @Override
            public float[] act(float[] observation, boolean explore) {
                double[] x = new double[observation.length];
                for (int i = 0; i < x.length; i++) {
                    x[i] = observation[i];
                }
                if (last != null) {
                    past.add(transition(last, lastU, x));
                }
                double u = RapidMotorAdaptation.this.act(weights, x, estimate(adapter, past));
                last = x;
                lastU = u;
                return new float[] {(float) u};
            }

            @Override
            public void startEpisode() {
                past.clear();
                last = null;
            }
        };
    }
}
