package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEvaluation;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Heterogeneous-agent trust-region learning (Kuba et al., ICLR 2022,
 * <a href="https://arxiv.org/abs/2109.11251">arXiv:2109.11251</a>): every agent has its own
 * policy network (no parameter sharing) and a centralised critic on the global state gives the
 * joint advantage {@code A}. Instead of updating all agents at once, the agents are updated one
 * at a time in a fresh random order, and agent {@code i_m}'s objective weights the advantage by
 * {@code M = prod_{k < m} pi_new^{i_k} / pi_old^{i_k}}, the probability ratios of the agents
 * already updated in this round (the multi-agent advantage decomposition); this is what makes
 * the joint return improve monotonically in the trust-region version.
 *
 * <ul>
 *   <li>{@link Mode#HAPPO}: PPO's clipped objective {@code min(r M A, clip(r) M A)};</li>
 *   <li>{@link Mode#HATRPO}: a KL-penalised surrogate {@code r M A - beta KL(pi_old || pi)}
 *       instead of the paper's constrained natural-gradient step (a simplification);</li>
 *   <li>{@link Mode#SIMULTANEOUS}: every agent updated from the same data with {@code M = 1},
 *       which is MAPPO without parameter sharing, the ablation.</li>
 * </ul>
 */
public final class Happo {

    /** The update scheme. */
    public enum Mode {
        /** sequential clipped updates */
        HAPPO,
        /** sequential KL-penalised updates */
        HATRPO,
        /** simultaneous clipped updates (MAPPO without sharing) */
        SIMULTANEOUS
    }

    private final Mode mode;
    private final Supplier<? extends MultiAgentEnvironment> factory;
    private final MultiAgentEnvironment env;
    private final int agents;
    private final int obsSize;
    private final int stateSize;
    private final int actions;
    private final List<Model> actors = new ArrayList<>();
    private final Model critic;
    private final List<Optimizer> actorOptimizers = new ArrayList<>();
    private final Optimizer criticOptimizer = Optimizer.adam(3e-3f);
    private int episodesPerIteration = 32;
    private int epochs = 5;
    private float clip = 0.2f;
    private float beta = 1f;
    private float gamma = 0.99f;
    private float entropy = 0.01f;
    private final SplittableRandom random;
    private long steps;

    private Happo(Mode mode, Supplier<? extends MultiAgentEnvironment> factory, int hidden, long seed) {
        this.mode = mode;
        this.factory = factory;
        this.env = factory.get();
        this.agents = env.agents();
        this.obsSize = env.observationSize();
        this.stateSize = env.stateSize();
        this.actions = env.actions();
        for (int i = 0; i < agents; i++) {
            actors.add(Nets.mlp(obsSize, actions, Activation.TANH, seed + 10L * i, hidden, hidden));
            actorOptimizers.add(Optimizer.adam(3e-3f));
        }
        critic = Nets.mlp(stateSize, 1, Activation.TANH, seed + 1, hidden, hidden);
        random = new SplittableRandom(seed);
    }

    /**
     * @param mode the update scheme
     * @param factory creates the environment
     * @param hidden hidden units of every actor and the critic
     * @param seed the seed
     * @return the learner
     */
    public static Happo of(Mode mode, Supplier<? extends MultiAgentEnvironment> factory, int hidden, long seed) {
        return new Happo(Objects.requireNonNull(mode), Objects.requireNonNull(factory), hidden, seed);
    }

    /** @param n episodes collected per update round @return this learner */
    public Happo episodesPerIteration(int n) {
        this.episodesPerIteration = n;
        return this;
    }

    /** @param n gradient epochs per agent per round @return this learner */
    public Happo epochs(int n) {
        this.epochs = n;
        return this;
    }

    /** @param k the KL penalty of {@link Mode#HATRPO} @return this learner */
    public Happo klPenalty(float k) {
        this.beta = k;
        return this;
    }

    /** @param coefficient the entropy bonus @return this learner */
    public Happo entropyBonus(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /** @return joint environment steps taken */
    public long steps() {
        return steps;
    }

    private float[] logits(int agent, float[] obs) {
        return RlTensors.row(actors.get(agent), obs);
    }

    private static double[] softmax(float[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (float v : x) {
            max = Math.max(max, v);
        }
        double[] p = new double[x.length];
        double z = 0;
        for (int i = 0; i < x.length; i++) {
            p[i] = Math.exp(x[i] - max);
            z += p[i];
        }
        for (int i = 0; i < x.length; i++) {
            p[i] /= z;
        }
        return p;
    }

    /**
     * @param observations one per agent
     * @param explore sample rather than take the most likely action
     * @return the joint action
     */
    public int[] act(float[][] observations, boolean explore) {
        int[] a = new int[agents];
        for (int i = 0; i < agents; i++) {
            double[] p = softmax(logits(i, observations[i]));
            if (explore) {
                double u = random.nextDouble();
                double c = 0;
                a[i] = actions - 1;
                for (int k = 0; k < actions; k++) {
                    c += p[k];
                    if (u < c) {
                        a[i] = k;
                        break;
                    }
                }
            } else {
                a[i] = RlTensors.argmax(p);
            }
        }
        return a;
    }

    /**
     * Trains for at least a number of joint environment steps.
     *
     * @param jointSteps joint steps
     * @return this learner
     */
    public Happo fit(long jointSteps) {
        long target = steps + jointSteps;
        while (steps < target) {
            iteration();
        }
        return this;
    }

    private void iteration() {
        List<float[][]> obs = new ArrayList<>();
        List<float[]> states = new ArrayList<>();
        List<int[]> acts = new ArrayList<>();
        List<Float> returns = new ArrayList<>();
        for (int e = 0; e < episodesPerIteration; e++) {
            float[][] o = env.reset(random.split());
            List<Float> rewards = new ArrayList<>();
            int start = obs.size();
            MultiAgentEnvironment.Step s;
            do {
                float[] st = env.state();
                int[] a = act(o, true);
                s = env.step(a);
                obs.add(o);
                states.add(st);
                acts.add(a);
                rewards.add(s.reward());
                o = s.observations();
                steps++;
            } while (!s.done());
            float g = 0;
            float[] rtg = new float[rewards.size()];
            for (int t = rewards.size() - 1; t >= 0; t--) {
                g = rewards.get(t) + gamma * g;
                rtg[t] = g;
            }
            for (int t = 0; t < rtg.length; t++) {
                returns.add(rtg[t]);
            }
            assert returns.size() == start + rtg.length;
        }
        int b = obs.size();
        float[] st = new float[b * stateSize];
        float[] ret = new float[b];
        for (int t = 0; t < b; t++) {
            System.arraycopy(states.get(t), 0, st, t * stateSize, stateSize);
            ret[t] = returns.get(t);
        }
        Tensor stateT = Tensor.of(Shape.of(b, stateSize), st);
        Tensor retT = Tensor.of(Shape.of(b, 1), ret);
        float[] values = critic.forward(Var.of(stateT), false).value().toArray();
        float[] adv = new float[b];
        double mean = 0;
        for (int t = 0; t < b; t++) {
            adv[t] = ret[t] - values[t];
            mean += adv[t];
        }
        mean /= b;
        double var = 0;
        for (int t = 0; t < b; t++) {
            var += (adv[t] - mean) * (adv[t] - mean);
        }
        double std = Math.sqrt(var / b) + 1e-6;
        for (int t = 0; t < b; t++) {
            adv[t] = (float) ((adv[t] - mean) / std);
        }
        for (int k = 0; k < epochs; k++) {
            Var loss = critic.forward(Var.of(stateT), true).sub(retT).square().mean();
            loss.backward();
            RlTensors.clip(critic.parameters(), 5f);
            criticOptimizer.step(critic.parameters());
        }
        int[] order = new int[agents];
        for (int i = 0; i < agents; i++) {
            order[i] = i;
        }
        if (mode != Mode.SIMULTANEOUS) {
            for (int i = agents - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
        float[] m = new float[b];
        java.util.Arrays.fill(m, 1f);
        for (int agent : order) {
            float[] o = new float[b * obsSize];
            float[] chosen = new float[b];
            for (int t = 0; t < b; t++) {
                System.arraycopy(obs.get(t)[agent], 0, o, t * obsSize, obsSize);
                chosen[t] = acts.get(t)[agent];
            }
            Tensor obsT = Tensor.of(Shape.of(b, obsSize), o);
            Model actor = actors.get(agent);
            Tensor hot = RlTensors.oneHot(chosen, actions, actor.device());
            Tensor oldLogPi = RlTensors.logSoftmax(actor.forward(Var.of(obsT), false)).value();
            Tensor oldLogp = oldLogPi.mul(hot).sum(1, true);
            float[] weighted = new float[b];
            for (int t = 0; t < b; t++) {
                weighted[t] = adv[t] * (mode == Mode.SIMULTANEOUS ? 1 : m[t]);
            }
            Tensor advT = Tensor.of(Shape.of(b, 1), weighted);
            for (int k = 0; k < epochs; k++) {
                Var logPi = RlTensors.logSoftmax(actor.forward(Var.of(obsT), true));
                Var ratio = logPi.mul(hot).sum(1, true).sub(oldLogp).exp();
                Var pi = logPi.exp();
                Var ent = pi.mul(logPi).sum(1, true).neg().mean();
                Var surrogate;
                if (mode == Mode.HATRPO) {
                    Var kl = logPi.neg().add(oldLogPi).mul(oldLogPi.exp()).sum(1, true).mean();
                    surrogate = ratio.mul(advT).mean().sub(kl.mul(beta));
                } else {
                    Var unclipped = ratio.mul(advT);
                    Var clipped = ratio.clamp(1 - clip, 1 + clip).mul(advT);
                    surrogate = unclipped.minimum(clipped).mean();
                }
                surrogate.add(ent.mul(entropy)).neg().backward();
                RlTensors.clip(actor.parameters(), 5f);
                actorOptimizers.get(agent).step(actor.parameters());
            }
            float[] newLogp = RlTensors.logSoftmax(actor.forward(Var.of(obsT), false)).value().mul(hot).sum(1, true)
                    .toArray();
            float[] old = oldLogp.toArray();
            for (int t = 0; t < b; t++) {
                m[t] *= (float) Math.exp(newLogp[t] - old[t]);
            }
        }
    }

    /**
     * @param episodes greedy episodes
     * @param seed the seed
     * @return the mean team return
     */
    public double evaluate(int episodes, long seed) {
        return MultiAgentEvaluation.teamReturn(factory.get(), this::act, episodes, seed);
    }

    @Override
    public String toString() {
        return mode.toString();
    }
}
