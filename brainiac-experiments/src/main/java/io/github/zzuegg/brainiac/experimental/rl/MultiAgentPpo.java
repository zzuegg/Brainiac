package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * PPO for cooperative multi-agent tasks ({@link MultiAgentEnvironment}), in the two forms
 * that turned out to be strong baselines: independent PPO with a shared actor and critic over
 * local observations (IPPO, de Witt et al. 2020,
 * <a href="https://arxiv.org/abs/2011.09533">arXiv:2011.09533</a>), and multi-agent PPO with a
 * centralised critic on the global state (MAPPO, Yu et al. 2022,
 * <a href="https://arxiv.org/abs/2103.01955">arXiv:2103.01955</a>). Both follow the MAPPO
 * recipe: parameter sharing (agents are told apart by the agent id in their observation), GAE
 * advantages on the shared team reward, value normalisation (the critic predicts returns
 * standardised by running statistics), a clipped surrogate with an entropy bonus, and few
 * epochs per batch. Episodes run in parallel environments so every network call is batched.
 * (The tasks here have no agent deaths, so death masking does not arise.)
 *
 * <pre>{@code
 * MultiAgentPpo mappo = MultiAgentPpo.mappo(CooperativeNavigation::new, 64, 1);
 * mappo.fit(40_000);
 * double teamReturn = mappo.evaluate(20, 7);
 * }</pre>
 */
public final class MultiAgentPpo {

    private final Supplier<? extends MultiAgentEnvironment> factory;
    private final boolean centralised;
    private final MultiAgentEnvironment[] envs;
    private final int agents;
    private final int obsSize;
    private final int stateSize;
    private final int actions;
    private final Model actor;
    private final Model critic;
    private final Optimizer actorOpt = Optimizer.adam(1e-3f);
    private final Optimizer criticOpt = Optimizer.adam(1e-3f);
    private final SplittableRandom random;
    private float gamma = 0.99f;
    private float lambda = 0.95f;
    private float clip = 0.2f;
    private float entropy = 0.01f;
    private int epochs = 4;
    private double retMean;
    private double retVar = 1;
    private long statCount;
    private long steps;

    private MultiAgentPpo(Supplier<? extends MultiAgentEnvironment> factory, boolean centralised, int hidden,
            int parallel, long seed) {
        this.factory = factory;
        this.centralised = centralised;
        this.envs = new MultiAgentEnvironment[parallel];
        for (int e = 0; e < parallel; e++) {
            envs[e] = factory.get();
        }
        MultiAgentEnvironment env = envs[0];
        this.agents = env.agents();
        this.obsSize = env.observationSize();
        this.stateSize = env.stateSize();
        this.actions = env.actions();
        this.actor = Nets.mlp(obsSize, actions, Activation.TANH, seed, hidden, hidden);
        this.critic = Nets.mlp(centralised ? stateSize : obsSize, 1, Activation.TANH, seed + 1, hidden, hidden);
        this.random = new SplittableRandom(seed);
    }

    /**
     * Independent PPO: a shared actor and a shared critic, both on local observations.
     *
     * @param factory creates environments (one per parallel episode)
     * @param hidden hidden units
     * @param seed the seed
     * @return the learner, with 16 parallel environments
     */
    public static MultiAgentPpo ippo(Supplier<? extends MultiAgentEnvironment> factory, int hidden, long seed) {
        return new MultiAgentPpo(Objects.requireNonNull(factory), false, hidden, 16, seed);
    }

    /**
     * Multi-agent PPO: a shared actor on local observations and a centralised critic on the
     * global state.
     *
     * @param factory creates environments (one per parallel episode)
     * @param hidden hidden units
     * @param seed the seed
     * @return the learner, with 16 parallel environments
     */
    public static MultiAgentPpo mappo(Supplier<? extends MultiAgentEnvironment> factory, int hidden, long seed) {
        return new MultiAgentPpo(Objects.requireNonNull(factory), true, hidden, 16, seed);
    }

    /**
     * @param epochCount PPO epochs per batch
     * @return this
     */
    public MultiAgentPpo epochs(int epochCount) {
        this.epochs = epochCount;
        return this;
    }

    /**
     * @param coefficient the entropy bonus
     * @return this
     */
    public MultiAgentPpo entropyBonus(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /** @return joint environment steps so far */
    public long steps() {
        return steps;
    }

    private static float[] flatten(float[][] rows) {
        int w = rows[0].length;
        float[] out = new float[rows.length * w];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, out, i * w, w);
        }
        return out;
    }

    private float[] values(float[] inputs, int rows) {
        int w = centralised ? stateSize : obsSize;
        float[] v = critic.forward(Var.of(Tensor.of(Shape.of(rows, w), inputs)), false).value().toArray();
        double std = Math.sqrt(retVar) + 1e-6;
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (v[i] * std + retMean);
        }
        return v;
    }

    /**
     * Trains for at least a number of joint environment steps, one batch of parallel episodes
     * at a time.
     *
     * @param jointSteps joint steps (summed over the parallel environments)
     * @return this
     */
    public MultiAgentPpo fit(long jointSteps) {
        long target = steps + jointSteps;
        while (steps < target) {
            iteration();
        }
        return this;
    }

    private void iteration() {
        int e = envs.length;
        int n = agents;
        java.util.List<float[]> obsRows = new java.util.ArrayList<>();
        java.util.List<float[]> stateRows = new java.util.ArrayList<>();
        java.util.List<int[]> actionRows = new java.util.ArrayList<>();
        java.util.List<float[]> logpRows = new java.util.ArrayList<>();
        java.util.List<float[]> rewardRows = new java.util.ArrayList<>();
        java.util.List<float[]> valueRows = new java.util.ArrayList<>();
        float[][][] obs = new float[e][][];
        for (int k = 0; k < e; k++) {
            obs[k] = envs[k].reset(random.split());
        }
        boolean done = false;
        while (!done) {
            float[] o = new float[e * n * obsSize];
            float[] s = new float[e * stateSize];
            for (int k = 0; k < e; k++) {
                System.arraycopy(flatten(obs[k]), 0, o, k * n * obsSize, n * obsSize);
                System.arraycopy(envs[k].state(), 0, s, k * stateSize, stateSize);
            }
            float[] logits = actor.forward(Var.of(Tensor.of(Shape.of(e * n, obsSize), o)), false).value().toArray();
            float[] v = centralised ? values(s, e) : values(o, e * n);
            int[] act = new int[e * n];
            float[] logp = new float[e * n];
            for (int i = 0; i < e * n; i++) {
                double[] p = softmax(logits, i * actions, actions);
                act[i] = sample(p);
                logp[i] = (float) Math.log(p[act[i]]);
            }
            float[] rew = new float[e];
            for (int k = 0; k < e; k++) {
                int[] a = java.util.Arrays.copyOfRange(act, k * n, (k + 1) * n);
                MultiAgentEnvironment.Step st = envs[k].step(a);
                obs[k] = st.observations();
                rew[k] = st.reward();
                done |= st.done();
            }
            obsRows.add(o);
            stateRows.add(s);
            actionRows.add(act);
            logpRows.add(logp);
            rewardRows.add(rew);
            valueRows.add(v);
            steps += e;
        }
        int t = obsRows.size();
        // bootstrap the time-limit cut with the final value
        float[] lastO = new float[e * n * obsSize];
        float[] lastS = new float[e * stateSize];
        for (int k = 0; k < e; k++) {
            System.arraycopy(flatten(obs[k]), 0, lastO, k * n * obsSize, n * obsSize);
            System.arraycopy(envs[k].state(), 0, lastS, k * stateSize, stateSize);
        }
        float[] lastV = centralised ? values(lastS, e) : values(lastO, e * n);
        int samples = t * e * n;
        float[] adv = new float[samples];
        float[] ret = new float[samples];
        int vPerStep = centralised ? e : e * n;
        float[] gae = new float[vPerStep];
        float[] retC = new float[t * vPerStep];
        for (int step = t - 1; step >= 0; step--) {
            float[] v = valueRows.get(step);
            float[] next = step == t - 1 ? lastV : valueRows.get(step + 1);
            for (int j = 0; j < vPerStep; j++) {
                int k = centralised ? j : j / n;
                float delta = rewardRows.get(step)[k] + gamma * next[j] - v[j];
                gae[j] = delta + gamma * lambda * gae[j];
                retC[step * vPerStep + j] = gae[j] + v[j];
                for (int i = 0; i < (centralised ? n : 1); i++) {
                    int idx = step * e * n + (centralised ? j * n + i : j);
                    adv[idx] = gae[j];
                    ret[idx] = gae[j] + v[j];
                }
            }
        }
        updateStats(retC);
        normalise(adv);
        float[] o = new float[samples * obsSize];
        int[] a = new int[samples];
        float[] oldLogp = new float[samples];
        for (int step = 0; step < t; step++) {
            System.arraycopy(obsRows.get(step), 0, o, step * e * n * obsSize, e * n * obsSize);
            System.arraycopy(actionRows.get(step), 0, a, step * e * n, e * n);
            System.arraycopy(logpRows.get(step), 0, oldLogp, step * e * n, e * n);
        }
        float[] cIn;
        int cRows;
        if (centralised) {
            cRows = t * e;
            cIn = new float[cRows * stateSize];
            for (int step = 0; step < t; step++) {
                System.arraycopy(stateRows.get(step), 0, cIn, step * e * stateSize, e * stateSize);
            }
        } else {
            cRows = samples;
            cIn = o;
        }
        float[] cTarget = new float[cRows];
        double std = Math.sqrt(retVar) + 1e-6;
        for (int i = 0; i < cRows; i++) {
            cTarget[i] = (float) ((retC[i] - retMean) / std);
        }
        float[] onehot = new float[samples * actions];
        for (int i = 0; i < samples; i++) {
            onehot[i * actions + a[i]] = 1;
        }
        Tensor oT = Tensor.of(Shape.of(samples, obsSize), o);
        Tensor maskT = Tensor.of(Shape.of(samples, actions), onehot);
        Tensor oldT = Tensor.of(Shape.of(samples, 1), oldLogp);
        Tensor advT = Tensor.of(Shape.of(samples, 1), adv);
        Tensor cInT = Tensor.of(Shape.of(cRows, centralised ? stateSize : obsSize), cIn);
        Tensor cTargetT = Tensor.of(Shape.of(cRows, 1), cTarget);
        for (int ep = 0; ep < epochs; ep++) {
            Var logp = RlTensors.logSoftmax(actor.forward(Var.of(oT), true));
            Var chosen = logp.mul(maskT).sum(1, true);
            Var ratio = chosen.sub(oldT).exp();
            Var surrogate = ratio.mul(advT).minimum(ratio.clamp(1 - clip, 1 + clip).mul(advT));
            Var ent = logp.exp().mul(logp).sum(1, true).neg();
            surrogate.add(ent.mul(entropy)).mean().neg().backward();
            RlTensors.clip(actor.parameters(), 0.5f);
            actorOpt.step(actor.parameters());
            critic.forward(Var.of(cInT), true).sub(cTargetT).square().mean().backward();
            RlTensors.clip(critic.parameters(), 0.5f);
            criticOpt.step(critic.parameters());
        }
    }

    private void updateStats(float[] returns) {
        for (float r : returns) {
            statCount++;
            double d = r - retMean;
            double w = Math.max(1.0 / statCount, 1e-3);
            retMean += w * d;
            retVar = (1 - w) * (retVar + w * d * d);
        }
    }

    private static void normalise(float[] x) {
        double m = 0;
        for (float v : x) {
            m += v;
        }
        m /= x.length;
        double s = 0;
        for (float v : x) {
            s += (v - m) * (v - m);
        }
        s = Math.sqrt(s / x.length) + 1e-8;
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) ((x[i] - m) / s);
        }
    }

    private static double[] softmax(float[] logits, int offset, int count) {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            max = Math.max(max, logits[offset + i]);
        }
        double[] p = new double[count];
        double z = 0;
        for (int i = 0; i < count; i++) {
            p[i] = Math.exp(logits[offset + i] - max);
            z += p[i];
        }
        for (int i = 0; i < count; i++) {
            p[i] /= z;
        }
        return p;
    }

    private int sample(double[] p) {
        double u = random.nextDouble();
        for (int i = 0; i < p.length; i++) {
            u -= p[i];
            if (u <= 0) {
                return i;
            }
        }
        return p.length - 1;
    }

    /**
     * Picks a joint action.
     *
     * @param observations every agent's observation
     * @param explore sample from the policy rather than take its most likely action
     * @return one action per agent
     */
    public int[] act(float[][] observations, boolean explore) {
        int n = observations.length;
        float[] logits = actor.forward(Var.of(Tensor.of(Shape.of(n, obsSize), flatten(observations))), false).value()
                .toArray();
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            double[] p = softmax(logits, i * actions, actions);
            out[i] = explore ? sample(p) : RlTensors.argmax(p);
        }
        return out;
    }

    /**
     * Mean undiscounted team return of the greedy joint policy.
     *
     * @param episodes episodes
     * @param seed the seed
     * @return the mean return
     */
    public double evaluate(int episodes, long seed) {
        return MultiAgentEvaluation.teamReturn(factory.get(), this::act, episodes, seed);
    }
}
