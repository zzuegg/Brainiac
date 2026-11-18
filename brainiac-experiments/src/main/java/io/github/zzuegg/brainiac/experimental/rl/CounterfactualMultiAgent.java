package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
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
 * COMA, counterfactual multi-agent policy gradients: decentralised actors (one shared
 * network, told apart by the agent id in the observation) trained with one centralised
 * critic {@code Q(s, u)} over the global state and the joint action. The critic is built to
 * output, for agent {@code i}, the values of all of {@code i}'s actions with the other agents'
 * actions held fixed, so the counterfactual advantage
 * {@code A_i = Q(s, u) - sum over a of pi_i(a) Q(s, (u_-i, a))} costs one forward pass. It
 * marginalises out only agent {@code i}'s action, which assigns the shared team reward to the
 * agents that caused it.
 *
 * <p>The critic is trained on on-policy batches with TD(lambda) targets from a target
 * network; the actors take one policy-gradient step per batch with an entropy bonus.
 *
 * <p>Foerster et al. 2018, "Counterfactual Multi-Agent Policy Gradients",
 * <a href="https://arxiv.org/abs/1705.08926">arXiv:1705.08926</a>.
 */
public final class CounterfactualMultiAgent {

    private final Supplier<? extends MultiAgentEnvironment> factory;
    private final MultiAgentEnvironment[] envs;
    private final int agents;
    private final int obsSize;
    private final int stateSize;
    private final int actions;
    private final int criticIn;
    private final Model actor;
    private final Model critic;
    private final Model target;
    private final Optimizer actorOpt = Optimizer.adam(1e-3f);
    private final Optimizer criticOpt = Optimizer.adam(1e-3f);
    private final SplittableRandom random;
    private float gamma = 0.99f;
    private float lambda = 0.8f;
    private float entropy = 0.01f;
    private int criticEpochs = 8;
    private float rewardScale = 0.1f;
    private long steps;

    private CounterfactualMultiAgent(Supplier<? extends MultiAgentEnvironment> factory, int hidden, int parallel,
            long seed) {
        this.factory = factory;
        this.envs = new MultiAgentEnvironment[parallel];
        for (int e = 0; e < parallel; e++) {
            envs[e] = factory.get();
        }
        this.agents = envs[0].agents();
        this.obsSize = envs[0].observationSize();
        this.stateSize = envs[0].stateSize();
        this.actions = envs[0].actions();
        this.criticIn = stateSize + (agents - 1) * actions + agents;
        this.actor = Nets.mlp(obsSize, actions, Activation.TANH, seed, hidden, hidden);
        this.critic = Nets.mlp(criticIn, actions, Activation.RELU, seed + 1, hidden, hidden);
        this.target = Nets.mlp(criticIn, actions, Activation.RELU, seed + 1, hidden, hidden);
        RlTensors.copy(critic.parameters(), target.parameters());
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param factory creates environments (one per parallel episode)
     * @param hidden hidden units
     * @param seed the seed
     * @return the learner, with 16 parallel environments
     */
    public static CounterfactualMultiAgent of(Supplier<? extends MultiAgentEnvironment> factory, int hidden,
            long seed) {
        return new CounterfactualMultiAgent(Objects.requireNonNull(factory), hidden, 16, seed);
    }

    /** @return joint environment steps so far */
    public long steps() {
        return steps;
    }

    private float[] criticRow(float[] state, int[] joint, int offset, int agent) {
        float[] row = new float[criticIn];
        System.arraycopy(state, 0, row, 0, stateSize);
        int k = stateSize;
        for (int j = 0; j < agents; j++) {
            if (j != agent) {
                row[k + joint[offset + j]] = 1;
                k += actions;
            }
        }
        row[k + agent] = 1;
        return row;
    }

    /**
     * Trains for at least a number of joint environment steps.
     *
     * @param jointSteps joint steps (summed over the parallel environments)
     * @return this
     */
    public CounterfactualMultiAgent fit(long jointSteps) {
        long end = steps + jointSteps;
        while (steps < end) {
            iteration();
        }
        return this;
    }

    private void iteration() {
        int e = envs.length;
        int n = agents;
        List<float[]> obsRows = new ArrayList<>();
        List<float[]> stateRows = new ArrayList<>();
        List<int[]> actionRows = new ArrayList<>();
        List<float[]> rewardRows = new ArrayList<>();
        float[][][] obs = new float[e][][];
        for (int k = 0; k < e; k++) {
            obs[k] = envs[k].reset(random.split());
        }
        boolean done = false;
        while (!done) {
            float[] o = new float[e * n * obsSize];
            float[] s = new float[e * stateSize];
            for (int k = 0; k < e; k++) {
                for (int i = 0; i < n; i++) {
                    System.arraycopy(obs[k][i], 0, o, (k * n + i) * obsSize, obsSize);
                }
                System.arraycopy(envs[k].state(), 0, s, k * stateSize, stateSize);
            }
            float[] logits = actor.forward(Var.of(Tensor.of(Shape.of(e * n, obsSize), o)), false).value().toArray();
            int[] act = new int[e * n];
            for (int i = 0; i < e * n; i++) {
                act[i] = sample(softmax(logits, i * actions, actions));
            }
            float[] rew = new float[e];
            for (int k = 0; k < e; k++) {
                MultiAgentEnvironment.Step st = envs[k].step(java.util.Arrays.copyOfRange(act, k * n, (k + 1) * n));
                obs[k] = st.observations();
                rew[k] = st.reward();
                done |= st.done();
            }
            obsRows.add(o);
            stateRows.add(s);
            actionRows.add(act);
            rewardRows.add(rew);
            steps += e;
        }
        int t = obsRows.size();
        int samples = t * e * n;
        float[] cIn = new float[samples * criticIn];
        float[] taken = new float[samples * actions];
        for (int step = 0; step < t; step++) {
            for (int k = 0; k < e; k++) {
                for (int i = 0; i < n; i++) {
                    int idx = (step * e + k) * n + i;
                    float[] row = criticRow(java.util.Arrays.copyOfRange(stateRows.get(step), k * stateSize,
                            (k + 1) * stateSize), actionRows.get(step), k * n, i);
                    System.arraycopy(row, 0, cIn, idx * criticIn, criticIn);
                    taken[idx * actions + actionRows.get(step)[k * n + i]] = 1;
                }
            }
        }
        Tensor cInT = Tensor.of(Shape.of(samples, criticIn), cIn);
        Tensor takenT = Tensor.of(Shape.of(samples, actions), taken);
        // TD(lambda) targets of Q(s, u) from the target network; the episode end is a time limit
        float[] qT = target.forward(Var.of(cInT), false).value().toArray();
        float[] qTaken = new float[samples];
        for (int i = 0; i < samples; i++) {
            int a = -1;
            for (int b = 0; b < actions; b++) {
                if (taken[i * actions + b] > 0) {
                    a = b;
                }
            }
            qTaken[i] = qT[i * actions + a];
        }
        float[] targets = new float[samples];
        for (int k = 0; k < e; k++) {
            for (int i = 0; i < n; i++) {
                float g = qTaken[((t - 1) * e + k) * n + i];
                for (int step = t - 1; step >= 0; step--) {
                    int idx = (step * e + k) * n + i;
                    float next = step == t - 1 ? qTaken[idx] : qTaken[((step + 1) * e + k) * n + i];
                    g = rewardScale * rewardRows.get(step)[k] + gamma * ((1 - lambda) * next + lambda * g);
                    targets[idx] = g;
                }
            }
        }
        Tensor targetT = Tensor.of(Shape.of(samples, 1), targets);
        for (int ep = 0; ep < criticEpochs; ep++) {
            critic.forward(Var.of(cInT), true).mul(takenT).sum(1, true).sub(targetT).square().mean().backward();
            RlTensors.clip(critic.parameters(), 1f);
            criticOpt.step(critic.parameters());
        }
        RlTensors.copy(critic.parameters(), target.parameters());
        float[] q = critic.forward(Var.of(cInT), false).value().toArray();
        float[] o = new float[samples * obsSize];
        for (int step = 0; step < t; step++) {
            System.arraycopy(obsRows.get(step), 0, o, step * e * n * obsSize, e * n * obsSize);
        }
        Tensor oT = Tensor.of(Shape.of(samples, obsSize), o);
        Var logp = RlTensors.logSoftmax(actor.forward(Var.of(oT), true));
        float[] pi = logp.value().exp().toArray();
        float[] adv = new float[samples];
        for (int i = 0; i < samples; i++) {
            double baseline = 0;
            double chosen = 0;
            for (int a = 0; a < actions; a++) {
                baseline += pi[i * actions + a] * q[i * actions + a];
                chosen += taken[i * actions + a] * q[i * actions + a];
            }
            adv[i] = (float) (chosen - baseline);
        }
        double scale = 0;
        for (float a : adv) {
            scale += a * a;
        }
        scale = Math.sqrt(scale / samples) + 1e-6;
        for (int i = 0; i < samples; i++) {
            adv[i] /= (float) scale;
        }
        Var ent = logp.exp().mul(logp).sum(1, true).neg();
        logp.mul(takenT).sum(1, true).mul(Tensor.of(Shape.of(samples, 1), adv)).add(ent.mul(entropy)).mean().neg()
                .backward();
        RlTensors.clip(actor.parameters(), 0.5f);
        actorOpt.step(actor.parameters());
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
     * @param explore sample rather than take the most likely action
     * @return one action per agent
     */
    public int[] act(float[][] observations, boolean explore) {
        int n = observations.length;
        float[] in = new float[n * obsSize];
        for (int i = 0; i < n; i++) {
            System.arraycopy(observations[i], 0, in, i * obsSize, obsSize);
        }
        float[] logits = actor.forward(Var.of(Tensor.of(Shape.of(n, obsSize), in)), false).value().toArray();
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            double[] p = softmax(logits, i * actions, actions);
            out[i] = explore ? sample(p) : RlTensors.argmax(p);
        }
        return out;
    }

    /**
     * @param episodes episodes
     * @param seed the seed
     * @return the mean greedy team return
     */
    public double evaluate(int episodes, long seed) {
        return MultiAgentEvaluation.teamReturn(factory.get(), this::act, episodes, seed);
    }
}
