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
 * MADDPG, multi-agent deep deterministic policy gradients: centralised training with
 * decentralised execution for continuous actions. Each agent acts with a deterministic actor
 * {@code mu(o_i)} on its own observation (the DDPG/TD3 actor, a tanh-squashed MLP), and is
 * trained through a critic {@code Q_i(s, a_1 ... a_n)} that sees the global state and every
 * agent's action. Because the critic conditions on the other agents' actions, its target does
 * not move when they change their policies, which is what makes independent DDPG unstable.
 * Off-policy replay, target networks with soft updates and Gaussian exploration noise as in
 * DDPG. Actors and critics are shared across agents with the agent id as input (the common
 * parameter-sharing variant), so each update is one batched call.
 *
 * <p>Works on {@link CooperativeNavigation} (its {@link CooperativeNavigation#stepContinuous}).
 *
 * <p>Lowe et al. 2017, "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive
 * Environments", <a href="https://arxiv.org/abs/1706.02275">arXiv:1706.02275</a>.
 */
public final class Maddpg {

    private final Supplier<CooperativeNavigation> factory;
    private final CooperativeNavigation[] envs;
    private final int n;
    private final int obsSize;
    private final int stateSize;
    private final int criticIn;
    private final Model actor;
    private final Model actorTarget;
    private final Model critic;
    private final Model criticTarget;
    private final Optimizer actorOpt = Optimizer.adam(1e-3f);
    private final Optimizer criticOpt = Optimizer.adam(1e-3f);
    private final SplittableRandom random;
    private final int capacity = 50_000;
    private final float[][] bState;
    private final float[][] bObs;
    private final float[][] bAct;
    private final float[] bRew;
    private final float[][] bNextState;
    private final float[][] bNextObs;
    private int size;
    private int pos;
    private float gamma = 0.95f;
    private float tau = 0.02f;
    private float noise = 0.3f;
    private int batch = 64;
    private int updateEvery = 2;
    private long steps;

    private Maddpg(Supplier<CooperativeNavigation> factory, int hidden, int parallel, long seed) {
        this.factory = factory;
        this.envs = new CooperativeNavigation[parallel];
        for (int e = 0; e < parallel; e++) {
            envs[e] = factory.get();
        }
        this.n = envs[0].agents();
        this.obsSize = envs[0].observationSize();
        this.stateSize = envs[0].stateSize();
        this.criticIn = stateSize + 2 * n + n;
        this.actor = Nets.mlp(obsSize, 2, Activation.RELU, seed, hidden, hidden);
        this.actorTarget = Nets.mlp(obsSize, 2, Activation.RELU, seed, hidden, hidden);
        this.critic = Nets.mlp(criticIn, 1, Activation.RELU, seed + 1, hidden, hidden);
        this.criticTarget = Nets.mlp(criticIn, 1, Activation.RELU, seed + 1, hidden, hidden);
        RlTensors.copy(actor.parameters(), actorTarget.parameters());
        RlTensors.copy(critic.parameters(), criticTarget.parameters());
        this.random = new SplittableRandom(seed);
        bState = new float[capacity][];
        bObs = new float[capacity][];
        bAct = new float[capacity][];
        bRew = new float[capacity];
        bNextState = new float[capacity][];
        bNextObs = new float[capacity][];
    }

    /**
     * @param factory creates environments (one per parallel episode)
     * @param hidden hidden units
     * @param seed the seed
     * @return the learner, with 8 parallel environments
     */
    public static Maddpg of(Supplier<CooperativeNavigation> factory, int hidden, long seed) {
        return new Maddpg(Objects.requireNonNull(factory), hidden, 8, seed);
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

    private float[] forces(Model net, float[] obs, int rows) {
        return net.forward(Var.of(Tensor.of(Shape.of(rows, obsSize), obs)), false).value().tanh().toArray();
    }

    /**
     * Trains for at least a number of joint environment steps.
     *
     * @param jointSteps joint steps (summed over the parallel environments)
     * @return this
     */
    public Maddpg fit(long jointSteps) {
        long end = steps + jointSteps;
        int e = envs.length;
        while (steps < end) {
            float[][][] obs = new float[e][][];
            for (int k = 0; k < e; k++) {
                obs[k] = envs[k].reset(random.split());
            }
            boolean done = false;
            int t = 0;
            while (!done) {
                float[] o = new float[e * n * obsSize];
                for (int k = 0; k < e; k++) {
                    System.arraycopy(flatten(obs[k]), 0, o, k * n * obsSize, n * obsSize);
                }
                float[] a = forces(actor, o, e * n);
                for (int i = 0; i < a.length; i++) {
                    a[i] = (float) Math.max(-1, Math.min(1, a[i] + noise * random.nextGaussian()));
                }
                for (int k = 0; k < e; k++) {
                    float[] s = envs[k].state();
                    float[][] f = new float[n][];
                    for (int i = 0; i < n; i++) {
                        f[i] = new float[] {a[2 * (k * n + i)], a[2 * (k * n + i) + 1]};
                    }
                    MultiAgentEnvironment.Step st = envs[k].stepContinuous(f);
                    bState[pos] = s;
                    bObs[pos] = java.util.Arrays.copyOfRange(o, k * n * obsSize, (k + 1) * n * obsSize);
                    bAct[pos] = java.util.Arrays.copyOfRange(a, 2 * k * n, 2 * (k + 1) * n);
                    bRew[pos] = st.reward();
                    bNextState[pos] = envs[k].state();
                    bNextObs[pos] = flatten(st.observations());
                    pos = (pos + 1) % capacity;
                    size = Math.min(size + 1, capacity);
                    obs[k] = st.observations();
                    done |= st.done();
                }
                steps += e;
                if (++t % updateEvery == 0 && size >= 4 * batch) {
                    update();
                }
            }
        }
        return this;
    }

    private void update() {
        int rows = batch * n;
        float[] s = new float[rows * stateSize];
        float[] ns = new float[rows * stateSize];
        float[] o = new float[rows * obsSize];
        float[] no = new float[rows * obsSize];
        float[] act = new float[rows * 2 * n];
        float[] mask = new float[rows * 2 * n];
        float[] id = new float[rows * n];
        float[] r = new float[rows];
        for (int b = 0; b < batch; b++) {
            int k = random.nextInt(size);
            for (int i = 0; i < n; i++) {
                int row = b * n + i;
                System.arraycopy(bState[k], 0, s, row * stateSize, stateSize);
                System.arraycopy(bNextState[k], 0, ns, row * stateSize, stateSize);
                System.arraycopy(bAct[k], 0, act, row * 2 * n, 2 * n);
                mask[row * 2 * n + 2 * i] = 1;
                mask[row * 2 * n + 2 * i + 1] = 1;
                id[row * n + i] = 1;
                r[row] = bRew[k];
            }
            // every agent's own observation, and all next observations for the target actions
            for (int i = 0; i < n; i++) {
                System.arraycopy(bObs[k], i * obsSize, o, (b * n + i) * obsSize, obsSize);
                System.arraycopy(bNextObs[k], i * obsSize, no, (b * n + i) * obsSize, obsSize);
            }
        }
        Tensor idT = Tensor.of(Shape.of(rows, n), id);
        Tensor tile = tile();
        // target: r + gamma Q'(s', mu'(o'_1..n), i); next actions of all agents of a sample
        float[] nextA = forces(actorTarget, no, rows);
        float[] nextJoint = new float[rows * 2 * n];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < n; i++) {
                System.arraycopy(nextA, 2 * b * n, nextJoint, (b * n + i) * 2 * n, 2 * n);
            }
        }
        Var nextIn = Var.concat(1, Var.of(Tensor.of(Shape.of(rows, stateSize), ns)),
                Var.of(Tensor.of(Shape.of(rows, 2 * n), nextJoint)), Var.of(idT));
        float[] nq = criticTarget.forward(nextIn, false).value().toArray();
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            y[i] = r[i] + gamma * nq[i];
        }
        Tensor sT = Tensor.of(Shape.of(rows, stateSize), s);
        Tensor actT = Tensor.of(Shape.of(rows, 2 * n), act);
        Var in = Var.concat(1, Var.of(sT), Var.of(actT), Var.of(idT));
        critic.forward(in, true).sub(Tensor.of(Shape.of(rows, 1), y)).square().mean().backward();
        RlTensors.clip(critic.parameters(), 1f);
        criticOpt.step(critic.parameters());
        // actor: replace agent i's action by mu(o_i) and ascend Q_i
        Tensor maskT = Tensor.of(Shape.of(rows, 2 * n), mask);
        Var mu = actor.forward(Var.of(Tensor.of(Shape.of(rows, obsSize), o)), true).tanh();
        Var joint = mu.matmul(Var.of(tile)).mul(maskT).add(actT.mul(maskT.neg().add(1f)));
        Var q = critic.forward(Var.concat(1, Var.of(sT), joint, Var.of(idT)), true);
        q.mean().neg().add(mu.square().mean().mul(1e-3f)).backward();
        RlTensors.clip(actor.parameters(), 1f);
        actorOpt.step(actor.parameters());
        RlTensors.zeroGrad(critic.parameters());
        RlTensors.blend(actor.parameters(), actorTarget.parameters(), tau);
        RlTensors.blend(critic.parameters(), criticTarget.parameters(), tau);
    }

    private Tensor tile() {
        float[] t = new float[2 * 2 * n];
        for (int i = 0; i < n; i++) {
            t[2 * i] = 1;
            t[2 * n + 2 * i + 1] = 1;
        }
        return Tensor.of(Shape.of(2, 2 * n), t);
    }

    /**
     * Picks every agent's force greedily (no noise).
     *
     * @param observations every agent's observation
     * @return one 2-D force per agent
     */
    public float[][] act(float[][] observations) {
        float[] a = forces(actor, flatten(observations), observations.length);
        float[][] out = new float[observations.length][];
        for (int i = 0; i < out.length; i++) {
            out[i] = new float[] {a[2 * i], a[2 * i + 1]};
        }
        return out;
    }

    /**
     * Greedy evaluation.
     *
     * @param episodes episodes
     * @param seed the seed
     * @return {@code {mean team return, mean final coverage}}
     */
    public double[] evaluate(int episodes, long seed) {
        CooperativeNavigation env = factory.get();
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        double covered = 0;
        for (int ep = 0; ep < episodes; ep++) {
            float[][] obs = env.reset(r.split());
            MultiAgentEnvironment.Step st;
            do {
                st = env.stepContinuous(act(obs));
                obs = st.observations();
                total += st.reward();
            } while (!st.done());
            covered += env.coverage();
        }
        return new double[] {total / episodes, covered / episodes};
    }
}
