package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * PQN, parallelised Q-learning (Gallici et al., "Simplifying Deep Temporal Difference Learning",
 * ICLR 2025, <a href="https://arxiv.org/abs/2407.04811">arXiv:2407.04811</a>). LayerNorm (with a
 * little ℓ2 regularisation) makes TD learning stable even off-policy with nonlinear function
 * approximation, so the stabilisers DQN needed go: no target network and no replay buffer.
 * Instead PQN steps many environments at once, computes Q(λ) returns over each short rollout,
 * and takes a few epochs of minibatch updates on it, like PPO with a Q-network.
 *
 * <p>The vectorised runner steps the environments on a fork-join pool when
 * {@link #parallel(boolean)} is on (the environments must be independent objects). With
 * {@link #layerNorm(boolean) layerNorm(false)} the network is a plain MLP: the ablation the
 * paper predicts diverges without a target network.
 */
public final class Pqn implements Learner {

    private final int obsDims;
    private final int actions;
    private final int hidden;
    private boolean layerNorm = true;
    private int envs = 16;
    private int rolloutSteps = 8;
    private int epochs = 4;
    private int minibatches = 4;
    private float lambda = 0.65f;
    private float gamma = 0.99f;
    private float lr = 3e-3f;
    private float l2 = 1e-4f;
    private float epsilonStart = 1f;
    private float epsilonEnd = 0.05f;
    private float epsilonFraction = 0.3f;
    private boolean parallel;
    private SplittableRandom random = new SplittableRandom(0);
    private long seed;
    private Model network;
    private Optimizer optimizer;
    private long updates;
    private long stepsTaken;
    private long plannedSteps = 1;
    private Supplier<? extends Environment> factory;

    private Pqn(int obsDims, int actions, int hidden) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.hidden = hidden;
    }

    /**
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param hidden hidden width (two layers)
     * @return the learner
     */
    public static Pqn of(int observationDims, int actions, int hidden) {
        return new Pqn(observationDims, actions, hidden);
    }

    /** @param on LayerNorm after every hidden layer @return this learner */
    public Pqn layerNorm(boolean on) {
        this.layerNorm = on;
        return this;
    }

    /**
     * @param factory builds each of the parallel environments
     * @param count how many run at once
     * @return this learner
     */
    public Pqn environments(Supplier<? extends Environment> factory, int count) {
        this.factory = factory;
        this.envs = Math.max(1, count);
        return this;
    }

    /** @param steps steps per environment per rollout @return this learner */
    public Pqn rolloutSteps(int steps) {
        this.rolloutSteps = Math.max(1, steps);
        return this;
    }

    /** @param n epochs over each rollout @param batches minibatches per epoch @return this learner */
    public Pqn epochs(int n, int batches) {
        this.epochs = Math.max(1, n);
        this.minibatches = Math.max(1, batches);
        return this;
    }

    /** @param value the Q(lambda) mixing (0 is one-step Q-learning) @return this learner */
    public Pqn lambda(float value) {
        this.lambda = value;
        return this;
    }

    /** @param discount gamma @return this learner */
    public Pqn discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public Pqn learningRate(float rate) {
        this.lr = rate;
        return this;
    }

    /** @param on whether to step the environments in parallel threads @return this learner */
    public Pqn parallel(boolean on) {
        this.parallel = on;
        return this;
    }

    /** @param fraction of the run over which epsilon falls to its floor @return this learner */
    public Pqn epsilonDecay(float fraction) {
        this.epsilonFraction = fraction;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Pqn seed(long seed) {
        this.seed = seed;
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the gradient updates taken */
    public long updates() {
        return updates;
    }

    private void build() {
        if (network != null) {
            return;
        }
        Model.Builder b = Model.sequential().input(obsDims);
        for (int i = 0; i < 2; i++) {
            b.dense(hidden);
            if (layerNorm) {
                b.layerNorm();
            }
            b.activation(Activation.RELU);
        }
        network = b.dense(actions).seed(seed * 1000 + 7).build();
        optimizer = Optimizer.adamW(lr, 0);
    }

    private float[][] values(float[][] obs) {
        Tensor x = RlTensors.rows(obs, obsDims, network.device());
        float[] v = Var.noGrad(() -> network.forward(Var.of(x), false)).value().toArray();
        float[][] out = new float[obs.length][actions];
        for (int i = 0; i < obs.length; i++) {
            System.arraycopy(v, i * actions, out[i], 0, actions);
        }
        return out;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        build();
        return new float[] {RlTensors.argmax(values(new float[][] {observation})[0])};
    }

    private double epsilon() {
        double p = Math.min(1, stepsTaken / (epsilonFraction * plannedSteps));
        return epsilonStart + (epsilonEnd - epsilonStart) * p;
    }

    /**
     * Trains on {@code count} copies of the environment built by the factory given to
     * {@link #environments}; when none was given, the passed environment is used alone.
     */
    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("PQN needs " + actions + " discrete actions");
        }
        build();
        plannedSteps = stepsTaken + steps;
        int n = factory == null ? 1 : envs;
        List<Environment> list = new ArrayList<>();
        if (factory == null) {
            list.add(env);
        } else {
            for (int i = 0; i < n; i++) {
                list.add(factory.get());
            }
        }
        Tracker recorder = new Tracker(listener);
        float[][] obs = new float[n][];
        SplittableRandom[] rngs = new SplittableRandom[n];
        for (int i = 0; i < n; i++) {
            rngs[i] = random.split();
            obs[i] = list.get(i).reset(rngs[i].split());
        }
        ForkJoinPool pool = parallel ? ForkJoinPool.commonPool() : null;
        while (recorder.steps() < steps) {
            int t = rolloutSteps;
            float[][][] o = new float[t][][];
            int[][] a = new int[t][n];
            float[][] r = new float[t][n];
            float[][] notDone = new float[t][n];
            float[][] reset = new float[t][n];
            float[][][] next = new float[t][][];
            for (int k = 0; k < t; k++) {
                o[k] = obs.clone();
                float[][] q = values(obs);
                double eps = epsilon();
                for (int i = 0; i < n; i++) {
                    a[k][i] = random.nextDouble() < eps ? random.nextInt(actions) : RlTensors.argmax(q[i]);
                }
                int kk = k;
                Environment.Step[] res = new Environment.Step[n];
                if (pool != null) {
                    pool.submit(() -> IntStream.range(0, n).parallel()
                            .forEach(i -> res[i] = list.get(i).step(a[kk][i]))).join();
                } else {
                    for (int i = 0; i < n; i++) {
                        res[i] = list.get(i).step(a[k][i]);
                    }
                }
                next[k] = new float[n][];
                for (int i = 0; i < n; i++) {
                    Environment.Step s = res[i];
                    stepsTaken++;
                    r[k][i] = s.reward();
                    notDone[k][i] = s.terminated() ? 0 : 1;
                    next[k][i] = s.observation();
                    boolean ended = recorder.step(s);
                    reset[k][i] = ended ? 1 : 0;
                    obs[i] = ended ? list.get(i).reset(rngs[i].split()) : s.observation();
                }
            }
            learn(o, a, r, notDone, reset, next, n, t);
        }
        return recorder.result();
    }

    /** Q(lambda) targets backwards through the rollout, then epochs of minibatch regression. */
    private void learn(float[][][] o, int[][] a, float[][] r, float[][] notDone, float[][] reset, float[][][] next,
            int n, int t) {
        float[][] target = new float[t][n];
        float[] lambdaReturn = new float[n];
        for (int k = t - 1; k >= 0; k--) {
            float[][] qNext = values(next[k]);
            for (int i = 0; i < n; i++) {
                float maxNext = qNext[i][RlTensors.argmax(qNext[i])];
                boolean cut = reset[k][i] == 1;     // episode ended at this step: no lambda carry-over
                float carry = k == t - 1 || cut ? maxNext : (1 - lambda) * maxNext + lambda * lambdaReturn[i];
                target[k][i] = r[k][i] + gamma * notDone[k][i] * carry;
            }
            System.arraycopy(target[k], 0, lambdaReturn, 0, n);
        }
        int total = n * t;
        float[][] flatObs = new float[total][];
        float[] flatA = new float[total];
        float[] flatY = new float[total];
        for (int k = 0; k < t; k++) {
            for (int i = 0; i < n; i++) {
                int j = k * n + i;
                flatObs[j] = o[k][i];
                flatA[j] = a[k][i];
                flatY[j] = target[k][i];
            }
        }
        int mb = Math.max(1, total / minibatches);
        int[] order = IntStream.range(0, total).toArray();
        for (int e = 0; e < epochs; e++) {
            for (int i = total - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
            for (int start = 0; start + mb <= total; start += mb) {
                float[][] xo = new float[mb][];
                float[] xa = new float[mb];
                float[] xy = new float[mb];
                for (int i = 0; i < mb; i++) {
                    xo[i] = flatObs[order[start + i]];
                    xa[i] = flatA[order[start + i]];
                    xy[i] = flatY[order[start + i]];
                }
                Tensor x = RlTensors.rows(xo, obsDims, network.device());
                Var q = network.forward(Var.of(x), true).mul(RlTensors.oneHot(xa, actions, network.device()))
                        .sum(1, true);
                Var loss = q.sub(RlTensors.column(xy, network.device())).square().mean().mul(0.5f);
                if (l2 > 0) {
                    Var reg = null;
                    for (Var p : network.parameters()) {
                        if (p.shape().rank() == 2) {
                            Var s = p.square().sum();
                            reg = reg == null ? s : reg.add(s);
                        }
                    }
                    if (reg != null) {
                        loss = loss.add(reg.mul(l2 * 0.5f));
                    }
                }
                loss.backward();
                RlTensors.clip(network.parameters(), 10f);
                optimizer.step(network.parameters());
                updates++;
            }
        }
    }

    @Override
    public String toString() {
        return "PQN(" + envs + " envs" + (layerNorm ? ", LayerNorm" : "") + ", lambda " + lambda + ")";
    }
}
