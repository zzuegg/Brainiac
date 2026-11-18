package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * IMPALA's V-trace actor-critic (Espeholt et al., 2018,
 * <a href="https://arxiv.org/abs/1802.01561">arXiv:1802.01561</a>), with a simulated
 * actor-learner lag: the actors collect experience with the policy of {@link #lag} updates
 * ago, as decoupled IMPALA actors do, and the learner corrects for the mismatch with
 * truncated importance weights. With target policy {@code pi}, behaviour {@code mu},
 * {@code rho_t = min(rhoBar, pi/mu)} and {@code c_t = lambda min(cBar, pi/mu)}:
 * <pre>
 * v_s = V(x_s) + sum_t gamma^(t-s) (prod_{i&lt;t} c_i) rho_t (r_t + gamma V(x_{t+1}) - V(x_t))
 * </pre>
 * is the value target, and {@code rho_s (r_s + gamma v_{s+1} - V(x_s))} the policy-gradient
 * advantage. {@link Correction#NONE} ignores the lag (plain n-step A2C), and
 * {@link Correction#IMPORTANCE_SAMPLING} weights the policy gradient by the untruncated
 * one-step ratio. {@link #retrace} gives the off-policy Q counterpart, Retrace(lambda)
 * (Munos et al., 2016, <a href="https://arxiv.org/abs/1606.02647">arXiv:1606.02647</a>).
 */
public final class VTraceLearner implements Learner {

    /** How off-policy experience is corrected. */
    public enum Correction {
        /** V-trace with truncated importance weights. */
        VTRACE,
        /** No correction: on-policy n-step returns, as A2C. */
        NONE,
        /** The policy gradient weighted by the plain one-step ratio; uncorrected value targets. */
        IMPORTANCE_SAMPLING
    }

    private final Model policy;
    private final Model value;
    private final Model behaviour;
    private final PolicyDistribution dist;
    private final PolicyDistribution behaviourDist;
    private Correction correction = Correction.VTRACE;
    private Optimizer optimizer = Optimizer.adam(2e-3f);
    private int lag;
    private int unroll = 16;
    private float gamma = 0.99f;
    private float lambda = 1f;
    private float rhoBar = 1f;
    private float cBar = 1f;
    private float entropy = 0.01f;
    private SplittableRandom random = new SplittableRandom(0);
    private final Deque<float[]> snapshots = new ArrayDeque<>();
    private double meanRho = Double.NaN;

    private VTraceLearner(Supplier<Model> policy, Model value, Space actions) {
        this.policy = policy.get();
        this.behaviour = policy.get();
        this.value = Objects.requireNonNull(value);
        this.dist = new PolicyDistribution(actions, -0.5f);
        this.behaviourDist = new PolicyDistribution(actions, -0.5f);
    }

    /**
     * @param policy makes the policy network (called twice: learner and actor copy)
     * @param value observations to one value
     * @param actions the action space
     * @return the learner: V-trace, no lag, 16-step unrolls
     */
    public static VTraceLearner of(Supplier<Model> policy, Model value, Space actions) {
        return new VTraceLearner(Objects.requireNonNull(policy), value, actions);
    }

    /** @param c the correction @return this learner */
    public VTraceLearner correction(Correction c) {
        this.correction = Objects.requireNonNull(c);
        return this;
    }

    /** @param updates how many updates the actors' policy lags behind the learner @return this learner */
    public VTraceLearner lag(int updates) {
        this.lag = Math.max(0, updates);
        return this;
    }

    /** @param steps steps per environment per update @return this learner */
    public VTraceLearner unroll(int steps) {
        this.unroll = steps;
        return this;
    }

    /** @param discount the discount @return this learner */
    public VTraceLearner discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /**
     * @param rho the truncation of the importance weight in the TD error
     * @param c the truncation of the trace coefficients
     * @return this learner
     */
    public VTraceLearner truncation(float rho, float c) {
        this.rhoBar = rho;
        this.cBar = c;
        return this;
    }

    /** @param o the optimizer @return this learner */
    public VTraceLearner optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param s the seed @return this learner */
    public VTraceLearner seed(long s) {
        this.random = new SplittableRandom(s);
        return this;
    }

    /** @return the mean truncated importance weight of the last update */
    public double meanRho() {
        return meanRho;
    }

    private List<Var> policyParams(Model m, PolicyDistribution d) {
        List<Var> p = new ArrayList<>(m.parameters());
        if (d.logStd() != null) {
            p.add(d.logStd());
        }
        return p;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return dist.toEnvironment(dist.sample(AgentKit.predict(policy, observation), random, explore));
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        return fit(() -> env, 1, steps, listener);
    }

    /**
     * Trains with several actor environments stepped in turn.
     *
     * @param environments makes each actor's environment
     * @param actors how many
     * @param steps total environment steps
     * @param listener told about every finished episode
     * @return every finished episode
     */
    public RlResult fit(Supplier<? extends Environment> environments, int actors, long steps, RlListener listener) {
        Tracker tracker = new Tracker(listener);
        List<Environment> envs = new ArrayList<>();
        float[][] obs = new float[actors][];
        for (int i = 0; i < actors; i++) {
            envs.add(environments.get());
            obs[i] = envs.get(i).reset(random.split());
        }
        List<Var> learnerParams = policyParams(policy, dist);
        List<Var> actorParams = policyParams(behaviour, behaviourDist);
        int actSize = dist.actionSize();
        while (tracker.steps() < steps) {
            snapshots.addLast(TrustRegion.flatten(learnerParams));
            while (snapshots.size() > lag + 1) {
                snapshots.removeFirst();
            }
            TrustRegion.assign(actorParams, snapshots.peekFirst());
            int n = actors * unroll;
            List<float[]> o = new ArrayList<>(n);
            float[] a = new float[n * actSize];
            float[] mu = new float[n];
            float[] r = new float[n];
            float[] done = new float[n];
            float[][] bootObs = new float[actors][];
            // sequences are stored actor by actor: index = actor * unroll + t
            float[][][] seqObs = new float[actors][unroll][];
            for (int t = 0; t < unroll; t++) {
                for (int i = 0; i < actors; i++) {
                    int k = i * unroll + t;
                    Var out = behaviour.forward(Var.of(AgentKit.row(obs[i])), false);
                    float[] act = behaviourDist.sample(AgentKit.values(out), random, true);
                    mu[k] = behaviourDist.logProb(out, Tensor.of(Shape.of(1, actSize), act)).value().item();
                    System.arraycopy(act, 0, a, k * actSize, actSize);
                    seqObs[i][t] = obs[i];
                    Environment.Step s = envs.get(i).step(behaviourDist.toEnvironment(act));
                    float rew = s.reward();
                    if (s.truncated() && !s.terminated()) {
                        rew += gamma * AgentKit.predict(value, s.observation())[0];
                    }
                    r[k] = rew;
                    done[k] = s.done() ? 1 : 0;
                    obs[i] = tracker.step(s) ? envs.get(i).reset(random.split()) : s.observation();
                }
            }
            for (int i = 0; i < actors; i++) {
                bootObs[i] = obs[i];
                for (int t = 0; t < unroll; t++) {
                    o.add(seqObs[i][t]);
                }
            }
            update(o, a, mu, r, done, bootObs, actors, actSize);
        }
        return tracker.result();
    }

    private void update(List<float[]> o, float[] a, float[] mu, float[] r, float[] done, float[][] boot, int actors,
            int actSize) {
        int n = o.size();
        Tensor x = AgentKit.rows(o);
        Var out = policy.forward(Var.of(x), true);
        Var logp = dist.logProb(out, Tensor.of(Shape.of(n, actSize), a));
        Var v = value.forward(Var.of(x), true);
        float[] pi = logp.value().toArray();
        float[] values = v.value().toArray();
        float[] bootV = new float[actors];
        for (int i = 0; i < actors; i++) {
            bootV[i] = AgentKit.predict(value, boot[i])[0];
        }
        float[] targets = new float[n];
        float[] advantages = new float[n];
        double rhoSum = 0;
        for (int i = 0; i < actors; i++) {
            int from = i * unroll;
            double[] ratio = new double[unroll];
            for (int t = 0; t < unroll; t++) {
                ratio[t] = Math.exp(pi[from + t] - mu[from + t]);
            }
            double[] rho = new double[unroll];
            double[] c = new double[unroll];
            double[] pgWeight = new double[unroll];
            for (int t = 0; t < unroll; t++) {
                switch (correction) {
                    case VTRACE -> {
                        rho[t] = Math.min(rhoBar, ratio[t]);
                        c[t] = lambda * Math.min(cBar, ratio[t]);
                        pgWeight[t] = rho[t];
                    }
                    case NONE -> {
                        rho[t] = 1;
                        c[t] = lambda;
                        pgWeight[t] = 1;
                    }
                    case IMPORTANCE_SAMPLING -> {
                        rho[t] = 1;
                        c[t] = lambda;
                        pgWeight[t] = Math.min(ratio[t], 100);
                    }
                }
                rhoSum += pgWeight[t];
            }
            double[] vs = vtrace(slice(values, from, unroll), bootV[i], slice(r, from, unroll),
                    slice(done, from, unroll), rho, c, gamma);
            for (int t = 0; t < unroll; t++) {
                int k = from + t;
                double next = t == unroll - 1 ? bootV[i] : vs[t + 1];
                double live = 1 - done[k];
                targets[k] = (float) vs[t];
                advantages[k] = (float) (pgWeight[t] * (r[k] + gamma * live * next - values[k]));
            }
        }
        meanRho = rhoSum / n;
        Var loss = logp.mul(AgentKit.column(advantages)).mean().neg()
                .add(v.sub(AgentKit.column(targets)).square().mean().mul(0.25f))
                .sub(dist.entropy(out).mean().mul(entropy));
        loss.backward();
        List<Var> params = policyParams(policy, dist);
        params.addAll(value.parameters());
        RlTensors.clip(params, 0.5f);
        optimizer.step(params);
    }

    private static double[] slice(float[] x, int from, int n) {
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[from + i];
        }
        return out;
    }

    /**
     * V-trace value targets for one sequence.
     *
     * @param values {@code V(x_t)} for every step
     * @param bootstrap {@code V(x_T)} after the last step
     * @param rewards the rewards
     * @param dones 1 where the episode ended after the step
     * @param rho the truncated importance weights {@code min(rhoBar, pi/mu)}
     * @param c the trace coefficients {@code lambda min(cBar, pi/mu)}
     * @param gamma the discount
     * @return the targets {@code v_s}
     */
    public static double[] vtrace(double[] values, double bootstrap, double[] rewards, double[] dones, double[] rho,
            double[] c, double gamma) {
        int n = values.length;
        double[] vs = new double[n];
        double nextVs = bootstrap;
        double nextV = bootstrap;
        for (int t = n - 1; t >= 0; t--) {
            double live = 1 - dones[t];
            double delta = rho[t] * (rewards[t] + gamma * live * nextV - values[t]);
            vs[t] = values[t] + delta + gamma * live * c[t] * (nextVs - nextV);
            nextVs = vs[t];
            nextV = values[t];
        }
        return vs;
    }

    /**
     * Retrace(lambda) targets for {@code Q(x_t, a_t)} along one sequence of off-policy
     * experience: {@code Q_ret(t) = r_t + gamma (E_pi Q(x_{t+1}, .) + c_{t+1} (Q_ret(t+1) - Q(x_{t+1}, a_{t+1})))}
     * with {@code c = lambda min(1, pi/mu)}. Safe for any behaviour policy, and efficient
     * when it is close to the target.
     *
     * @param q {@code Q(x_t, a_t)} for every step
     * @param expectedNext {@code E_{a~pi} Q(x_{t+1}, a)} for every step (0 after a terminal)
     * @param ratios {@code pi(a_t|x_t) / mu(a_t|x_t)} for every step
     * @param rewards the rewards
     * @param dones 1 where the episode ended after the step
     * @param gamma the discount
     * @param lambda the trace decay
     * @return the targets
     */
    public static double[] retrace(double[] q, double[] expectedNext, double[] ratios, double[] rewards, double[] dones,
            double gamma, double lambda) {
        int n = q.length;
        double[] out = new double[n];
        double next = 0;
        for (int t = n - 1; t >= 0; t--) {
            double live = 1 - dones[t];
            double target = rewards[t] + gamma * live * expectedNext[t];
            if (t < n - 1 && live > 0) {
                double c = lambda * Math.min(1, ratios[t + 1]);
                target += gamma * c * (next - q[t + 1]);
            }
            out[t] = target;
            next = target;
        }
        return out;
    }

    @Override
    public String toString() {
        return "V-trace(" + correction + ", lag " + lag + ")";
    }
}
