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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/**
 * Asynchronous advantage actor-critic, A3C (Mnih et al., 2016,
 * <a href="https://arxiv.org/abs/1602.01783">arXiv:1602.01783</a>): several worker threads each
 * run their own environment and a local copy of the actor and critic. A worker copies the
 * shared parameters, collects {@code nSteps} transitions, computes the n-step actor-critic
 * gradient on its local copy and applies it to the shared parameters through a shared
 * optimizer, without waiting for the others. Workers therefore act with slightly stale
 * parameters, and the decorrelation between workers replaces a replay buffer.
 *
 * <p>Gradients are computed in parallel; only the short update of the shared parameters
 * holds a lock, because the core optimizers keep unsynchronized per-parameter state (the
 * paper's lock-free Hogwild updates would race on it).
 *
 * <pre>{@code
 * AsyncActorCritic a3c = AsyncActorCritic.of(() -> Nets.mlp(4, 2, 1, 32), () -> Nets.mlp(4, 1, 2, 32),
 *         Space.discrete(2));
 * a3c.fit(CartPole::new, 4, 40_000, RlListener.NONE);
 * }</pre>
 */
public final class AsyncActorCritic implements Learner {

    private final Supplier<Model> policyFactory;
    private final Supplier<Model> valueFactory;
    private final Model policy;
    private final Model value;
    private final Space actions;
    private final PolicyDistribution dist;
    private final Object lock = new Object();
    private Optimizer optimizer = Optimizer.adam(2e-3f);
    private int nSteps = 8;
    private float gamma = 0.99f;
    private float entropy = 0.01f;
    private float valueCoefficient = 0.5f;
    private float maxGradNorm = 0.5f;
    private long seed;
    private SplittableRandom random = new SplittableRandom(0);
    private final AtomicLong updates = new AtomicLong();

    private AsyncActorCritic(Supplier<Model> policy, Supplier<Model> value, Space actions) {
        this.policyFactory = Objects.requireNonNull(policy);
        this.valueFactory = Objects.requireNonNull(value);
        this.policy = policy.get();
        this.value = value.get();
        this.actions = actions;
        this.dist = new PolicyDistribution(actions, -0.5f);
    }

    /**
     * @param policy makes the actor network (called once per worker; every call must give
     *     the same layout)
     * @param value makes the critic network
     * @param actions the action space
     * @return the learner: 8-step returns, Adam 2e-3, entropy bonus 0.01
     */
    public static AsyncActorCritic of(Supplier<Model> policy, Supplier<Model> value, Space actions) {
        return new AsyncActorCritic(policy, value, actions);
    }

    /** @param n steps per worker update @return this learner */
    public AsyncActorCritic nSteps(int n) {
        this.nSteps = n;
        return this;
    }

    /** @param discount the discount @return this learner */
    public AsyncActorCritic discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param coefficient the entropy bonus @return this learner */
    public AsyncActorCritic entropyBonus(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /** @param o the shared optimizer @return this learner */
    public AsyncActorCritic optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param s the seed @return this learner */
    public AsyncActorCritic seed(long s) {
        this.seed = s;
        this.random = new SplittableRandom(s);
        return this;
    }

    /** @return the number of shared-parameter updates applied */
    public long updates() {
        return updates.get();
    }

    private List<Var> shared() {
        List<Var> p = new ArrayList<>(policy.parameters());
        if (dist.logStd() != null) {
            p.add(dist.logStd());
        }
        p.addAll(value.parameters());
        return p;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return dist.toEnvironment(dist.sample(AgentKit.predict(policy, observation), random, explore));
    }

    /** Trains with a single worker on the given environment. */
    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        return fit(() -> env, 1, steps, listener);
    }

    /**
     * Trains with several asynchronous workers.
     *
     * @param environments makes one environment per worker
     * @param workers the number of worker threads
     * @param steps the total environment steps over all workers
     * @param listener told about every finished episode (from any worker)
     * @return every finished episode
     */
    public RlResult fit(Supplier<? extends Environment> environments, int workers, long steps, RlListener listener) {
        Tracker tracker = new Tracker(listener);
        AtomicLong taken = new AtomicLong();
        List<Thread> threads = new ArrayList<>();
        List<Throwable> failures = new ArrayList<>();
        for (int w = 0; w < workers; w++) {
            Environment env = environments.get();
            Worker worker = new Worker(env, new SplittableRandom(seed * 7919 + w));
            Thread t = Thread.ofPlatform().name("a3c-worker-" + w).unstarted(() -> worker.run(tracker, taken, steps));
            t.setUncaughtExceptionHandler((th, e) -> {
                synchronized (failures) {
                    failures.add(e);
                }
            });
            threads.add(t);
        }
        if (workers == 1) {
            threads.get(0).run();
        } else {
            threads.forEach(Thread::start);
            for (Thread t : threads) {
                try {
                    t.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException("interrupted", e);
                }
            }
        }
        if (!failures.isEmpty()) {
            throw new IllegalStateException("a worker failed", failures.get(0));
        }
        return tracker.result();
    }

    private final class Worker {
        final Environment env;
        final SplittableRandom random;
        final Model localPolicy = policyFactory.get();
        final Model localValue = valueFactory.get();
        final PolicyDistribution localDist = new PolicyDistribution(actions, -0.5f);

        Worker(Environment env, SplittableRandom random) {
            this.env = env;
            this.random = random;
        }

        List<Var> local() {
            List<Var> p = new ArrayList<>(localPolicy.parameters());
            if (localDist.logStd() != null) {
                p.add(localDist.logStd());
            }
            p.addAll(localValue.parameters());
            return p;
        }

        void run(Tracker tracker, AtomicLong taken, long steps) {
            List<Var> sharedParams = shared();
            List<Var> localParams = local();
            float[] obs = env.reset(random.split());
            int actSize = localDist.actionSize();
            while (taken.get() < steps) {
                synchronized (lock) {
                    for (int i = 0; i < sharedParams.size(); i++) {
                        localParams.get(i).assign(sharedParams.get(i).value().copy());
                    }
                }
                List<float[]> o = new ArrayList<>();
                List<float[]> a = new ArrayList<>();
                List<Float> r = new ArrayList<>();
                boolean ended = false;
                for (int t = 0; t < nSteps && taken.get() < steps; t++) {
                    float[] act = localDist.sample(AgentKit.predict(localPolicy, obs), random, true);
                    Environment.Step s = env.step(localDist.toEnvironment(act));
                    taken.incrementAndGet();
                    o.add(obs);
                    a.add(act);
                    r.add(s.reward());
                    boolean done;
                    synchronized (tracker) {
                        done = tracker.step(s);
                    }
                    if (done) {
                        ended = true;
                        obs = s.truncated() && !s.terminated() ? s.observation() : null;
                        float boot = obs == null ? 0 : AgentKit.predict(localValue, obs)[0];
                        update(o, a, r, boot, actSize, localParams, sharedParams);
                        obs = env.reset(random.split());
                        break;
                    }
                    obs = s.observation();
                }
                if (!ended && !o.isEmpty()) {
                    float boot = AgentKit.predict(localValue, obs)[0];
                    update(o, a, r, boot, actSize, localParams, sharedParams);
                }
            }
        }

        void update(List<float[]> o, List<float[]> a, List<Float> r, float bootstrap, int actSize,
                List<Var> localParams, List<Var> sharedParams) {
            int n = o.size();
            float[] returns = new float[n];
            float g = bootstrap;
            for (int t = n - 1; t >= 0; t--) {
                g = r.get(t) + gamma * g;
                returns[t] = g;
            }
            Tensor x = AgentKit.rows(o);
            float[] flat = new float[n * actSize];
            for (int i = 0; i < n; i++) {
                System.arraycopy(a.get(i), 0, flat, i * actSize, actSize);
            }
            Var out = localPolicy.forward(Var.of(x), true);
            Var v = localValue.forward(Var.of(x), true);
            Tensor ret = AgentKit.column(returns);
            Tensor adv = ret.sub(v.value());
            Var logp = localDist.logProb(out, Tensor.of(Shape.of(n, actSize), flat));
            Var loss = logp.mul(adv).mean().neg()
                    .add(v.sub(ret).square().mean().mul(0.5f * valueCoefficient))
                    .sub(localDist.entropy(out).mean().mul(entropy));
            loss.backward();
            RlTensors.clip(localParams, maxGradNorm);
            synchronized (lock) {
                for (int i = 0; i < sharedParams.size(); i++) {
                    Tensor lg = localParams.get(i).grad();
                    if (lg != null) {
                        sharedParams.get(i).setGrad(lg);
                    }
                }
                optimizer.step(sharedParams);
            }
            RlTensors.zeroGrad(localParams);
            updates.incrementAndGet();
        }
    }

    @Override
    public String toString() {
        return "A3C";
    }
}
