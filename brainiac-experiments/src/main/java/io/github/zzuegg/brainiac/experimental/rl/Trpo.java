package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
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
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Trust region policy optimization (Schulman et al., 2015,
 * <a href="https://arxiv.org/abs/1502.05477">arXiv:1502.05477</a>): every update maximizes the
 * importance-weighted surrogate advantage subject to a hard bound on the mean KL divergence
 * between the old and new policy. The step direction is the natural gradient
 * {@code F^-1 g}, found by conjugate gradient from Fisher-vector products (Hessian-vector
 * products of the KL, by double back-propagation); it is scaled to the edge of the trust
 * region and then shrunk by a backtracking line search until the KL bound holds and the
 * surrogate improves. The value network is fitted separately by regression on GAE returns.
 *
 * <p>Compared with PPO's clipped objective, TRPO enforces the KL bound exactly at every
 * update ({@link #klHistory}), at the price of second-order computation.
 *
 * <pre>{@code
 * Trpo trpo = Trpo.of(Nets.mlp(4, 2, 1, 32), Nets.mlp(4, 1, 2, 32), env.actionSpace()).maxKl(0.01f);
 * trpo.fit(new CartPole(), 20_000);
 * }</pre>
 */
public final class Trpo implements Learner {

    private final Model policy;
    private final Model value;
    private final PolicyDistribution dist;
    private Optimizer valueOptimizer = Optimizer.adam(3e-3f);
    private float maxKl = 0.01f;
    private float damping = 0.1f;
    private int cgIterations = 10;
    private int backtracks = 10;
    private int rolloutSteps = 1000;
    private int valueEpochs = 30;
    private float gamma = 0.99f;
    private float lambda = 0.97f;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Double> klHistory = new ArrayList<>();

    private Trpo(Model policy, Model value, Space actions) {
        this.policy = Objects.requireNonNull(policy, "policy");
        this.value = Objects.requireNonNull(value, "value");
        this.dist = new PolicyDistribution(actions, -0.5f);
        if (policy.outputSize() != dist.outputs() || value.outputSize() != 1) {
            throw new IllegalArgumentException("the policy must output " + dist.outputs()
                    + " values and the value network one");
        }
    }

    /**
     * @param policy observations to logits (discrete) or Gaussian means (continuous)
     * @param value observations to one value
     * @param actions the action space
     * @return the learner: max KL 0.01, 1000 steps per update, damping 0.1
     */
    public static Trpo of(Model policy, Model value, Space actions) {
        return new Trpo(policy, value, actions);
    }

    /** @param kl the trust region: the largest mean KL per update @return this learner */
    public Trpo maxKl(float kl) {
        this.maxKl = kl;
        return this;
    }

    /** @param d the Fisher damping @return this learner */
    public Trpo damping(float d) {
        this.damping = d;
        return this;
    }

    /** @param n conjugate gradient iterations @return this learner */
    public Trpo cgIterations(int n) {
        this.cgIterations = n;
        return this;
    }

    /** @param steps environment steps per update @return this learner */
    public Trpo rolloutSteps(int steps) {
        this.rolloutSteps = steps;
        return this;
    }

    /** @param discount the discount @return this learner */
    public Trpo discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param l the GAE lambda @return this learner */
    public Trpo gaeLambda(float l) {
        this.lambda = l;
        return this;
    }

    /** @param epochs value regression steps per update @return this learner */
    public Trpo valueEpochs(int epochs) {
        this.valueEpochs = epochs;
        return this;
    }

    /** @param o the value network's optimizer @return this learner */
    public Trpo valueOptimizer(Optimizer o) {
        this.valueOptimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param seed the seed @return this learner */
    public Trpo seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the mean KL between old and new policy, one value per update */
    public List<Double> klHistory() {
        return List.copyOf(klHistory);
    }

    /** @return the policy network */
    public Model policy() {
        return policy;
    }

    List<Var> policyParameters() {
        List<Var> p = new ArrayList<>(policy.parameters());
        if (dist.logStd() != null) {
            p.add(dist.logStd());
        }
        return p;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return dist.toEnvironment(dist.sample(AgentKit.predict(policy, observation), random, explore));
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        while (tracker.steps() < steps) {
            OnPolicyRollout r = OnPolicyRollout.collect(env, obs, policy, dist, value, null, rolloutSteps, gamma,
                    tracker, random, 0).gae(gamma, lambda);
            obs = r.nextObservation();
            klHistory.add(update(r));
            OnPolicyRollout.fitValue(value, valueOptimizer, r.observations(), r.returns(), valueEpochs);
        }
        return tracker.result();
    }

    private double update(OnPolicyRollout r) {
        List<Var> params = policyParameters();
        Tensor obs = r.observations();
        Tensor acts = r.actions();
        Tensor oldLogp = r.logProbs();
        Tensor adv = AgentKit.column(OnPolicyRollout.normalize(r.advantages()));
        Tensor oldOut = policy.forward(Var.of(obs), false).value();
        Tensor oldLogStd = dist.logStd() == null ? null : dist.logStd().value();
        Supplier<Var> surrogate = () -> dist.logProb(policy.forward(Var.of(obs), true), acts).sub(oldLogp).exp()
                .mul(adv).mean();
        Supplier<Var> kl = () -> dist.kl(oldOut, oldLogStd, policy.forward(Var.of(obs), true));
        float[] g = TrustRegion.gradient(surrogate.get(), params);
        float[] step = TrustRegion.conjugateGradient(
                v -> TrustRegion.fisherVectorProduct(kl, params, v, damping), g, cgIterations);
        double shs = TrustRegion.dot(step, TrustRegion.fisherVectorProduct(kl, params, step, damping));
        if (!(shs > 0)) {
            return 0;
        }
        double beta = Math.sqrt(2 * maxKl / shs);
        float[] theta = TrustRegion.flatten(params);
        double before = surrogate.get().value().item();
        for (int k = 0; k < backtracks; k++) {
            double frac = beta * Math.pow(0.5, k);
            TrustRegion.assign(params, TrustRegion.axpy(theta, frac, step));
            double after = surrogate.get().value().item();
            double d = kl.get().value().item();
            if (d <= 1.5 * maxKl && after > before) {
                return d;
            }
        }
        TrustRegion.assign(params, theta);
        return 0;
    }

    @Override
    public String toString() {
        return "TRPO(max KL " + maxKl + ")";
    }
}
