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
import java.util.function.Supplier;

/**
 * Constrained policy optimization: maximize the return subject to an expected total cost
 * per episode of at most {@code d}, on a {@link CostedEnvironment}. Two methods:
 * <ul>
 *   <li>{@link Method#CPO} (Achiam et al., 2017, <a href="https://arxiv.org/abs/1705.10528">arXiv:1705.10528</a>):
 *       a TRPO step whose direction solves the trust-region problem with the linearized
 *       cost constraint in closed form via its dual; when the policy is infeasible it takes
 *       a pure recovery step that lowers the cost. It reuses {@link TrustRegion}'s
 *       Fisher-vector products and conjugate gradient.</li>
 *   <li>{@link Method#PPO_LAGRANGIAN} (Ray, Achiam and Amodei, 2019,
 *       <a href="https://cdn.openai.com/safexp-short.pdf">Safety Gym</a>): PPO on the
 *       advantage {@code (A_r - lambda A_c) / (1 + lambda)} with a Lagrange multiplier
 *       {@code lambda} that grows while the measured episode cost exceeds the limit and
 *       shrinks back once it does not.</li>
 * </ul>
 * Both keep a second critic for the cost. Fixed reward shaping ({@code r - k c}) needs
 * {@code k} tuned per task; the multiplier finds it.
 */
public final class ConstrainedPolicyOptimization implements Learner {

    /** How the constraint is enforced. */
    public enum Method {
        /** Trust-region step with the constraint solved in closed form. */
        CPO,
        /** PPO with a learned Lagrange multiplier. */
        PPO_LAGRANGIAN
    }

    private static final double EPS = 1e-8;

    private final Method method;
    private final Model policy;
    private final Model value;
    private final Model costValue;
    private final PolicyDistribution dist;
    private final float costLimit;
    private Optimizer policyOptimizer = Optimizer.adam(3e-3f);
    private final Optimizer valueOptimizer = Optimizer.adam(3e-3f);
    private final Optimizer costOptimizer = Optimizer.adam(3e-3f);
    private float maxKl = 0.01f;
    private float damping = 0.1f;
    private int rolloutSteps = 1000;
    private int epochs = 8;
    private int minibatch = 100;
    private float clip = 0.2f;
    private float gamma = 0.99f;
    private float lambdaGae = 0.95f;
    private float lagrangeRate = 0.05f;
    private double multiplier;
    private double costEstimate;
    private boolean haveCost;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Double> episodeCosts = new ArrayList<>();
    private final List<Double> multipliers = new ArrayList<>();

    private ConstrainedPolicyOptimization(Method method, Model policy, Model value, Model costValue, Space actions,
            float costLimit) {
        this.method = Objects.requireNonNull(method);
        this.policy = Objects.requireNonNull(policy);
        this.value = Objects.requireNonNull(value);
        this.costValue = Objects.requireNonNull(costValue);
        this.dist = new PolicyDistribution(actions, -0.5f);
        this.costLimit = costLimit;
    }

    /**
     * @param method CPO or PPO-Lagrangian
     * @param policy observations to logits or means
     * @param value observations to the reward value
     * @param costValue observations to the cost value
     * @param actions the action space
     * @param costLimit the largest allowed expected total cost per episode
     * @return the learner
     */
    public static ConstrainedPolicyOptimization of(Method method, Model policy, Model value, Model costValue,
            Space actions, float costLimit) {
        return new ConstrainedPolicyOptimization(method, policy, value, costValue, actions, costLimit);
    }

    /** @param steps environment steps per update @return this learner */
    public ConstrainedPolicyOptimization rolloutSteps(int steps) {
        this.rolloutSteps = steps;
        return this;
    }

    /** @param kl the CPO trust region @return this learner */
    public ConstrainedPolicyOptimization maxKl(float kl) {
        this.maxKl = kl;
        return this;
    }

    /** @param rate the step size of the Lagrange multiplier (PPO-Lagrangian) @return this learner */
    public ConstrainedPolicyOptimization lagrangeRate(float rate) {
        this.lagrangeRate = rate;
        return this;
    }

    /** @param lr the PPO policy learning rate @return this learner */
    public ConstrainedPolicyOptimization learningRate(float lr) {
        this.policyOptimizer = Optimizer.adam(lr);
        return this;
    }

    /** @param discount the discount @return this learner */
    public ConstrainedPolicyOptimization discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the seed @return this learner */
    public ConstrainedPolicyOptimization seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the total cost of every training episode */
    public List<Double> episodeCosts() {
        return List.copyOf(episodeCosts);
    }

    /** @return the Lagrange multiplier after every update (PPO-Lagrangian), or CPO's dual nu */
    public List<Double> multipliers() {
        return List.copyOf(multipliers);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return dist.toEnvironment(dist.sample(AgentKit.predict(policy, observation), random, explore));
    }

    private List<Var> policyParameters() {
        List<Var> p = new ArrayList<>(policy.parameters());
        if (dist.logStd() != null) {
            p.add(dist.logStd());
        }
        return p;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        double carried = 0;
        while (tracker.steps() < steps) {
            int before = tracker.result().episodes().size();
            OnPolicyRollout r = OnPolicyRollout.collect(env, obs, policy, dist, value, costValue, rolloutSteps,
                    gamma, tracker, random, carried).gae(gamma, lambdaGae);
            obs = r.nextObservation();
            carried = r.carriedCost();
            episodeCosts.addAll(r.episodeCosts());
            int finished = tracker.result().episodes().size() - before;
            if (!r.episodeCosts().isEmpty()) {
                double c = r.episodeCosts().stream().mapToDouble(Double::doubleValue).average().orElse(0);
                costEstimate = haveCost ? 0.5 * costEstimate + 0.5 * c : c;
                haveCost = true;
            }
            double meanLength = finished == 0 ? rolloutSteps : (double) rolloutSteps / finished;
            if (method == Method.CPO) {
                cpoStep(r, meanLength);
            } else {
                multiplier = Math.max(0, multiplier + lagrangeRate * (costEstimate - costLimit));
                multipliers.add(multiplier);
                ppoStep(r);
            }
            OnPolicyRollout.fitValue(value, valueOptimizer, r.observations(), r.returns(), 30);
            OnPolicyRollout.fitValue(costValue, costOptimizer, r.observations(), r.costReturns(), 30);
        }
        return tracker.result();
    }

    private void ppoStep(OnPolicyRollout r) {
        int n = r.size();
        float[] ar = OnPolicyRollout.normalize(r.advantages());
        float[] ac = OnPolicyRollout.normalize(r.costAdvantages());
        float[] adv = new float[n];
        for (int i = 0; i < n; i++) {
            adv[i] = (float) ((ar[i] - multiplier * ac[i]) / (1 + multiplier));
        }
        float[] obs = r.observations().toArray();
        float[] acts = r.actions().toArray();
        float[] logp = r.logProbs().toArray();
        int obsSize = obs.length / n;
        int actSize = acts.length / n;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        List<Var> params = policyParameters();
        for (int e = 0; e < epochs; e++) {
            for (int i = n - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int from = 0; from + minibatch <= n; from += minibatch) {
                float[] o = new float[minibatch * obsSize];
                float[] a = new float[minibatch * actSize];
                float[] lp = new float[minibatch];
                float[] ad = new float[minibatch];
                for (int b = 0; b < minibatch; b++) {
                    int k = order[from + b];
                    System.arraycopy(obs, k * obsSize, o, b * obsSize, obsSize);
                    System.arraycopy(acts, k * actSize, a, b * actSize, actSize);
                    lp[b] = logp[k];
                    ad[b] = adv[k];
                }
                Tensor advT = AgentKit.column(ad);
                Var out = policy.forward(Var.of(Tensor.of(Shape.of(minibatch, obsSize), o)), true);
                Var ratio = dist.logProb(out, Tensor.of(Shape.of(minibatch, actSize), a))
                        .sub(AgentKit.column(lp)).exp();
                Var surrogate = ratio.mul(advT).minimum(ratio.clamp(1 - clip, 1 + clip).mul(advT));
                Var loss = surrogate.mean().neg().sub(dist.entropy(out).mean().mul(0.01f));
                loss.backward();
                RlTensors.clip(params, 0.5f);
                policyOptimizer.step(params);
            }
        }
    }

    private static float[] center(float[] x) {
        double m = 0;
        for (float v : x) {
            m += v;
        }
        m /= x.length;
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) (x[i] - m);
        }
        return out;
    }

    private void cpoStep(OnPolicyRollout r, double meanLength) {
        List<Var> params = policyParameters();
        Tensor obs = r.observations();
        Tensor acts = r.actions();
        Tensor oldLogp = r.logProbs();
        Tensor advR = AgentKit.column(OnPolicyRollout.normalize(r.advantages()));
        Tensor advC = AgentKit.column(center(r.costAdvantages()));
        Tensor oldOut = policy.forward(Var.of(obs), false).value();
        Tensor oldLogStd = dist.logStd() == null ? null : dist.logStd().value();
        Supplier<Var> ratio = () -> dist.logProb(policy.forward(Var.of(obs), true), acts).sub(oldLogp).exp();
        Supplier<Var> kl = () -> dist.kl(oldOut, oldLogStd, policy.forward(Var.of(obs), true));
        float[] g = TrustRegion.gradient(ratio.get().mul(advR).mean(), params);
        float[] b = TrustRegion.gradient(ratio.get().mul(advC).mean(), params);
        java.util.function.UnaryOperator<float[]> fvp = v -> TrustRegion.fisherVectorProduct(kl, params, v, damping);
        float[] v = TrustRegion.conjugateGradient(fvp, g, 10);
        double q = TrustRegion.dot(g, v);
        // the constraint in surrogate units: per-step cost change times the episode length
        double c = (costEstimate - costLimit) / Math.max(1, meanLength);
        double delta = maxKl;
        double lam;
        double nu;
        float[] w = null;
        double r0 = 0;
        double s = 0;
        int optimCase;
        if (TrustRegion.dot(b, b) <= 1e-12 && c < 0) {
            optimCase = 4;
        } else {
            w = TrustRegion.conjugateGradient(fvp, b, 10);
            r0 = TrustRegion.dot(b, v);
            s = TrustRegion.dot(b, w);
            double bb = 2 * delta - c * c / (s + EPS);
            if (c < 0 && bb < 0) {
                optimCase = 3;
            } else if (c < 0) {
                optimCase = 2;
            } else if (bb >= 0) {
                optimCase = 1;
            } else {
                optimCase = 0;
            }
        }
        if (optimCase >= 3) {
            lam = Math.sqrt(q / (2 * delta));
            nu = 0;
        } else if (optimCase >= 1) {
            double a = q - r0 * r0 / (s + EPS);
            double bb = 2 * delta - c * c / (s + EPS);
            double[] la = {0, r0 / c};
            double[] lb = {r0 / c, Double.POSITIVE_INFINITY};
            if (c < 0) {
                double[] t = la;
                la = lb;
                lb = t;
            }
            double lamA = project(Math.sqrt(Math.max(0, a / bb)), la);
            double lamB = project(Math.sqrt(q / (2 * delta)), lb);
            double fa = -0.5 * (a / (lamA + EPS) + bb * lamA) - r0 * c / (s + EPS);
            double fb = -0.5 * (q / (lamB + EPS) + 2 * delta * lamB);
            lam = fa >= fb ? lamA : lamB;
            nu = Math.max(0, lam * c - r0) / (s + EPS);
        } else {
            lam = 0;
            nu = Math.sqrt(2 * delta / (s + EPS));
        }
        multipliers.add(nu);
        float[] step;
        if (optimCase > 0) {
            step = w == null ? v.clone() : TrustRegion.axpy(v, -nu, w);
            for (int i = 0; i < step.length; i++) {
                step[i] = (float) (step[i] / (lam + EPS));
            }
        } else {
            step = new float[w.length];
            for (int i = 0; i < step.length; i++) {
                step[i] = (float) (-nu * w[i]);
            }
        }
        float[] theta = TrustRegion.flatten(params);
        double rewardBefore = ratio.get().mul(advR).mean().value().item();
        double costBefore = ratio.get().mul(advC).mean().value().item();
        for (int k = 0; k < 10; k++) {
            double frac = Math.pow(0.8, k);
            TrustRegion.assign(params, TrustRegion.axpy(theta, frac, step));
            Var rat = ratio.get();
            double rewardAfter = rat.mul(advR).mean().value().item();
            double costAfter = rat.mul(advC).mean().value().item();
            double d = kl.get().value().item();
            boolean improves = optimCase == 0 || rewardAfter >= rewardBefore;
            if (d <= delta * 1.5 && improves && costAfter - costBefore <= Math.max(-c, 0)) {
                return;
            }
        }
        TrustRegion.assign(params, theta);
    }

    private static double project(double x, double[] range) {
        return Math.max(range[0], Math.min(range[1], x));
    }

    @Override
    public String toString() {
        return method == Method.CPO ? "CPO" : "PPO-Lagrangian";
    }
}
