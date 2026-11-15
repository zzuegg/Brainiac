package io.github.zzuegg.brainiac.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Proximal policy optimization (Schulman et al., 2017) and advantage actor-critic (A2C).
 * An actor (policy network) and a critic (value network) learn from short rollouts of
 * one or several environments run side by side. Advantages come from generalized
 * advantage estimation (GAE). PPO reuses each rollout for several epochs of minibatches,
 * clipping how far the policy may move from the one that collected it.
 *
 * <pre>{@code
 * Ppo ppo = Ppo.of(Nets.mlp(4, 2, 1, 64, 64), Nets.mlp(4, 1, 2, 64, 64), env.actionSpace())
 *         .rolloutSteps(256).epochs(10).minibatchSize(64)
 *         .discount(0.99f).gaeLambda(0.95f).clipRange(0.2f)
 *         .entropyBonus(0.0f).valueCoefficient(0.5f).clipGradientNorm(0.5f)
 *         .learningRate(3e-4f).annealLearningRate(true).targetKl(0.02f);
 * ppo.fit(CartPole::new, 8, 200_000, RlListener.every(50));   // 8 environments in step
 * }</pre>
 *
 * <p>{@link #a2c} gives the A2C settings: short rollouts, one pass, no clipping.
 */
public final class Ppo implements Learner {

    private final Model policy;
    private final Model value;
    private final PolicyHead head;
    private Optimizer optimizer;
    private float learningRate = 3e-4f;
    private boolean anneal;
    private int rolloutSteps = 256;
    private int epochs = 10;
    private int minibatch = 64;
    private float gamma = 0.99f;
    private float lambda = 0.95f;
    private float clip = 0.2f;
    private boolean clipValue;
    private float valueCoefficient = 0.5f;
    private float entropy;
    private float maxGradNorm = 0.5f;
    private float targetKl;
    private boolean normalize = true;
    private SplittableRandom random = new SplittableRandom(0);
    private final Map<String, Double> diagnostics = new LinkedHashMap<>();

    private Ppo(Model policy, Model value, Space actions) {
        this.policy = Objects.requireNonNull(policy, "policy");
        this.value = Objects.requireNonNull(value, "value");
        this.head = new PolicyHead(actions, 0f, policy.device());
        int expected = actions instanceof Space.Discrete d ? d.n() : actions.size();
        if (policy.outputSize() != expected) {
            throw new IllegalArgumentException("the policy must output " + expected + " values for " + actions
                    + ", it outputs " + policy.outputSize());
        }
        if (value.outputSize() != 1) {
            throw new IllegalArgumentException("the value network must output one value");
        }
    }

    /**
     * PPO with the usual settings: 256 steps per environment per rollout, 10 epochs of
     * minibatches of 64, clip range 0.2, GAE lambda 0.95, Adam 3e-4.
     *
     * @param policy observations to action logits (discrete) or Gaussian means (continuous)
     * @param value observations to one value
     * @param actions the action space
     * @return the learner
     */
    public static Ppo of(Model policy, Model value, Space actions) {
        return new Ppo(policy, value, actions);
    }

    /**
     * A2C: 8 steps per environment per rollout, one pass over the whole rollout, no
     * ratio clipping, GAE lambda 1 (plain n-step returns), an entropy bonus of 0.01 and
     * Adam 2e-3.
     *
     * @param policy observations to action logits or means
     * @param value observations to one value
     * @param actions the action space
     * @return the learner
     */
    public static Ppo a2c(Model policy, Model value, Space actions) {
        Ppo p = new Ppo(policy, value, actions);
        p.rolloutSteps = 8;
        p.epochs = 1;
        p.minibatch = 0;
        p.clip = 0;
        p.lambda = 1;
        p.entropy = 0.01f;
        p.learningRate = 2e-3f;
        p.normalize = false;
        return p;
    }

    /**
     * @param steps steps taken in every environment before each update
     * @return this learner
     */
    public Ppo rolloutSteps(int steps) {
        this.rolloutSteps = positive(steps, "rolloutSteps");
        return this;
    }

    /**
     * @param n passes over every rollout
     * @return this learner
     */
    public Ppo epochs(int n) {
        this.epochs = positive(n, "epochs");
        return this;
    }

    /**
     * @param size samples per minibatch; 0 for the whole rollout at once
     * @return this learner
     */
    public Ppo minibatchSize(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("minibatchSize must not be negative");
        }
        this.minibatch = size;
        return this;
    }

    /**
     * @param discount how much future rewards count (gamma)
     * @return this learner
     */
    public Ppo discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /**
     * @param lambda the GAE trade-off: 0 is one-step TD (low variance, more bias), 1 is
     *     the full return
     * @return this learner
     */
    public Ppo gaeLambda(float lambda) {
        this.lambda = lambda;
        return this;
    }

    /**
     * @param range how far the probability ratio may move from 1; 0 turns clipping off
     * @return this learner
     */
    public Ppo clipRange(float range) {
        this.clip = range;
        return this;
    }

    /**
     * @param on whether to clip value updates by the clip range too
     * @return this learner
     */
    public Ppo clipValue(boolean on) {
        this.clipValue = on;
        return this;
    }

    /**
     * @param c weight of the value loss
     * @return this learner
     */
    public Ppo valueCoefficient(float c) {
        this.valueCoefficient = c;
        return this;
    }

    /**
     * @param coefficient weight of an entropy bonus that keeps the policy exploring
     * @return this learner
     */
    public Ppo entropyBonus(float coefficient) {
        this.entropy = coefficient;
        return this;
    }

    /**
     * @param norm rescale gradients longer than this; 0 for none
     * @return this learner
     */
    public Ppo clipGradientNorm(float norm) {
        this.maxGradNorm = norm;
        return this;
    }

    /**
     * @param kl stop an update's epochs early once the policy moved more than about this
     *     (approximate KL divergence); 0 for no limit
     * @return this learner
     */
    public Ppo targetKl(float kl) {
        this.targetKl = kl;
        return this;
    }

    /**
     * @param on whether to scale advantages to mean 0 and standard deviation 1 per
     *     minibatch
     * @return this learner
     */
    public Ppo normalizeAdvantages(boolean on) {
        this.normalize = on;
        return this;
    }

    /**
     * @param rate the Adam learning rate (ignored with {@link #optimizer})
     * @return this learner
     */
    public Ppo learningRate(float rate) {
        this.learningRate = rate;
        return this;
    }

    /**
     * @param on whether to lower the learning rate linearly to 0 over training
     * @return this learner
     */
    public Ppo annealLearningRate(boolean on) {
        this.anneal = on;
        return this;
    }

    /**
     * @param o the optimizer for both networks
     * @return this learner
     */
    public Ppo optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param logStd the starting log standard deviation of continuous actions
     * @return this learner
     */
    public Ppo initialLogStd(float logStd) {
        if (head.logStd != null) {
            Tensor s = head.logStd.value();
            head.logStd.assign(Tensor.full(s.shape(), logStd, s.dtype(), s.device()));
        }
        return this;
    }

    /**
     * @param seed the seed for sampling actions, shuffling and episodes
     * @return this learner
     */
    public Ppo seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    private static int positive(int n, String what) {
        if (n < 1) {
            throw new IllegalArgumentException(what + " must be at least 1, got " + n);
        }
        return n;
    }

    /** @return the policy network */
    public Model policy() {
        return policy;
    }

    /** @return the value network */
    public Model value() {
        return value;
    }

    /**
     * @return statistics of the last update: policy loss, value loss, entropy,
     *     approximate KL, clip fraction, explained variance and epochs run
     */
    public Map<String, Double> diagnostics() {
        return Map.copyOf(diagnostics);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return head.toEnvironment(head.sample(Nets.row(policy, observation), random, explore));
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        return run(List.of(env), steps, listener);
    }

    /**
     * Trains on several copies of an environment at once; their steps are batched
     * through the networks together.
     *
     * @param factory makes each environment
     * @param count how many to run side by side
     * @param steps the total environment steps, over all copies
     * @param listener told about every finished episode
     * @return every finished episode
     */
    public RlResult fit(Supplier<? extends Environment> factory, int count, long steps, RlListener listener) {
        List<Environment> envs = new ArrayList<>();
        for (int i = 0; i < positive(count, "count"); i++) {
            envs.add(factory.get());
        }
        return run(envs, steps, listener);
    }

    private List<Var> parameters() {
        List<Var> p = new ArrayList<>(policy.parameters());
        if (head.logStd != null) {
            p.add(head.logStd);
        }
        p.addAll(value.parameters());
        return p;
    }

    private RlResult run(List<Environment> envs, long steps, RlListener listener) {
        if (optimizer == null) {
            optimizer = Optimizer.adam(learningRate);
        }
        float startRate = optimizer.learningRate();
        int e = envs.size();
        int obsSize = envs.get(0).observationSpace().size();
        int actSize = head.actionSize();
        Device device = policy.device();
        Tracker tracker = new Tracker(listener);
        float[][] obs = new float[e][];
        for (int i = 0; i < e; i++) {
            obs[i] = envs.get(i).reset(random.split());
        }
        int n = rolloutSteps * e;
        float[] bObs = new float[n * obsSize];
        float[] bAct = new float[n * actSize];
        float[] bLogp = new float[n];
        float[] bVal = new float[n];
        float[] bRew = new float[n];
        float[] bDone = new float[n];
        while (tracker.steps() < steps) {
            if (anneal) {
                optimizer.learningRate((float) Math.max(1e-8,
                        startRate * (1 - (double) tracker.steps() / steps)));
            }
            for (int t = 0; t < rolloutSteps; t++) {
                Tensor x = Nets.rows(obs, obsSize, device);
                Var out = policy.forward(Var.of(x), false);
                float[] outRows = out.value().to(DType.FLOAT32).toArray();
                float[] values = value.forward(Var.of(x), false).value().to(DType.FLOAT32).toArray();
                int width = outRows.length / e;
                float[][] actions = new float[e][];
                for (int i = 0; i < e; i++) {
                    actions[i] = head.sample(java.util.Arrays.copyOfRange(outRows, i * width, (i + 1) * width),
                            random, true);
                }
                float[] logp = head.logProb(out, Nets.rows(actions, actSize, device)).value()
                        .to(DType.FLOAT32).toArray();
                for (int i = 0; i < e; i++) {
                    int k = t * e + i;
                    System.arraycopy(obs[i], 0, bObs, k * obsSize, obsSize);
                    System.arraycopy(actions[i], 0, bAct, k * actSize, actSize);
                    bLogp[k] = logp[i];
                    bVal[k] = values[i];
                    Environment.Step s = envs.get(i).step(head.toEnvironment(actions[i]));
                    float r = s.reward();
                    if (s.truncated() && !s.terminated()) {
                        r += gamma * Nets.row(value, s.observation())[0];
                    }
                    bRew[k] = r;
                    bDone[k] = s.done() ? 1 : 0;
                    tracker.step(s);
                    obs[i] = s.done() ? envs.get(i).reset(random.split()) : s.observation();
                }
            }
            float[] last = value.forward(Var.of(Nets.rows(obs, obsSize, device)), false).value()
                    .to(DType.FLOAT32).toArray();
            float[] adv = new float[n];
            float[] ret = new float[n];
            for (int i = 0; i < e; i++) {
                float gae = 0;
                for (int t = rolloutSteps - 1; t >= 0; t--) {
                    int k = t * e + i;
                    float next = t == rolloutSteps - 1 ? last[i] : bVal[k + e];
                    float live = 1 - bDone[k];
                    float delta = bRew[k] + gamma * next * live - bVal[k];
                    gae = delta + gamma * lambda * live * gae;
                    adv[k] = gae;
                    ret[k] = gae + bVal[k];
                }
            }
            update(n, obsSize, actSize, bObs, bAct, bLogp, bVal, adv, ret, device);
        }
        if (anneal) {
            optimizer.learningRate(startRate);
        }
        return tracker.result;
    }

    private void update(int n, int obsSize, int actSize, float[] bObs, float[] bAct, float[] bLogp, float[] bVal,
                        float[] adv, float[] ret, Device device) {
        int mb = minibatch <= 0 || minibatch > n ? n : minibatch;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        double policyLoss = 0;
        double valueLoss = 0;
        double ent = 0;
        double kl = 0;
        double clipped = 0;
        int batches = 0;
        int epochsRun = 0;
        List<Var> params = parameters();
        for (int epoch = 0; epoch < epochs; epoch++) {
            epochsRun++;
            for (int i = n - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
            double epochKl = 0;
            int epochBatches = 0;
            for (int from = 0; from + mb <= n; from += mb) {
                float[] o = new float[mb * obsSize];
                float[] a = new float[mb * actSize];
                float[] lp = new float[mb];
                float[] v0 = new float[mb];
                float[] ad = new float[mb];
                float[] r = new float[mb];
                for (int b = 0; b < mb; b++) {
                    int k = order[from + b];
                    System.arraycopy(bObs, k * obsSize, o, b * obsSize, obsSize);
                    System.arraycopy(bAct, k * actSize, a, b * actSize, actSize);
                    lp[b] = bLogp[k];
                    v0[b] = bVal[k];
                    ad[b] = adv[k];
                    r[b] = ret[k];
                }
                if (normalize && mb > 1) {
                    double mean = 0;
                    for (float x : ad) {
                        mean += x;
                    }
                    mean /= mb;
                    double var = 0;
                    for (float x : ad) {
                        var += (x - mean) * (x - mean);
                    }
                    double std = Math.sqrt(var / mb) + 1e-8;
                    for (int b = 0; b < mb; b++) {
                        ad[b] = (float) ((ad[b] - mean) / std);
                    }
                }
                Tensor x = Tensor.of(Shape.of(mb, obsSize), o).to(device);
                Tensor advantages = Nets.column(ad, device);
                Var out = policy.forward(Var.of(x), true);
                Var logp = head.logProb(out, Tensor.of(Shape.of(mb, actSize), a).to(device));
                Var ratio = logp.sub(Nets.column(lp, device)).exp();
                Var surrogate = ratio.mul(advantages);
                if (clip > 0) {
                    surrogate = surrogate.minimum(ratio.clamp(1 - clip, 1 + clip).mul(advantages));
                }
                Var pg = surrogate.mean().neg();
                Var v = value.forward(Var.of(x), true);
                Tensor returns = Nets.column(r, device);
                Var vl = v.sub(returns).square();
                if (clipValue && clip > 0) {
                    Tensor old = Nets.column(v0, device);
                    Var vc = v.sub(old).clamp(-clip, clip).add(old);
                    vl = vl.maximum(vc.sub(returns).square());
                }
                vl = vl.mean().mul(0.5f);
                Var en = head.entropy(out).mean();
                Var loss = pg.add(vl.mul(valueCoefficient));
                if (entropy != 0) {
                    loss = loss.sub(en.mul(entropy));
                }
                loss.backward();
                Nets.clip(params, maxGradNorm);
                optimizer.step(params);
                float[] newLogp = logp.value().to(DType.FLOAT32).toArray();
                float[] ratios = ratio.value().to(DType.FLOAT32).toArray();
                double batchKl = 0;
                for (int b = 0; b < mb; b++) {
                    batchKl += (ratios[b] - 1) - (newLogp[b] - lp[b]);
                    if (clip > 0 && Math.abs(ratios[b] - 1) > clip) {
                        clipped++;
                    }
                }
                batchKl /= mb;
                epochKl += batchKl;
                epochBatches++;
                policyLoss += pg.value().item();
                valueLoss += vl.value().item();
                ent += en.value().item();
                kl += batchKl;
                batches++;
            }
            if (targetKl > 0 && epochBatches > 0 && epochKl / epochBatches > 1.5 * targetKl) {
                break;
            }
        }
        double varRet = variance(ret);
        double[] residual = new double[n];
        for (int i = 0; i < n; i++) {
            residual[i] = ret[i] - bVal[i];
        }
        diagnostics.put("policyLoss", policyLoss / batches);
        diagnostics.put("valueLoss", valueLoss / batches);
        diagnostics.put("entropy", ent / batches);
        diagnostics.put("approxKl", kl / batches);
        diagnostics.put("clipFraction", clipped / ((double) batches * mb));
        diagnostics.put("explainedVariance", varRet == 0 ? 0 : 1 - variance(residual) / varRet);
        diagnostics.put("epochs", (double) epochsRun);
        diagnostics.put("learningRate", (double) optimizer.learningRate());
    }

    private static double variance(float[] x) {
        double[] d = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            d[i] = x[i];
        }
        return variance(d);
    }

    private static double variance(double[] x) {
        double mean = 0;
        for (double v : x) {
            mean += v;
        }
        mean /= x.length;
        double var = 0;
        for (double v : x) {
            var += (v - mean) * (v - mean);
        }
        return var / x.length;
    }

    @Override
    public String toString() {
        return clip > 0 ? "PPO" : "A2C";
    }
}
