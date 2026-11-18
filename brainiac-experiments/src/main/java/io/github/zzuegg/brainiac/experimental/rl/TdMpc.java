package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TD-MPC2 (Hansen et al., 2024, <a href="https://arxiv.org/abs/2310.16828">arXiv:2310.16828</a>),
 * reduced to its parts: a decoder-free latent world model planned with MPPI.
 * <ul>
 *   <li>An encoder {@code z = E(o)} (layer-normalized), latent dynamics
 *       {@code z' = D(z, a)}, a reward head {@code R(z, a)}, a {@link CriticEnsemble} of
 *       {@code Q(z, a)} and a tanh-Gaussian policy prior {@code pi(z)}.</li>
 *   <li>Trained on short replayed sequences by three losses summed over the unrolled
 *       horizon with decaying weights: <b>consistency</b> of the predicted latent with the
 *       (stop-gradient) encoding of the real next observation, <b>reward</b> regression and
 *       <b>TD</b> regression of every Q towards {@code r + gamma min Q'(z', pi(z'))}. No
 *       observation is ever reconstructed.</li>
 *   <li>The policy prior maximizes {@code Q(z, pi(z))} plus an entropy bonus.</li>
 *   <li>Acting runs MPPI: sampled action sequences (some from the policy prior) are scored
 *       by {@code sum gamma^t R + gamma^H Q(z_H, pi(z_H))} with the learned terminal value,
 *       the distribution is refit with softmax weights of the elites, and the first action
 *       of the mean is executed. The plan warm-starts the next step.</li>
 * </ul>
 * Episode ends are treated as truncations (the TD target always bootstraps), which suits
 * continuing control tasks such as the pendulum.
 */
public final class TdMpc implements Learner {

    private static final float LOG_STD_MIN = -5f;
    private static final float LOG_STD_MAX = 1f;

    private final int obsDims;
    private final int actDims;
    private final int latent;
    private final Model encoder;
    private final Model dynamics;
    private final Model rewardHead;
    private final Model policy;
    private final CriticEnsemble critics;
    private final List<Var> modelParams = new ArrayList<>();
    private final Optimizer modelOpt = Optimizer.adam(3e-3f);
    private final Optimizer policyOpt = Optimizer.adam(3e-3f);
    private float gamma = 0.99f;
    private float tau = 0.01f;
    private float rho = 0.5f;
    private float entropy = 1e-3f;
    private int horizon = 3;
    private int samples = 64;
    private int policySamples = 8;
    private int elites = 8;
    private int iterations = 3;
    private float temperature = 0.5f;
    private int batchSize = 64;
    private int learningStarts = 300;
    private boolean plan = true;
    private float[] low;
    private float[] high;
    private float[] warm;
    private SplittableRandom random;
    private final List<float[]> obsLog = new ArrayList<>();
    private final List<float[]> actLog = new ArrayList<>();
    private final List<Float> rewLog = new ArrayList<>();
    private final List<Boolean> lastLog = new ArrayList<>();

    private TdMpc(int obsDims, int actDims, int latent, int hidden, long seed) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.latent = latent;
        this.random = new SplittableRandom(seed);
        this.encoder = Model.sequential().input(obsDims).dense(hidden, Activation.RELU).dense(latent)
                .layer(LayerNorm.of(latent)).seed(seed).build();
        this.dynamics = Model.sequential().input(latent + actDims).dense(hidden, Activation.RELU)
                .dense(hidden, Activation.RELU).dense(latent).layer(LayerNorm.of(latent)).seed(seed + 1).build();
        this.rewardHead = Nets.mlp(latent + actDims, 1, Activation.RELU, seed + 2, hidden);
        this.policy = Nets.mlp(latent, 2 * actDims, Activation.RELU, seed + 3, hidden);
        this.critics = new CriticEnsemble(latent + actDims, 2, 0, true, seed + 4, hidden);
        modelParams.addAll(encoder.parameters());
        modelParams.addAll(dynamics.parameters());
        modelParams.addAll(rewardHead.parameters());
        modelParams.addAll(critics.parameters());
    }

    /**
     * @param obsDims values per observation
     * @param actDims values per action
     * @param latent latent size
     * @param hidden hidden width
     * @param seed the seed
     * @return the learner
     */
    public static TdMpc of(int obsDims, int actDims, int latent, int hidden, long seed) {
        return new TdMpc(obsDims, actDims, latent, hidden, seed);
    }

    /** @param steps the planning and training horizon @return this learner */
    public TdMpc horizon(int steps) {
        this.horizon = steps;
        return this;
    }

    /**
     * @param sampled random action sequences per MPPI iteration
     * @param fromPolicy sequences from the policy prior
     * @param best elites
     * @param rounds MPPI iterations
     * @return this learner
     */
    public TdMpc search(int sampled, int fromPolicy, int best, int rounds) {
        this.samples = sampled;
        this.policySamples = fromPolicy;
        this.elites = best;
        this.iterations = rounds;
        return this;
    }

    /** @param on plan with MPPI (true) or act with the policy prior alone @return this learner */
    public TdMpc planning(boolean on) {
        this.plan = on;
        return this;
    }

    /** @param steps random steps before learning starts @return this learner */
    public TdMpc learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    private Tensor row(float[] o) {
        return Tensor.of(Shape.of(1, o.length), o);
    }

    private Var[] policyOut(Var z) {
        Var out = policy.forward(z, true);
        Var mean = out.slice(1, 0, actDims);
        Var logStd = out.slice(1, actDims, 2 * actDims).tanh().add(1f).mul(0.5f * (LOG_STD_MAX - LOG_STD_MIN))
                .add(LOG_STD_MIN);
        return new Var[] {mean, logStd};
    }

    /** A squashed policy sample in [-1, 1] and its log-probability. */
    private Var[] samplePolicy(Var z, int rows) {
        Var[] ms = policyOut(z);
        Tensor noise = Tensor.normal(Shape.of(rows, actDims), 0, 1, random);
        Var u = ms[0].add(ms[1].exp().mul(noise));
        Var a = u.tanh();
        Var gauss = ms[1].add((float) (0.5 * Math.log(2 * Math.PI))).neg().sub(Var.of(noise.square().mul(0.5f)));
        Var squash = u.neg().sub(u.mul(-2f).softplus()).add((float) Math.log(2)).mul(2f);
        return new Var[] {a, gauss.sub(squash).sum(1, true)};
    }

    private float[] scale(float[] unit) {
        float[] a = new float[unit.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = low[i] + (Math.max(-1, Math.min(1, unit[i])) + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    @Override
    public void startEpisode() {
        warm = null;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return scale(unitAction(observation, explore));
    }

    private float[] unitAction(float[] observation, boolean explore) {
        Tensor z = encoder.forward(Var.of(row(observation)), false).value();
        if (!plan) {
            Var[] ms = policyOut(Var.of(z));
            float[] m = ms[0].value().to(DType.FLOAT32).toArray();
            float[] ls = ms[1].value().to(DType.FLOAT32).toArray();
            float[] a = new float[actDims];
            for (int i = 0; i < actDims; i++) {
                a[i] = (float) Math.tanh(m[i] + (explore ? Math.exp(ls[i]) * random.nextGaussian() : 0));
            }
            return a;
        }
        return mppi(z, explore);
    }

    private float[] mppi(Tensor z0, boolean explore) {
        int n = samples + policySamples;
        int len = horizon * actDims;
        float[] mean = new float[len];
        float[] std = new float[len];
        for (int i = 0; i < len; i++) {
            mean[i] = warm == null ? 0 : (i + actDims < len ? warm[i + actDims] : 0);
            std[i] = 1f;
        }
        // policy-prior trajectories
        float[][] seqs = new float[n][len];
        Tensor zp = Rows.repeat(z0, policySamples);
        for (int t = 0; t < horizon; t++) {
            float[] a = samplePolicy(Var.of(zp), policySamples)[0].value().to(DType.FLOAT32).toArray();
            for (int p = 0; p < policySamples; p++) {
                System.arraycopy(a, p * actDims, seqs[samples + p], t * actDims, actDims);
            }
            zp = dynamics.forward(Var.of(Tensor.concat(1, zp, Tensor.of(Shape.of(policySamples, actDims), a))), false)
                    .value();
        }
        double[] value = new double[n];
        Tensor zs = Rows.repeat(z0, n);
        Integer[] order = new Integer[n];
        for (int it = 0; it < iterations; it++) {
            for (int p = 0; p < samples; p++) {
                for (int i = 0; i < len; i++) {
                    seqs[p][i] = (float) Math.max(-1, Math.min(1, mean[i] + std[i] * random.nextGaussian()));
                }
            }
            Arrays.fill(value, 0);
            Tensor z = zs;
            double disc = 1;
            for (int t = 0; t < horizon; t++) {
                float[] a = new float[n * actDims];
                for (int p = 0; p < n; p++) {
                    System.arraycopy(seqs[p], t * actDims, a, p * actDims, actDims);
                }
                Tensor za = Tensor.concat(1, z, Tensor.of(Shape.of(n, actDims), a));
                float[] r = rewardHead.forward(Var.of(za), false).value().to(DType.FLOAT32).toArray();
                for (int p = 0; p < n; p++) {
                    value[p] += disc * r[p];
                }
                z = dynamics.forward(Var.of(za), false).value();
                disc *= gamma;
            }
            Tensor aEnd = samplePolicy(Var.of(z), n)[0].value();
            float[] q = critics.mean(Var.of(Tensor.concat(1, z, aEnd)), false).value().to(DType.FLOAT32).toArray();
            for (int p = 0; p < n; p++) {
                value[p] += disc * q[p];
                order[p] = p;
            }
            Arrays.sort(order, (x, y) -> Double.compare(value[y], value[x]));
            double best = value[order[0]];
            double[] w = new double[elites];
            double wsum = 0;
            for (int e = 0; e < elites; e++) {
                w[e] = Math.exp(temperature * (value[order[e]] - best));
                wsum += w[e];
            }
            for (int i = 0; i < len; i++) {
                double m = 0;
                for (int e = 0; e < elites; e++) {
                    m += w[e] * seqs[order[e]][i];
                }
                m /= wsum;
                double v = 0;
                for (int e = 0; e < elites; e++) {
                    v += w[e] * (seqs[order[e]][i] - m) * (seqs[order[e]][i] - m);
                }
                mean[i] = (float) m;
                std[i] = (float) Math.max(0.05, Math.sqrt(v / wsum));
            }
        }
        warm = mean;
        float[] a = Arrays.copyOf(mean, actDims);
        if (explore) {
            for (int i = 0; i < actDims; i++) {
                a[i] = (float) Math.max(-1, Math.min(1, a[i] + std[i] * random.nextGaussian()));
            }
        }
        return a;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Box box) || box.size() != actDims
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("needs " + obsDims + " observation values and a box of "
                    + actDims + " actions");
        }
        low = box.low();
        high = box.high();
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        startEpisode();
        long taken = 0;
        while (recorder.steps() < steps) {
            float[] u;
            if (taken < learningStarts) {
                u = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    u[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                u = unitAction(obs, true);
            }
            Environment.Step s = env.step(scale(u));
            taken++;
            obsLog.add(obs);
            actLog.add(u);
            rewLog.add(s.reward());
            lastLog.add(false);
            boolean over = recorder.step(s);
            if (over) {
                obsLog.add(s.observation());
                actLog.add(new float[actDims]);
                rewLog.add(0f);
                lastLog.add(true);
                obs = env.reset(random.split());
                startEpisode();
            } else {
                obs = s.observation();
            }
            if (taken >= learningStarts) {
                update();
            }
        }
        return recorder.result();
    }

    /** A start index whose next {@code horizon} transitions stay within one episode. */
    private int sampleStart() {
        while (true) {
            int i = random.nextInt(obsLog.size() - horizon);
            boolean ok = true;
            for (int t = 0; t < horizon; t++) {
                if (lastLog.get(i + t)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                return i;
            }
        }
    }

    private void update() {
        int b = batchSize;
        int[] starts = new int[b];
        for (int i = 0; i < b; i++) {
            starts[i] = sampleStart();
        }
        Tensor[] o = new Tensor[horizon + 1];
        Tensor[] a = new Tensor[horizon];
        Tensor[] r = new Tensor[horizon];
        for (int t = 0; t <= horizon; t++) {
            float[] ov = new float[b * obsDims];
            float[] av = new float[b * actDims];
            float[] rv = new float[b];
            for (int i = 0; i < b; i++) {
                System.arraycopy(obsLog.get(starts[i] + t), 0, ov, i * obsDims, obsDims);
                if (t < horizon) {
                    System.arraycopy(actLog.get(starts[i] + t), 0, av, i * actDims, actDims);
                    rv[i] = rewLog.get(starts[i] + t);
                }
            }
            o[t] = Tensor.of(Shape.of(b, obsDims), ov);
            if (t < horizon) {
                a[t] = Tensor.of(Shape.of(b, actDims), av);
                r[t] = Tensor.of(Shape.of(b, 1), rv);
            }
        }
        // targets from the real next observations, without gradients
        Tensor allNext = encoder.forward(Var.of(Tensor.concat(0, java.util.Arrays.copyOfRange(o, 1, horizon + 1))),
                false).value();
        Tensor allA = samplePolicy(Var.of(allNext), b * horizon)[0].value();
        Tensor allQ = critics.targetMin(Tensor.concat(1, allNext, allA), 2, random);
        Tensor[] next = new Tensor[horizon];
        Tensor[] td = new Tensor[horizon];
        for (int t = 0; t < horizon; t++) {
            next[t] = allNext.slice(0, t * b, (t + 1) * b);
            td[t] = r[t].add(allQ.slice(0, t * b, (t + 1) * b).mul(gamma));
        }
        Var z = encoder.forward(Var.of(o[0]), true);
        Var loss = null;
        float w = 1;
        for (int t = 0; t < horizon; t++) {
            Var za = Var.concat(1, z, Var.of(a[t]));
            Var rew = rewardHead.forward(za, true).sub(r[t]).square().mean();
            Var val = critics.loss(za, td[t]);
            Var zNext = dynamics.forward(za, true);
            Var cons = zNext.sub(next[t]).square().sum(1, true).mean();
            Var term = cons.mul(20f).add(rew.mul(0.1f)).add(val.mul(0.1f)).mul(w);
            loss = loss == null ? term : loss.add(term);
            z = zNext;
            w *= rho;
        }
        RlTensors.zeroGrad(modelParams);
        loss.mul(1f / horizon).backward();
        RlTensors.clip(modelParams, 20f);
        modelOpt.step(modelParams);
        critics.blend(tau);

        // policy prior on the detached latents
        Var zs = Var.of(encoder.forward(Var.of(o[0]), false).value());
        Var[] pa = samplePolicy(zs, b);
        Var q = critics.mean(Var.concat(1, zs, pa[0]), true);
        Var policyLoss = pa[1].mul(entropy).sub(q).mean();
        RlTensors.zeroGrad(policy.parameters());
        policyLoss.backward();
        policyOpt.step(policy.parameters());
        RlTensors.zeroGrad(critics.parameters());
    }

    @Override
    public String toString() {
        return "TD-MPC(latent " + latent + ", horizon " + horizon + ")";
    }

    /** Row repetition for batched planning. */
    private static final class Rows {

        private Rows() {
        }

        static Tensor repeat(Tensor row, int n) {
            float[] v = row.to(DType.FLOAT32).toArray();
            float[] out = new float[n * v.length];
            for (int i = 0; i < n; i++) {
                System.arraycopy(v, 0, out, i * v.length, v.length);
            }
            return Tensor.of(Shape.of(n, v.length), out);
        }
    }
}
