package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A generative world model trained with shortcut forcing, after Dreamer 4 (Hafner et al.,
 * 2025, <a href="https://arxiv.org/abs/2509.24527">arXiv:2509.24527</a>): the next observation
 * is <em>sampled</em> by a flow from noise, conditioned on the observation and action, and
 * the flow network also takes the step size, so it can jump in one or a few steps where plain
 * flow matching needs many. The targets are those of shortcut models (Frans et al., 2024, as
 * in {@code experimental.nn.diffusion.MeanFlow#shortcutLoss}, here with a condition): flow
 * matching at step size 0 for part of the batch, and self-consistency for the rest: one step
 * of size {@code 2d} must equal two chained steps of size {@code d} (stop-gradient target).
 * With {@link #shortcut(boolean) shortcut off} it is plain conditional flow matching, the
 * baseline. A small MLP stands in for Dreamer 4's transformer.
 *
 * <p>Conventions follow core flow matching: time 0 is data, 1 is noise. The model generates
 * the standardized change of observation; a separate head predicts the reward. It implements
 * {@link ImaginationTrainer.Simulator} so a policy can be trained purely in imagination.
 */
public final class ShortcutWorldModel implements ImaginationTrainer.Simulator {

    private static final int FREQS = 8;

    private final int obsDims;
    private final int actions;
    private final Model velocity;
    private final Model rewardHead;
    private final Optimizer optimizer = Optimizer.adam(2e-3f);
    private final float[] mean;
    private final float[] std;
    private final float[] obsMean;
    private final float[] obsStd;
    private boolean shortcut = true;
    private float flowShare = 0.75f;
    private int levels = 4;
    private int sampleSteps = 2;
    private SplittableRandom random;

    private ShortcutWorldModel(int obsDims, int actions, int width, long seed) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.random = new SplittableRandom(seed);
        int in = obsDims + obsDims + actions + 4 * FREQS;
        this.velocity = Nets.mlp(in, obsDims, Activation.RELU, seed, width, width, width);
        this.rewardHead = Nets.mlp(obsDims + actions, 1, Activation.RELU, seed + 1, width);
        this.mean = new float[obsDims];
        this.std = new float[obsDims];
        java.util.Arrays.fill(std, 1f);
        this.obsMean = new float[obsDims];
        this.obsStd = new float[obsDims];
        java.util.Arrays.fill(obsStd, 1f);
    }

    /**
     * @param obsDims values per observation
     * @param actions discrete actions
     * @param width hidden width
     * @param seed the seed
     * @return the model
     */
    public static ShortcutWorldModel of(int obsDims, int actions, int width, long seed) {
        return new ShortcutWorldModel(obsDims, actions, width, seed);
    }

    /** @param on train self-consistency targets (false: plain flow matching) @return this model */
    public ShortcutWorldModel shortcut(boolean on) {
        this.shortcut = on;
        return this;
    }

    /** @param steps sampling steps used by {@link #step} @return this model */
    public ShortcutWorldModel sampleSteps(int steps) {
        this.sampleSteps = steps;
        return this;
    }

    private Tensor features(float[] t, float[] gap) {
        int n = t.length;
        float[] f = new float[n * 4 * FREQS];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < FREQS; k++) {
                double w = Math.pow(8, (double) k / (FREQS - 1));
                f[i * 4 * FREQS + k] = (float) Math.sin(w * t[i]);
                f[i * 4 * FREQS + FREQS + k] = (float) Math.cos(w * t[i]);
                f[i * 4 * FREQS + 2 * FREQS + k] = (float) Math.sin(w * gap[i]);
                f[i * 4 * FREQS + 3 * FREQS + k] = (float) Math.cos(w * gap[i]);
            }
        }
        return Tensor.of(Shape.of(n, 4 * FREQS), f);
    }

    private Var u(Tensor x, Tensor cond, float[] r, float[] t, boolean training) {
        float[] gap = new float[t.length];
        for (int i = 0; i < t.length; i++) {
            gap[i] = t[i] - r[i];
        }
        return velocity.forward(Var.of(Tensor.concat(1, x, cond, features(t, gap))), training);
    }

    private Tensor condition(Tensor obs, int[] acts) {
        float[] o = obs.to(DType.FLOAT32).toArray();
        int n = acts.length;
        float[] c = new float[n * (obsDims + actions)];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < obsDims; k++) {
                c[i * (obsDims + actions) + k] = (o[i * obsDims + k] - obsMean[k]) / obsStd[k];
            }
            c[i * (obsDims + actions) + obsDims + acts[i]] = 1;
        }
        return Tensor.of(Shape.of(n, obsDims + actions), c);
    }

    /**
     * Trains on logged transitions (discrete actions).
     *
     * @param data the transitions
     * @param steps gradient steps
     * @param batch transitions per step
     * @return this model
     */
    public ShortcutWorldModel train(List<Transition> data, int steps, int batch) {
        double[] s = new double[obsDims];
        double[] sq = new double[obsDims];
        double[] os = new double[obsDims];
        double[] osq = new double[obsDims];
        for (Transition tr : data) {
            for (int k = 0; k < obsDims; k++) {
                double d = tr.nextObservation()[k] - tr.observation()[k];
                s[k] += d;
                sq[k] += d * d;
                os[k] += tr.observation()[k];
                osq[k] += tr.observation()[k] * tr.observation()[k];
            }
        }
        for (int k = 0; k < obsDims; k++) {
            obsMean[k] = (float) (os[k] / data.size());
            obsStd[k] = (float) Math.sqrt(Math.max(1e-6, osq[k] / data.size() - obsMean[k] * obsMean[k]));
        }
        float[] dMean = new float[obsDims];
        float[] dStd = new float[obsDims];
        for (int k = 0; k < obsDims; k++) {
            dMean[k] = (float) (s[k] / data.size());
            dStd[k] = (float) Math.sqrt(Math.max(1e-6, sq[k] / data.size() - dMean[k] * dMean[k]));
        }
        System.arraycopy(dMean, 0, mean, 0, obsDims);
        System.arraycopy(dStd, 0, std, 0, obsDims);
        List<Var> params = new java.util.ArrayList<>(velocity.parameters());
        params.addAll(rewardHead.parameters());
        for (int it = 0; it < steps; it++) {
            float[] obs = new float[batch * obsDims];
            float[] x0 = new float[batch * obsDims];
            float[] rw = new float[batch];
            int[] acts = new int[batch];
            for (int i = 0; i < batch; i++) {
                Transition tr = data.get(random.nextInt(data.size()));
                acts[i] = Math.round(tr.action()[0]);
                rw[i] = tr.reward();
                for (int k = 0; k < obsDims; k++) {
                    obs[i * obsDims + k] = tr.observation()[k];
                    x0[i * obsDims + k] = (tr.nextObservation()[k] - tr.observation()[k] - mean[k]) / std[k];
                }
            }
            Tensor cond = condition(Tensor.of(Shape.of(batch, obsDims), obs), acts);
            Var loss = flowLoss(Tensor.of(Shape.of(batch, obsDims), x0), cond, batch);
            Var rl = rewardHead.forward(Var.of(cond), true).sub(Tensor.of(Shape.of(batch, 1), rw)).square().mean();
            loss = loss.add(rl);
            RlTensors.zeroGrad(params);
            loss.backward();
            RlTensors.clip(params, 10f);
            optimizer.step(params);
        }
        return this;
    }

    private Var flowLoss(Tensor x0, Tensor cond, int batch) {
        int flow = shortcut ? Math.max(1, Math.min(batch - 1, Math.round(batch * flowShare))) : batch;
        int self = batch - flow;
        Tensor noise = Tensor.normal(x0.shape(), 0, 1, random);
        float[] t = new float[batch];
        float[] r = new float[batch];
        float[] half = new float[self];
        for (int i = 0; i < batch; i++) {
            if (i < flow) {
                t[i] = (float) random.nextDouble();
                r[i] = t[i];
            } else {
                float size = (float) Math.pow(0.5, random.nextInt(levels));
                int slots = Math.round(1 / size);
                t[i] = size * (1 + random.nextInt(slots));
                r[i] = t[i] - size;
                half[i - flow] = size / 2;
            }
        }
        Tensor tc = Tensor.of(Shape.of(batch, 1), t.clone());
        Tensor zt = x0.mul(tc.neg().add(1f)).add(noise.mul(tc));
        Var u = u(zt, cond, r, t, true);
        Var flowErr = u.slice(0, 0, flow).sub(Var.of(noise.sub(x0).slice(0, 0, flow))).square().mean();
        if (self == 0) {
            return flowErr;
        }
        Tensor zs = zt.slice(0, flow, batch);
        Tensor cs = cond.slice(0, flow, batch);
        float[] ts = java.util.Arrays.copyOfRange(t, flow, batch);
        float[] mid = new float[self];
        float[] end = new float[self];
        for (int i = 0; i < self; i++) {
            mid[i] = ts[i] - half[i];
            end[i] = mid[i] - half[i];
        }
        Tensor h = Tensor.of(Shape.of(self, 1), half);
        Tensor u1 = u(zs, cs, mid, ts, false).value();
        Tensor zm = zs.sub(u1.mul(h));
        Tensor u2 = u(zm, cs, end, mid, false).value();
        Tensor target = u1.add(u2).mul(0.5f);
        Var selfErr = u.slice(0, flow, batch).sub(Var.of(target)).square().mean();
        return flowErr.mul((float) flow / batch).add(selfErr.mul((float) self / batch));
    }

    /**
     * Samples next observations.
     *
     * @param obs observations {@code [n, obs]}
     * @param acts actions
     * @param steps flow steps (1 for a single jump)
     * @return sampled next observations
     */
    public Tensor sample(Tensor obs, int[] acts, int steps) {
        int n = acts.length;
        Tensor cond = condition(obs, acts);
        Tensor x = Tensor.normal(Shape.of(n, obsDims), 0, 1, random);
        for (int i = 0; i < steps; i++) {
            float tv = 1 - (float) i / steps;
            float rv = 1 - (float) (i + 1) / steps;
            float[] t = new float[n];
            float[] r = new float[n];
            java.util.Arrays.fill(t, tv);
            java.util.Arrays.fill(r, rv);
            x = x.sub(u(x, cond, r, t, false).value().mul(tv - rv));
        }
        float[] d = x.to(DType.FLOAT32).toArray();
        float[] o = obs.to(DType.FLOAT32).toArray();
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < obsDims; k++) {
                o[i * obsDims + k] += d[i * obsDims + k] * std[k] + mean[k];
            }
        }
        return Tensor.of(Shape.of(n, obsDims), o);
    }

    @Override
    public Tensor[] step(Tensor obs, int[] acts) {
        Tensor next = sample(obs, acts, sampleSteps);
        Tensor reward = rewardHead.forward(Var.of(condition(obs, acts)), false).value();
        return new Tensor[] {next, reward};
    }
}
