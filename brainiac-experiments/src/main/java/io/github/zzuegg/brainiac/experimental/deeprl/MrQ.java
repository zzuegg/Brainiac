package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.HlGaussValue;
import io.github.zzuegg.brainiac.experimental.rl.MlpWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MR.Q (Fujimoto et al., "Towards General-Purpose Model-Free Reinforcement Learning", ICLR 2025,
 * <a href="https://arxiv.org/abs/2501.16142">arXiv:2501.16142</a>): model-based representations
 * for model-free Q-learning, with one hyperparameter set for continuous and discrete actions.
 *
 * <p>A state encoder {@code z_s = f(s)} and a state-action encoder {@code z_sa = g(z_s, a)}
 * (both AvgL1-normalised) are trained, over a short unrolled horizon, by three model-based
 * losses through a linear predictor on {@code z_sa}: the next state's embedding (mean squared
 * error to a target encoder), the reward (cross-entropy to a two-hot encoding over
 * symexp-spaced bins, reusing {@link HlGaussValue} and {@link MlpWorldModel#symlog(float)}) and
 * termination. Because these targets make the value approximately linear in {@code z_sa}, the
 * value networks on top of the frozen-for-the-value-update embeddings learn fast. The value is
 * TD3-style (two critics, clipped double Q, target policy noise) on multi-step returns with a
 * Huber loss, rewards are divided by the buffer's mean absolute reward, replay is
 * {@link LapReplayBuffer}, and discrete actions use a Gumbel-softmax policy whose relaxed
 * one-hot output feeds {@code g}.
 *
 * <p>{@link #modelLosses(boolean) modelLosses(false)} is the ablation: the same networks
 * trained end to end through the value loss only.
 */
public final class MrQ implements Learner {

    private static final int BINS = 41;

    private final int obsDims;
    private final int actDims;
    private final boolean discrete;
    private final int hidden;
    private final int zDim;
    private boolean modelLosses = true;
    private int horizon = 3;
    private int nStep = 3;
    private float gamma = 0.99f;
    private float lr = 1e-3f;
    private int batchSize = 64;
    private int learningStarts = 500;
    private int targetEvery = 250;
    private double epsilon = 0.05;
    private SplittableRandom random = new SplittableRandom(0);
    private long seed;
    private final HlGaussValue rewardBins = HlGaussValue.of(-10f, 10f, BINS);

    private Model f;
    private Model g;
    private Model predictor;
    private Model fTarget;
    private Model gTarget;
    private Model q1;
    private Model q2;
    private Model q1Target;
    private Model q2Target;
    private Model policy;
    private Model policyTarget;
    private Optimizer encOpt;
    private Optimizer valueOpt;
    private Optimizer policyOpt;
    private LapReplayBuffer buffer;
    private float[] low;
    private float[] high;
    private long updates;
    private long stepsTaken;
    private double rewardScale = 1;

    private MrQ(int obsDims, int actDims, boolean discrete, int hidden, int zDim) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.discrete = discrete;
        this.hidden = hidden;
        this.zDim = zDim;
    }

    /**
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param hidden hidden width
     * @param embedding embedding size
     * @return MR.Q for box actions
     */
    public static MrQ continuous(int observationDims, int actionDims, int hidden, int embedding) {
        return new MrQ(observationDims, actionDims, false, hidden, embedding);
    }

    /**
     * @param observationDims values per observation
     * @param actions the number of discrete actions
     * @param hidden hidden width
     * @param embedding embedding size
     * @return MR.Q for discrete actions
     */
    public static MrQ discrete(int observationDims, int actions, int hidden, int embedding) {
        return new MrQ(observationDims, actions, true, hidden, embedding);
    }

    /** @param on whether the encoders learn from the model losses (else end to end from the value) @return this */
    public MrQ modelLosses(boolean on) {
        this.modelLosses = on;
        return this;
    }

    /** @param steps the encoder's unrolled horizon @return this learner */
    public MrQ horizon(int steps) {
        this.horizon = Math.max(1, steps);
        return this;
    }

    /** @param n steps per value target @return this learner */
    public MrQ nStep(int n) {
        this.nStep = Math.max(1, n);
        return this;
    }

    /** @param seed the seed @return this learner */
    public MrQ seed(long seed) {
        this.seed = seed;
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @param steps random steps first @return this learner */
    public MrQ learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public MrQ learningRate(float rate) {
        this.lr = rate;
        return this;
    }

    /** @param updates gradient updates between hard target updates @return this learner */
    public MrQ targetEvery(int updates) {
        this.targetEvery = updates;
        return this;
    }

    /** @param discount gamma @return this learner */
    public MrQ discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @return the gradient updates taken */
    public long updates() {
        return updates;
    }

    private Model net(int in, int out, long s) {
        return RlNetworkFactory.mlp(Activation.RELU, hidden, hidden).create(in, out, seed * 1000 + s);
    }

    private void build() {
        if (f != null) {
            return;
        }
        f = net(obsDims, zDim, 1);
        g = net(zDim + actDims, zDim, 2);
        fTarget = net(obsDims, zDim, 1);
        gTarget = net(zDim + actDims, zDim, 2);
        predictor = Model.sequential().input(zDim).dense(zDim + BINS + 1).seed(seed * 1000 + 3).build();
        q1 = net(zDim, 1, 4);
        q2 = net(zDim, 1, 5);
        q1Target = net(zDim, 1, 4);
        q2Target = net(zDim, 1, 5);
        policy = net(zDim, actDims, 6);
        policyTarget = net(zDim, actDims, 6);
        encOpt = Optimizer.adam(lr);
        valueOpt = Optimizer.adam(lr);
        policyOpt = Optimizer.adam(lr);
    }

    private static Var avgL1(Var x) {
        return x.div(x.abs().mean(1, true).add(1e-6f));
    }

    private Tensor embed(Model enc, Tensor s) {
        return Var.noGrad(() -> avgL1(enc.forward(Var.of(s), false))).value();
    }

    private Tensor embedSa(Model enc, Tensor zs, Tensor a) {
        return Var.noGrad(() -> avgL1(enc.forward(Var.of(Tensor.concat(1, zs, a)), false))).value();
    }

    /** The policy's action for embeddings: tanh for boxes, a (relaxed) one-hot for discrete actions. */
    private Var policyAction(Model pi, Var zs, boolean gumbel) {
        Var out = pi.forward(zs, true);
        if (!discrete) {
            return out.tanh();
        }
        if (!gumbel) {
            Tensor hard = RlTensors.oneHot(argmaxRows(out.value()), actDims, out.value().device());
            return Var.of(hard);
        }
        Tensor u = Tensor.uniform(out.shape(), 1e-6f, 1 - 1e-6f, random);
        Tensor gumbelNoise = u.log().neg().log().neg();
        return RlTensors.softmax(out.add(gumbelNoise).div(1f));
    }

    private static float[] argmaxRows(Tensor logits) {
        int rows = logits.shape().dim(0);
        int cols = logits.shape().dim(1);
        float[] v = logits.toArray();
        float[] out = new float[rows];
        for (int r = 0; r < rows; r++) {
            int best = 0;
            for (int c = 1; c < cols; c++) {
                if (v[r * cols + c] > v[r * cols + best]) {
                    best = c;
                }
            }
            out[r] = best;
        }
        return out;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        build();
        Tensor zs = embed(f, Tensor.of(Shape.of(1, obsDims), observation));
        float[] out = Var.noGrad(() -> policy.forward(Var.of(zs), false)).value().toArray();
        if (discrete) {
            int a = explore && random.nextDouble() < epsilon ? random.nextInt(actDims) : RlTensors.argmax(out);
            return new float[] {a};
        }
        float[] unit = new float[actDims];
        for (int i = 0; i < actDims; i++) {
            double u = Math.tanh(out[i]) + (explore ? 0.2 * random.nextGaussian() : 0);
            unit[i] = (float) Math.max(-1, Math.min(1, u));
        }
        return low == null ? unit : scale(unit);
    }

    private float[] scale(float[] unit) {
        float[] a = new float[unit.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = low[i] + (unit[i] + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    private float[] stored(float[] envAction) {
        if (discrete) {
            float[] h = new float[actDims];
            h[Math.round(envAction[0])] = 1;
            return h;
        }
        float[] u = new float[actDims];
        for (int i = 0; i < u.length; i++) {
            u[i] = 2 * (envAction[i] - low[i]) / (high[i] - low[i]) - 1;
        }
        return u;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        build();
        if (discrete) {
            if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actDims) {
                throw new IllegalArgumentException("MR.Q (discrete) needs " + actDims + " actions");
            }
        } else {
            if (!(env.actionSpace() instanceof Space.Box box) || box.size() != actDims) {
                throw new IllegalArgumentException("MR.Q needs a box of " + actDims + " actions");
            }
            low = box.low();
            high = box.high();
        }
        if (buffer == null) {
            buffer = new LapReplayBuffer(100_000, obsDims, actDims, 0.4, true);
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        while (recorder.steps() < steps) {
            float[] a;
            if (stepsTaken < learningStarts) {
                if (discrete) {
                    a = new float[] {random.nextInt(actDims)};
                } else {
                    a = new float[actDims];
                    for (int i = 0; i < actDims; i++) {
                        a[i] = (float) random.nextDouble(-1, 1);
                    }
                    a = scale(a);
                }
            } else {
                a = act(obs, true);
            }
            Environment.Step s = env.step(a);
            stepsTaken++;
            buffer.add(obs, stored(a), s.reward(), s.observation(), s.terminated(), s.done());
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (stepsTaken >= learningStarts && buffer.size() >= batchSize + Math.max(horizon, nStep)) {
                update();
            }
        }
        return recorder.result();
    }

    private void update() {
        Device device = f.device();
        int span = Math.max(horizon, nStep);
        int[] idx = buffer.sample(batchSize, span, random);
        LapReplayBuffer.Batch[] seq = new LapReplayBuffer.Batch[span];
        for (int k = 0; k < span; k++) {
            seq[k] = buffer.gather(idx, k, device);
        }
        if (updates % targetEvery == 0) {
            rewardScale = Math.max(1e-3, buffer.meanAbsReward());
        }
        List<Var> encParams = new ArrayList<>(f.parameters());
        encParams.addAll(g.parameters());
        encParams.addAll(predictor.parameters());
        if (modelLosses) {
            Var zs = avgL1(f.forward(Var.of(seq[0].observations()), true));
            Var loss = null;
            for (int k = 0; k < horizon; k++) {
                LapReplayBuffer.Batch b = seq[k];
                Var zsa = avgL1(g.forward(Var.concat(1, zs, Var.of(b.actions())), true));
                Var pred = predictor.forward(zsa, true);
                Var nextZ = pred.slice(1, 0, zDim);
                Var rewardLogits = pred.slice(1, zDim, zDim + BINS);
                Var done = pred.slice(1, zDim + BINS, zDim + BINS + 1);
                Tensor targetZ = embed(fTarget, b.nextObservations());
                float[] symlog = new float[batchSize];
                for (int i = 0; i < batchSize; i++) {
                    symlog[i] = MlpWorldModel.symlog(b.rawRewards()[i]);
                }
                Tensor symlogR = Tensor.of(Shape.of(batchSize, 1), symlog).to(device);
                Tensor m = b.mask();
                Var dyn = nextZ.sub(targetZ).square().mean(1, true).mul(m).mean();
                Var rew = rewardBins.loss(rewardLogits.mul(m), symlogR.mul(m));
                Var term = done.sub(Tensor.ones(m.shape()).sub(b.notDone())).square().mul(m).mean();
                Var stepLoss = dyn.add(rew.mul(0.1f)).add(term.mul(0.1f));
                loss = loss == null ? stepLoss : loss.add(stepLoss);
                zs = nextZ;
            }
            loss.backward();
            encOpt.step(encParams);
        }
        // multi-step targets
        float[] ret = new float[batchSize];
        float[] boot = new float[batchSize];
        float[][] bootObs = new float[batchSize][];
        float[][] nextObsRows = new float[nStep][];
        for (int k = 0; k < nStep; k++) {
            nextObsRows[k] = seq[k].nextObservations().toArray();
        }
        for (int i = 0; i < batchSize; i++) {
            float disc = 1;
            int last = 0;
            boolean alive = true;
            for (int k = 0; k < nStep && alive; k++) {
                if (seq[k].rawMask()[i] == 0) {
                    break;
                }
                ret[i] += (float) (disc * seq[k].rawRewards()[i] / rewardScale);
                disc *= gamma;
                last = k;
                if (seq[k].rawNotDone()[i] == 0) {
                    alive = false;
                    disc = 0;
                }
            }
            boot[i] = disc;
            bootObs[i] = new float[obsDims];
            System.arraycopy(nextObsRows[last], i * obsDims, bootObs[i], 0, obsDims);
        }
        Tensor bootObsT = RlTensors.rows(bootObs, obsDims, device);
        Tensor zsn = embed(fTarget, bootObsT);
        Tensor an = Var.noGrad(() -> policyAction(policyTarget, Var.of(zsn), false)).value();
        if (!discrete) {
            Tensor noise = Tensor.normal(an.shape(), 0, 0.2f, random).maximum(Tensor.scalar(-0.3f))
                    .minimum(Tensor.scalar(0.3f));
            an = an.add(noise).maximum(Tensor.scalar(-1)).minimum(Tensor.scalar(1));
        }
        Tensor zsan = embedSa(gTarget, zsn, an);
        Tensor qn = Var.noGrad(() -> q1Target.forward(Var.of(zsan), false)).value()
                .minimum(Var.noGrad(() -> q2Target.forward(Var.of(zsan), false)).value());
        Tensor y = RlTensors.column(ret, device).add(RlTensors.column(boot, device).mul(qn));
        LapReplayBuffer.Batch b0 = seq[0];
        Var zsaIn;
        List<Var> valueParams = new ArrayList<>(q1.parameters());
        valueParams.addAll(q2.parameters());
        if (modelLosses) {
            zsaIn = Var.of(embedSa(g, embed(f, b0.observations()), b0.actions()));
        } else {
            Var zs = avgL1(f.forward(Var.of(b0.observations()), true));
            zsaIn = avgL1(g.forward(Var.concat(1, zs, Var.of(b0.actions())), true));
            valueParams.addAll(f.parameters());
            valueParams.addAll(g.parameters());
        }
        Var d1 = q1.forward(zsaIn, true).sub(y);
        Var d2 = q2.forward(zsaIn, true).sub(y);
        Var vLoss = RlTensors.huber(d1, 1f).add(RlTensors.huber(d2, 1f)).mean();
        vLoss.backward();
        valueOpt.step(valueParams);
        float[] e1 = d1.value().abs().toArray();
        float[] e2 = d2.value().abs().toArray();
        float[] e = new float[batchSize];
        for (int i = 0; i < batchSize; i++) {
            e[i] = Math.max(e1[i], e2[i]);
        }
        buffer.updatePriorities(idx, e);

        Tensor zsPolicy = embed(f, b0.observations());
        Var pa = policyAction(policy, Var.of(zsPolicy), true);
        Var zsaPi = avgL1(g.forward(Var.concat(1, Var.of(zsPolicy), pa), false));
        Var q = q1.forward(zsaPi, false).add(q2.forward(zsaPi, false)).mul(0.5f);
        Var pLoss = q.mean().neg();
        pLoss.backward();
        policyOpt.step(policy.parameters());
        RlTensors.zeroGrad(valueParams);
        RlTensors.zeroGrad(encParams);
        updates++;
        if (updates % targetEvery == 0) {
            RlTensors.copy(f.parameters(), fTarget.parameters());
            RlTensors.copy(g.parameters(), gTarget.parameters());
            RlTensors.copy(q1.parameters(), q1Target.parameters());
            RlTensors.copy(q2.parameters(), q2Target.parameters());
            RlTensors.copy(policy.parameters(), policyTarget.parameters());
        }
    }

    @Override
    public String toString() {
        return "MR.Q(" + (discrete ? "discrete" : "continuous") + (modelLosses ? "" : ", no model losses") + ")";
    }
}
