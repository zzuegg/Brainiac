package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Soft actor-critic with a high update-to-data ratio (UTD: critic gradient steps per
 * environment step), in the two forms that make many updates per sample stable:
 * <ul>
 *   <li>REDQ (Chen et al., 2021, <a href="https://arxiv.org/abs/2101.05982">arXiv:2101.05982</a>):
 *       an ensemble of {@code N} critics whose target is the minimum of a random subset of
 *       {@code M}, which keeps the bias of the target near zero;</li>
 *   <li>DroQ (Hiraoka et al., 2022, <a href="https://arxiv.org/abs/2110.02034">arXiv:2110.02034</a>):
 *       only two critics, each with dropout and layer normalization, the cheaper way to the
 *       same effect.</li>
 * </ul>
 * With {@link #sac} it is plain SAC with clipped double Q at any UTD, the baseline. The
 * actor and temperature are updated once per environment step, after the critic steps. The
 * critics are a {@link CriticEnsemble}. {@link #qBias} measures how far the critics'
 * estimate is from the Monte Carlo return of the policy, the overestimation REDQ and DroQ
 * fight.
 *
 * <p>A separate class rather than a {@link Sac} option because {@code Sac} is final and has
 * exactly two critics.
 */
public final class HighUtdSac implements Learner {

    private static final float LOG_STD_MIN = -5f;
    private static final float LOG_STD_MAX = 2f;
    private static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2 * Math.PI));

    private final String name;
    private final int obsDims;
    private final int actDims;
    private final Model actor;
    private final CriticEnsemble critics;
    private final int subset;
    private final Var logAlpha;
    private Optimizer actorOpt = Optimizer.adam(3e-3f);
    private Optimizer criticOpt = Optimizer.adam(3e-3f);
    private Optimizer alphaOpt = Optimizer.adam(3e-3f);
    private float gamma = 0.99f;
    private float tau = 0.005f;
    private int batchSize = 64;
    private int learningStarts = 300;
    private int utd = 1;
    private final float targetEntropy;
    private float[] low;
    private float[] high;
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long updates;

    private HighUtdSac(String name, int obsDims, int actDims, CriticEnsemble critics, int subset, int[] hidden) {
        this.name = name;
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.actor = Nets.mlp(obsDims, 2 * actDims, Activation.RELU, 11, hidden);
        this.critics = critics;
        this.subset = subset;
        this.logAlpha = Var.param(Tensor.full(Shape.of(1, 1), (float) Math.log(0.2)));
        this.targetEntropy = -actDims;
    }

    /**
     * Plain SAC: two critics, the minimum of both in the target.
     *
     * @param obsDims values per observation
     * @param actDims values per action
     * @param hidden the hidden layer sizes of actor and critics
     * @return the learner
     */
    public static HighUtdSac sac(int obsDims, int actDims, int... hidden) {
        return new HighUtdSac("SAC", obsDims, actDims, new CriticEnsemble(obsDims + actDims, 2, 0, false, 12, hidden),
                2, hidden.clone());
    }

    /**
     * REDQ: {@code ensemble} critics, the minimum of a random {@code subset} in the target.
     *
     * @param obsDims values per observation
     * @param actDims values per action
     * @param ensemble the number of critics (10 in the paper)
     * @param subset critics in the target minimum (2 in the paper)
     * @param hidden the hidden layer sizes of actor and critics
     * @return the learner
     */
    public static HighUtdSac redq(int obsDims, int actDims, int ensemble, int subset, int... hidden) {
        return new HighUtdSac("REDQ", obsDims, actDims,
                new CriticEnsemble(obsDims + actDims, ensemble, 0, false, 12, hidden), subset, hidden.clone());
    }

    /**
     * DroQ: two critics with dropout and layer normalization.
     *
     * @param obsDims values per observation
     * @param actDims values per action
     * @param dropout the dropout rate (0.01 in the paper)
     * @param hidden the hidden layer sizes of actor and critics
     * @return the learner
     */
    public static HighUtdSac droq(int obsDims, int actDims, float dropout, int... hidden) {
        return new HighUtdSac("DroQ", obsDims, actDims,
                new CriticEnsemble(obsDims + actDims, 2, dropout, true, 12, hidden), 2, hidden.clone());
    }

    /** @param ratio critic gradient steps per environment step @return this learner */
    public HighUtdSac updateToData(int ratio) {
        this.utd = Math.max(1, ratio);
        return this;
    }

    /** @param lr the learning rate of actor, critics and temperature @return this learner */
    public HighUtdSac learningRate(float lr) {
        actorOpt = Optimizer.adam(lr);
        criticOpt = Optimizer.adam(lr);
        alphaOpt = Optimizer.adam(lr);
        return this;
    }

    /** @param discount the discount gamma @return this learner */
    public HighUtdSac discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public HighUtdSac batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param steps random steps before learning starts @return this learner */
    public HighUtdSac learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param seed the seed @return this learner */
    public HighUtdSac seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return critic gradient steps taken */
    public long updates() {
        return updates;
    }

    private Var[] sample(Var obs, Tensor noise) {
        Var out = actor.forward(obs, true);
        Var mean = out.slice(1, 0, actDims);
        Var logStd = out.slice(1, actDims, 2 * actDims).tanh()
                .add(1f).mul(0.5f * (LOG_STD_MAX - LOG_STD_MIN)).add(LOG_STD_MIN);
        Var u = mean.add(logStd.exp().mul(noise));
        Var a = u.tanh();
        Var gauss = logStd.add(HALF_LOG_TWO_PI).neg().sub(Var.of(noise.square().mul(0.5f)));
        Var squash = u.neg().sub(u.mul(-2f).softplus()).add((float) Math.log(2)).mul(2f);
        return new Var[] {a, gauss.sub(squash).sum(1, true)};
    }

    private Tensor noise(int rows) {
        return Tensor.normal(Shape.of(rows, actDims), 0, 1, random);
    }

    private float[] unit(float[] observation, boolean explore) {
        float[] out = RlTensors.row(actor, observation);
        float[] unit = new float[actDims];
        for (int i = 0; i < actDims; i++) {
            double m = out[i];
            if (explore) {
                double ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (Math.tanh(out[actDims + i]) + 1);
                m += Math.exp(ls) * random.nextGaussian();
            }
            unit[i] = (float) Math.tanh(m);
        }
        return unit;
    }

    private float[] scale(float[] unit) {
        if (low == null) {
            return unit;
        }
        float[] a = new float[unit.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = low[i] + (unit[i] + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        return scale(unit(observation, explore));
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
        if (buffer == null) {
            buffer = new ReplayBuffer((int) Math.max(1000, steps), obsDims, actDims);
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        long taken = 0;
        while (recorder.steps() < steps) {
            float[] unit;
            if (taken < learningStarts) {
                unit = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    unit[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                unit = unit(obs, true);
            }
            Environment.Step s = env.step(scale(unit));
            taken++;
            buffer.add(new Transition(obs, unit, s.reward(), s.observation(), s.terminated(), gamma));
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (taken >= learningStarts && buffer.size() >= batchSize) {
                for (int g = 0; g < utd; g++) {
                    criticStep();
                }
                actorStep();
            }
        }
        return recorder.result();
    }

    private void criticStep() {
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, actor.device());
        float alpha = (float) Math.exp(logAlpha.value().item());
        Var[] next = sample(Var.of(b.nextObservations()), noise(batchSize));
        Tensor nextIn = Tensor.concat(1, b.nextObservations(), next[0].value());
        Tensor soft = critics.targetMin(nextIn, subset, random).sub(next[1].value().mul(alpha));
        Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(soft));
        RlTensors.zeroGrad(actor.parameters());
        critics.update(Tensor.concat(1, b.observations(), b.actions()), y, criticOpt);
        critics.blend(tau);
        updates++;
    }

    private void actorStep() {
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, actor.device());
        float alpha = (float) Math.exp(logAlpha.value().item());
        Var obs = Var.of(b.observations());
        Var[] cur = sample(obs, noise(batchSize));
        Var q = critics.mean(Var.concat(1, obs, cur[0]), true);
        Var loss = cur[1].mul(alpha).sub(q).mean();
        loss.backward();
        actorOpt.step(actor.parameters());
        RlTensors.zeroGrad(critics.parameters());
        Tensor err = cur[1].value().add(targetEntropy);
        Var alphaLoss = logAlpha.mul(err).mean().neg();
        alphaLoss.backward();
        alphaOpt.step(List.of(logAlpha));
    }

    /**
     * The critics' normalized bias: on fresh episodes of the exploring policy, the mean of
     * {@code Q(s_t, a_t) - G_t} over the first half of each episode, divided by the mean
     * {@code |G_t|}, where {@code G_t} is the discounted Monte Carlo return to the end of the
     * episode. Positive means overestimation.
     *
     * @param env the environment (episodes should be long enough for the discount)
     * @param episodes episodes to average over
     * @param seed the episode seed
     * @return the normalized bias
     */
    public double qBias(Environment env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double diff = 0;
        double scale = 0;
        int n = 0;
        for (int e = 0; e < episodes; e++) {
            List<float[]> ins = new ArrayList<>();
            List<Double> rewards = new ArrayList<>();
            float[] o = env.reset(r.split());
            Environment.Step s;
            do {
                float[] u = unit(o, true);
                float[] in = new float[obsDims + actDims];
                System.arraycopy(o, 0, in, 0, obsDims);
                System.arraycopy(u, 0, in, obsDims, actDims);
                ins.add(in);
                s = env.step(scale(u));
                rewards.add((double) s.reward());
                o = s.observation();
            } while (!s.done());
            double[] g = new double[rewards.size()];
            double acc = 0;
            for (int t = g.length - 1; t >= 0; t--) {
                acc = rewards.get(t) + gamma * acc;
                g[t] = acc;
            }
            int half = g.length / 2;
            Tensor x = RlTensors.rows(ins.subList(0, half).toArray(new float[0][]), obsDims + actDims,
                    actor.device());
            float[] q = critics.mean(Var.of(x), false).value().to(DType.FLOAT32).toArray();
            for (int t = 0; t < half; t++) {
                diff += q[t] - g[t];
                scale += Math.abs(g[t]);
                n++;
            }
        }
        return n == 0 || scale == 0 ? 0 : diff / scale;
    }

    @Override
    public String toString() {
        return name + "(UTD " + utd + ", " + critics.size() + " critics)";
    }
}
