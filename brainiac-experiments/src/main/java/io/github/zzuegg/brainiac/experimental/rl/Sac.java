package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Soft actor-critic (Haarnoja et al., 2018, <a href="https://arxiv.org/abs/1801.01290">arXiv:1801.01290</a>
 * and <a href="https://arxiv.org/abs/1812.05905">arXiv:1812.05905</a>) for continuous actions: a
 * stochastic tanh-squashed Gaussian actor maximizes reward plus entropy, judged by two Q
 * critics whose minimum (clipped double Q) fights overestimation, with Polyak-averaged
 * target critics. The entropy temperature {@code alpha} tunes itself so the policy's
 * entropy stays near a target (by default {@code -actionDims}).
 *
 * <p>SAC is the standard off-policy baseline for continuous control: sample efficient and
 * robust to hyperparameters, because the entropy bonus keeps exploring without hand-tuned
 * noise.
 *
 * <pre>{@code
 * Sac sac = Sac.of(3, 1, RlNetworkFactory.mlp(Activation.RELU, 64, 64)).seed(1);
 * sac.fit(new Pendulum(), 10_000);
 * }</pre>
 */
public final class Sac implements Learner {

    private static final float LOG_STD_MIN = -5f;
    private static final float LOG_STD_MAX = 2f;
    private static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2 * Math.PI));

    private final int obsDims;
    private final int actDims;
    private final Model actor;
    private final Model q1;
    private final Model q2;
    private final Model q1Target;
    private final Model q2Target;
    private final Var logAlpha;
    private Optimizer actorOpt = Optimizer.adam(1e-3f);
    private Optimizer criticOpt = Optimizer.adam(1e-3f);
    private Optimizer alphaOpt = Optimizer.adam(1e-3f);
    private float gamma = 0.99f;
    private float tau = 0.005f;
    private int batchSize = 64;
    private int bufferSize = 100_000;
    private int learningStarts = 500;
    private int gradientSteps = 1;
    private boolean autoAlpha = true;
    private float targetEntropy;
    private float[] low;
    private float[] high;
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long updates;
    private double lastCriticLoss = Double.NaN;

    private Sac(int obsDims, int actDims, RlNetworkFactory factory) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.actor = factory.create(obsDims, 2 * actDims, 11);
        this.q1 = factory.create(obsDims + actDims, 1, 12);
        this.q2 = factory.create(obsDims + actDims, 1, 13);
        this.q1Target = factory.create(obsDims + actDims, 1, 12);
        this.q2Target = factory.create(obsDims + actDims, 1, 13);
        RlTensors.copy(q1.parameters(), q1Target.parameters());
        RlTensors.copy(q2.parameters(), q2Target.parameters());
        this.logAlpha = Var.param(Tensor.full(Shape.of(1, 1), (float) Math.log(0.2)));
        this.targetEntropy = -actDims;
    }

    /**
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param networks builds the actor ({@code obs -> 2 * actionDims}: means and log
     *     standard deviations) and the critics ({@code obs + action -> 1})
     * @return the learner
     */
    public static Sac of(int observationDims, int actionDims, RlNetworkFactory networks) {
        Objects.requireNonNull(networks, "networks");
        return new Sac(observationDims, actionDims, networks);
    }

    /** @param lr the learning rate of actor, critics and temperature @return this learner */
    public Sac learningRate(float lr) {
        actorOpt = Optimizer.adam(lr);
        criticOpt = Optimizer.adam(lr);
        alphaOpt = Optimizer.adam(lr);
        return this;
    }

    /** @param discount the discount gamma @return this learner */
    public Sac discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param tau the Polyak factor for the target critics @return this learner */
    public Sac tau(float tau) {
        this.tau = tau;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public Sac batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param size transitions kept @return this learner */
    public Sac bufferSize(int size) {
        this.bufferSize = size;
        return this;
    }

    /** @param steps random steps before learning starts @return this learner */
    public Sac learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param n gradient steps per environment step @return this learner */
    public Sac gradientSteps(int n) {
        this.gradientSteps = Math.max(1, n);
        return this;
    }

    /**
     * Fixes the temperature instead of tuning it.
     *
     * @param alpha the entropy weight
     * @return this learner
     */
    public Sac fixedAlpha(float alpha) {
        this.autoAlpha = false;
        logAlpha.assign(Tensor.full(Shape.of(1, 1), (float) Math.log(alpha)));
        return this;
    }

    /** @param entropy the entropy the temperature tuning aims for @return this learner */
    public Sac targetEntropy(float entropy) {
        this.targetEntropy = entropy;
        return this;
    }

    /** @param seed the seed for exploration, replay and episodes @return this learner */
    public Sac seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the current entropy temperature */
    public float alpha() {
        return (float) Math.exp(logAlpha.value().item());
    }

    /** @return the actor network */
    public Model actor() {
        return actor;
    }

    /** @return the critic loss of the last update */
    public double lastCriticLoss() {
        return lastCriticLoss;
    }

    /** @return the gradient steps taken */
    public long updates() {
        return updates;
    }

    /** Samples a squashed action and its log-probability for a batch of observations. */
    private Var[] sample(Model policy, Var obs, Tensor noise) {
        Var out = policy.forward(obs, true);
        Var mean = out.slice(1, 0, actDims);
        Var logStd = out.slice(1, actDims, 2 * actDims).tanh()
                .add(1f).mul(0.5f * (LOG_STD_MAX - LOG_STD_MIN)).add(LOG_STD_MIN);
        Var u = mean.add(logStd.exp().mul(noise));
        Var a = u.tanh();
        // log N(u) - log(1 - tanh(u)^2), the latter as 2 (log 2 - u - softplus(-2u))
        Var gauss = logStd.add(HALF_LOG_TWO_PI).neg().sub(Var.of(noise.square().mul(0.5f)));
        Var squash = u.neg().sub(u.mul(-2f).softplus()).add((float) Math.log(2)).mul(2f);
        Var logp = gauss.sub(squash).sum(1, true);
        return new Var[] {a, logp};
    }

    private Tensor noise(int rows) {
        return Tensor.normal(Shape.of(rows, actDims), 0, 1, random);
    }

    private float[] scale(float[] unit) {
        float[] a = new float[unit.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = low[i] + (unit[i] + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        Tensor x = Tensor.of(Shape.of(1, obsDims), observation).to(actor.device());
        float[] out = actor.forward(Var.of(x), false).value().to(DType.FLOAT32).toArray();
        float[] unit = new float[actDims];
        for (int i = 0; i < actDims; i++) {
            double m = out[i];
            if (explore) {
                double ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (Math.tanh(out[actDims + i]) + 1);
                m += Math.exp(ls) * random.nextGaussian();
            }
            unit[i] = (float) Math.tanh(m);
        }
        return low == null ? unit : scale(unit);
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Box box) || box.size() != actDims
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("SAC needs " + obsDims + " observation values and a box of "
                    + actDims + " actions");
        }
        low = box.low();
        high = box.high();
        if (buffer == null) {
            buffer = new ReplayBuffer(bufferSize, obsDims, actDims);
        }
        EpisodeRecorder recorder = new EpisodeRecorder(listener);
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
                unit = unitAction(act(obs, true));
            }
            Environment.Step s = env.step(scale(unit));
            taken++;
            buffer.add(new Transition(obs, unit, s.reward(), s.observation(), s.terminated(), gamma));
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (taken >= learningStarts && buffer.size() >= batchSize) {
                for (int g = 0; g < gradientSteps; g++) {
                    update();
                }
            }
        }
        return recorder.result();
    }

    private float[] unitAction(float[] scaled) {
        float[] u = new float[scaled.length];
        for (int i = 0; i < u.length; i++) {
            u[i] = (2 * (scaled[i] - low[i]) / (high[i] - low[i])) - 1;
        }
        return u;
    }

    private void update() {
        Device device = actor.device();
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, device);
        Var obs = Var.of(b.observations());
        Var next = Var.of(b.nextObservations());
        float alpha = alpha();

        Var[] nextSample = sample(actor, next, noise(batchSize));
        Tensor nextA = nextSample[0].value();
        Tensor nextIn = Tensor.concat(1, b.nextObservations(), nextA);
        Tensor minNext = q1Target.forward(Var.of(nextIn), false).value()
                .minimum(q2Target.forward(Var.of(nextIn), false).value());
        Tensor softNext = minNext.sub(nextSample[1].value().mul(alpha));
        Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(softNext));
        RlTensors.zeroGrad(actor.parameters());

        Var in = Var.of(Tensor.concat(1, b.observations(), b.actions()));
        Var l1 = q1.forward(in, true).sub(y).square().mean();
        Var l2 = q2.forward(in, true).sub(y).square().mean();
        Var criticLoss = l1.add(l2);
        criticLoss.backward();
        List<Var> criticParams = new java.util.ArrayList<>(q1.parameters());
        criticParams.addAll(q2.parameters());
        criticOpt.step(criticParams);
        lastCriticLoss = criticLoss.value().item();

        Var[] cur = sample(actor, obs, noise(batchSize));
        Var curIn = Var.concat(1, obs, cur[0]);
        Var minQ = q1.forward(curIn, true).minimum(q2.forward(curIn, true));
        Var actorLoss = cur[1].mul(alpha).sub(minQ).mean();
        actorLoss.backward();
        actorOpt.step(actor.parameters());
        RlTensors.zeroGrad(criticParams);

        if (autoAlpha) {
            Tensor err = cur[1].value().add(targetEntropy);
            Var alphaLoss = logAlpha.mul(err).mean().neg();
            alphaLoss.backward();
            alphaOpt.step(List.of(logAlpha));
        }
        RlTensors.blend(q1.parameters(), q1Target.parameters(), tau);
        RlTensors.blend(q2.parameters(), q2Target.parameters(), tau);
        updates++;
    }

    @Override
    public String toString() {
        return "SAC(alpha " + String.format("%.3f", alpha()) + ")";
    }
}
