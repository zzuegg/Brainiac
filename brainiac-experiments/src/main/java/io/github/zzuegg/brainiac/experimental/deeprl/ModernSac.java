package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.HlGaussValue;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

/**
 * Soft actor-critic with the switches the 2023-2025 off-policy recipes turn: the shared
 * engine of {@link CrossQ}, {@link Bro} and {@link SimbaV2}, and a plain SAC when every switch
 * is left at its default (two MLP critics, clipped double Q, Polyak target networks, UTD 1).
 *
 * <ul>
 *   <li>{@link #jointBatch(boolean)} and {@link #targetNetworks(boolean)}: CrossQ's single
 *       critic pass over the concatenated current and next state-action batches, so batch
 *       normalisation sees both, with no target network.</li>
 *   <li>{@link #critics(int)}, {@link #pessimism(float)} and {@link #quantiles(int)}: an
 *       ensemble whose value is {@code mean - beta * std} instead of the minimum of two, and
 *       quantile-regression critics (BRO).</li>
 *   <li>{@link #optimism(float, float)}: BRO's second, optimistic actor that maximises
 *       {@code mean + beta_o * std}, explores, and is held near the pessimistic actor by a KL
 *       penalty with an adaptive weight.</li>
 *   <li>{@link #resetEvery(int)}: full parameter resets that keep the replay buffer.</li>
 *   <li>{@link #categorical(HlGaussValue)}, {@link #afterStep(Consumer)},
 *       {@link #returnScaling(boolean)} and {@link #learningRateDecay(boolean)}: SimbaV2's
 *       categorical critic, weight projection after each update, reward scaling by the running
 *       return statistics and linearly decaying learning rate.</li>
 *   <li>{@link #augment(UnaryOperator)} and {@link #nStep(int)}: image augmentation of the
 *       observation batches (DrQ) and multi-step targets.</li>
 * </ul>
 *
 * <p>Papers: Haarnoja et al., <a href="https://arxiv.org/abs/1812.05905">arXiv:1812.05905</a>
 * (SAC); see the recipe classes for the rest.
 */
public final class ModernSac implements Learner {

    private static final float LOG_STD_MIN = -5f;
    private static final float LOG_STD_MAX = 2f;
    private static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2 * Math.PI));

    private final int obsDims;
    private final int actDims;
    private final RlNetworkFactory actorNets;
    private final RlNetworkFactory criticNets;
    private int ensemble = 2;
    private boolean targets = true;
    private boolean joint;
    private int quantiles;
    private HlGaussValue categorical;
    private float pessimism = -1;
    private float optimism;
    private float klTarget = 0.05f;
    private int resetEvery;
    private float weightDecay;
    private int utd = 1;
    private int nStep = 1;
    private float lr = 1e-3f;
    private float beta1 = 0.9f;
    private float gamma = 0.99f;
    private float tau = 0.005f;
    private int batchSize = 64;
    private int bufferSize = 100_000;
    private int learningStarts = 500;
    private boolean returnScaling;
    private boolean lrDecay;
    private Consumer<Model> afterStep;
    private UnaryOperator<Tensor> augment;
    private java.util.function.Supplier<Optimizer> optimizerFactory;
    private SplittableRandom random = new SplittableRandom(0);
    private long netSeed = 0;

    private Model actor;
    private Model optimisticActor;
    private List<Model> critics;
    private List<Model> criticTargets;
    private Var logAlpha;
    private float klWeight = 1f;
    private Optimizer actorOpt;
    private Optimizer optimisticOpt;
    private Optimizer criticOpt;
    private Optimizer alphaOpt;
    private ReplayBuffer buffer;
    private float[] low;
    private float[] high;
    private long updates;
    private long resets;
    private long plannedSteps = 1;
    private long stepsTaken;
    private double lastCriticLoss = Double.NaN;
    private final RunningReturnScale returnScale = new RunningReturnScale();

    private ModernSac(int obsDims, int actDims, RlNetworkFactory actorNets, RlNetworkFactory criticNets) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.actorNets = actorNets;
        this.criticNets = criticNets;
    }

    /**
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param actorNets builds the actor ({@code obs -> 2 * actionDims})
     * @param criticNets builds each critic ({@code obs + action -> 1}, or quantiles or bins)
     * @return a plain SAC until switches are turned
     */
    public static ModernSac of(int observationDims, int actionDims, RlNetworkFactory actorNets,
            RlNetworkFactory criticNets) {
        return new ModernSac(observationDims, actionDims, Objects.requireNonNull(actorNets, "actorNets"),
                Objects.requireNonNull(criticNets, "criticNets"));
    }

    /** @param n critics in the ensemble @return this learner */
    public ModernSac critics(int n) {
        this.ensemble = Math.max(1, n);
        return this;
    }

    /** @param on whether to keep Polyak-averaged target critics @return this learner */
    public ModernSac targetNetworks(boolean on) {
        this.targets = on;
        return this;
    }

    /** @param on whether to pass current and next batches through the critic together @return this learner */
    public ModernSac jointBatch(boolean on) {
        this.joint = on;
        return this;
    }

    /** @param n quantiles per critic, 0 for a scalar critic @return this learner */
    public ModernSac quantiles(int n) {
        this.quantiles = Math.max(0, n);
        return this;
    }

    /** @param bins the categorical value bins, or null for a scalar critic @return this learner */
    public ModernSac categorical(HlGaussValue bins) {
        this.categorical = bins;
        return this;
    }

    /**
     * @param beta the ensemble's value is {@code mean - beta * std}; negative for the minimum
     *     (clipped double Q, the default)
     * @return this learner
     */
    public ModernSac pessimism(float beta) {
        this.pessimism = beta;
        return this;
    }

    /**
     * Turns on BRO's optimistic exploration actor.
     *
     * @param beta the optimistic actor maximises {@code mean + beta * std}
     * @param kl the target KL between the optimistic and the pessimistic actor
     * @return this learner
     */
    public ModernSac optimism(float beta, float kl) {
        this.optimism = beta;
        this.klTarget = kl;
        return this;
    }

    /** @param updates gradient updates between full resets, 0 for none @return this learner */
    public ModernSac resetEvery(int updates) {
        this.resetEvery = Math.max(0, updates);
        return this;
    }

    /** @param decay AdamW's decoupled weight decay @return this learner */
    public ModernSac weightDecay(float decay) {
        this.weightDecay = decay;
        return this;
    }

    /** @param ratio gradient updates per environment step @return this learner */
    public ModernSac updateToData(int ratio) {
        this.utd = Math.max(1, ratio);
        return this;
    }

    /** @param n steps per return target @return this learner */
    public ModernSac nStep(int n) {
        this.nStep = Math.max(1, n);
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public ModernSac learningRate(float rate) {
        this.lr = rate;
        return this;
    }

    /**
     * @param factory builds the optimizer of each network (actor, critics), replacing Adam; for
     *     example {@code () -> PsgdKron.of(3e-4f)}
     * @return this learner
     */
    public ModernSac optimizer(java.util.function.Supplier<Optimizer> factory) {
        this.optimizerFactory = factory;
        return this;
    }

    /** @param beta Adam's first-moment decay (CrossQ uses 0.5) @return this learner */
    public ModernSac adamBeta1(float beta) {
        this.beta1 = beta;
        return this;
    }

    /** @param discount gamma @return this learner */
    public ModernSac discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param polyak the target-network averaging rate @return this learner */
    public ModernSac tau(float polyak) {
        this.tau = polyak;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public ModernSac batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param steps random steps before learning @return this learner */
    public ModernSac learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param on whether to divide rewards by the running return's standard deviation @return this learner */
    public ModernSac returnScaling(boolean on) {
        this.returnScaling = on;
        return this;
    }

    /** @param on whether the learning rate decays linearly to a tenth over the run @return this learner */
    public ModernSac learningRateDecay(boolean on) {
        this.lrDecay = on;
        return this;
    }

    /** @param hook run on every network after each of its optimizer steps @return this learner */
    public ModernSac afterStep(Consumer<Model> hook) {
        this.afterStep = hook;
        return this;
    }

    /** @param transform applied to observation batches before each update @return this learner */
    public ModernSac augment(UnaryOperator<Tensor> transform) {
        this.augment = transform;
        return this;
    }

    /** @param seed the seed of networks, exploration and replay @return this learner */
    public ModernSac seed(long seed) {
        this.random = new SplittableRandom(seed);
        this.netSeed = seed * 1000;
        return this;
    }

    /** @return the gradient updates taken */
    public long updates() {
        return updates;
    }

    /** @return the full resets done */
    public long resets() {
        return resets;
    }

    /** @return the critic loss of the last update */
    public double lastCriticLoss() {
        return lastCriticLoss;
    }

    /** @return the (pessimistic) actor */
    public Model actor() {
        build();
        return actor;
    }

    /** @return the critics */
    public List<Model> criticModels() {
        build();
        return List.copyOf(critics);
    }

    /** @return the replay buffer, null before training */
    public ReplayBuffer buffer() {
        return buffer;
    }

    private int criticOutputs() {
        return quantiles > 0 ? quantiles : categorical != null ? categorical.bins() : 1;
    }

    private void build() {
        if (actor != null) {
            return;
        }
        long s = netSeed + resets * 101;
        actor = actorNets.create(obsDims, 2 * actDims, s + 1);
        optimisticActor = optimism > 0 ? actorNets.create(obsDims, 2 * actDims, s + 2) : null;
        critics = new ArrayList<>();
        criticTargets = new ArrayList<>();
        for (int i = 0; i < ensemble; i++) {
            Model c = criticNets.create(obsDims + actDims, criticOutputs(), s + 10 + i);
            critics.add(c);
            if (targets && !joint) {
                Model t = criticNets.create(obsDims + actDims, criticOutputs(), s + 10 + i);
                RlTensors.copy(c.parameters(), t.parameters());
                criticTargets.add(t);
            }
        }
        logAlpha = Var.param(Tensor.full(Shape.of(1, 1), (float) Math.log(0.1)));
        actorOpt = optimizer();
        optimisticOpt = optimizer();
        criticOpt = optimizer();
        alphaOpt = Optimizer.adam(lr);
        runHooks();
    }

    private Optimizer optimizer() {
        if (optimizerFactory != null) {
            return optimizerFactory.get();
        }
        return Optimizer.adamW(lr, beta1, 0.999f, 1e-8f, weightDecay);
    }

    private void runHooks() {
        if (afterStep == null) {
            return;
        }
        afterStep.accept(actor);
        if (optimisticActor != null) {
            afterStep.accept(optimisticActor);
        }
        critics.forEach(afterStep);
    }

    /** Re-initialises every network and optimizer, keeping the replay buffer. */
    public void reset() {
        resets++;
        actor = null;
        build();
    }

    private float alpha() {
        return (float) Math.exp(logAlpha.value().item());
    }

    /** Samples squashed actions: {action, log-probability, mean, log-std}. */
    private Var[] sample(Model policy, Var obs, boolean training) {
        Var out = policy.forward(obs, training);
        Var mean = out.slice(1, 0, actDims);
        Var logStd = out.slice(1, actDims, 2 * actDims).tanh()
                .add(1f).mul(0.5f * (LOG_STD_MAX - LOG_STD_MIN)).add(LOG_STD_MIN);
        Tensor noise = Tensor.normal(Shape.of(obs.shape().dim(0), actDims), 0, 1, random);
        Var u = mean.add(logStd.exp().mul(noise));
        Var a = u.tanh();
        Var gauss = logStd.add(HALF_LOG_TWO_PI).neg().sub(Var.of(noise.square().mul(0.5f)));
        Var squash = u.neg().sub(u.mul(-2f).softplus()).add((float) Math.log(2)).mul(2f);
        return new Var[] {a, gauss.sub(squash).sum(1, true), mean, logStd};
    }

    /** One critic's value per row from its raw output. */
    private Var value(Var out) {
        if (quantiles > 0) {
            return out.mean(1, true);
        }
        return categorical != null ? categorical.value(out) : out;
    }

    /** The ensemble's aggregate of per-critic values {@code [B, k]} (k critics stacked on axis 1). */
    private Var aggregate(Var stacked, float beta) {
        return beta < 0 ? minimum(stacked) : meanStd(stacked, beta);
    }

    private static Var minimum(Var stacked) {
        Var m = stacked.slice(1, 0, 1);
        for (int i = 1; i < stacked.shape().dim(1); i++) {
            m = m.minimum(stacked.slice(1, i, i + 1));
        }
        return m;
    }

    /** {@code mean - beta * std} across the critics (beta may be negative: optimism). */
    private static Var meanStd(Var stacked, float beta) {
        Var mean = stacked.mean(1, true);
        if (beta == 0 || stacked.shape().dim(1) < 2) {
            return mean;
        }
        Var std = stacked.sub(mean).square().mean(1, true).add(1e-8f).sqrt();
        return mean.add(std.mul(-beta));
    }

    private Var stackValues(List<Var> outs) {
        Var[] v = new Var[outs.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = value(outs.get(i));
        }
        return Var.concat(1, v);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        build();
        Model policy = explore && optimisticActor != null ? optimisticActor : actor;
        float[] out = RlTensors.row(policy, observation);
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

    private float[] scale(float[] unit) {
        float[] a = new float[unit.length];
        for (int i = 0; i < a.length; i++) {
            a[i] = low[i] + (unit[i] + 1) * 0.5f * (high[i] - low[i]);
        }
        return a;
    }

    private float[] unitAction(float[] scaled) {
        float[] u = new float[scaled.length];
        for (int i = 0; i < u.length; i++) {
            u[i] = (2 * (scaled[i] - low[i]) / (high[i] - low[i])) - 1;
        }
        return u;
    }

    private record Pending(float[] obs, float[] action, float reward) {
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Box box) || box.size() != actDims
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("needs " + obsDims + " observation values and a box of "
                    + actDims + " actions");
        }
        build();
        low = box.low();
        high = box.high();
        if (buffer == null) {
            buffer = new ReplayBuffer(bufferSize, obsDims, actDims);
        }
        plannedSteps = stepsTaken + steps;
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        ArrayDeque<Pending> window = new ArrayDeque<>();
        while (recorder.steps() < steps) {
            float[] unit;
            if (stepsTaken < learningStarts) {
                unit = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    unit[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                unit = unitAction(act(obs, true));
            }
            Environment.Step s = env.step(scale(unit));
            stepsTaken++;
            returnScale.observe(s.reward(), gamma, s.done());
            window.addLast(new Pending(obs, unit, s.reward()));
            if (window.size() == nStep || s.done()) {
                flush(window, s, s.done());
            }
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (stepsTaken >= learningStarts && buffer.size() >= batchSize) {
                for (int g = 0; g < utd; g++) {
                    update();
                }
            }
        }
        return recorder.result();
    }

    /** Writes n-step transitions; at an episode end, every pending start is written. */
    private void flush(ArrayDeque<Pending> window, Environment.Step s, boolean all) {
        do {
            float ret = 0;
            float disc = 1;
            for (Pending p : window) {
                ret += disc * p.reward();
                disc *= gamma;
            }
            Pending first = window.removeFirst();
            buffer.add(new Transition(first.obs(), first.action(), ret, s.observation(), s.terminated(), disc));
        } while (all && !window.isEmpty());
    }

    private Tensor maybeAugment(Tensor x) {
        return augment == null ? x : augment.apply(x);
    }

    private void update() {
        if (lrDecay) {
            float frac = 1 - 0.9f * Math.min(1f, (float) stepsTaken / plannedSteps);
            for (Optimizer o : List.of(actorOpt, optimisticOpt, criticOpt)) {
                o.learningRate(lr * frac);
            }
        }
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, actor.device());
        Tensor obsT = maybeAugment(b.observations());
        Tensor nextT = maybeAugment(b.nextObservations());
        Var obs = Var.of(obsT);
        float alpha = alpha();
        Tensor rewards = returnScaling ? b.rewards().mul(returnScale.scale()) : b.rewards();
        Tensor bootstrap = b.discounts().mul(b.notDone());

        Var[] nextSample = Var.noGrad(() -> sample(actor, Var.of(nextT), false));
        Tensor nextA = nextSample[0].value();
        Tensor nextLogp = nextSample[1].value();
        Tensor in = Tensor.concat(1, obsT, b.actions());
        Tensor nextIn = Tensor.concat(1, nextT, nextA);

        List<Var> preds = new ArrayList<>();
        List<Tensor> nextOuts = new ArrayList<>();
        int rows = batchSize;
        for (int i = 0; i < critics.size(); i++) {
            Model c = critics.get(i);
            if (joint) {
                Var both = c.forward(Var.of(Tensor.concat(0, in, nextIn)), true);
                preds.add(both.slice(0, 0, rows));
                nextOuts.add(both.slice(0, rows, 2 * rows).value());
            } else {
                preds.add(c.forward(Var.of(in), true));
                Model t = targets ? criticTargets.get(i) : c;
                nextOuts.add(Var.noGrad(() -> t.forward(Var.of(nextIn), false)).value());
            }
        }
        Var criticLoss;
        if (quantiles > 0) {
            Tensor targetQ = quantileTarget(nextOuts, rewards, bootstrap, nextLogp, alpha);
            criticLoss = null;
            for (Var p : preds) {
                Var l = quantileHuber(p, targetQ);
                criticLoss = criticLoss == null ? l : criticLoss.add(l);
            }
        } else {
            List<Var> nv = new ArrayList<>();
            for (Tensor t : nextOuts) {
                nv.add(Var.of(t));
            }
            Tensor nextQ = aggregate(stackValues(nv), pessimism).value();
            Tensor y = rewards.add(bootstrap.mul(nextQ.sub(nextLogp.mul(alpha))));
            criticLoss = null;
            for (Var p : preds) {
                Var l = categorical != null ? categorical.loss(p, y) : p.sub(y).square().mean();
                criticLoss = criticLoss == null ? l : criticLoss.add(l);
            }
        }
        criticLoss.backward();
        List<Var> criticParams = new ArrayList<>();
        critics.forEach(c -> criticParams.addAll(c.parameters()));
        criticOpt.step(criticParams);
        lastCriticLoss = criticLoss.value().item();

        Var[] cur = sample(actor, obs, true);
        Var qPi = aggregate(criticValues(Var.concat(1, obs, cur[0])), pessimism);
        Var actorLoss = cur[1].mul(alpha).sub(qPi).mean();
        actorLoss.backward();
        actorOpt.step(actor.parameters());
        RlTensors.zeroGrad(criticParams);

        Tensor err = cur[1].value().add((float) -actDims);
        Var alphaLoss = logAlpha.mul(err).mean().neg();
        alphaLoss.backward();
        alphaOpt.step(List.of(logAlpha));

        if (optimisticActor != null) {
            Var[] opt = sample(optimisticActor, obs, true);
            Var qOpt = meanStd(criticValues(Var.concat(1, obs, opt[0])), -optimism);
            Var kl = gaussianKl(opt[2], opt[3], cur[2].value(), cur[3].value());
            Var loss = qOpt.neg().add(kl.mul(klWeight)).mean();
            loss.backward();
            optimisticOpt.step(optimisticActor.parameters());
            RlTensors.zeroGrad(criticParams);
            RlTensors.zeroGrad(actor.parameters());
            double klNow = kl.value().mean().item();
            klWeight = (float) Math.max(1e-3, Math.min(1e3, klWeight * Math.exp(0.01 * Math.signum(klNow - klTarget))));
        }
        if (targets && !joint) {
            for (int i = 0; i < critics.size(); i++) {
                RlTensors.blend(critics.get(i).parameters(), criticTargets.get(i).parameters(), tau);
                RlTensors.copy(critics.get(i).buffers(), criticTargets.get(i).buffers());
            }
        }
        runHooks();
        updates++;
        if (resetEvery > 0 && updates % resetEvery == 0) {
            reset();
        }
    }

    /** Each critic's value of {@code input} stacked as {@code [B, k]}, with evaluation-mode normalisation. */
    private Var criticValues(Var input) {
        List<Var> outs = new ArrayList<>();
        for (Model c : critics) {
            outs.add(c.forward(input, false));
        }
        return stackValues(outs);
    }

    private Tensor quantileTarget(List<Tensor> nextOuts, Tensor rewards, Tensor bootstrap, Tensor nextLogp,
            float alpha) {
        // per-quantile ensemble aggregate: mean - beta * std across critics (min when beta < 0)
        Tensor agg;
        if (pessimism < 0) {
            agg = nextOuts.get(0);
            for (int i = 1; i < nextOuts.size(); i++) {
                agg = agg.minimum(nextOuts.get(i));
            }
        } else {
            Tensor mean = nextOuts.get(0);
            for (int i = 1; i < nextOuts.size(); i++) {
                mean = mean.add(nextOuts.get(i));
            }
            mean = mean.div(nextOuts.size());
            Tensor var = Tensor.zeros(mean.shape());
            for (Tensor t : nextOuts) {
                var = var.add(t.sub(mean).square());
            }
            agg = mean.sub(var.div(nextOuts.size()).add(1e-8f).sqrt().mul(pessimism));
        }
        return rewards.add(bootstrap.mul(agg.sub(nextLogp.mul(alpha))));
    }

    /** Quantile Huber loss of predicted quantiles {@code [B, N]} against target samples {@code [B, M]}. */
    private Var quantileHuber(Var pred, Tensor target) {
        int b = pred.shape().dim(0);
        int n = pred.shape().dim(1);
        int m = target.shape().dim(1);
        float[] taus = new float[n];
        for (int i = 0; i < n; i++) {
            taus[i] = (i + 0.5f) / n;
        }
        Tensor tau = Tensor.of(Shape.of(1, n, 1), taus).to(pred.value().device());
        Var u = Var.of(target.reshape(b, 1, m)).sub(pred.reshape(b, n, 1));
        Tensor negative = u.value().less(Tensor.zeros(Shape.of(1)));
        Tensor weight = tau.sub(negative).abs();
        Var h = RlTensors.huber(u, 1f).mul(weight);
        return h.mean(2, false).sum(1, false).mean();
    }

    /** KL between diagonal Gaussians (pre-squash) of the optimistic and the (fixed) pessimistic actor. */
    private static Var gaussianKl(Var meanO, Var logStdO, Tensor meanP, Tensor logStdP) {
        Var varO = logStdO.mul(2f).exp();
        Tensor varP = logStdP.mul(2f).exp();
        Var term = varO.add(meanO.sub(meanP).square()).div(varP.mul(2f));
        return Var.of(logStdP).sub(logStdO).add(term).sub(0.5f).sum(1, true);
    }

    @Override
    public String toString() {
        return "ModernSac(critics " + ensemble + (joint ? ", joint batch" : "") + (quantiles > 0 ? ", quantiles "
                + quantiles : "") + (optimism > 0 ? ", optimistic" : "") + ", UTD " + utd + ")";
    }

    /**
     * Running statistics of the discounted return, for SimbaV2's reward scaling: rewards are
     * divided by the standard deviation of {@code G_t = gamma G_{t-1} + r_t}.
     */
    static final class RunningReturnScale {
        private double g;
        private long n;
        private double mean;
        private double m2;

        void observe(float reward, float gamma, boolean done) {
            g = gamma * g + reward;
            n++;
            double d = g - mean;
            mean += d / n;
            m2 += d * (g - mean);
            if (done) {
                g = 0;
            }
        }

        float scale() {
            double var = n > 1 ? m2 / (n - 1) : 1;
            return (float) (1 / Math.max(1e-2, Math.sqrt(var + 1e-8)));
        }
    }
}
