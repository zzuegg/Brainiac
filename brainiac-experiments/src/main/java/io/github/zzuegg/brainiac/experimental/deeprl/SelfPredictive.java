package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.experimental.train.MomentumTeacher;
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
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

/**
 * A DQN-family learner for discrete actions with the sample-efficiency recipes of the Atari 100k
 * line, each a switch, so the same agent is the plain baseline, SPR, BBF or DrQ.
 *
 * <ul>
 *   <li>Self-predictive representations (SPR; Schwarzer et al., ICLR 2021,
 *       <a href="https://arxiv.org/abs/2007.05929">arXiv:2007.05929</a>), {@link #spr(int, float)}:
 *       a latent transition model, rolled forward {@code k} steps from the online encoder with
 *       the actions taken, must predict (cosine similarity after a projection and a predictor)
 *       the momentum target encoder's embeddings of the future observations
 *       ({@link MomentumTeacher}).</li>
 *   <li>BBF (Schwarzer et al., ICML 2023, <a href="https://arxiv.org/abs/2305.19452">arXiv:2305.19452</a>):
 *       a high {@link #replayRatio(int) replay ratio} made safe by periodic
 *       {@link #resets(int, float) resets} (the Q head re-initialised, encoder and transition model
 *       shrunk and perturbed towards fresh weights), an update horizon annealed from 10 to 3 and a
 *       discount annealed from 0.97 to 0.997 ({@link #bbfSchedules(int)}), and weight decay.</li>
 *   <li>DrQ (Kostrikov et al., ICLR 2021, <a href="https://arxiv.org/abs/2004.13649">arXiv:2004.13649</a>),
 *       {@link #augment(UnaryOperator, int, int)}: observations augmented by random shifts
 *       ({@link DrQ#randomShift}), the target averaged over {@code K} and the estimate over
 *       {@code M} augmentations.</li>
 * </ul>
 *
 * <p>The base is double DQN with a Polyak (EMA) target network, n-step returns, a Huber loss and
 * an epsilon-greedy schedule. For plasticity experiments the encoder and head are exposed, a hook
 * runs after every update, and {@link #injectPlasticity()} adds plasticity injection.
 */
public final class SelfPredictive implements Learner {

    private final int obsDims;
    private final int actions;
    private final int latent;
    private final RlNetworkFactory encoderNets;
    private final RlNetworkFactory headNets;
    private int sprSteps;
    private float sprWeight = 1f;
    private int replayRatio = 1;
    private int nStep = 3;
    private boolean bbf;
    private int annealUpdates = 2_000;
    private int resetEvery;
    private float shrink = 0.5f;
    private float weightDecay;
    private float lr = 1e-3f;
    private float gamma = 0.99f;
    private float tau = 0.01f;
    private int batchSize = 32;
    private int learningStarts = 500;
    private int epsilonSteps = 2_000;
    private float epsilonEnd = 0.05f;
    private UnaryOperator<Tensor> augment;
    private int augK = 1;
    private int augM = 1;
    private Consumer<SelfPredictive> onUpdate;
    private SplittableRandom random = new SplittableRandom(0);
    private long seed;

    private Model encoder;
    private Model head;
    private Model transition;
    private Model projection;
    private Model predictor;
    private Model encoderTarget;
    private Model headTarget;
    private Model projectionTarget;
    private Model injected;
    private Model injectedFrozen;
    private Model injectedTarget;
    private MomentumTeacher teacher;
    private Optimizer optimizer;
    private LapReplayBuffer buffer;
    private long updates;
    private long resetsDone;
    private long stepsTaken;
    private double lastLoss = Double.NaN;

    private SelfPredictive(int obsDims, int actions, RlNetworkFactory encoderNets, int latent,
            RlNetworkFactory headNets) {
        this.obsDims = obsDims;
        this.actions = actions;
        this.encoderNets = encoderNets;
        this.latent = latent;
        this.headNets = headNets;
    }

    /**
     * @param observationDims values per observation
     * @param actions discrete actions
     * @param encoder builds the encoder ({@code obs -> latent})
     * @param latent the embedding size
     * @param headWidth the Q head's hidden width
     * @return a plain double DQN until switches are turned
     */
    public static SelfPredictive of(int observationDims, int actions, RlNetworkFactory encoder, int latent,
            int headWidth) {
        return new SelfPredictive(observationDims, actions, Objects.requireNonNull(encoder, "encoder"), latent,
                RlNetworkFactory.mlp(Activation.RELU, headWidth));
    }

    /**
     * An MLP encoder with a LayerNorm on the embedding.
     *
     * @param hidden the hidden width
     * @return the factory
     */
    public static RlNetworkFactory mlpEncoder(int hidden) {
        return (in, out, seed) -> Model.sequential().input(in).dense(hidden, Activation.RELU).dense(out).layerNorm()
                .seed(seed).build();
    }

    /**
     * A small convolutional encoder for {@code [channels, height, width]} images (flattened
     * input).
     *
     * @param channels image channels
     * @param height image height
     * @param width image width
     * @param filters convolution filters
     * @return the factory
     */
    public static RlNetworkFactory convEncoder(int channels, int height, int width, int filters) {
        return (in, out, seed) -> Model.sequential().input(in).reshape(channels, height, width)
                .conv2d(filters, 3, Activation.RELU).maxPool(2, 2).flatten().dense(out).layerNorm().seed(seed).build();
    }

    /** @param steps SPR's prediction horizon k (0 for none) @param weight its loss weight @return this */
    public SelfPredictive spr(int steps, float weight) {
        this.sprSteps = Math.max(0, steps);
        this.sprWeight = weight;
        return this;
    }

    /** @param ratio gradient updates per environment step @return this learner */
    public SelfPredictive replayRatio(int ratio) {
        this.replayRatio = Math.max(1, ratio);
        return this;
    }

    /** @param n steps per return target @return this learner */
    public SelfPredictive nStep(int n) {
        this.nStep = Math.max(1, n);
        return this;
    }

    /**
     * BBF's schedules: after each reset the update horizon anneals from 10 to 3 and the
     * discount from 0.97 to 0.997 over {@code updates} gradient updates.
     *
     * @param updates the annealing length
     * @return this learner
     */
    public SelfPredictive bbfSchedules(int updates) {
        this.bbf = true;
        this.annealUpdates = Math.max(1, updates);
        return this;
    }

    /**
     * Periodic resets: the Q head is re-initialised; encoder and transition model are shrunk
     * and perturbed, {@code w <- alpha w + (1 - alpha) w_fresh}.
     *
     * @param every gradient updates between resets (0 for none)
     * @param alpha the shrink factor (0.5 in BBF)
     * @return this learner
     */
    public SelfPredictive resets(int every, float alpha) {
        this.resetEvery = Math.max(0, every);
        this.shrink = alpha;
        return this;
    }

    /** @param decay AdamW weight decay @return this learner */
    public SelfPredictive weightDecay(float decay) {
        this.weightDecay = decay;
        return this;
    }

    /**
     * DrQ: augment every observation batch; the target is averaged over {@code k} and the
     * estimate over {@code m} augmentations.
     *
     * @param transform the augmentation
     * @param k target augmentations
     * @param m estimate augmentations
     * @return this learner
     */
    public SelfPredictive augment(UnaryOperator<Tensor> transform, int k, int m) {
        this.augment = transform;
        this.augK = Math.max(1, k);
        this.augM = Math.max(1, m);
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public SelfPredictive learningRate(float rate) {
        this.lr = rate;
        return this;
    }

    /** @param discount gamma (the start of BBF's schedule overrides it) @return this learner */
    public SelfPredictive discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public SelfPredictive batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param steps random steps first @return this learner */
    public SelfPredictive learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param steps environment steps over which epsilon falls from 1 @param end its final value @return this */
    public SelfPredictive epsilon(int steps, float end) {
        this.epsilonSteps = Math.max(1, steps);
        this.epsilonEnd = end;
        return this;
    }

    /** @param polyak the target network's EMA rate @return this learner */
    public SelfPredictive targetTau(float polyak) {
        this.tau = polyak;
        return this;
    }

    /** @param hook run after every gradient update @return this learner */
    public SelfPredictive onUpdate(Consumer<SelfPredictive> hook) {
        this.onUpdate = hook;
        return this;
    }

    /** @param seed the seed @return this learner */
    public SelfPredictive seed(long seed) {
        this.seed = seed;
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the gradient updates taken */
    public long updates() {
        return updates;
    }

    /** @return the TD loss of the last update */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return the encoder */
    public Model encoder() {
        build();
        return encoder;
    }

    /** @return the Q head */
    public Model head() {
        build();
        return head;
    }

    /**
     * @param n observations to draw
     * @return a uniform sample of stored observations {@code [n, obs]}, or null before training
     */
    public Tensor sampleObservations(int n) {
        if (buffer == null || buffer.size() == 0) {
            return null;
        }
        return buffer.gather(buffer.sample(n, 1, random), 0, encoder.device()).observations();
    }

    private Model fresh(RlNetworkFactory f, int in, int out, long salt) {
        return f.create(in, out, seed * 1000 + salt + 97 * resetsDone);
    }

    private void build() {
        if (encoder != null) {
            return;
        }
        encoder = fresh(encoderNets, obsDims, latent, 1);
        head = fresh(headNets, latent, actions, 2);
        encoderTarget = fresh(encoderNets, obsDims, latent, 1);
        headTarget = fresh(headNets, latent, actions, 2);
        if (sprSteps > 0) {
            transition = fresh(RlNetworkFactory.mlp(Activation.RELU, latent), latent + actions, latent, 3);
            projection = fresh(RlNetworkFactory.mlp(Activation.RELU, latent), latent, latent, 4);
            predictor = fresh(RlNetworkFactory.mlp(Activation.RELU, latent), latent, latent, 5);
            projectionTarget = fresh(RlNetworkFactory.mlp(Activation.RELU, latent), latent, latent, 4);
        }
        syncTeacher();
        optimizer = weightDecay > 0 ? Optimizer.adamW(lr, weightDecay) : Optimizer.adam(lr);
    }

    private void syncTeacher() {
        List<Var> student = new ArrayList<>(encoder.parameters());
        List<Var> target = new ArrayList<>(encoderTarget.parameters());
        student.addAll(head.parameters());
        target.addAll(headTarget.parameters());
        if (projection != null) {
            student.addAll(projection.parameters());
            target.addAll(projectionTarget.parameters());
        }
        if (injected != null) {
            student.addAll(injected.parameters());
            target.addAll(injectedTarget.parameters());
        }
        RlTensors.copy(student, target);
        teacher = new MomentumTeacher(target, student, 1 - tau);
    }

    private List<Var> trainable() {
        List<Var> p = new ArrayList<>(encoder.parameters());
        if (injected == null) {
            p.addAll(head.parameters());
        } else {
            p.addAll(injected.parameters());
        }
        if (transition != null) {
            p.addAll(transition.parameters());
            p.addAll(projection.parameters());
            p.addAll(predictor.parameters());
        }
        return p;
    }

    /**
     * Plasticity injection (Nikishin et al., NeurIPS 2023, <a href="https://arxiv.org/abs/2305.15555">arXiv:2305.15555</a>):
     * the current head is frozen and a fresh head minus a frozen copy of it is added, so the
     * output is unchanged at the moment of injection while new trainable capacity appears.
     */
    public void injectPlasticity() {
        build();
        long salt = 50 + updates;
        injected = fresh(headNets, latent, actions, salt);
        injectedFrozen = fresh(headNets, latent, actions, salt);
        injectedTarget = fresh(headNets, latent, actions, salt);
        syncTeacher();
        optimizer = weightDecay > 0 ? Optimizer.adamW(lr, weightDecay) : Optimizer.adam(lr);
    }

    /** @return whether plasticity was injected */
    public boolean injected() {
        return injected != null;
    }

    private Var q(Model enc, Model h, Model inj, Model injFrozen, Var obs, boolean training) {
        Var z = enc.forward(obs, training);
        Var out = h.forward(z, training);
        if (inj != null) {
            Tensor frozen = Var.noGrad(() -> injFrozen.forward(Var.of(z.value()), false)).value();
            out = out.add(inj.forward(z, training)).sub(frozen);
        }
        return out;
    }

    /**
     * @param observation an observation
     * @return the action values
     */
    public float[] actionValues(float[] observation) {
        build();
        Tensor x = Tensor.of(Shape.of(1, obsDims), observation);
        return Var.noGrad(() -> q(encoder, head, injected, injectedFrozen, Var.of(x), false)).value().toArray();
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        double eps = Math.max(epsilonEnd, 1 - (1 - epsilonEnd) * (double) stepsTaken / epsilonSteps);
        if (explore && random.nextDouble() < eps) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {RlTensors.argmax(actionValues(observation))};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions
                || env.observationSpace().size() != obsDims) {
            throw new IllegalArgumentException("needs " + obsDims + " observation values and " + actions
                    + " actions");
        }
        build();
        if (buffer == null) {
            buffer = new LapReplayBuffer(100_000, obsDims, 1, 0, false);
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        int span = Math.max(sprSteps + 1, bbf ? 10 : nStep);
        while (recorder.steps() < steps) {
            float[] a = stepsTaken < learningStarts ? new float[] {random.nextInt(actions)} : act(obs, true);
            Environment.Step s = env.step(a);
            stepsTaken++;
            buffer.add(obs, a, s.reward(), s.observation(), s.terminated(), s.done());
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (stepsTaken >= learningStarts && buffer.size() > batchSize + span) {
                for (int g = 0; g < replayRatio; g++) {
                    update(span);
                }
            }
        }
        return recorder.result();
    }

    private Tensor aug(Tensor x) {
        return augment == null ? x : augment.apply(x);
    }

    private int horizonNow() {
        if (!bbf) {
            return nStep;
        }
        double p = Math.min(1, (updates % Math.max(1, resetEvery > 0 ? resetEvery : Long.MAX_VALUE)) / (double) annealUpdates);
        return (int) Math.round(10 * Math.pow(3 / 10.0, p));
    }

    private float gammaNow() {
        if (!bbf) {
            return gamma;
        }
        double p = Math.min(1, (updates % Math.max(1, resetEvery > 0 ? resetEvery : Long.MAX_VALUE)) / (double) annealUpdates);
        // anneal 1 - gamma geometrically from 0.03 to 0.003
        return (float) (1 - 0.03 * Math.pow(0.1, p));
    }

    private void update(int span) {
        Device device = encoder.device();
        int n = horizonNow();
        float g = gammaNow();
        int[] idx = buffer.sample(batchSize, span, random);
        LapReplayBuffer.Batch[] seq = new LapReplayBuffer.Batch[span];
        for (int k = 0; k < span; k++) {
            seq[k] = buffer.gather(idx, k, device);
        }
        // n-step returns and bootstrap observations
        float[] ret = new float[batchSize];
        float[] boot = new float[batchSize];
        float[] bootObs = new float[batchSize * obsDims];
        float[][] nextRows = new float[n][];
        for (int k = 0; k < n; k++) {
            nextRows[k] = seq[k].nextObservations().toArray();
        }
        for (int i = 0; i < batchSize; i++) {
            float disc = 1;
            int last = 0;
            for (int k = 0; k < n; k++) {
                if (seq[k].rawMask()[i] == 0) {
                    break;
                }
                ret[i] += disc * seq[k].rawRewards()[i];
                disc *= g;
                last = k;
                if (seq[k].rawNotDone()[i] == 0) {
                    disc = 0;
                    break;
                }
            }
            boot[i] = disc;
            System.arraycopy(nextRows[last], i * obsDims, bootObs, i * obsDims, obsDims);
        }
        Tensor bootT = Tensor.of(Shape.of(batchSize, obsDims), bootObs).to(device);
        Tensor target = null;
        for (int k = 0; k < augK; k++) {
            Tensor nb = aug(bootT);
            Tensor online = Var.noGrad(() -> q(encoder, head, injected, injectedFrozen, Var.of(nb), false)).value();
            Tensor best = RlTensors.oneHot(argmaxRows(online), actions, device);
            Tensor tq = Var.noGrad(() -> q(encoderTarget, headTarget, injectedTarget, injectedFrozen, Var.of(nb), false))
                    .value().mul(best).sum(1, true);
            target = target == null ? tq : target.add(tq);
        }
        target = target.div(augK);
        Tensor y = RlTensors.column(ret, device).add(RlTensors.column(boot, device).mul(target));
        Tensor actionHot = RlTensors.oneHot(seq[0].actions().toArray(), actions, device);
        Var loss = null;
        Var firstLatent = null;
        for (int m = 0; m < augM; m++) {
            Var obs = Var.of(aug(seq[0].observations()));
            Var z = encoder.forward(obs, true);
            if (m == 0) {
                firstLatent = z;
            }
            Var out = head.forward(z, true);
            if (injected != null) {
                Tensor frozen = injectedFrozen.forward(Var.of(z.value()), false).value();
                out = out.add(injected.forward(z, true)).sub(frozen);
            }
            Var qa = out.mul(actionHot).sum(1, true);
            Var l = RlTensors.huber(qa.sub(y), 1f).mean();
            loss = loss == null ? l : loss.add(l);
        }
        loss = loss.div(augM);
        lastLoss = loss.value().item();
        if (sprSteps > 0) {
            Var z = firstLatent;
            Var spr = null;
            for (int k = 1; k <= sprSteps; k++) {
                Tensor a = RlTensors.oneHot(seq[k - 1].actions().toArray(), actions, device);
                z = transition.forward(Var.concat(1, z, Var.of(a)), true);
                Var pred = l2(predictor.forward(projection.forward(z, true), true));
                Tensor futureObs = aug(seq[k].observations());
                Tensor tgt = Var.noGrad(() -> l2(projectionTarget.forward(encoderTarget.forward(Var.of(futureObs),
                        false), false))).value();
                Var cos = pred.mul(tgt).sum(1, true).mul(seq[k].mask()).mean().neg();
                spr = spr == null ? cos : spr.add(cos);
            }
            loss = loss.add(spr.mul(sprWeight / sprSteps));
        }
        loss.backward();
        List<Var> params = trainable();
        RlTensors.clip(params, 10f);
        optimizer.step(params);
        RlTensors.zeroGrad(head.parameters());
        teacher.update();
        updates++;
        if (onUpdate != null) {
            onUpdate.accept(this);
        }
        if (resetEvery > 0 && updates % resetEvery == 0) {
            reset();
        }
    }

    private static Var l2(Var x) {
        return x.div(x.square().sum(1, true).add(1e-6f).sqrt());
    }

    /** BBF's reset: a fresh Q head; encoder and transition model shrunk and perturbed. */
    public void reset() {
        build();
        resetsDone++;
        Model freshHead = fresh(headNets, latent, actions, 2);
        RlTensors.copy(freshHead.parameters(), head.parameters());
        shrinkPerturb(encoder, fresh(encoderNets, obsDims, latent, 1), shrink);
        if (transition != null) {
            shrinkPerturb(transition, fresh(RlNetworkFactory.mlp(Activation.RELU, latent), latent + actions, latent, 3),
                    shrink);
        }
        injected = null;
        injectedFrozen = null;
        injectedTarget = null;
        syncTeacher();
        optimizer = weightDecay > 0 ? Optimizer.adamW(lr, weightDecay) : Optimizer.adam(lr);
    }

    /**
     * Shrink and perturb (Ash and Adams, 2020): {@code w <- alpha w + (1 - alpha) w_fresh}.
     *
     * @param model the model changed in place
     * @param freshModel a freshly initialised model of the same layout
     * @param alpha the shrink factor
     */
    public static void shrinkPerturb(Model model, Model freshModel, float alpha) {
        List<Var> p = model.parameters();
        List<Var> f = freshModel.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(p.get(i).value().mul(alpha).add(f.get(i).value().mul(1 - alpha)));
        }
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
    public String toString() {
        return "SelfPredictive(spr " + sprSteps + ", replay ratio " + replayRatio + (resetEvery > 0 ? ", resets" : "")
                + (augment != null ? ", DrQ" : "") + ")";
    }
}
