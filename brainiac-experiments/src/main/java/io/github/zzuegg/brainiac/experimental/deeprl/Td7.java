package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
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
 * TD7 (Fujimoto et al., "For SALE: State-Action Representation Learning for Deep Reinforcement
 * Learning", NeurIPS 2023, <a href="https://arxiv.org/abs/2306.02451">arXiv:2306.02451</a>):
 * TD3 plus four additions, each switchable for ablations.
 *
 * <ul>
 *   <li>SALE: a state encoder {@code z_s = f(s)} and a state-action encoder
 *       {@code z_sa = g(z_s, a)}, both ending in AvgL1Norm, trained so that {@code z_sa}
 *       predicts the next state's {@code z_s'}; their outputs are concatenated to the critic's
 *       and actor's inputs. Frozen copies of the encoders feed the networks and are refreshed at
 *       every target update.</li>
 *   <li>LAP: {@link LapReplayBuffer} prioritised replay with a Huber loss.</li>
 *   <li>Value clipping: target values are clipped to the range of targets seen in the previous
 *       target-update interval, which stops extrapolation error from compounding.</li>
 *   <li>Policy checkpoints: training is deferred to episode ends; a fixed policy is assessed
 *       for several episodes, and the best one by its minimum return is kept and deployed for
 *       evaluation, so the evaluated policy is not whichever the last update happened to
 *       produce.</li>
 * </ul>
 */
public final class Td7 implements Learner {

    /**
     * Which components are on.
     *
     * @param sale state-action learned embeddings
     * @param lap loss-adjusted prioritised replay
     * @param clipping value clipping
     * @param checkpoints policy checkpoints
     */
    public record Components(boolean sale, boolean lap, boolean clipping, boolean checkpoints) {
        /** @return everything on */
        public static Components all() {
            return new Components(true, true, true, true);
        }

        /** @return everything off: TD3 with hard target updates */
        public static Components none() {
            return new Components(false, false, false, false);
        }
    }

    private final int obsDims;
    private final int actDims;
    private final int hidden;
    private final int zDim;
    private final Components parts;
    private float gamma = 0.99f;
    private int batchSize = 64;
    private int learningStarts = 500;
    private int targetEvery = 250;
    private float lr = 1e-3f;
    private int checkpointEpisodes = 3;
    private SplittableRandom random = new SplittableRandom(0);
    private long seed;

    private Model f;
    private Model g;
    private Model fFixed;
    private Model gFixed;
    private Model fFixedTarget;
    private Model gFixedTarget;
    private Model actor;
    private Model actorTarget;
    private Model q1;
    private Model q2;
    private Model q1Target;
    private Model q2Target;
    private Model ckActor;
    private Model ckEncoder;
    private Optimizer encOpt;
    private Optimizer actorOpt;
    private Optimizer criticOpt;
    private LapReplayBuffer buffer;
    private float[] low;
    private float[] high;
    private long updates;
    private double qMin = 0;
    private double qMax = 0;
    private double seenMin = Double.POSITIVE_INFINITY;
    private double seenMax = Double.NEGATIVE_INFINITY;
    private long stepsTaken;

    private Td7(int obsDims, int actDims, int hidden, int zDim, Components parts) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.hidden = hidden;
        this.zDim = parts.sale() ? zDim : 0;
        this.parts = parts;
    }

    /**
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param hidden hidden width of every network
     * @param embedding the SALE embedding size
     * @param components the parts in use
     * @return the learner
     */
    public static Td7 of(int observationDims, int actionDims, int hidden, int embedding, Components components) {
        return new Td7(observationDims, actionDims, hidden, embedding, components);
    }

    /** @param seed the seed @return this learner */
    public Td7 seed(long seed) {
        this.seed = seed;
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @param steps random steps first @return this learner */
    public Td7 learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param rate the learning rate @return this learner */
    public Td7 learningRate(float rate) {
        this.lr = rate;
        return this;
    }

    /** @param updates gradient updates between hard target updates @return this learner */
    public Td7 targetEvery(int updates) {
        this.targetEvery = updates;
        return this;
    }

    /** @param episodes episodes a checkpoint candidate is assessed for @return this learner */
    public Td7 checkpointEpisodes(int episodes) {
        this.checkpointEpisodes = Math.max(1, episodes);
        return this;
    }

    /** @param discount gamma @return this learner */
    public Td7 discount(float discount) {
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
        if (actor != null) {
            return;
        }
        if (zDim > 0) {
            f = net(obsDims, zDim, 1);
            g = net(zDim + actDims, zDim, 2);
            fFixed = net(obsDims, zDim, 1);
            gFixed = net(zDim + actDims, zDim, 2);
            fFixedTarget = net(obsDims, zDim, 1);
            gFixedTarget = net(zDim + actDims, zDim, 2);
            ckEncoder = net(obsDims, zDim, 1);
            encOpt = Optimizer.adam(lr);
        }
        actor = net(obsDims + zDim, actDims, 3);
        actorTarget = net(obsDims + zDim, actDims, 3);
        ckActor = net(obsDims + zDim, actDims, 3);
        int qIn = obsDims + actDims + 2 * zDim;
        q1 = net(qIn, 1, 4);
        q2 = net(qIn, 1, 5);
        q1Target = net(qIn, 1, 4);
        q2Target = net(qIn, 1, 5);
        actorOpt = Optimizer.adam(lr);
        criticOpt = Optimizer.adam(lr);
    }

    private static Var avgL1(Var x) {
        return x.div(x.abs().mean(1, true).add(1e-6f));
    }

    private static Tensor avgL1(Tensor x) {
        return x.div(x.abs().mean(1, true).add(1e-6f));
    }

    private Tensor zs(Model enc, Tensor s) {
        return avgL1(Var.noGrad(() -> enc.forward(Var.of(s), false)).value());
    }

    private Tensor zsa(Model enc, Tensor zs, Tensor a) {
        Tensor in = Tensor.concat(1, zs, a);
        return avgL1(Var.noGrad(() -> enc.forward(Var.of(in), false)).value());
    }

    private Tensor actorInput(Tensor s, Tensor zs) {
        return zDim > 0 ? Tensor.concat(1, s, zs) : s;
    }

    private Tensor criticInput(Tensor s, Tensor a, Tensor zsa, Tensor zs) {
        return zDim > 0 ? Tensor.concat(1, s, a, zsa, zs) : Tensor.concat(1, s, a);
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        build();
        Tensor s = Tensor.of(Shape.of(1, obsDims), observation);
        boolean useCheckpoint = !explore && parts.checkpoints() && updates > 0;
        Model pi = useCheckpoint ? ckActor : actor;
        Tensor zs = zDim > 0 ? zs(useCheckpoint ? ckEncoder : fFixed, s) : null;
        Tensor in = actorInput(s, zs);
        float[] out = Var.noGrad(() -> pi.forward(Var.of(in), false)).value().toArray();
        float[] unit = new float[actDims];
        for (int i = 0; i < actDims; i++) {
            double u = Math.tanh(out[i]) + (explore ? 0.1 * random.nextGaussian() : 0);
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

    private float[] unit(float[] scaled) {
        float[] u = new float[scaled.length];
        for (int i = 0; i < u.length; i++) {
            u[i] = 2 * (scaled[i] - low[i]) / (high[i] - low[i]) - 1;
        }
        return u;
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Box box) || box.size() != actDims) {
            throw new IllegalArgumentException("TD7 needs a box of " + actDims + " actions");
        }
        build();
        low = box.low();
        high = box.high();
        if (buffer == null) {
            buffer = new LapReplayBuffer(100_000, obsDims, actDims, 0.4, parts.lap());
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        int pending = 0;
        int assessed = 0;
        double minReturn = Double.POSITIVE_INFINITY;
        double best = Double.NEGATIVE_INFINITY;
        double episodeReturn = 0;
        while (recorder.steps() < steps) {
            float[] u;
            if (stepsTaken < learningStarts) {
                u = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    u[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                u = unit(act(obs, true));
            }
            Environment.Step s = env.step(scale(u));
            stepsTaken++;
            episodeReturn += s.reward();
            buffer.add(obs, u, s.reward(), s.observation(), s.terminated(), s.done());
            boolean ended = recorder.step(s);
            obs = ended ? env.reset(random.split()) : s.observation();
            boolean learning = stepsTaken >= learningStarts && buffer.size() >= batchSize;
            if (!parts.checkpoints()) {
                if (learning) {
                    update();
                }
                continue;
            }
            if (learning) {
                pending++;
            }
            if (ended) {
                assessed++;
                minReturn = Math.min(minReturn, episodeReturn);
                episodeReturn = 0;
                boolean worse = minReturn < best;
                if (pending > 0 && (worse || assessed >= checkpointEpisodes)) {
                    if (!worse) {
                        best = minReturn;
                        RlTensors.copy(actor.parameters(), ckActor.parameters());
                        if (zDim > 0) {
                            RlTensors.copy(fFixed.parameters(), ckEncoder.parameters());
                        }
                    }
                    for (int i = 0; i < pending; i++) {
                        update();
                    }
                    pending = 0;
                    assessed = 0;
                    minReturn = Double.POSITIVE_INFINITY;
                }
            }
        }
        if (parts.checkpoints() && updates > 0 && best == Double.NEGATIVE_INFINITY) {
            RlTensors.copy(actor.parameters(), ckActor.parameters());
            if (zDim > 0) {
                RlTensors.copy(fFixed.parameters(), ckEncoder.parameters());
            }
        }
        return recorder.result();
    }

    private void update() {
        Device device = actor.device();
        int[] idx = buffer.sample(batchSize, 1, random);
        LapReplayBuffer.Batch b = buffer.gather(idx, 0, device);
        Tensor s = b.observations();
        Tensor a = b.actions();
        Tensor sn = b.nextObservations();
        if (zDim > 0) {
            Tensor target = zs(f, sn);
            Var zsOnline = avgL1(f.forward(Var.of(s), true));
            Var zsaOnline = avgL1(g.forward(Var.concat(1, zsOnline, Var.of(a)), true));
            Var encLoss = zsaOnline.sub(target).square().mean();
            encLoss.backward();
            List<Var> p = new ArrayList<>(f.parameters());
            p.addAll(g.parameters());
            encOpt.step(p);
        }
        Tensor zs = zDim > 0 ? zs(fFixed, s) : null;
        Tensor zsa = zDim > 0 ? zsa(gFixed, zs, a) : null;
        Tensor zsn = zDim > 0 ? zs(fFixedTarget, sn) : null;
        Tensor nextIn = actorInput(sn, zsn);
        Tensor an = Var.noGrad(() -> actorTarget.forward(Var.of(nextIn), false)).value().tanh();
        Tensor noise = Tensor.normal(an.shape(), 0, 0.2f, random).maximum(Tensor.scalar(-0.5f))
                .minimum(Tensor.scalar(0.5f));
        an = an.add(noise).maximum(Tensor.scalar(-1)).minimum(Tensor.scalar(1));
        Tensor zsan = zDim > 0 ? zsa(gFixedTarget, zsn, an) : null;
        Tensor qIn = criticInput(sn, an, zsan, zsn);
        Tensor qNext = Var.noGrad(() -> q1Target.forward(Var.of(qIn), false)).value()
                .minimum(Var.noGrad(() -> q2Target.forward(Var.of(qIn), false)).value());
        if (parts.clipping()) {
            seenMin = Math.min(seenMin, qNext.min().item());
            seenMax = Math.max(seenMax, qNext.max().item());
            if (updates >= targetEvery) {
                qNext = qNext.maximum(Tensor.scalar((float) qMin)).minimum(Tensor.scalar((float) qMax));
            }
        }
        Tensor y = b.rewards().add(b.notDone().mul(qNext).mul(gamma));
        Var in = Var.of(criticInput(s, a, zsa, zs));
        Var d1 = q1.forward(in, true).sub(y);
        Var d2 = q2.forward(in, true).sub(y);
        Var loss = parts.lap() ? RlTensors.huber(d1, 1f).add(RlTensors.huber(d2, 1f)).mean()
                : d1.square().add(d2.square()).mean();
        loss.backward();
        List<Var> criticParams = new ArrayList<>(q1.parameters());
        criticParams.addAll(q2.parameters());
        criticOpt.step(criticParams);
        if (parts.lap()) {
            float[] e1 = d1.value().abs().toArray();
            float[] e2 = d2.value().abs().toArray();
            float[] e = new float[e1.length];
            for (int i = 0; i < e.length; i++) {
                e[i] = Math.max(e1[i], e2[i]);
            }
            buffer.updatePriorities(idx, e);
        }
        if (updates % 2 == 0) {
            Var pi = actor.forward(Var.of(actorInput(s, zs)), true).tanh();
            Var piIn;
            if (zDim > 0) {
                Var zsaPi = avgL1(gFixed.forward(Var.concat(1, Var.of(zs), pi), false));
                piIn = Var.concat(1, Var.of(s), pi, zsaPi, Var.of(zs));
            } else {
                piIn = Var.concat(1, Var.of(s), pi);
            }
            Var actorLoss = q1.forward(piIn, false).mean().neg();
            actorLoss.backward();
            actorOpt.step(actor.parameters());
            RlTensors.zeroGrad(criticParams);
            if (zDim > 0) {
                RlTensors.zeroGrad(gFixed.parameters());
            }
        }
        updates++;
        if (updates % targetEvery == 0) {
            RlTensors.copy(actor.parameters(), actorTarget.parameters());
            RlTensors.copy(q1.parameters(), q1Target.parameters());
            RlTensors.copy(q2.parameters(), q2Target.parameters());
            if (zDim > 0) {
                RlTensors.copy(fFixed.parameters(), fFixedTarget.parameters());
                RlTensors.copy(gFixed.parameters(), gFixedTarget.parameters());
                RlTensors.copy(f.parameters(), fFixed.parameters());
                RlTensors.copy(g.parameters(), gFixed.parameters());
            }
            if (seenMin <= seenMax) {
                qMin = seenMin;
                qMax = seenMax;
            }
            seenMin = Double.POSITIVE_INFINITY;
            seenMax = Double.NEGATIVE_INFINITY;
        }
    }

    @Override
    public String toString() {
        return "TD7" + parts;
    }
}
