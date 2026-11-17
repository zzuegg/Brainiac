package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
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
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Deterministic actor-critic for continuous actions: TD3 (Fujimoto et al., 2018,
 * <a href="https://arxiv.org/abs/1802.09477">arXiv:1802.09477</a>) and, with its three tricks
 * switched off, DDPG (Lillicrap et al., 2015, <a href="https://arxiv.org/abs/1509.02971">arXiv:1509.02971</a>).
 *
 * <p>A deterministic actor {@code mu(s)} (tanh-squashed) is trained to climb the critic's
 * {@code Q(s, mu(s))}; the critic learns from replayed transitions against slowly moving
 * target networks. TD3 adds the three fixes that made DDPG reliable:
 * <ul>
 *   <li><b>clipped double Q</b>: two critics, the target uses their minimum;</li>
 *   <li><b>delayed policy updates</b>: the actor and targets move every {@code d} critic steps;</li>
 *   <li><b>target policy smoothing</b>: clipped noise on the target action, so the
 *       critic cannot exploit sharp peaks.</li>
 * </ul>
 *
 * <pre>{@code
 * Td3 td3 = Td3.td3(3, 1, RlNetworkFactory.mlp(Activation.RELU, 64, 64));
 * Td3 ddpg = Td3.ddpg(3, 1, RlNetworkFactory.mlp(Activation.RELU, 64, 64));
 * }</pre>
 */
public final class Td3 implements Learner {

    private final int obsDims;
    private final int actDims;
    private final boolean twin;
    private final Model actor;
    private final Model actorTarget;
    private final Model q1;
    private final Model q1Target;
    private final Model q2;
    private final Model q2Target;
    private Optimizer actorOpt = Optimizer.adam(1e-3f);
    private Optimizer criticOpt = Optimizer.adam(1e-3f);
    private float gamma = 0.99f;
    private float tau = 0.005f;
    private int policyDelay;
    private float targetNoise;
    private float noiseClip = 0.5f;
    private float explorationNoise = 0.1f;
    private int batchSize = 64;
    private int bufferSize = 100_000;
    private int learningStarts = 500;
    private float[] low;
    private float[] high;
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long updates;
    private double lastCriticLoss = Double.NaN;

    private Td3(int obsDims, int actDims, RlNetworkFactory f, boolean twin, int delay, float smoothing) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.twin = twin;
        this.policyDelay = delay;
        this.targetNoise = smoothing;
        actor = f.create(obsDims, actDims, 21);
        actorTarget = f.create(obsDims, actDims, 21);
        q1 = f.create(obsDims + actDims, 1, 22);
        q1Target = f.create(obsDims + actDims, 1, 22);
        q2 = twin ? f.create(obsDims + actDims, 1, 23) : null;
        q2Target = twin ? f.create(obsDims + actDims, 1, 23) : null;
        RlTensors.copy(actor.parameters(), actorTarget.parameters());
        RlTensors.copy(q1.parameters(), q1Target.parameters());
        if (twin) {
            RlTensors.copy(q2.parameters(), q2Target.parameters());
        }
    }

    /**
     * TD3 with twin critics, policy delay 2 and target noise 0.2.
     *
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param networks builds actor ({@code obs -> actionDims}) and critics ({@code obs + action -> 1})
     * @return the learner
     */
    public static Td3 td3(int observationDims, int actionDims, RlNetworkFactory networks) {
        return new Td3(observationDims, actionDims, Objects.requireNonNull(networks), true, 2, 0.2f);
    }

    /**
     * DDPG: one critic, the actor updated every step, no target smoothing.
     *
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param networks builds actor and critic
     * @return the learner
     */
    public static Td3 ddpg(int observationDims, int actionDims, RlNetworkFactory networks) {
        return new Td3(observationDims, actionDims, Objects.requireNonNull(networks), false, 1, 0f);
    }

    /** @param lr the learning rate of actor and critics @return this learner */
    public Td3 learningRate(float lr) {
        actorOpt = Optimizer.adam(lr);
        criticOpt = Optimizer.adam(lr);
        return this;
    }

    /** @param discount the discount gamma @return this learner */
    public Td3 discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param tau the Polyak factor of the targets @return this learner */
    public Td3 tau(float tau) {
        this.tau = tau;
        return this;
    }

    /** @param delay critic steps per actor step @return this learner */
    public Td3 policyDelay(int delay) {
        this.policyDelay = Math.max(1, delay);
        return this;
    }

    /**
     * @param noise standard deviation of the target smoothing noise (in unit action space)
     * @param clip its clip range
     * @return this learner
     */
    public Td3 targetSmoothing(float noise, float clip) {
        this.targetNoise = noise;
        this.noiseClip = clip;
        return this;
    }

    /** @param std Gaussian exploration noise in unit action space @return this learner */
    public Td3 explorationNoise(float std) {
        this.explorationNoise = std;
        return this;
    }

    /** @param size transitions per update @return this learner */
    public Td3 batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param size transitions kept @return this learner */
    public Td3 bufferSize(int size) {
        this.bufferSize = size;
        return this;
    }

    /** @param steps random steps before learning @return this learner */
    public Td3 learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param seed the seed @return this learner */
    public Td3 seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the actor */
    public Model actor() {
        return actor;
    }

    /** @return the gradient steps taken */
    public long updates() {
        return updates;
    }

    /** @return the critic loss of the last update */
    public double lastCriticLoss() {
        return lastCriticLoss;
    }

    private float[] unit(float[] observation, boolean explore) {
        Tensor x = Tensor.of(Shape.of(1, obsDims), observation).to(actor.device());
        float[] out = actor.forward(Var.of(x), false).tanh().value().to(DType.FLOAT32).toArray();
        if (explore) {
            for (int i = 0; i < out.length; i++) {
                out[i] = (float) Math.max(-1, Math.min(1, out[i] + explorationNoise * random.nextGaussian()));
            }
        }
        return out;
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
            throw new IllegalArgumentException(this + " needs " + obsDims + " observation values and a box of "
                    + actDims + " actions");
        }
        low = box.low();
        high = box.high();
        if (buffer == null) {
            buffer = new ReplayBuffer(bufferSize, obsDims, actDims);
        }
        Tracker recorder = new Tracker(listener);
        float[] obs = env.reset(random.split());
        long taken = 0;
        while (recorder.steps() < steps) {
            float[] u;
            if (taken < learningStarts) {
                u = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    u[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                u = unit(obs, true);
            }
            Environment.Step s = env.step(scale(u));
            taken++;
            buffer.add(new Transition(obs, u, s.reward(), s.observation(), s.terminated(), gamma));
            obs = recorder.step(s) ? env.reset(random.split()) : s.observation();
            if (taken >= learningStarts && buffer.size() >= batchSize) {
                update();
            }
        }
        return recorder.result();
    }

    private void update() {
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, actor.device());
        Tensor nextA = actorTarget.forward(Var.of(b.nextObservations()), false).value().tanh();
        if (targetNoise > 0) {
            Tensor noise = Tensor.normal(Shape.of(batchSize, actDims), 0, targetNoise, random)
                    .maximum(Tensor.full(Shape.of(1, 1), -noiseClip))
                    .minimum(Tensor.full(Shape.of(1, 1), noiseClip));
            nextA = nextA.add(noise).maximum(Tensor.full(Shape.of(1, 1), -1f))
                    .minimum(Tensor.full(Shape.of(1, 1), 1f));
        }
        Var nextIn = Var.of(Tensor.concat(1, b.nextObservations(), nextA));
        Tensor nextQ = q1Target.forward(nextIn, false).value();
        if (twin) {
            nextQ = nextQ.minimum(q2Target.forward(nextIn, false).value());
        }
        Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(nextQ));
        Var in = Var.of(Tensor.concat(1, b.observations(), b.actions()));
        Var loss = q1.forward(in, true).sub(y).square().mean();
        List<Var> critics = new ArrayList<>(q1.parameters());
        if (twin) {
            loss = loss.add(q2.forward(in, true).sub(y).square().mean());
            critics.addAll(q2.parameters());
        }
        loss.backward();
        criticOpt.step(critics);
        lastCriticLoss = loss.value().item();
        updates++;
        if (updates % policyDelay == 0) {
            Var obs = Var.of(b.observations());
            Var a = actor.forward(obs, true).tanh();
            Var actorLoss = q1.forward(Var.concat(1, obs, a), true).mean().neg()
                    .add(a.square().mean().mul(1e-3f));
            actorLoss.backward();
            actorOpt.step(actor.parameters());
            RlTensors.zeroGrad(critics);
            RlTensors.blend(actor.parameters(), actorTarget.parameters(), tau);
            RlTensors.blend(q1.parameters(), q1Target.parameters(), tau);
            if (twin) {
                RlTensors.blend(q2.parameters(), q2Target.parameters(), tau);
            }
        }
    }

    @Override
    public String toString() {
        return twin ? "TD3" : "DDPG";
    }
}
