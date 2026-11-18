package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.RlTensors;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * TD3 (Fujimoto et al., 2018, <a href="https://arxiv.org/abs/1802.09477">arXiv:1802.09477</a>)
 * driven one call at a time, with a replay buffer whose rewards can be rewritten: the engine of
 * the hierarchical, preference-based and goal-relabelling learners, which must store
 * transitions they construct themselves (subgoal transitions, hindsight goals) and relabel the
 * whole buffer when a learned reward model changes. Core {@code Td3}, {@code Sac} and
 * {@code ModernSac} only train through {@code fit}, and core {@code ReplayBuffer} cannot
 * rewrite rewards, hence this class.
 *
 * <p>Actions are in {@code [-1, 1]} (a tanh actor); {@link #fit} scales them to the
 * environment's box, so the class is also a plain {@link Learner}.
 */
public final class StepwiseTd3 implements Learner {

    /**
     * A stored transition; its reward may be rewritten by {@link #relabel}.
     *
     * @param observation the observation
     * @param action the action in {@code [-1, 1]}
     * @param reward the reward
     * @param next the next observation
     * @param terminal whether the episode terminated
     */
    public record Entry(float[] observation, float[] action, float reward, float[] next, boolean terminal) {
    }

    /** Rewrites the reward of a stored transition. */
    @FunctionalInterface
    public interface Relabel {
        /**
         * @param entry the transition
         * @return its new reward
         */
        float reward(Entry entry);
    }

    private final int obsDims;
    private final int actDims;
    private final Model actor;
    private final Model actorTarget;
    private final Model q1;
    private final Model q2;
    private final Model q1Target;
    private final Model q2Target;
    private final Optimizer actorOpt;
    private final Optimizer criticOpt;
    private final List<Entry> buffer = new ArrayList<>();
    private int capacity = 100_000;
    private int position;
    private float gamma = 0.98f;
    private float tau = 0.01f;
    private float noise = 0.2f;
    private float targetNoise = 0.2f;
    private float noiseClip = 0.5f;
    private int batchSize = 64;
    private int policyDelay = 2;
    private int learningStarts = 200;
    private long updates;
    private SplittableRandom random;
    private Space.Box box;

    private StepwiseTd3(int obsDims, int actDims, RlNetworkFactory nets, float lr, long seed) {
        this.obsDims = obsDims;
        this.actDims = actDims;
        this.random = new SplittableRandom(seed);
        actor = nets.create(obsDims, actDims, seed + 1);
        actorTarget = nets.create(obsDims, actDims, seed + 1);
        q1 = nets.create(obsDims + actDims, 1, seed + 2);
        q2 = nets.create(obsDims + actDims, 1, seed + 3);
        q1Target = nets.create(obsDims + actDims, 1, seed + 2);
        q2Target = nets.create(obsDims + actDims, 1, seed + 3);
        RlTensors.copy(actor.parameters(), actorTarget.parameters());
        RlTensors.copy(q1.parameters(), q1Target.parameters());
        RlTensors.copy(q2.parameters(), q2Target.parameters());
        actorOpt = Optimizer.adam(lr);
        criticOpt = Optimizer.adam(lr);
    }

    /**
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param networks builds the actor and critics
     * @param learningRate Adam's rate for both
     * @param seed the seed
     * @return the learner
     */
    public static StepwiseTd3 of(int observationDims, int actionDims, RlNetworkFactory networks, float learningRate,
            long seed) {
        return new StepwiseTd3(observationDims, actionDims, Objects.requireNonNull(networks), learningRate, seed);
    }

    /** @param discount gamma @return this learner */
    public StepwiseTd3 discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param std the exploration noise standard deviation @return this learner */
    public StepwiseTd3 explorationNoise(float std) {
        this.noise = std;
        return this;
    }

    /** @param steps random steps before learning (in {@link #fit}) @return this learner */
    public StepwiseTd3 learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param size the batch size @return this learner */
    public StepwiseTd3 batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @return gradient updates taken */
    public long updates() {
        return updates;
    }

    /** @return the stored transitions (live view) */
    public List<Entry> buffer() {
        return buffer;
    }

    /** @return the actor */
    public Model actor() {
        return actor;
    }

    /**
     * @param observation an observation
     * @param explore add Gaussian exploration noise
     * @return an action in {@code [-1, 1]}
     */
    public float[] unitAction(float[] observation, boolean explore) {
        float[] a = RlTensors.row(actor, observation);
        for (int i = 0; i < a.length; i++) {
            double v = Math.tanh(a[i]);
            if (explore) {
                v += noise * random.nextGaussian();
            }
            a[i] = (float) Math.max(-1, Math.min(1, v));
        }
        return a;
    }

    /**
     * Stores a transition.
     *
     * @param observation the observation
     * @param action the action in {@code [-1, 1]}
     * @param reward the reward
     * @param next the next observation
     * @param terminal whether the episode terminated there
     */
    public void remember(float[] observation, float[] action, float reward, float[] next, boolean terminal) {
        Entry e = new Entry(observation.clone(), action.clone(), reward, next.clone(), terminal);
        if (buffer.size() < capacity) {
            buffer.add(e);
        } else {
            buffer.set(position, e);
            position = (position + 1) % capacity;
        }
    }

    /**
     * Rewrites every stored reward.
     *
     * @param relabel the new reward of a transition
     */
    public void relabel(Relabel relabel) {
        for (int i = 0; i < buffer.size(); i++) {
            Entry e = buffer.get(i);
            buffer.set(i, new Entry(e.observation(), e.action(), relabel.reward(e), e.next(), e.terminal()));
        }
    }

    /**
     * One critic update (and an actor update every second call), if the buffer holds a batch.
     */
    public void update() {
        if (buffer.size() < batchSize) {
            return;
        }
        int b = batchSize;
        float[] o = new float[b * obsDims];
        float[] n = new float[b * obsDims];
        float[] a = new float[b * actDims];
        float[] r = new float[b];
        float[] notDone = new float[b];
        float[] eps = new float[b * actDims];
        for (int i = 0; i < b; i++) {
            Entry e = buffer.get(random.nextInt(buffer.size()));
            System.arraycopy(e.observation(), 0, o, i * obsDims, obsDims);
            System.arraycopy(e.next(), 0, n, i * obsDims, obsDims);
            System.arraycopy(e.action(), 0, a, i * actDims, actDims);
            r[i] = e.reward();
            notDone[i] = e.terminal() ? 0 : 1;
        }
        for (int i = 0; i < eps.length; i++) {
            eps[i] = (float) Math.max(-noiseClip, Math.min(noiseClip, targetNoise * random.nextGaussian()));
        }
        Tensor obs = Tensor.of(Shape.of(b, obsDims), o);
        Tensor next = Tensor.of(Shape.of(b, obsDims), n);
        Tensor act = Tensor.of(Shape.of(b, actDims), a);
        Tensor nextA = actorTarget.forward(Var.of(next), false).value().tanh().add(Tensor.of(Shape.of(b, actDims), eps))
                .minimum(1f).maximum(-1f);
        Tensor nextIn = Tensor.concat(1, next, nextA);
        Tensor target = q1Target.forward(Var.of(nextIn), false).value()
                .minimum(q2Target.forward(Var.of(nextIn), false).value());
        Tensor y = Tensor.of(Shape.of(b, 1), r).add(Tensor.of(Shape.of(b, 1), notDone).mul(gamma).mul(target));
        Tensor in = Tensor.concat(1, obs, act);
        Var loss = q1.forward(Var.of(in), true).sub(y).square().mean()
                .add(q2.forward(Var.of(in), true).sub(y).square().mean());
        loss.backward();
        List<Var> critics = new ArrayList<>(q1.parameters());
        critics.addAll(q2.parameters());
        RlTensors.clip(critics, 10f);
        criticOpt.step(critics);
        updates++;
        if (updates % policyDelay == 0) {
            Var pa = actor.forward(Var.of(obs), true).tanh();
            Var actorLoss = q1.forward(Var.concat(1, Var.of(obs), pa), false).mean().neg()
                    .add(pa.square().mean().mul(1e-3f));
            actorLoss.backward();
            RlTensors.zeroGrad(critics);
            actorOpt.step(actor.parameters());
            RlTensors.blend(actor.parameters(), actorTarget.parameters(), tau);
            RlTensors.blend(q1.parameters(), q1Target.parameters(), tau);
            RlTensors.blend(q2.parameters(), q2Target.parameters(), tau);
        }
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] u = unitAction(observation, explore);
        return box == null ? u : scale(box, u);
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        box = (Space.Box) env.actionSpace();
        Tracker t = new Tracker(listener);
        float[] obs = env.reset(random.split());
        long taken = 0;
        while (t.steps() < steps) {
            float[] unit;
            if (buffer.size() < learningStarts) {
                unit = new float[actDims];
                for (int i = 0; i < actDims; i++) {
                    unit[i] = (float) random.nextDouble(-1, 1);
                }
            } else {
                unit = unitAction(obs, true);
            }
            Environment.Step s = env.step(scale(box, unit));
            remember(obs, unit, s.reward(), s.observation(), s.terminated());
            obs = t.step(s) ? env.reset(random.split()) : s.observation();
            if (++taken > learningStarts) {
                update();
            }
        }
        return t.result();
    }

    /**
     * @param box the action box
     * @param unit an action in {@code [-1, 1]}
     * @return the action in the box
     */
    public static float[] scale(Space.Box box, float[] unit) {
        float[] out = new float[unit.length];
        for (int i = 0; i < unit.length; i++) {
            out[i] = box.low()[i] + (unit[i] + 1) / 2 * (box.high()[i] - box.low()[i]);
        }
        return out;
    }

    @Override
    public String toString() {
        return "StepwiseTd3";
    }
}
