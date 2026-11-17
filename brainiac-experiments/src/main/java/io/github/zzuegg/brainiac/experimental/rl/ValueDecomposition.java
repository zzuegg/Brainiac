package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Cooperative multi-agent Q-learning with centralized training and decentralized
 * execution. Every agent picks its action greedily from its own Q-network (shared
 * weights; the observation carries the agent's id), but training can see the team:
 * <ul>
 *   <li>{@link Mixer#INDEPENDENT}: independent Q-learners (Tan, 1993), each treating the
 *       others as part of the environment and learning from the team reward alone;</li>
 *   <li>{@link Mixer#VDN}: value decomposition networks (Sunehag et al., 2017,
 *       <a href="https://arxiv.org/abs/1706.05296">arXiv:1706.05296</a>), {@code Q_tot = sum_i Q_i}
 *       trained end-to-end on the team reward;</li>
 *   <li>{@link Mixer#QMIX}: QMIX (Rashid et al., 2018, <a href="https://arxiv.org/abs/1803.11485">arXiv:1803.11485</a>),
 *       a mixing network whose weights come from hypernetworks of the global state and are
 *       kept non-negative, so {@code Q_tot} is monotonic in every {@code Q_i}: each agent's
 *       greedy action is still the team's greedy action, yet far richer joint values fit.</li>
 * </ul>
 */
public final class ValueDecomposition {

    /** How agent values combine into the team value. */
    public enum Mixer {
        /** no mixing: every agent learns on its own */
        INDEPENDENT,
        /** sum of agent values */
        VDN,
        /** monotonic, state-conditioned mixing network */
        QMIX
    }

    private record Sample(float[][] obs, float[] state, int[] actions, float reward, float[][] next,
            float[] nextState, boolean terminated) {
    }

    private final Mixer mixer;
    private final int agents;
    private final int obsSize;
    private final int stateSize;
    private final int actions;
    private final Model q;
    private final Model qTarget;
    private final List<Var> mix = new ArrayList<>();
    private final List<Var> mixTarget = new ArrayList<>();
    private final int embed = 8;
    private Optimizer optimizer = Optimizer.adam(5e-3f);
    private Schedule epsilon = Schedule.linear(1, 0.05, 2_000);
    private float gamma = 0.99f;
    private int batchSize = 32;
    private int bufferSize = 5_000;
    private int targetEvery = 50;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Sample> buffer = new ArrayList<>();
    private int position;
    private long steps;
    private long updates;

    private ValueDecomposition(Mixer mixer, MultiAgentEnvironment env, RlNetworkFactory nets) {
        this.mixer = mixer;
        this.agents = env.agents();
        this.obsSize = env.observationSize();
        this.stateSize = env.stateSize();
        this.actions = env.actions();
        q = nets.create(obsSize, actions, 61);
        qTarget = nets.create(obsSize, actions, 61);
        RlTensors.copy(q.parameters(), qTarget.parameters());
        if (mixer == Mixer.QMIX) {
            SplittableRandom r = new SplittableRandom(62);
            int[][] shapes = {{stateSize, agents * embed}, {stateSize, embed}, {stateSize, embed}, {stateSize, embed},
                    {embed, 1}};
            for (int[] s : shapes) {
                float std = (float) Math.sqrt(1.0 / s[0]);
                Tensor init = Tensor.normal(Shape.of(s[0], s[1]), 0, std, r);
                mix.add(Var.param(init));
                mixTarget.add(Var.param(init));
            }
        }
    }

    /**
     * @param mixer how agent values combine
     * @param env the environment (for sizes)
     * @param networks builds the shared agent Q-network ({@code observation -> actions})
     * @return the learner
     */
    public static ValueDecomposition of(Mixer mixer, MultiAgentEnvironment env, RlNetworkFactory networks) {
        return new ValueDecomposition(Objects.requireNonNull(mixer), env, Objects.requireNonNull(networks));
    }

    /** @param s exploration by step @return this learner */
    public ValueDecomposition epsilon(Schedule s) {
        this.epsilon = Objects.requireNonNull(s);
        return this;
    }

    /** @param discount gamma @return this learner */
    public ValueDecomposition discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param o the optimizer @return this learner */
    public ValueDecomposition optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param seed the seed @return this learner */
    public ValueDecomposition seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /**
     * Picks every agent's action from its own observation.
     *
     * @param observations one observation per agent
     * @param explore whether to act epsilon-greedily
     * @return the joint action
     */
    public int[] act(float[][] observations, boolean explore) {
        int[] a = new int[agents];
        for (int i = 0; i < agents; i++) {
            if (explore && random.nextDouble() < epsilon.at(steps)) {
                a[i] = random.nextInt(actions);
            } else {
                a[i] = RlTensors.argmax(RlTensors.row(q, observations[i]));
            }
        }
        return a;
    }

    /**
     * Trains for a number of joint steps.
     *
     * @param env the environment
     * @param totalSteps joint environment steps
     * @return the mean team return of the last 50 episodes
     */
    public double fit(MultiAgentEnvironment env, long totalSteps) {
        EpisodeRecorder rec = new EpisodeRecorder(null);
        float[][] obs = env.reset(random.split());
        float[] state = env.state();
        while (rec.steps() < totalSteps) {
            int[] a = act(obs, true);
            MultiAgentEnvironment.Step s = env.step(a);
            steps++;
            float[] nextState = env.state();
            Sample sample = new Sample(obs, state, a, s.reward(), s.observations(), nextState, s.terminated());
            if (buffer.size() < bufferSize) {
                buffer.add(sample);
            } else {
                buffer.set(position, sample);
                position = (position + 1) % bufferSize;
            }
            if (rec.step(s.reward(), s.done())) {
                obs = env.reset(random.split());
                state = env.state();
            } else {
                obs = s.observations();
                state = nextState;
            }
            if (buffer.size() >= batchSize) {
                learn();
            }
        }
        return rec.meanReturn(50);
    }

    /** Team value from agent values {@code [b, agents]} and states {@code [b, stateSize]}. */
    private Var mixValues(Var qs, Tensor states, List<Var> params, int b) {
        return switch (mixer) {
            case INDEPENDENT -> qs;
            case VDN -> qs.sum(1, true);
            case QMIX -> {
                Var s = Var.of(states);
                Var w1 = s.matmul(params.get(0)).abs().reshape(b, agents, embed);
                Var b1 = s.matmul(params.get(1));
                Var hidden = qs.reshape(b, agents, 1).mul(w1).sum(1, false).add(b1);
                Var elu = Var.where(hidden.value().greater(Tensor.full(Shape.of(1, 1), 0f)), hidden,
                        hidden.minimum(0f).exp().sub(1f));
                Var w2 = s.matmul(params.get(2)).abs();
                Var v = s.matmul(params.get(3)).relu().matmul(params.get(4));
                yield elu.mul(w2).sum(1, true).add(v);
            }
        };
    }

    private void learn() {
        int b = batchSize;
        float[] o = new float[b * agents * obsSize];
        float[] n = new float[b * agents * obsSize];
        float[] st = new float[b * stateSize];
        float[] nst = new float[b * stateSize];
        float[] hot = new float[b * agents * actions];
        float[] r = new float[b];
        float[] notDone = new float[b];
        for (int i = 0; i < b; i++) {
            Sample s = buffer.get(random.nextInt(buffer.size()));
            for (int k = 0; k < agents; k++) {
                System.arraycopy(s.obs()[k], 0, o, (i * agents + k) * obsSize, obsSize);
                System.arraycopy(s.next()[k], 0, n, (i * agents + k) * obsSize, obsSize);
                hot[(i * agents + k) * actions + s.actions()[k]] = 1;
            }
            System.arraycopy(s.state(), 0, st, i * stateSize, stateSize);
            System.arraycopy(s.nextState(), 0, nst, i * stateSize, stateSize);
            r[i] = s.reward();
            notDone[i] = s.terminated() ? 0 : 1;
        }
        Tensor nextObs = Tensor.of(Shape.of(b * agents, obsSize), n);
        float[] pick = q.forward(Var.of(nextObs), false).value().argmax(1).to(DType.FLOAT32).toArray();
        Tensor nextQ = qTarget.forward(Var.of(nextObs), false).value()
                .mul(RlTensors.oneHot(pick, actions, q.device())).sum(1, true).reshape(b, agents);
        Tensor nextTot = mixValues(Var.of(nextQ), Tensor.of(Shape.of(b, stateSize), nst), mixTarget, b).value();
        Tensor y = Tensor.of(Shape.of(b, 1), r).add(Tensor.of(Shape.of(b, 1), notDone).mul(gamma).mul(nextTot));
        Var chosen = q.forward(Var.of(Tensor.of(Shape.of(b * agents, obsSize), o)), true)
                .mul(Tensor.of(Shape.of(b * agents, actions), hot)).sum(1, true).reshape(b, agents);
        Var tot = mixValues(chosen, Tensor.of(Shape.of(b, stateSize), st), mix, b);
        Var loss = tot.sub(y).square().mean();
        loss.backward();
        List<Var> params = new ArrayList<>(q.parameters());
        params.addAll(mix);
        RlTensors.clip(params, 10f);
        optimizer.step(params);
        if (++updates % targetEvery == 0) {
            RlTensors.copy(q.parameters(), qTarget.parameters());
            if (!mix.isEmpty()) {
                RlTensors.copy(mix, mixTarget);
            }
        }
    }

    /**
     * Plays one greedy episode.
     *
     * @param env the environment
     * @param seed the episode seed
     * @return the team return
     */
    public double evaluate(MultiAgentEnvironment env, long seed) {
        float[][] obs = env.reset(new SplittableRandom(seed));
        double total = 0;
        MultiAgentEnvironment.Step s;
        do {
            s = env.step(act(obs, false));
            total += s.reward();
            obs = s.observations();
        } while (!s.done());
        return total;
    }

    @Override
    public String toString() {
        return mixer.toString();
    }
}
