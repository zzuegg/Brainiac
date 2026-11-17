package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.ReplayBuffer;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Hindsight experience replay (Andrychowicz et al., 2017,
 * <a href="https://arxiv.org/abs/1707.01495">arXiv:1707.01495</a>) with a goal-conditioned DQN.
 *
 * <p>In a sparse-reward goal task, a failed episode teaches nothing: every reward is
 * {@code -1}. HER replays each transition a second (third, ...) time with the goal
 * replaced by one the agent actually reached later in the same episode (the "future"
 * strategy), so those copies end in success and the agent learns how to reach <i>some</i>
 * goals, which generalizes to the goals it is asked for. {@link #relabel} is the reusable
 * part and works for any off-policy learner over a {@link GoalEnvironment}.
 *
 * <pre>{@code
 * HindsightReplay her = HindsightReplay.dqn(() -> Nets.mlp(16, 8, Activation.RELU, 1, 128), 8).relabelCount(4);
 * her.fit(new BitFlip(8), 20_000);
 * }</pre>
 */
public final class HindsightReplay implements Learner {

    private final Model online;
    private final Model target;
    private final int actions;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private int relabel = 4;
    private int updatesPerEpisode = 8;
    private int batchSize = 64;
    private int bufferSize = 50_000;
    private int targetEvery = 50;
    private float gamma = 0.98f;
    private Schedule epsilon = Schedule.linear(0.5, 0.05, 2_000);
    private SplittableRandom random = new SplittableRandom(0);
    private ReplayBuffer buffer;
    private long steps;
    private long updates;

    private HindsightReplay(Supplier<Model> network, int actions) {
        this.online = network.get();
        this.target = network.get();
        this.actions = actions;
        RlTensors.copy(online.parameters(), target.parameters());
    }

    /**
     * @param network makes the goal-conditioned Q-network ({@code state + goal -> actions}), called twice
     * @param actions discrete actions
     * @return the learner, relabelling 4 future goals per transition
     */
    public static HindsightReplay dqn(Supplier<Model> network, int actions) {
        return new HindsightReplay(Objects.requireNonNull(network), actions);
    }

    /** @param k hindsight copies per transition; 0 turns HER off @return this learner */
    public HindsightReplay relabelCount(int k) {
        this.relabel = k;
        return this;
    }

    /** @param n gradient steps after every episode @return this learner */
    public HindsightReplay updatesPerEpisode(int n) {
        this.updatesPerEpisode = n;
        return this;
    }

    /** @param s exploration by step @return this learner */
    public HindsightReplay epsilon(Schedule s) {
        this.epsilon = Objects.requireNonNull(s);
        return this;
    }

    /** @param discount gamma @return this learner */
    public HindsightReplay discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param seed the seed @return this learner */
    public HindsightReplay seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /**
     * Relabels an episode with achieved goals ("future" strategy).
     *
     * @param env the goal environment (for goal size and rewards)
     * @param episode the episode's transitions, in order
     * @param k how many relabelled copies per transition
     * @param random picks the future steps
     * @return the relabelled transitions (not including the originals)
     */
    public static List<Transition> relabel(GoalEnvironment env, List<Transition> episode, int k,
            SplittableRandom random) {
        List<Transition> out = new ArrayList<>();
        int g = env.goalSize();
        for (int t = 0; t < episode.size(); t++) {
            Transition tr = episode.get(t);
            for (int c = 0; c < k; c++) {
                int future = t + random.nextInt(episode.size() - t);
                float[] goal = env.achievedGoal(episode.get(future).nextObservation());
                float[] achieved = env.achievedGoal(tr.nextObservation());
                out.add(new Transition(withGoal(tr.observation(), goal, g), tr.action(), env.reward(achieved, goal),
                        withGoal(tr.nextObservation(), goal, g), env.success(achieved, goal), tr.discount()));
            }
        }
        return out;
    }

    private static float[] withGoal(float[] obs, float[] goal, int g) {
        float[] o = obs.clone();
        System.arraycopy(goal, 0, o, o.length - g, g);
        return o;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        if (explore && random.nextDouble() < epsilon.at(steps)) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {RlTensors.argmax(RlTensors.row(online, observation))};
    }

    @Override
    public RlResult fit(Environment env, long totalSteps, RlListener listener) {
        if (!(env instanceof GoalEnvironment goals) || !(env.actionSpace() instanceof Space.Discrete d)
                || d.n() != actions) {
            throw new IllegalArgumentException("needs a goal environment with " + actions + " discrete actions");
        }
        if (buffer == null) {
            buffer = new ReplayBuffer(bufferSize, env.observationSpace().size(), 1);
        }
        Tracker recorder = new Tracker(listener);
        List<Transition> episode = new ArrayList<>();
        float[] obs = env.reset(random.split());
        while (recorder.steps() < totalSteps) {
            float[] a = act(obs, true);
            Environment.Step s = env.step(a);
            steps++;
            Transition tr = new Transition(obs, a, s.reward(), s.observation(), s.terminated(), gamma);
            episode.add(tr);
            buffer.add(tr);
            if (recorder.step(s)) {
                if (relabel > 0) {
                    relabel(goals, episode, relabel, random).forEach(buffer::add);
                }
                episode.clear();
                for (int u = 0; u < updatesPerEpisode && buffer.size() >= batchSize; u++) {
                    learn();
                }
                obs = env.reset(random.split());
            } else {
                obs = s.observation();
            }
        }
        return recorder.result();
    }

    private void learn() {
        Device device = online.device();
        ReplayBuffer.Batch b = buffer.sample(batchSize, random, device);
        float[] pick = online.forward(Var.of(b.nextObservations()), false).value().argmax(1)
                .to(DType.FLOAT32).toArray();
        Tensor next = target.forward(Var.of(b.nextObservations()), false).value()
                .mul(RlTensors.oneHot(pick, actions, device)).sum(1, true);
        Tensor y = b.rewards().add(b.discounts().mul(b.notDone()).mul(next));
        Var q = online.forward(Var.of(b.observations()), true)
                .mul(RlTensors.oneHot(b.actions().toArray(), actions, device)).sum(1, true);
        Var loss = RlTensors.huber(q.sub(y), 1f).mean();
        loss.backward();
        optimizer.step(online.parameters());
        if (++updates % targetEvery == 0) {
            RlTensors.copy(online.parameters(), target.parameters());
        }
    }
}
