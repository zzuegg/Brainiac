package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Schedule;
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
 * R2D2, recurrent experience replay in distributed RL (Kapturowski et al., 2019,
 * <a href="https://openreview.net/forum?id=r1lyTjAqYX">ICLR 2019</a>), single-process: a
 * double DQN whose Q-network has a GRU core, so it can act on partially observable tasks.
 * Whole episodes are replayed as fixed-length sequences, with the recurrent state the actor
 * had at the time stored next to every step. A training sequence starts from that stored
 * state, runs a {@link #burnIn burn-in} prefix through the current network without
 * gradients to refresh the stale state, and then learns on the rest by back-propagation
 * through time. {@link #storedState(boolean) storedState(false)} with no burn-in gives the
 * zero-state replay the paper improves on.
 *
 * <pre>{@code
 * RecurrentReplayDqn r2d2 = RecurrentReplayDqn.of(4, 3, 32, 1).burnIn(4).trainLength(6);
 * r2d2.fit(new CueCorridor(6), 6_000);
 * }</pre>
 */
public final class RecurrentReplayDqn implements Learner {

    private record Record(float[][] observations, int[] actions, float[] rewards, boolean terminated,
            float[][] states) {
        int length() {
            return actions.length;
        }
    }

    private final int obsSize;
    private final int actions;
    private final RlGruCell cell;
    private final RlGruCell targetCell;
    private final Model head;
    private final Model targetHead;
    private Optimizer optimizer = Optimizer.adam(2e-3f);
    private int burnIn = 4;
    private int trainLength = 8;
    private int batchSize = 16;
    private int capacity = 2000;
    private int learningStarts = 300;
    private int trainEvery = 2;
    private int targetEvery = 100;
    private boolean storedState = true;
    private float gamma = 0.97f;
    private Schedule epsilon = Schedule.linear(1, 0.05, 3000);
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Record> buffer = new ArrayList<>();
    private float[] state;
    private long steps;
    private long updates;
    private double lastLoss = Double.NaN;

    private RecurrentReplayDqn(int obsSize, int actions, int units, long seed) {
        this.obsSize = obsSize;
        this.actions = actions;
        this.cell = new RlGruCell(obsSize, units, seed);
        this.targetCell = new RlGruCell(obsSize, units, seed);
        this.head = Model.sequential().input(units).dense(actions).seed(seed + 1).build();
        this.targetHead = Model.sequential().input(units).dense(actions).seed(seed + 1).build();
        this.state = new float[units];
        syncTarget();
    }

    /**
     * @param observations values per observation
     * @param actions the number of discrete actions
     * @param units the GRU state size
     * @param seed the seed
     * @return the learner: stored state, burn-in 4, sequences of 8
     */
    public static RecurrentReplayDqn of(int observations, int actions, int units, long seed) {
        return new RecurrentReplayDqn(observations, actions, units, seed);
    }

    /** @param steps the burn-in prefix length @return this learner */
    public RecurrentReplayDqn burnIn(int steps) {
        this.burnIn = Math.max(0, steps);
        return this;
    }

    /** @param steps the trained part of every sequence @return this learner */
    public RecurrentReplayDqn trainLength(int steps) {
        this.trainLength = steps;
        return this;
    }

    /**
     * @param on start sequences from the actor's stored state (true) or from zeros (false)
     * @return this learner
     */
    public RecurrentReplayDqn storedState(boolean on) {
        this.storedState = on;
        return this;
    }

    /** @param size sequences per update @return this learner */
    public RecurrentReplayDqn batchSize(int size) {
        this.batchSize = size;
        return this;
    }

    /** @param steps random-ish steps before learning @return this learner */
    public RecurrentReplayDqn learningStarts(int steps) {
        this.learningStarts = steps;
        return this;
    }

    /** @param s the exploration rate over steps @return this learner */
    public RecurrentReplayDqn epsilon(Schedule s) {
        this.epsilon = Objects.requireNonNull(s);
        return this;
    }

    /** @param discount the discount @return this learner */
    public RecurrentReplayDqn discount(float discount) {
        this.gamma = discount;
        return this;
    }

    /** @param o the optimizer @return this learner */
    public RecurrentReplayDqn optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @param seed the seed @return this learner */
    public RecurrentReplayDqn seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    /** @return the loss of the last update */
    public double lastLoss() {
        return lastLoss;
    }

    /** @return the updates made */
    public long updates() {
        return updates;
    }

    private void syncTarget() {
        RlTensors.copy(cell.parameters(), targetCell.parameters());
        RlTensors.copy(head.parameters(), targetHead.parameters());
    }

    private List<Var> parameters() {
        List<Var> p = new ArrayList<>(cell.parameters());
        p.addAll(head.parameters());
        return p;
    }

    @Override
    public void startEpisode() {
        state = new float[cell.units()];
    }

    /**
     * @param observation the next observation; advances the recurrent state
     * @return the action values
     */
    public float[] actionValues(float[] observation) {
        Var h = cell.step(Var.of(AgentKit.row(observation)), Var.of(AgentKit.row(state)));
        state = AgentKit.values(h);
        return AgentKit.values(head.forward(Var.of(h.value()), false));
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        float[] q = actionValues(observation);
        if (explore && random.nextDouble() < epsilon.at(steps)) {
            return new float[] {random.nextInt(actions)};
        }
        return new float[] {AgentKit.argmax(q)};
    }

    @Override
    public RlResult fit(Environment env, long totalSteps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("needs " + actions + " discrete actions");
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        startEpisode();
        List<float[]> os = new ArrayList<>();
        List<float[]> hs = new ArrayList<>();
        List<Integer> as = new ArrayList<>();
        List<Float> rs = new ArrayList<>();
        os.add(obs);
        while (tracker.steps() < totalSteps) {
            hs.add(state.clone());
            int a = Math.round(act(obs, true)[0]);
            Environment.Step s = env.step(a);
            steps++;
            as.add(a);
            rs.add(s.reward());
            os.add(s.observation());
            if (tracker.step(s)) {
                store(os, hs, as, rs, s.terminated());
                obs = env.reset(random.split());
                startEpisode();
                os = new ArrayList<>();
                hs = new ArrayList<>();
                as = new ArrayList<>();
                rs = new ArrayList<>();
                os.add(obs);
            } else {
                obs = s.observation();
            }
            if (steps >= learningStarts && steps % trainEvery == 0 && !buffer.isEmpty()) {
                learn();
            }
        }
        return tracker.result();
    }

    private void store(List<float[]> os, List<float[]> hs, List<Integer> as, List<Float> rs, boolean terminated) {
        int[] a = new int[as.size()];
        float[] r = new float[rs.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = as.get(i);
            r[i] = rs.get(i);
        }
        buffer.add(new Record(os.toArray(new float[0][]), a, r, terminated, hs.toArray(new float[0][])));
        if (buffer.size() > capacity) {
            buffer.remove(0);
        }
    }

    private void learn() {
        int b = batchSize;
        int units = cell.units();
        Record[] eps = new Record[b];
        int[] start = new int[b];
        int[] from = new int[b];
        float[] h0 = new float[b * units];
        for (int i = 0; i < b; i++) {
            Record e = buffer.get(random.nextInt(buffer.size()));
            eps[i] = e;
            start[i] = random.nextInt(e.length());
            from[i] = Math.max(0, start[i] - burnIn);
            if (storedState) {
                System.arraycopy(e.states()[from[i]], 0, h0, i * units, units);
            }
        }
        Tensor h0t = Tensor.of(Shape.of(b, units), h0);
        Var[] online = unroll(cell, head, eps, start, from, h0t, true);
        Var[] target = unroll(targetCell, targetHead, eps, start, from, h0t, false);
        Var loss = null;
        float count = 0;
        for (int t = 0; t < trainLength; t++) {
            float[] y = new float[b];
            float[] mask = new float[b];
            float[] act = new float[b];
            float[] qNextOnline = online[t + 1].value().toArray();
            float[] qNextTarget = target[t + 1].value().toArray();
            for (int i = 0; i < b; i++) {
                int k = start[i] + t;
                if (k >= eps[i].length()) {
                    continue;
                }
                mask[i] = 1;
                count++;
                act[i] = eps[i].actions()[k];
                boolean terminal = k == eps[i].length() - 1 && eps[i].terminated();
                int best = 0;
                for (int a = 1; a < actions; a++) {
                    if (qNextOnline[i * actions + a] > qNextOnline[i * actions + best]) {
                        best = a;
                    }
                }
                y[i] = eps[i].rewards()[k] + (terminal ? 0 : gamma * qNextTarget[i * actions + best]);
            }
            Var q = online[t].mul(RlTensors.oneHot(act, actions, online[t].value().device())).sum(1, true);
            Var term = q.sub(AgentKit.column(y)).square().mul(AgentKit.column(mask)).sum();
            loss = loss == null ? term : loss.add(term);
        }
        loss = loss.div(Math.max(1, count));
        lastLoss = loss.value().item();
        loss.backward();
        List<Var> params = parameters();
        RlTensors.clip(params, 10f);
        optimizer.step(params);
        updates++;
        if (updates % targetEvery == 0) {
            syncTarget();
        }
    }

    /** Runs burn-in then {@code trainLength + 1} steps; returns the Q-values of the latter. */
    private Var[] unroll(RlGruCell c, Model q, Record[] eps, int[] start, int[] from, Tensor h0, boolean grad) {
        int b = eps.length;
        Var h = Var.of(h0);
        for (int t = -burnIn; t < 0; t++) {
            float[] x = new float[b * obsSize];
            float[] m = new float[b];
            for (int i = 0; i < b; i++) {
                int k = start[i] + t;
                if (k >= from[i]) {
                    System.arraycopy(eps[i].observations()[k], 0, x, i * obsSize, obsSize);
                    m[i] = 1;
                }
            }
            Var next = c.step(Var.of(Tensor.of(Shape.of(b, obsSize), x)), h);
            h = Var.of(h.value().add(next.value().sub(h.value()).mul(AgentKit.column(m))));
        }
        Var[] out = new Var[trainLength + 1];
        for (int t = 0; t <= trainLength; t++) {
            float[] x = new float[b * obsSize];
            for (int i = 0; i < b; i++) {
                int k = Math.min(start[i] + t, eps[i].observations().length - 1);
                System.arraycopy(eps[i].observations()[k], 0, x, i * obsSize, obsSize);
            }
            h = c.step(Var.of(Tensor.of(Shape.of(b, obsSize), x)), h);
            if (!grad) {
                h = Var.of(h.value());
            }
            out[t] = q.forward(h, grad);
            if (!grad) {
                out[t] = Var.of(out[t].value());
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "R2D2(" + (storedState ? "stored state" : "zero state") + ", burn-in " + burnIn + ")";
    }
}
