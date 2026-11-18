package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.GoalEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.HindsightReplay;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Goal-conditioned hierarchical RL on a {@link GoalEnvironment} whose achieved goal is the
 * agent's position, with two TD3 levels ({@link StepwiseTd3}):
 *
 * <ul>
 *   <li>{@link Mode#HIRO} (Nachum et al., NeurIPS 2018,
 *       <a href="https://arxiv.org/abs/1805.08296">arXiv:1805.08296</a>): every {@code c} steps
 *       the high level, which sees the observation and the task goal, proposes a relative
 *       subgoal {@code g}; the low level sees its position and the subgoal still to go
 *       ({@code g_{t+1} = s_t + g_t - s_{t+1}}) and is rewarded {@code -||s_t + g_t - s_{t+1}||}.
 *       The high level learns from {@code (obs, g, summed reward, obs after c steps)}; because
 *       the low level keeps changing, those stored subgoals go stale, so HIRO's off-policy
 *       correction periodically replaces each stored subgoal by the candidate (the original,
 *       the achieved offset and eight Gaussian perturbations of it) under which the current low
 *       policy best explains the actions actually taken;</li>
 *   <li>{@link Mode#HIRO_NO_CORRECTION}: the same without the relabelling;</li>
 *   <li>{@link Mode#HAC} (Levy et al., ICLR 2019,
 *       <a href="https://arxiv.org/abs/1712.00948">arXiv:1712.00948</a>): hindsight action
 *       transitions (the high level's action replaced by the offset the low level actually
 *       achieved, so the high level learns as if the low level were already optimal),
 *       hindsight goal transitions at both levels, and subgoal testing (with probability 0.3 a
 *       subgoal is pursued without noise and penalised with {@code -c} if missed). This keeps
 *       HIRO's relative subgoals and dense low-level reward rather than HAC's absolute,
 *       sparse ones;</li>
 *   <li>{@link Mode#FLAT_HER}: one TD3 on the observation with hindsight goal relabelling
 *       ({@link HindsightReplay#relabel}), the flat baseline.</li>
 * </ul>
 */
public final class Hiro implements Learner {

    /** The variant. */
    public enum Mode {
        /** two levels with off-policy subgoal correction */
        HIRO,
        /** two levels without correction */
        HIRO_NO_CORRECTION,
        /** hierarchical actor-critic style hindsight at both levels */
        HAC,
        /** flat TD3 with hindsight experience replay */
        FLAT_HER
    }

    private record Segment(float[][] states, float[][] actions) {
    }

    private final Mode mode;
    private final GoalEnvironment env;
    private final int obsDims;
    private final StepwiseTd3 high;
    private final StepwiseTd3 low;
    private final List<Segment> segments = new ArrayList<>();
    private final int c;
    private float range = 0.5f;
    private int correctEvery = 500;
    private final SplittableRandom random;
    private float[] subgoal;
    private int sinceHigh;

    private Hiro(Mode mode, GoalEnvironment env, RlNetworkFactory nets, int c, long seed) {
        this.mode = mode;
        this.env = env;
        this.c = c;
        this.obsDims = env.observationSpace().size();
        this.random = new SplittableRandom(seed);
        if (mode == Mode.FLAT_HER) {
            high = null;
            low = StepwiseTd3.of(obsDims, 2, nets, 1e-3f, seed).discount(0.95f);
        } else {
            high = StepwiseTd3.of(obsDims, 2, nets, 1e-3f, seed).discount(0.95f).explorationNoise(0.3f);
            low = StepwiseTd3.of(4, 2, nets, 1e-3f, seed + 100).discount(0.9f);
        }
    }

    /**
     * @param mode the variant
     * @param env the goal environment; its achieved goal must be a 2-D position and its actions
     *     a 2-D box
     * @param networks builds every actor and critic
     * @param interval the high level's period {@code c}
     * @param seed the seed
     * @return the learner
     */
    public static Hiro of(Mode mode, GoalEnvironment env, RlNetworkFactory networks, int interval, long seed) {
        return new Hiro(Objects.requireNonNull(mode), env, networks, interval, seed);
    }

    /** @param r the largest subgoal offset per axis @return this learner */
    public Hiro subgoalRange(float r) {
        this.range = r;
        return this;
    }

    /** @return gradient updates over both levels */
    public long updates() {
        return low.updates() + (high == null ? 0 : high.updates());
    }

    private float[] lowObs(float[] state, float[] goal) {
        return new float[] {state[0], state[1], goal[0] / range, goal[1] / range};
    }

    @Override
    public void startEpisode() {
        sinceHigh = 0;
        subgoal = null;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        Space.Box box = (Space.Box) env.actionSpace();
        if (mode == Mode.FLAT_HER) {
            return StepwiseTd3.scale(box, low.unitAction(observation, explore));
        }
        float[] s = env.achievedGoal(observation);
        if (subgoal == null || sinceHigh % c == 0) {
            float[] u = high.unitAction(observation, explore);
            subgoal = new float[] {s[0] + u[0] * range, s[1] + u[1] * range};
        }
        sinceHigh++;
        float[] togo = {subgoal[0] - s[0], subgoal[1] - s[1]};
        return StepwiseTd3.scale(box, low.unitAction(lowObs(s, togo), explore));
    }

    @Override
    public RlResult fit(Environment e, long steps, RlListener listener) {
        Space.Box box = (Space.Box) e.actionSpace();
        Tracker t = new Tracker(listener);
        long taken = 0;
        while (t.steps() < steps) {
            float[] obs = e.reset(random.split());
            startEpisode();
            List<Transition> flat = new ArrayList<>();
            boolean done = false;
            while (!done && t.steps() < steps) {
                float[] hiObs = obs;
                float[] s0 = env.achievedGoal(obs);
                float[] u = mode == Mode.FLAT_HER ? null : high.unitAction(obs, true);
                boolean testing = mode == Mode.HAC && random.nextDouble() < 0.3;
                if (testing) {
                    u = high.unitAction(obs, false);
                }
                float[] g = u == null ? null : new float[] {u[0] * range, u[1] * range};
                float[] target = g == null ? null : new float[] {s0[0] + g[0], s0[1] + g[1]};
                float total = 0;
                List<float[]> states = new ArrayList<>();
                List<float[]> acts = new ArrayList<>();
                List<float[]> lowObsList = new ArrayList<>();
                int k = 0;
                while (k < c && !done && t.steps() < steps) {
                    float[] s = env.achievedGoal(obs);
                    float[] a;
                    float[] lo = null;
                    if (mode == Mode.FLAT_HER) {
                        a = low.buffer().size() < 200 ? randomAction() : low.unitAction(obs, true);
                    } else {
                        float[] togo = {target[0] - s[0], target[1] - s[1]};
                        lo = lowObs(s, togo);
                        a = low.buffer().size() < 200 ? randomAction() : low.unitAction(lo, !testing);
                    }
                    Environment.Step st = e.step(StepwiseTd3.scale(box, a));
                    taken++;
                    total += st.reward();
                    done = t.step(st);
                    float[] s1 = env.achievedGoal(st.observation());
                    if (mode == Mode.FLAT_HER) {
                        low.remember(obs, a, st.reward(), st.observation(), st.terminated());
                        flat.add(new Transition(obs, a, st.reward(), st.observation(), st.terminated(), 1));
                    } else {
                        float[] togo1 = {target[0] - s1[0], target[1] - s1[1]};
                        float lr = (float) -Math.hypot(togo1[0], togo1[1]);
                        low.remember(lo, a, lr, lowObs(s1, togo1), false);
                        states.add(s);
                        acts.add(a);
                        lowObsList.add(lo);
                    }
                    obs = st.observation();
                    k++;
                    if (taken > 200) {
                        low.update();
                        if (high != null && taken % 2 == 0) {
                            high.update();
                        }
                    }
                }
                if (mode != Mode.FLAT_HER) {
                    float[] s1 = env.achievedGoal(obs);
                    states.add(s1);
                    storeHigh(hiObs, u, total, obs, done && isTerminal(obs), states, acts, target, testing,
                            lowObsList);
                }
            }
            if (mode == Mode.FLAT_HER) {
                for (Transition tr : HindsightReplay.relabel(env, flat, 4, random)) {
                    low.remember(tr.observation(), tr.action(), tr.reward(), tr.nextObservation(), tr.terminated());
                }
            }
            if (mode == Mode.HIRO && taken / correctEvery != (taken - c) / correctEvery) {
                correct();
            }
        }
        return t.result();
    }

    private boolean isTerminal(float[] obs) {
        int g = env.goalSize();
        float[] desired = java.util.Arrays.copyOfRange(obs, obs.length - g, obs.length);
        return env.success(env.achievedGoal(obs), desired);
    }

    private float[] randomAction() {
        return new float[] {(float) random.nextDouble(-1, 1), (float) random.nextDouble(-1, 1)};
    }

    private void storeHigh(float[] hiObs, float[] u, float total, float[] next, boolean terminal, List<float[]> states,
            List<float[]> acts, float[] target, boolean testing, List<float[]> lowObs) {
        float[] s0 = states.get(0);
        float[] s1 = states.get(states.size() - 1);
        high.remember(hiObs, u, total, next, terminal);
        segments.add(new Segment(states.toArray(new float[0][]), acts.toArray(new float[0][])));
        if (mode != Mode.HAC) {
            return;
        }
        // hindsight action: the offset actually achieved
        float[] achievedU = {clip((s1[0] - s0[0]) / range), clip((s1[1] - s0[1]) / range)};
        high.remember(hiObs, achievedU, total, next, terminal);
        segments.add(null);
        // hindsight goal: pretend the task goal was where the agent ended up
        int g = env.goalSize();
        float[] relObs = hiObs.clone();
        float[] relNext = next.clone();
        System.arraycopy(s1, 0, relObs, relObs.length - g, g);
        System.arraycopy(s1, 0, relNext, relNext.length - g, g);
        high.remember(relObs, achievedU, 0, relNext, true);
        segments.add(null);
        // subgoal testing penalty
        if (testing && Math.hypot(s1[0] - target[0], s1[1] - target[1]) > 0.1) {
            high.remember(hiObs, u, -c, next, false);
            segments.add(null);
        }
        // low-level hindsight: the reached position as the subgoal of the whole segment
        for (int i = 0; i < acts.size(); i++) {
            float[] s = states.get(i);
            float[] sn = states.get(i + 1);
            float[] togo = {s1[0] - s[0], s1[1] - s[1]};
            float[] togo1 = {s1[0] - sn[0], s1[1] - sn[1]};
            low.remember(lowObs(s, togo), acts.get(i), (float) -Math.hypot(togo1[0], togo1[1]), lowObs(sn, togo1),
                    false);
        }
    }

    private static float clip(float v) {
        return Math.max(-1, Math.min(1, v));
    }

    /** HIRO's off-policy correction over every stored high-level transition. */
    private void correct() {
        List<StepwiseTd3.Entry> buf = high.buffer();
        for (int i = 0; i < buf.size() && i < segments.size(); i++) {
            Segment seg = segments.get(i);
            if (seg == null || seg.actions().length == 0) {
                continue;
            }
            float[] s0 = seg.states()[0];
            float[] sEnd = seg.states()[seg.states().length - 1];
            List<float[]> candidates = new ArrayList<>();
            candidates.add(buf.get(i).action());
            float[] achieved = {clip((sEnd[0] - s0[0]) / range), clip((sEnd[1] - s0[1]) / range)};
            candidates.add(achieved);
            for (int k = 0; k < 8; k++) {
                candidates.add(new float[] {clip(achieved[0] + 0.5f * (float) random.nextGaussian()),
                    clip(achieved[1] + 0.5f * (float) random.nextGaussian())});
            }
            int n = seg.actions().length;
            float[] rows = new float[candidates.size() * n * 4];
            for (int cidx = 0; cidx < candidates.size(); cidx++) {
                float[] u = candidates.get(cidx);
                float[] target = {s0[0] + u[0] * range, s0[1] + u[1] * range};
                for (int j = 0; j < n; j++) {
                    float[] s = seg.states()[j];
                    float[] lo = lowObs(s, new float[] {target[0] - s[0], target[1] - s[1]});
                    System.arraycopy(lo, 0, rows, (cidx * n + j) * 4, 4);
                }
            }
            float[] mu = low.actor().forward(Var.of(Tensor.of(Shape.of(candidates.size() * n, 4), rows)), false)
                    .value().tanh().toArray();
            int best = 0;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int cidx = 0; cidx < candidates.size(); cidx++) {
                double score = 0;
                for (int j = 0; j < n; j++) {
                    float[] a = seg.actions()[j];
                    for (int d = 0; d < 2; d++) {
                        double diff = a[d] - mu[(cidx * n + j) * 2 + d];
                        score -= diff * diff;
                    }
                }
                if (score > bestScore) {
                    bestScore = score;
                    best = cidx;
                }
            }
            StepwiseTd3.Entry e = buf.get(i);
            buf.set(i, new StepwiseTd3.Entry(e.observation(), candidates.get(best), e.reward(), e.next(), e.terminal()));
        }
    }

    @Override
    public String toString() {
        return mode.toString();
    }
}
