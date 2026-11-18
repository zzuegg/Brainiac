package io.github.zzuegg.brainiac.experimental.explore;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.SplittableRandom;

/**
 * Planning with a learned tabular model, as a core {@link Learner} for one-hot-state
 * environments:
 *
 * <ul>
 *   <li>{@link Method#Q_LEARNING}: one-step Q-learning, no planning (the {@code n = 0} baseline);</li>
 *   <li>{@link Method#DYNA_Q}: Dyna-Q (Sutton, ICML 1990): after each real step, {@code n}
 *       simulated Q-updates on previously seen state-action pairs drawn from a deterministic
 *       learned model;</li>
 *   <li>{@link Method#DYNA_Q_PLUS}: Dyna-Q+: simulated rewards get the bonus
 *       {@code kappa sqrt(tau)} for pairs not tried for {@code tau} steps, and untried actions of
 *       seen states are modelled as self-loops, so the agent re-checks parts of the world that
 *       may have changed;</li>
 *   <li>{@link Method#PRIORITIZED_SWEEPING}: prioritised sweeping (Moore &amp; Atkeson, Machine
 *       Learning 1993; Peng &amp; Williams 1993): simulated updates are ordered by the size of
 *       the predicted value change in a priority queue, and predecessors of an updated state are
 *       queued, so new information propagates backwards from where it arrived.</li>
 * </ul>
 *
 * <p>{@link Maze} provides Sutton &amp; Barto's Dyna maze and the blocking and shortcut mazes,
 * whose walls change mid-run.
 */
public final class DynaPlanning implements Learner {

    /** The planning scheme. */
    public enum Method {
        /** no planning */
        Q_LEARNING,
        /** uniform replay of the learned model */
        DYNA_Q,
        /** Dyna-Q with an exploration bonus for stale pairs */
        DYNA_Q_PLUS,
        /** replay ordered by predicted value change */
        PRIORITIZED_SWEEPING
    }

    private final Method method;
    private final int states;
    private final int actions;
    private final int planningSteps;
    private final SplittableRandom random;
    private final double[] q;
    private final int[] modelNext;
    private final double[] modelReward;
    private final boolean[] modelTerminal;
    private final boolean[] known;
    private final long[] lastTried;
    private final List<List<Integer>> predecessors;
    private final List<Integer> seenPairs = new ArrayList<>();
    private final boolean[] seenState;
    private double alpha = 0.1;
    private double gamma = 0.95;
    private double epsilon = 0.1;
    private double kappa = 1e-3;
    private double threshold = 1e-4;
    private long time;
    private long simulatedUpdates;

    private DynaPlanning(Method method, int states, int actions, int planningSteps, long seed) {
        this.method = Objects.requireNonNull(method, "method");
        this.states = states;
        this.actions = actions;
        this.planningSteps = method == Method.Q_LEARNING ? 0 : planningSteps;
        this.random = new SplittableRandom(seed);
        int pairs = states * actions;
        q = new double[pairs];
        modelNext = new int[pairs];
        modelReward = new double[pairs];
        modelTerminal = new boolean[pairs];
        known = new boolean[pairs];
        lastTried = new long[pairs];
        seenState = new boolean[states];
        predecessors = new ArrayList<>();
        for (int s = 0; s < states; s++) {
            predecessors.add(new ArrayList<>());
        }
    }

    /**
     * @param method the scheme
     * @param states the number of states (one-hot observation length)
     * @param actions the number of actions
     * @param planningSteps simulated updates per real step ({@code n})
     * @param seed the seed
     * @return the learner
     */
    public static DynaPlanning of(Method method, int states, int actions, int planningSteps, long seed) {
        return new DynaPlanning(method, states, actions, planningSteps, seed);
    }

    /** @param a the step size @return this learner */
    public DynaPlanning alpha(double a) {
        this.alpha = a;
        return this;
    }

    /** @param g the discount @return this learner */
    public DynaPlanning discount(double g) {
        this.gamma = g;
        return this;
    }

    /** @param e the dithering rate @return this learner */
    public DynaPlanning epsilon(double e) {
        this.epsilon = e;
        return this;
    }

    /** @param k the Dyna-Q+ bonus weight @return this learner */
    public DynaPlanning kappa(double k) {
        this.kappa = k;
        return this;
    }

    /** @param t the prioritised-sweeping queue threshold @return this learner */
    public DynaPlanning threshold(double t) {
        this.threshold = t;
        return this;
    }

    /** @return simulated (planning) updates so far */
    public long simulatedUpdates() {
        return simulatedUpdates;
    }

    private static int state(float[] observation) {
        int best = 0;
        for (int i = 1; i < observation.length; i++) {
            if (observation[i] > observation[best]) {
                best = i;
            }
        }
        return best;
    }

    private int greedy(int s) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < actions; a++) {
            double v = q[s * actions + a];
            double b = q[s * actions + best];
            if (v > b + 1e-12) {
                best = a;
                ties = 1;
            } else if (Math.abs(v - b) <= 1e-12 && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }

    private double max(int s) {
        double m = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < actions; a++) {
            m = Math.max(m, q[s * actions + a]);
        }
        return m;
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        int s = state(observation);
        return new float[] {explore && random.nextDouble() < epsilon ? random.nextInt(actions) : greedy(s)};
    }

    @Override
    public RlResult fit(Environment env, long steps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions
                || env.observationSpace().size() != states) {
            throw new IllegalArgumentException("needs " + states + " one-hot states and " + actions + " actions");
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        PriorityQueue<double[]> queue = new PriorityQueue<>((x, y) -> Double.compare(y[0], x[0]));
        while (tracker.steps() < steps) {
            int s = state(obs);
            int a = (int) act(obs, true)[0];
            Environment.Step st = env.step(a);
            time++;
            int next = state(st.observation());
            double r = st.reward();
            boolean term = st.terminated();
            int i = s * actions + a;
            double target = r + (term ? 0 : gamma * max(next));
            double tdError = target - q[i];
            q[i] += alpha * tdError;
            remember(s, a, r, next, term);
            switch (method) {
                case DYNA_Q, DYNA_Q_PLUS -> planUniform();
                case PRIORITIZED_SWEEPING -> {
                    double p = Math.abs(r + (term ? 0 : gamma * max(next)) - q[i]);
                    if (p > threshold) {
                        queue.add(new double[] {p, i});
                    }
                    planSweeping(queue);
                }
                default -> {
                }
            }
            if (tracker.step(st)) {
                obs = env.reset(random.split());
            } else {
                obs = st.observation();
            }
        }
        return tracker.result();
    }

    private void remember(int s, int a, double r, int next, boolean term) {
        if (method == Method.DYNA_Q_PLUS && !seenState[s]) {
            for (int b = 0; b < actions; b++) {
                int j = s * actions + b;
                if (!known[j]) {
                    known[j] = true;
                    modelNext[j] = s;
                    modelReward[j] = 0;
                    seenPairs.add(j);
                }
            }
        }
        seenState[s] = true;
        int i = s * actions + a;
        if (!known[i]) {
            known[i] = true;
            seenPairs.add(i);
        }
        if (modelNext[i] != next || !predecessors.get(next).contains(i)) {
            predecessors.get(next).add(i);
        }
        modelNext[i] = next;
        modelReward[i] = r;
        modelTerminal[i] = term;
        lastTried[i] = time;
    }

    private void planUniform() {
        for (int k = 0; k < planningSteps && !seenPairs.isEmpty(); k++) {
            int i = seenPairs.get(random.nextInt(seenPairs.size()));
            double r = modelReward[i];
            if (method == Method.DYNA_Q_PLUS) {
                r += kappa * Math.sqrt(time - lastTried[i]);
            }
            double target = r + (modelTerminal[i] ? 0 : gamma * max(modelNext[i]));
            q[i] += alpha * (target - q[i]);
            simulatedUpdates++;
        }
    }

    private void planSweeping(PriorityQueue<double[]> queue) {
        for (int k = 0; k < planningSteps && !queue.isEmpty(); k++) {
            int i = (int) queue.poll()[1];
            double target = modelReward[i] + (modelTerminal[i] ? 0 : gamma * max(modelNext[i]));
            q[i] += alpha * (target - q[i]);
            simulatedUpdates++;
            int s = i / actions;
            double vs = max(s);
            for (int j : predecessors.get(s)) {
                if (modelNext[j] != s) {
                    continue;
                }
                double p = Math.abs(modelReward[j] + (modelTerminal[j] ? 0 : gamma * vs) - q[j]);
                if (p > threshold) {
                    queue.add(new double[] {p, j});
                }
            }
        }
    }

    @Override
    public String toString() {
        return method.name();
    }

    /**
     * A grid maze with walls that may change after a number of steps (Sutton &amp; Barto,
     * "Reinforcement Learning", 2nd ed., examples 8.1-8.3): one-hot observations, actions 0 up,
     * 1 right, 2 down, 3 left, reward 1 for reaching the goal (which ends the episode).
     */
    public static final class Maze implements Environment {
        private final int rows;
        private final int cols;
        private final int start;
        private final int goal;
        private final boolean[] before;
        private final boolean[] after;
        private final long changeAt;
        private long steps;
        private int pos;

        private Maze(int rows, int cols, int start, int goal, boolean[] before, boolean[] after, long changeAt) {
            this.rows = rows;
            this.cols = cols;
            this.start = start;
            this.goal = goal;
            this.before = before;
            this.after = after;
            this.changeAt = changeAt;
        }

        private static boolean[] walls(int rows, int cols, int[][] cells) {
            boolean[] w = new boolean[rows * cols];
            for (int[] c : cells) {
                w[c[0] * cols + c[1]] = true;
            }
            return w;
        }

        private static int[][] row(int r, int from, int to) {
            int[][] cells = new int[to - from + 1][];
            for (int c = from; c <= to; c++) {
                cells[c - from] = new int[] {r, c};
            }
            return cells;
        }

        /** @return the 6 x 9 Dyna maze (example 8.1) */
        public static Maze dyna() {
            boolean[] w = walls(6, 9, new int[][] {{1, 2}, {2, 2}, {3, 2}, {4, 5}, {0, 7}, {1, 7}, {2, 7}});
            return new Maze(6, 9, 2 * 9, 8, w, w, Long.MAX_VALUE);
        }

        /**
         * @param changeAt the step at which the gap moves from the right to the left
         * @return the blocking maze (example 8.2)
         */
        public static Maze blocking(long changeAt) {
            return new Maze(6, 9, 5 * 9 + 3, 8, walls(6, 9, row(3, 0, 7)), walls(6, 9, row(3, 1, 8)), changeAt);
        }

        /**
         * @param changeAt the step at which a shortcut opens on the right
         * @return the shortcut maze (example 8.3)
         */
        public static Maze shortcut(long changeAt) {
            return new Maze(6, 9, 5 * 9 + 3, 8, walls(6, 9, row(3, 1, 8)), walls(6, 9, row(3, 1, 7)), changeAt);
        }

        /** @return the number of cells */
        public int cells() {
            return rows * cols;
        }

        /** @return steps taken so far (over all episodes) */
        public long steps() {
            return steps;
        }

        @Override
        public Space observationSpace() {
            return Space.box(rows * cols, 0, 1);
        }

        @Override
        public Space actionSpace() {
            return Space.discrete(4);
        }

        private float[] obs() {
            float[] o = new float[rows * cols];
            o[pos] = 1;
            return o;
        }

        @Override
        public float[] reset(SplittableRandom random) {
            pos = start;
            return obs();
        }

        @Override
        public Step step(float[] action) {
            int a = Math.round(action[0]);
            boolean[] w = steps >= changeAt ? after : before;
            steps++;
            int r = pos / cols;
            int c = pos % cols;
            int nr = r + (a == 0 ? -1 : a == 2 ? 1 : 0);
            int nc = c + (a == 1 ? 1 : a == 3 ? -1 : 0);
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && !w[nr * cols + nc]) {
                pos = nr * cols + nc;
            }
            boolean done = pos == goal;
            return new Step(obs(), done ? 1 : 0, done, false);
        }

        @Override
        public String toString() {
            return "Maze" + Arrays.toString(new int[] {rows, cols});
        }
    }
}
