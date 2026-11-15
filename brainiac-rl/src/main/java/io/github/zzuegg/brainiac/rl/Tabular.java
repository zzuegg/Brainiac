package io.github.zzuegg.brainiac.rl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Tabular reinforcement learning: a table of action values for every observation seen,
 * for environments with few distinct observations such as a {@link GridWorld}.
 *
 * <ul>
 *   <li>{@link Method#Q_LEARNING Q-learning} learns the value of acting greedily
 *       (off-policy);</li>
 *   <li>{@link Method#SARSA SARSA} learns the value of the policy it follows, exploration
 *       included (on-policy), so it avoids risky states;</li>
 *   <li>{@link Method#EXPECTED_SARSA expected SARSA} averages over the next action instead
 *       of sampling it.</li>
 * </ul>
 *
 * <pre>{@code
 * Tabular q = Tabular.qLearning(4).learningRate(0.5).epsilon(Schedule.constant(0.1));
 * q.fit(GridWorld.cliff(), 20_000, RlListener.NONE);
 * }</pre>
 */
public final class Tabular implements Learner {

    /** The update rule. */
    public enum Method {
        /** off-policy: bootstrap from the best next action */
        Q_LEARNING,
        /** on-policy: bootstrap from the next action actually taken */
        SARSA,
        /** on-policy: bootstrap from the expected value under the epsilon-greedy policy */
        EXPECTED_SARSA
    }

    private record Key(float[] values) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Key k && Arrays.equals(values, k.values);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(values);
        }
    }

    private final Method method;
    private final int actions;
    private final Map<Key, double[]> table = new HashMap<>();
    private double alpha = 0.1;
    private double gamma = 0.99;
    private double initial;
    private Schedule epsilon = Schedule.constant(0.1);
    private SplittableRandom random = new SplittableRandom(0);
    private long steps;

    private Tabular(Method method, int actions) {
        this.method = Objects.requireNonNull(method, "method");
        if (actions < 1) {
            throw new IllegalArgumentException("actions must be at least 1");
        }
        this.actions = actions;
    }

    /**
     * @param method the update rule
     * @param actions the number of discrete actions
     * @return the learner
     */
    public static Tabular of(Method method, int actions) {
        return new Tabular(method, actions);
    }

    /**
     * @param actions the number of discrete actions
     * @return a Q-learning learner
     */
    public static Tabular qLearning(int actions) {
        return new Tabular(Method.Q_LEARNING, actions);
    }

    /**
     * @param actions the number of discrete actions
     * @return a SARSA learner
     */
    public static Tabular sarsa(int actions) {
        return new Tabular(Method.SARSA, actions);
    }

    /**
     * @param rate how far each update moves a value towards its target (alpha)
     * @return this learner
     */
    public Tabular learningRate(double rate) {
        this.alpha = rate;
        return this;
    }

    /**
     * @param discount how much future rewards count (gamma), in {@code [0, 1]}
     * @return this learner
     */
    public Tabular discount(double discount) {
        this.gamma = discount;
        return this;
    }

    /**
     * @param schedule the chance of a random action, by step
     * @return this learner
     */
    public Tabular epsilon(Schedule schedule) {
        this.epsilon = Objects.requireNonNull(schedule, "schedule");
        return this;
    }

    /**
     * @param value the starting value of every action; optimistic values encourage
     *     exploration
     * @return this learner
     */
    public Tabular initialValue(double value) {
        this.initial = value;
        return this;
    }

    /**
     * @param seed the seed for exploration and episodes
     * @return this learner
     */
    public Tabular seed(long seed) {
        this.random = new SplittableRandom(seed);
        return this;
    }

    private double[] values(float[] observation) {
        return table.computeIfAbsent(new Key(observation.clone()), k -> {
            double[] v = new double[actions];
            Arrays.fill(v, initial);
            return v;
        });
    }

    /**
     * @param observation an observation
     * @return the learned value of every action there (a copy)
     */
    public double[] actionValues(float[] observation) {
        return values(observation).clone();
    }

    /** @return the number of distinct observations seen */
    public int states() {
        return table.size();
    }

    private int greedy(double[] q) {
        int best = 0;
        int ties = 1;
        for (int a = 1; a < q.length; a++) {
            if (q[a] > q[best]) {
                best = a;
                ties = 1;
            } else if (q[a] == q[best] && random.nextInt(++ties) == 0) {
                best = a;
            }
        }
        return best;
    }

    private int choose(float[] observation, double eps) {
        return random.nextDouble() < eps ? random.nextInt(actions) : greedy(values(observation));
    }

    @Override
    public float[] act(float[] observation, boolean explore) {
        double[] q = values(observation);
        if (explore) {
            return new float[] {choose(observation, epsilon.at(steps))};
        }
        int best = 0;
        for (int a = 1; a < q.length; a++) {
            if (q[a] > q[best]) {
                best = a;
            }
        }
        return new float[] {best};
    }

    @Override
    public RlResult fit(Environment env, long totalSteps, RlListener listener) {
        if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != actions) {
            throw new IllegalArgumentException("tabular learning needs " + actions + " discrete actions, got "
                    + env.actionSpace());
        }
        Tracker tracker = new Tracker(listener);
        float[] obs = env.reset(random.split());
        int action = choose(obs, epsilon.at(steps));
        while (tracker.steps() < totalSteps) {
            Environment.Step s = env.step(action);
            double eps = epsilon.at(steps);
            steps++;
            double[] q = values(obs);
            int nextAction = s.done() ? 0 : choose(s.observation(), eps);
            double future = 0;
            if (!s.terminated()) {
                double[] next = values(s.observation());
                future = switch (method) {
                    case Q_LEARNING -> Arrays.stream(next).max().orElseThrow();
                    case SARSA -> s.done() ? Arrays.stream(next).max().orElseThrow() : next[nextAction];
                    case EXPECTED_SARSA -> {
                        double max = Arrays.stream(next).max().orElseThrow();
                        long best = Arrays.stream(next).filter(v -> v == max).count();
                        double e = 0;
                        for (double v : next) {
                            e += v * (eps / actions + (v == max ? (1 - eps) / best : 0));
                        }
                        yield e;
                    }
                };
            }
            q[action] += alpha * (s.reward() + gamma * future - q[action]);
            boolean ended = tracker.step(s);
            if (ended) {
                obs = env.reset(random.split());
                action = choose(obs, epsilon.at(steps));
            } else {
                obs = s.observation();
                action = nextAction;
            }
        }
        return tracker.result;
    }

    @Override
    public String toString() {
        return "Tabular(" + method + ", " + table.size() + " states)";
    }
}
