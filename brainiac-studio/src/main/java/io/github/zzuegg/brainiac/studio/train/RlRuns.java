package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Lander;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.MountainCar;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Reinforce;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tabular;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Reinforcement learning runs: an agent learns one of the built-in environments with one of the
 * learners of the reinforcement learning module, on the CPU and in a thread of its own. When a
 * run ends, its agent is scored on fresh episodes and one of its episodes is kept as pictures to
 * watch. Runs live in memory: they are gone when the studio closes.
 */
public final class RlRuns implements AutoCloseable {

    /** The most runs that learn at once. */
    public static final int MAX_ACTIVE = 2;

    /** The most environment steps of a run. */
    public static final long MAX_STEPS = 5_000_000;

    /** The most pictures kept of the episode to watch. */
    static final int MAX_FRAMES = 400;

    /** The most points of a run's curve. */
    static final int CURVE_POINTS = 200;

    private static final System.Logger LOG = System.getLogger(RlRuns.class.getName());

    /** A built-in environment. */
    private record Env(String id, String name, String what, boolean grid, Supplier<Environment> make,
            Supplier<Environment> small) {
    }

    /** A learner. */
    private record Algorithm(String id, String name, String what, boolean table, boolean discreteOnly) {
    }

    private static final List<Env> ENVS = List.of(
            new Env("cartpole", "CartPole", "Balance a pole on a cart by pushing it left or right. 500 is a perfect episode.",
                    false, CartPole::new, CartPole::new),
            new Env("lander", "Lander", "Land a craft softly between the flags with its main and side engines.",
                    false, Lander::new, Lander::new),
            new Env("mountaincar", "MountainCar", "Rock a weak car out of a valley. Rewards are shaped by its height.",
                    false, () -> new MountainCar().shaped(true), () -> new MountainCar().shaped(true)),
            new Env("pendulum", "Pendulum", "Swing a pendulum up and hold it there, with a continuous torque.",
                    false, Pendulum::new, Pendulum::new),
            new Env("cliff", "Cliff walk", "Walk a grid from start to goal along a cliff; falling costs 100.",
                    true, GridWorld::cliff, () -> GridWorld.cliff().coordinates(true)),
            new Env("lake", "Frozen lake", "Cross a grid of ice to the goal without falling into a hole.",
                    true, GridWorld::lake, () -> GridWorld.lake().coordinates(true)));

    private static final List<Algorithm> ALGORITHMS = List.of(
            new Algorithm("ppo", "PPO", "Proximal policy optimization: the dependable default, for any environment.", false, false),
            new Algorithm("a2c", "A2C", "Advantage actor-critic: simpler and quicker per step than PPO.", false, false),
            new Algorithm("dqn", "DQN", "Deep Q-learning with a replay buffer, for environments with a few actions.", false, true),
            new Algorithm("reinforce", "REINFORCE", "The plain policy gradient, with a value baseline.", false, true),
            new Algorithm("qlearning", "Q-learning", "A table of action values, for the grid worlds.", true, true),
            new Algorithm("sarsa", "SARSA", "A table that learns the policy it follows, for the grid worlds.", true, true));

    /** Thrown inside a learner to end a run the user stopped. */
    private static final class Stopped extends RuntimeException {
        private static final long serialVersionUID = 1L;

        Stopped() {
            super("stopped", null, false, false);
        }
    }

    /** One run. */
    private final class Run {
        final String id;
        final Env env;
        final Algorithm algorithm;
        final long steps;
        final long seed;
        final int[] hidden;
        final long created = System.currentTimeMillis();
        final List<double[]> episodes = new ArrayList<>();   // {total steps, return}
        String state = "running";
        String error;
        long ended;
        long done;
        Double score;
        List<String> frames = List.of();
        volatile boolean stopping;

        Run(String id, Env env, Algorithm algorithm, long steps, long seed, int[] hidden) {
            this.id = id;
            this.env = env;
            this.algorithm = algorithm;
            this.steps = steps;
            this.seed = seed;
            this.hidden = hidden;
        }

        synchronized boolean active() {
            return state.equals("running");
        }

        synchronized void episode(Episode e) {
            episodes.add(new double[] {e.totalSteps(), e.totalReward()});
            done = e.totalSteps();
        }

        synchronized Map<String, Object> toJson(boolean full) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("env", env.id());
            m.put("envName", env.name());
            m.put("algorithm", algorithm.id());
            m.put("algorithmName", algorithm.name());
            m.put("steps", steps);
            m.put("seed", seed);
            m.put("hidden", java.util.Arrays.stream(hidden).boxed().toList());
            m.put("state", state);
            m.put("created", created);
            m.put("ended", ended);
            m.put("done", Math.min(done, steps));
            m.put("episodes", episodes.size());
            int n = episodes.size();
            if (n > 0) {
                int last = Math.min(n, 20);
                double sum = 0;
                double best = Double.NEGATIVE_INFINITY;
                for (int i = n - last; i < n; i++) {
                    sum += episodes.get(i)[1];
                }
                for (double[] e : episodes) {
                    best = Math.max(best, e[1]);
                }
                m.put("meanReturn", sum / last);
                m.put("bestReturn", best);
            }
            if (score != null) {
                m.put("score", score);
            }
            if (error != null) {
                m.put("error", error);
            }
            m.put("watchable", !frames.isEmpty());
            if (full) {
                m.put("curve", curve());
                m.put("frames", frames);
            }
            return m;
        }

        // the return averaged over stretches of episodes, so a long run stays a short list
        private List<List<Double>> curve() {
            int n = episodes.size();
            int per = Math.max(1, (int) Math.ceil(n / (double) CURVE_POINTS));
            List<List<Double>> out = new ArrayList<>();
            for (int i = 0; i < n; i += per) {
                int to = Math.min(n, i + per);
                double sum = 0;
                for (int j = i; j < to; j++) {
                    sum += episodes.get(j)[1];
                }
                out.add(List.of(episodes.get(to - 1)[0], sum / (to - i)));
            }
            return out;
        }
    }

    private final Map<String, Run> runs = new ConcurrentHashMap<>();
    private int counter;

    /**
     * Returns what can be learned and with what, and every run, newest first.
     *
     * @return {@code {environments, algorithms, maxSteps, runs}}
     */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("environments", ENVS.stream().map(e -> {
            Environment x = e.make().get();
            Map<String, Object> j = new LinkedHashMap<>();
            j.put("id", e.id());
            j.put("name", e.name());
            j.put("what", e.what());
            j.put("grid", e.grid());
            j.put("continuous", !(x.actionSpace() instanceof Space.Discrete));
            j.put("observations", e.small().get().observationSpace().size());
            j.put("actions", x.actionSpace() instanceof Space.Discrete d ? d.n() : x.actionSpace().size());
            return j;
        }).toList());
        m.put("algorithms", ALGORITHMS.stream().map(a -> Map.of("id", a.id(), "name", a.name(), "what", a.what(),
                "table", a.table(), "discreteOnly", a.discreteOnly())).toList());
        m.put("maxSteps", MAX_STEPS);
        m.put("runs", runs.values().stream().sorted(Comparator.comparingLong((Run r) -> r.created).reversed())
                .map(r -> r.toJson(false)).toList());
        return m;
    }

    /**
     * Returns one run with its curve and the episode to watch.
     *
     * @param id the run
     * @return the run, or empty
     */
    public Optional<Map<String, Object>> get(String id) {
        return Optional.ofNullable(runs.get(id)).map(r -> r.toJson(true));
    }

    /**
     * Starts a run.
     *
     * @param env the environment's id
     * @param algorithm the learner's id
     * @param steps how many environment steps to learn for
     * @param seed the random seed
     * @param hidden the sizes of the hidden layers of its networks, one or two of 4 to 512; ignored by the tables
     * @return the run
     * @throws IllegalArgumentException if the two do not fit or a number is out of range
     */
    public synchronized Map<String, Object> start(String env, String algorithm, long steps, long seed, List<Integer> hidden) {
        Env e = ENVS.stream().filter(x -> x.id().equals(env)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("there is no environment " + env));
        Algorithm a = ALGORITHMS.stream().filter(x -> x.id().equals(algorithm)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("there is no learner " + algorithm));
        if (steps < 100 || steps > MAX_STEPS) {
            throw new IllegalArgumentException("steps must be between 100 and " + MAX_STEPS);
        }
        if (a.table() && !e.grid()) {
            throw new IllegalArgumentException(a.name() + " keeps a table, which only the grid worlds fit; use PPO or DQN for "
                    + e.name());
        }
        if (a.discreteOnly() && !(e.make().get().actionSpace() instanceof Space.Discrete)) {
            throw new IllegalArgumentException(a.name() + " picks one of a few actions, and " + e.name()
                    + " takes a continuous one; use PPO or A2C");
        }
        List<Integer> sizes = hidden == null || hidden.isEmpty() ? List.of(64, 64) : hidden;
        if (sizes.size() > 2 || sizes.stream().anyMatch(h -> h == null || h < 4 || h > 512)) {
            throw new IllegalArgumentException("give one or two hidden layers of 4 to 512 units");
        }
        if (runs.values().stream().filter(Run::active).count() >= MAX_ACTIVE) {
            throw new IllegalArgumentException(MAX_ACTIVE + " runs are learning already; stop one or wait");
        }
        Run r = new Run(String.valueOf(++counter), e, a, steps, seed, sizes.stream().mapToInt(Integer::intValue).toArray());
        runs.put(r.id, r);
        Thread t = new Thread(() -> work(r), "studio-rl-" + r.id);
        t.setDaemon(true);
        t.start();
        return r.toJson(false);
    }

    /**
     * Stops a run after the episode it is in.
     *
     * @param id the run
     * @return false if there is no such run
     */
    public boolean stop(String id) {
        Run r = runs.get(id);
        if (r == null) {
            return false;
        }
        r.stopping = true;
        return true;
    }

    /**
     * Forgets a run that has ended.
     *
     * @param id the run
     * @return false if there is no such run
     * @throws IllegalArgumentException if it is still learning
     */
    public boolean delete(String id) {
        Run r = runs.get(id);
        if (r == null) {
            return false;
        }
        if (r.active()) {
            throw new IllegalArgumentException("stop the run before removing it");
        }
        runs.remove(id);
        return true;
    }

    /** Stops every run. */
    @Override
    public void close() {
        runs.values().forEach(r -> r.stopping = true);
    }

    private void work(Run r) {
        Learner learner = null;
        String state = "done";
        String error = null;
        // the tables take the grid as one cell in many; the networks take its row and column
        Supplier<Environment> world = r.algorithm.table() ? r.env.make() : r.env.small();
        try {
            learner = learner(r, world.get());
            learner.fit(world.get(), r.steps, (e, all) -> {
                r.episode(e);
                if (r.stopping) {
                    throw new Stopped();
                }
            });
        } catch (Stopped e) {
            state = "stopped";
        } catch (RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "reinforcement learning run " + r.id + " failed", e);
            state = "failed";
            error = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
        }
        Double score = null;
        List<String> frames = List.of();
        if (learner != null && !state.equals("failed")) {
            try {
                Agent agent = learner;
                score = Rl.evaluate(agent, world.get(), 10, r.seed + 1);
                frames = Rl.watch(agent, world.get(), r.seed + 2, MAX_FRAMES);
            } catch (RuntimeException e) {
                LOG.log(System.Logger.Level.WARNING, "cannot score run " + r.id, e);
            }
        }
        synchronized (r) {
            r.state = state;
            r.error = error;
            r.score = score;
            r.frames = frames;
            r.ended = System.currentTimeMillis();
        }
    }

    private static Learner learner(Run r, Environment env) {
        Space actions = env.actionSpace();
        int in = env.observationSpace().size();
        int out = actions instanceof Space.Discrete d ? d.n() : actions.size();
        return switch (r.algorithm.id()) {
            case "qlearning" -> Tabular.qLearning(out).learningRate(0.5).seed(r.seed);
            case "sarsa" -> Tabular.sarsa(out).learningRate(0.5).seed(r.seed);
            case "dqn" -> Dqn.of(() -> Nets.mlp(in, out, Activation.RELU, r.seed, r.hidden), out).seed(r.seed);
            case "reinforce" -> Reinforce.of(Nets.mlp(in, out, r.seed, r.hidden), actions)
                    .baseline(Nets.mlp(in, 1, r.seed + 1, r.hidden)).seed(r.seed);
            case "a2c" -> Ppo.a2c(Nets.mlp(in, out, r.seed, r.hidden), Nets.mlp(in, 1, r.seed + 1, r.hidden), actions).seed(r.seed);
            default -> Ppo.of(Nets.mlp(in, out, r.seed, r.hidden), Nets.mlp(in, 1, r.seed + 1, r.hidden), actions).seed(r.seed);
        };
    }
}
