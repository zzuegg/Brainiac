package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.deeprl.ModernSac;
import io.github.zzuegg.brainiac.experimental.deeprl.PixelObservations;
import io.github.zzuegg.brainiac.experimental.explore.BootstrappedDqn;
import io.github.zzuegg.brainiac.experimental.explore.CountBonuses;
import io.github.zzuegg.brainiac.experimental.explore.DeepSeaChain;
import io.github.zzuegg.brainiac.experimental.explore.PointMaze;
import io.github.zzuegg.brainiac.experimental.explore.Rlsvi;
import io.github.zzuegg.brainiac.experimental.explore.SkillDiscovery;
import io.github.zzuegg.brainiac.experimental.nn.NoisyDense;
import io.github.zzuegg.brainiac.experimental.rl.CuriosityEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.IntrinsicReward;
import io.github.zzuegg.brainiac.experimental.rl.KeyDoorGrid;
import io.github.zzuegg.brainiac.experimental.rl.RainbowDqn;
import io.github.zzuegg.brainiac.experimental.rl.RandomNetworkDistillation;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.MountainCar;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.LongFunction;

/**
 * Wave 11 items 11.6.6-11.6.8: exploration when rewards are sparse or absent.
 *
 * <ul>
 *   <li>{@link #deepSea} (11.6.6): bootstrapped DQN with and without randomised priors and
 *       tabular RLSVI against epsilon-greedy DQN, NoisyNet DQN and DQN with an RND bonus on the
 *       Deep Sea of side {@code N}: episodes to the first reward and the share of seeds whose
 *       greedy policy solves it; {@link #riverSwim} the same agents' cousins on RiverSwim.</li>
 *   <li>{@link #countBonuses} (11.6.7): DQN with SimHash, learned-code SimHash, pseudo-count,
 *       exact-count and RND bonuses against plain epsilon-greedy DQN, on the Deep Sea (where
 *       exact counts are the oracle), sparse mountain car (state coverage) and a key-and-door
 *       grid seen as pixels.</li>
 *   <li>{@link #skills} (11.6.8): DIAYN, METRA and CSF skills pretrained with SAC without
 *       reward on an open point maze, against random skills and RND-driven SAC: grid cells
 *       covered by eight evaluation skills and the share of held-out goals one of them ends
 *       near (zero-shot skill selection).</li>
 * </ul>
 */
public final class DeepExploration {

    private DeepExploration() {
    }

    /** Records the extrinsic return of every episode and the cells a projection visits. */
    static final class Logged implements Environment {
        private final Environment inner;
        private final Function<float[], Object> cell;
        final List<Double> returns = new ArrayList<>();
        final Set<Object> cells = new HashSet<>();
        private double current;

        Logged(Environment inner, Function<float[], Object> cell) {
            this.inner = inner;
            this.cell = cell;
        }

        @Override
        public Space observationSpace() {
            return inner.observationSpace();
        }

        @Override
        public Space actionSpace() {
            return inner.actionSpace();
        }

        @Override
        public float[] reset(SplittableRandom random) {
            current = 0;
            float[] o = inner.reset(random);
            if (cell != null) {
                cells.add(cell.apply(o));
            }
            return o;
        }

        @Override
        public Step step(float[] action) {
            Step s = inner.step(action);
            current += s.reward();
            if (cell != null) {
                cells.add(cell.apply(s.observation()));
            }
            if (s.done()) {
                returns.add(current);
            }
            return s;
        }

        int firstEpisodeAbove(double threshold) {
            for (int i = 0; i < returns.size(); i++) {
                if (returns.get(i) > threshold) {
                    return i + 1;
                }
            }
            return -1;
        }
    }

    private static Dqn dqn(int obs, int actions, long steps, long seed) {
        return Dqn.of(() -> Nets.mlp(obs, actions, Activation.RELU, seed, 32), actions).learningStarts(64)
                .epsilon(Schedule.linear(1, 0.05, Math.max(1, steps / 2))).targetUpdateEvery(100).seed(seed);
    }

    /** The agents of 11.6.6 for a Deep Sea of a given side. */
    private static Map<String, LongFunction<Learner>> deepSeaAgents(int n, long steps) {
        int obs = n * n;
        RlNetworkFactory heads = RlNetworkFactory.mlp(Activation.RELU, 32);
        Map<String, LongFunction<Learner>> m = new LinkedHashMap<>();
        m.put("DQN eps-greedy", s -> dqn(obs, 2, steps, s));
        m.put("NoisyNet DQN", s -> RainbowDqn.of(() -> Model.sequential().input(obs).dense(32, Activation.RELU)
                .add(NoisyDense.spec(2, 0.5f, s)).seed(s).build(), 2).seed(s));
        m.put("Bootstrapped DQN + prior", s -> BootstrappedDqn.of(obs, 2, 5, heads).priorScale(3f).trainEvery(2)
                .seed(s));
        m.put("Bootstrapped DQN (no prior)", s -> BootstrappedDqn.of(obs, 2, 5, heads).priorScale(0f).trainEvery(2)
                .seed(s));
        m.put("RLSVI tabular", s -> Rlsvi.tabular(2).noise(0.1, 1).sweeps(n + 2).seed(s));
        return m;
    }

    /**
     * 11.6.6: deep exploration on the Deep Sea.
     *
     * @param n the side
     * @param steps environment steps per run
     * @param seeds seeds
     * @param rnd whether to add the DQN + RND row
     * @return agent name to {@code {mean episodes to first reward (episodes run + 1 when never),
     *     share of seeds whose greedy policy reaches the corner}}
     */
    public static Map<String, double[]> deepSea(int n, long steps, int seeds, boolean rnd) {
        Map<String, LongFunction<Learner>> agents = deepSeaAgents(n, steps);
        if (rnd) {
            agents.put("DQN + RND", s -> dqn(n * n, 2, steps, s));
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, LongFunction<Learner>> e : agents.entrySet()) {
            double first = 0;
            double solved = 0;
            for (int s = 0; s < seeds; s++) {
                DeepSeaChain sea = DeepSeaChain.deepSea(n, 7 + s);
                Logged log = new Logged(sea, null);
                Environment train = log;
                if (e.getKey().contains("RND")) {
                    train = CuriosityEnvironment.of(log, RandomNetworkDistillation.of(n * n, 16, 32, s), 0.1f);
                }
                Learner l = e.getValue().apply(s + 1);
                l.fit(train, steps);
                int f = log.firstEpisodeAbove(0.5);
                first += f < 0 ? log.returns.size() + 1 : f;
                solved += Rl.evaluate(l, DeepSeaChain.deepSea(n, 7 + s), 1, 1) > 0.5 ? 1 : 0;
            }
            out.put(e.getKey(), new double[] {first / seeds, solved / seeds});
        }
        return out;
    }

    /**
     * 11.6.6: RiverSwim, the greedy return after training.
     *
     * @param steps environment steps
     * @param seeds seeds
     * @return agent name to the mean greedy return per 20-step episode
     */
    public static Map<String, Double> riverSwim(long steps, int seeds) {
        Map<String, LongFunction<Learner>> agents = new LinkedHashMap<>();
        agents.put("Q-learning eps-greedy (core Tabular)", s -> Tabular.qLearning(2).discount(0.95).seed(s));
        agents.put("RLSVI tabular", s -> Rlsvi.tabular(2).discount(0.95).noise(0.3, 3).seed(s));
        agents.put("RLSVI linear", s -> Rlsvi.linear(2).discount(0.95).noise(0.3, 3).sweeps(20).seed(s));
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, LongFunction<Learner>> e : agents.entrySet()) {
            double total = 0;
            for (int s = 0; s < seeds; s++) {
                Learner l = e.getValue().apply(s + 1);
                l.fit(DeepSeaChain.riverSwim(6, 20), steps);
                total += Rl.evaluate(l, DeepSeaChain.riverSwim(6, 20), 5, 3);
            }
            out.put(e.getKey(), total / seeds);
        }
        return out;
    }

    // ---------------------------------------------------------------- 11.6.7

    /** The tasks of 11.6.7. */
    public enum CountTask {
        /** Deep Sea of side 12 with one-hot observations */
        DEEP_SEA,
        /** mountain car (reward -1 per step until the flag) */
        MOUNTAIN_CAR,
        /** a 4x4 key-and-door grid rendered at 2 pixels per cell */
        PIXEL_KEY_DOOR
    }

    private static Environment countEnv(CountTask task, long seed) {
        return switch (task) {
            case DEEP_SEA -> DeepSeaChain.deepSea(12, 7 + seed);
            case MOUNTAIN_CAR -> new MountainCar(200);
            case PIXEL_KEY_DOOR -> PixelObservations.keyDoor(new KeyDoorGrid(4, 30, 30), 4, 2);
        };
    }

    private static Object cellOf(CountTask task, float[] o) {
        return switch (task) {
            case MOUNTAIN_CAR -> (int) Math.floor((o[0] + 1.2) / 1.8 * 20) * 100 + (int) Math.floor((o[1] + 0.07)
                    / 0.14 * 20);
            default -> java.util.Arrays.toString(o);
        };
    }

    /**
     * 11.6.7: count bonuses on one task.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @return bonus name to {@code {mean episodes to first success (episodes run + 1 when never),
     *     mean distinct cells visited}}
     */
    public static Map<String, double[]> countBonuses(CountTask task, long steps, int seeds) {
        Environment probe = countEnv(task, 0);
        int obs = probe.observationSpace().size();
        int actions = ((Space.Discrete) probe.actionSpace()).n();
        Map<String, LongFunction<IntrinsicReward>> bonuses = new LinkedHashMap<>();
        bonuses.put("eps-greedy (no bonus)", s -> null);
        bonuses.put("SimHash k=16", s -> CountBonuses.simHash(obs, 16, s));
        bonuses.put("pseudo-count", s -> {
            Space.Box box = (Space.Box) probe.observationSpace();
            return CountBonuses.pseudoCount(task == CountTask.MOUNTAIN_CAR ? new float[] {-1.2f, -0.07f} : box.low(),
                    task == CountTask.MOUNTAIN_CAR ? new float[] {0.6f, 0.07f} : box.high(), 12);
        });
        if (task != CountTask.MOUNTAIN_CAR) {
            bonuses.put("exact counts (oracle)", s -> CountBonuses.tabular(obs));
        }
        if (task == CountTask.PIXEL_KEY_DOOR) {
            bonuses.put("learned-code SimHash", s -> CountBonuses.learnedSimHash(obs, 16, 16, 64, s));
        }
        bonuses.put("RND", s -> RandomNetworkDistillation.of(obs, 16, 32, s));
        double threshold = switch (task) {
            case DEEP_SEA -> 0.5;
            case MOUNTAIN_CAR -> -199.5;
            case PIXEL_KEY_DOOR -> 0.5;
        };
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, LongFunction<IntrinsicReward>> e : bonuses.entrySet()) {
            double first = 0;
            double cells = 0;
            for (int s = 0; s < seeds; s++) {
                Logged log = new Logged(countEnv(task, s), o -> cellOf(task, o));
                IntrinsicReward b = e.getValue().apply(s + 1);
                Environment train = b == null ? log : CuriosityEnvironment.of(log, b, task == CountTask.MOUNTAIN_CAR
                        ? 1f : 0.1f);
                dqn(obs, actions, steps, s + 1).fit(train, steps);
                int f = log.firstEpisodeAbove(threshold);
                first += f < 0 ? log.returns.size() + 1 : f;
                cells += log.cells.size();
            }
            out.put(e.getKey(), new double[] {first / seeds, cells / seeds});
        }
        return out;
    }

    // ---------------------------------------------------------------- 11.6.8

    /**
     * 11.6.8: skill discovery on an open point maze.
     *
     * @param steps pretraining steps per method
     * @param seeds seeds
     * @return method name to {@code {cells covered (10 x 10 grid), share of 8 held-out goals
     *     reached within 0.3 by the best of 8 skills}}
     */
    public static Map<String, double[]> skills(long steps, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        String[] rows = {"DIAYN", "METRA", "CSF", "random skills", "RND-driven SAC"};
        for (String row : rows) {
            double cover = 0;
            double reach = 0;
            for (int s = 0; s < seeds; s++) {
                PointMaze maze = new PointMaze(PointMaze.Layout.OPEN, 30, new float[] {0, 0}, new float[] {5, 5});
                SkillDiscovery.Method m = switch (row) {
                    case "DIAYN" -> SkillDiscovery.Method.DIAYN;
                    case "METRA" -> SkillDiscovery.Method.METRA;
                    case "CSF" -> SkillDiscovery.Method.CSF;
                    default -> SkillDiscovery.Method.RANDOM_SKILLS;
                };
                int dims = m == SkillDiscovery.Method.DIAYN ? 8 : 2;
                SkillDiscovery env = SkillDiscovery.of(m, maze, o -> new float[] {o[0], o[1]}, 2, dims, 31 + s);
                Environment train = row.startsWith("RND")
                        ? CuriosityEnvironment.of(env, RandomNetworkDistillation.of(4 + dims, 16, 32, s), 1f) : env;
                RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.RELU, 32, 32);
                ModernSac sac = ModernSac.of(4 + dims, 2, nets, nets).learningStarts(200).learningRate(1e-3f)
                        .seed(s + 1);
                sac.fit(train, steps);
                double[] c = env.coverage(sac, 8, 10, 5);
                cover += c[0];
                SplittableRandom g = new SplittableRandom(77);
                int hit = 0;
                for (int k = 0; k < 8; k++) {
                    double gx = g.nextDouble(-0.9, 0.9);
                    double gy = g.nextDouble(-0.9, 0.9);
                    double best = Double.POSITIVE_INFINITY;
                    for (int z = 0; z < 8; z++) {
                        best = Math.min(best, Math.hypot(c[1 + 2 * z] - gx, c[2 + 2 * z] - gy));
                    }
                    hit += best < 0.3 ? 1 : 0;
                }
                reach += hit / 8.0;
            }
            out.put(row, new double[] {cover / seeds, reach / seeds});
        }
        return out;
    }

    /**
     * Prints the tables at a larger budget.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (int n : new int[] {10, 14, 20}) {
            System.out.println("deep sea " + n + ": " + fmt(deepSea(n, 400L * n * n / 4, 5, true)));
        }
        for (CountTask t : CountTask.values()) {
            System.out.println(t + ": " + fmt(countBonuses(t, 10_000, 3)));
        }
        System.out.println("skills: " + fmt(skills(10_000, 3)));
    }

    /**
     * @param table a table
     * @return one line
     */
    public static String fmt(Map<String, double[]> table) {
        StringBuilder b = new StringBuilder();
        table.forEach((k, v) -> {
            b.append(k).append(' ');
            for (int i = 0; i < v.length; i++) {
                b.append(i == 0 ? "" : "/").append(String.format("%.2f", v[i]));
            }
            b.append("; ");
        });
        return b.toString();
    }
}
