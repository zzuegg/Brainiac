package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.explore.CountBonuses;
import io.github.zzuegg.brainiac.experimental.explore.DeepSeaChain;
import io.github.zzuegg.brainiac.experimental.explore.DynaPlanning;
import io.github.zzuegg.brainiac.experimental.explore.LaplacianOptions;
import io.github.zzuegg.brainiac.experimental.explore.Rlsvi;
import io.github.zzuegg.brainiac.experimental.explore.TabularExplorers;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlResult;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.LongFunction;

/**
 * Wave 12 items 12.6.1-12.6.3: tabular exploration, planning and options.
 *
 * <ul>
 *   <li>{@link #exploration} (12.6.1): cumulative regret and episodes to the first reward on Deep
 *       Sea of growing size for R-max, UCRL2, PSRL and optimistic Q-learning (Hoeffding and
 *       Bernstein), against epsilon-greedy Q-learning, Q-learning with a tabular count bonus
 *       ({@code CountBonuses}) and wave 11 tabular RLSVI; {@link #riverSwim} the mean return per
 *       episode on RiverSwim.</li>
 *   <li>{@link #dynaMaze} and {@link #changingMaze} (12.6.2): episodes completed within a budget of
 *       real steps on Sutton &amp; Barto's Dyna maze for Q-learning, Dyna-Q, Dyna-Q+ and prioritised sweeping (with their simulated
 *       updates), and the reward collected after the blocking and shortcut mazes change.</li>
 *   <li>{@link #options} (12.6.3): random-walk cover time and far-goal hitting time on four rooms
 *       with primitive actions only, eigenoptions and covering options, and the subspace cosines of
 *       the learned (graph-drawing) Laplacian representation against the exact eigenvectors.</li>
 * </ul>
 */
public final class TabularExploration {

    private TabularExploration() {
    }

    /** The learners of the exploration table, by name, for a chain with the given state count and horizon. */
    private static Map<String, LongFunction<Learner>> explorers(int states, int horizon) {
        Map<String, LongFunction<Learner>> rows = new LinkedHashMap<>();
        rows.put("EPSILON_GREEDY_Q", s -> TabularExplorers.of(TabularExplorers.Method.EPSILON_GREEDY_Q, states, 2,
                horizon, s));
        rows.put("COUNT_BONUS_Q", s -> TabularExplorers.of(TabularExplorers.Method.EPSILON_GREEDY_Q, states, 2,
                horizon, s).intrinsic(CountBonuses.tabular(states), 0.1));
        rows.put("RLSVI", s -> Rlsvi.tabular(2).discount(1).seed(s));
        rows.put("RMAX", s -> TabularExplorers.of(TabularExplorers.Method.RMAX, states, 2, horizon, s).knownVisits(1));
        rows.put("UCRL2", s -> TabularExplorers.of(TabularExplorers.Method.UCRL2, states, 2, horizon, s)
                .bonusScale(0.1));
        rows.put("PSRL", s -> TabularExplorers.of(TabularExplorers.Method.PSRL, states, 2, horizon, s));
        rows.put("UCB_Q_HOEFFDING", s -> TabularExplorers.of(TabularExplorers.Method.UCB_Q_HOEFFDING, states, 2,
                horizon, s).bonusScale(0.05));
        rows.put("UCB_Q_BERNSTEIN", s -> TabularExplorers.of(TabularExplorers.Method.UCB_Q_BERNSTEIN, states, 2,
                horizon, s).bonusScale(0.05));
        return rows;
    }

    /**
     * 12.6.1: Deep Sea of the given size.
     *
     * @param size the side {@code N}
     * @param episodes episodes per run
     * @param seeds runs (each with its own action mapping)
     * @return method to {@code {mean cumulative regret, mean episodes to first reward (episodes + 1 if never)}}
     */
    public static Map<String, double[]> exploration(int size, int episodes, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        int horizon = size;
        for (Map.Entry<String, LongFunction<Learner>> row : explorers(size * size, horizon).entrySet()) {
            double regret = 0;
            double first = 0;
            for (int s = 0; s < seeds; s++) {
                DeepSeaChain env = DeepSeaChain.deepSea(size, 100 + s);
                RlResult res = row.getValue().apply(s).fit(env, (long) episodes * (size - 1));
                List<Episode> eps = res.episodes();
                int firstReward = episodes + 1;
                double total = 0;
                for (int e = 0; e < eps.size(); e++) {
                    total += env.optimalReturn() - eps.get(e).totalReward();
                    if (firstReward > episodes && eps.get(e).totalReward() > 0.5) {
                        firstReward = e + 1;
                    }
                }
                regret += total;
                first += firstReward;
            }
            out.put(row.getKey(), new double[] {regret / seeds, first / seeds});
        }
        return out;
    }

    /**
     * 12.6.1: RiverSwim.
     *
     * @param states chain length
     * @param length steps per episode
     * @param episodes episodes per run
     * @param seeds runs
     * @return method to the mean return per episode
     */
    public static Map<String, Double> riverSwim(int states, int length, int episodes, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (Map.Entry<String, LongFunction<Learner>> row : explorers(states, length).entrySet()) {
            double mean = 0;
            for (int s = 0; s < seeds; s++) {
                DeepSeaChain env = DeepSeaChain.riverSwim(states, length);
                RlResult res = row.getValue().apply(s).fit(env, (long) episodes * length);
                mean += res.meanReturn(episodes);
            }
            out.put(row.getKey(), mean / seeds);
        }
        return out;
    }

    /**
     * 12.6.2: the Dyna maze with a fixed budget of real steps.
     *
     * @param planningSteps simulated updates per real step
     * @param steps real steps per run
     * @param seeds runs
     * @return scheme to {@code {episodes completed, mean length of the last 5 episodes, simulated updates}}
     */
    public static Map<String, double[]> dynaMaze(int planningSteps, int steps, int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (DynaPlanning.Method m : DynaPlanning.Method.values()) {
            double episodes = 0;
            double last = 0;
            double sim = 0;
            for (int s = 0; s < seeds; s++) {
                DynaPlanning.Maze maze = DynaPlanning.Maze.dyna();
                DynaPlanning agent = DynaPlanning.of(m, maze.cells(), 4, planningSteps, s).kappa(1e-3);
                List<Episode> eps = agent.fit(maze, steps).episodes();
                episodes += eps.size();
                double tail = 0;
                int count = Math.min(5, eps.size());
                for (int e = eps.size() - count; e < eps.size(); e++) {
                    tail += eps.get(e).steps();
                }
                last += count == 0 ? steps : tail / count;
                sim += agent.simulatedUpdates();
            }
            out.put(m.name(), new double[] {episodes / seeds, last / seeds, sim / seeds});
        }
        return out;
    }

    /**
     * 12.6.2: the blocking or shortcut maze.
     *
     * @param shortcut the shortcut maze (else the blocking maze)
     * @param changeAt the step of the change
     * @param steps total real steps
     * @param planningSteps simulated updates per step
     * @param seeds runs
     * @return scheme to {@code {reward before the change, reward after it}}
     */
    public static Map<String, double[]> changingMaze(boolean shortcut, int changeAt, int steps, int planningSteps,
            int seeds) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (DynaPlanning.Method m : DynaPlanning.Method.values()) {
            double before = 0;
            double after = 0;
            for (int s = 0; s < seeds; s++) {
                DynaPlanning.Maze maze = shortcut ? DynaPlanning.Maze.shortcut(changeAt)
                        : DynaPlanning.Maze.blocking(changeAt);
                DynaPlanning agent = DynaPlanning.of(m, maze.cells(), 4, planningSteps, s).kappa(1e-3);
                RlResult a = agent.fit(maze, changeAt);
                RlResult b = agent.fit(maze, steps - changeAt);
                before += a.episodes().size();
                after += b.episodes().size();
            }
            out.put(m.name(), new double[] {before / seeds, after / seeds});
        }
        return out;
    }

    /**
     * 12.6.3: options on a rooms grid.
     *
     * @param roomsPerSide 2 for four rooms, 3 for nine rooms
     * @param roomSize free cells per room side
     * @param runs random walks per measurement
     * @param seed the seed
     * @return row to {@code {cover time, far-goal hitting time, options}}; the row
     *     {@code "learned Laplacian cosines"} holds the cosines of the first eigenvectors
     */
    public static Map<String, double[]> options(int roomsPerSide, int roomSize, int runs, long seed) {
        LaplacianOptions.GridGraph g = LaplacianOptions.GridGraph.rooms(roomsPerSide, roomSize);
        int start = 0;
        int goal = g.nodes() - 1;
        int cap = 200_000;
        Map<String, List<LaplacianOptions.Option>> sets = new LinkedHashMap<>();
        sets.put("primitive", List.of());
        sets.put("eigenoptions (4 vectors)", LaplacianOptions.eigenoptions(g, 4, 0.99));
        sets.put("covering options (4 pairs)", LaplacianOptions.coveringOptions(g, 4));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<LaplacianOptions.Option>> e : sets.entrySet()) {
            double cover = LaplacianOptions.coverTime(g, e.getValue(), start, runs, cap, seed);
            double hit = LaplacianOptions.hittingTime(g, e.getValue(), start, goal, runs, cap, seed + 1);
            out.put(e.getKey(), new double[] {cover, hit, e.getValue().size()});
        }
        double[][] learned = LaplacianOptions.learnedLaplacian(g, 4, 200_000, 1.0, 0.003, seed);
        out.put("learned Laplacian cosines", LaplacianOptions.subspaceCosines(g, learned));
        return out;
    }
}
