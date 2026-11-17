package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.GoExplore;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.MountainCar;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Go-Explore against random exploration with the same number of environment steps (returns
 * included). In a serpentine corridor lined with holes, every wrong step ends the episode, so
 * random walks die near the start; Go-Explore keeps every stepping stone in its archive and
 * returns to the frontier. The sparse-reward mountain car is the classic easier case.
 */
public final class HardExploration {

    /** A serpentine corridor between holes; about 40 steps from start to goal. */
    public static final String[] DEADLY_MAZE = {
        "S...........",
        "HHHHHHHHHHH.",
        "............",
        ".HHHHHHHHHHH",
        "............",
        "HHHHHHHHHHH.",
        "HHHHHHHHHHHG",
    };

    /**
     * @param goExploreCells cells Go-Explore found
     * @param randomCells cells random exploration visited
     * @param goExploreSolved whether Go-Explore finished an episode (reached the goal or a hole)
     * @param randomSolved how many random episodes reached the goal
     * @param bestReturn the return of Go-Explore's best path, replayed
     * @param steps the environment steps both used
     */
    public record Result(int goExploreCells, int randomCells, boolean goExploreSolved, int randomSolved,
            double bestReturn, long steps) {
    }

    private HardExploration() {
    }

    /**
     * @param iterations Go-Explore iterations
     * @param seed the seed
     * @return the outcome in {@link #DEADLY_MAZE}
     */
    public static Result maze(int iterations, long seed) {
        Environment env = GridWorld.of(DEADLY_MAZE).stepReward(-0.01f).maxSteps(200).coordinates(true);
        return compare(env, Arrays::toString, iterations, 20, s -> s.terminated() && s.reward() > 0, seed);
    }

    /**
     * @param iterations Go-Explore iterations
     * @param seed the seed
     * @return the outcome on the mountain car
     */
    public static Result mountainCar(int iterations, long seed) {
        Function<float[], Object> cell = GoExplore.grid(new float[] {-1.2f, -0.07f}, new float[] {0.6f, 0.07f}, 12);
        return compare(new MountainCar(200), cell, iterations, 40, Environment.Step::terminated, seed);
    }

    private static Result compare(Environment env, Function<float[], Object> cell, int iterations, int explore,
            Predicate<Environment.Step> success, long seed) {
        int actions = ((Space.Discrete) env.actionSpace()).n();
        GoExplore ge = GoExplore.of(cell, seed).explorationSteps(explore).explore(env, iterations);
        long budget = ge.environmentSteps();
        SplittableRandom random = new SplittableRandom(seed);
        Set<Object> seen = new HashSet<>();
        int solved = 0;
        long used = 0;
        while (used < budget) {
            float[] obs = env.reset(random.split());
            seen.add(cell.apply(obs));
            int a = random.nextInt(actions);
            boolean done = false;
            while (!done && used < budget) {
                if (random.nextDouble() >= 0.9) {
                    a = random.nextInt(actions);
                }
                Environment.Step s = env.step(new float[] {a});
                used++;
                seen.add(cell.apply(s.observation()));
                done = s.done();
                if (success.test(s)) {
                    solved++;
                }
            }
        }
        return new Result(ge.cells(), seen.size(), ge.solved(), solved, ge.replayBest(env), budget);
    }
}
