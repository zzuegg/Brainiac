package io.github.zzuegg.brainiac.experiments.memory;

import io.github.zzuegg.brainiac.experimental.rl.EpisodicControl;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.List;

/**
 * Experiment X6: an agent that remembers states and their outcomes across episodes against
 * memoryless ones, on a GridWorld maze. The episodic-control agent keeps, per action, the
 * best return seen from each visited cell; the baselines are the same agent with its memory
 * wiped at every reset, and a random agent.
 */
public final class EpisodicAgentComparison {

    /**
     * The outcome.
     *
     * @param episodicLate the episodic agent's mean return over its last 20 training episodes
     * @param memorylessLate the memoryless agent's mean return over its last 20 training episodes
     * @param episodicGreedy the episodic agent's greedy return after training
     * @param randomGreedy the random agent's mean return
     * @param episodicEpisodes how many episodes the episodic agent finished in the step budget
     * @param memorylessEpisodes how many episodes the memoryless agent finished
     */
    public record Result(double episodicLate, double memorylessLate, double episodicGreedy, double randomGreedy,
            int episodicEpisodes, int memorylessEpisodes) {
    }

    private EpisodicAgentComparison() {
    }

    /** @return the maze used: a 5x8 grid with dead ends, reward only at the goal */
    static GridWorld maze() {
        return GridWorld.of(
                "S..#....",
                ".#.#.##.",
                ".#...#..",
                ".####.#.",
                "......#G").coordinates(true).stepReward(0).maxSteps(100);
    }

    /**
     * Runs the comparison.
     *
     * @param steps the environment steps each agent gets
     * @param seed the random seed
     * @return the result
     */
    public static Result run(long steps, long seed) {
        GridWorld env = maze();
        EpisodicControl episodic = EpisodicControl.of(env, 0, seed).epsilon(0.1);
        episodic.fit(env, steps);
        EpisodicControl memoryless = EpisodicControl.of(env, 0, seed).epsilon(0.1)
                .rememberAcrossEpisodes(false);
        memoryless.fit(env, steps);
        double greedy = Rl.evaluate(episodic, env, 5, seed);
        double random = Rl.evaluate(Agent.random(env.actionSpace(), seed), env, 20, seed);
        return new Result(late(episodic.episodeReturns()), late(memoryless.episodeReturns()), greedy, random,
                episodic.episodeReturns().size(), memoryless.episodeReturns().size());
    }

    private static double late(List<Double> returns) {
        int n = Math.min(20, returns.size());
        double s = 0;
        for (int i = returns.size() - n; i < returns.size(); i++) {
            s += returns.get(i);
        }
        return s / Math.max(1, n);
    }
}
