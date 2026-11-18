package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.OptionCritic;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Episode;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * W6 3.12: option-critic in the four-rooms grid with a goal switch (Bacon et al.'s transfer
 * experiment). Every learner first trains with the goal in the bottom-right room, then the
 * goal moves to the bottom-left room and training continues. Reported: the mean episode length
 * over the first episodes after the switch (how fast it re-adapts), the final greedy episode
 * length, and for option-critic the mean option duration. Baselines: flat tabular Q-learning
 * and the core A2C at the same steps.
 *
 * <p>Paper: Bacon, Harb and Precup, <a href="https://arxiv.org/abs/1609.05140">arXiv:1609.05140</a>.
 */
public final class OptionsAndTransfer {

    /** Four rooms, goal in the bottom-right room. */
    public static final String[] ROOMS_A = {
        "S..#...",
        ".......",
        "...#...",
        "#.###.#",
        "...#...",
        ".......",
        "...#..G",
    };

    /** The same rooms with the goal moved to the bottom-left room. */
    public static final String[] ROOMS_B = {
        "S..#...",
        ".......",
        "...#...",
        "#.###.#",
        "...#...",
        ".......",
        "G..#...",
    };

    /**
     * The outcome of one learner.
     *
     * @param adaptLength mean episode length over the first episodes after the switch
     * @param finalLength greedy episode length on the new goal at the end (200 = never reached)
     * @param optionDuration mean option duration (NaN for flat learners)
     */
    public record Outcome(double adaptLength, double finalLength, double optionDuration) {
    }

    private OptionsAndTransfer() {
    }

    private static GridWorld rooms(String[] map) {
        return GridWorld.of(map).stepReward(-0.01f).maxSteps(200);
    }

    /**
     * @param steps steps on each goal
     * @param adaptEpisodes episodes after the switch that the adaptation length averages
     * @param seeds seeds to average
     * @return learner name to outcome
     */
    public static Map<String, Outcome> run(int steps, int adaptEpisodes, int seeds) {
        String[] names = {"option-critic (4 options)", "Q-learning (core)", "A2C (core)"};
        double[][] sums = new double[names.length][3];
        for (int s = 0; s < seeds; s++) {
            long seed = 1 + s;
            int cells = rooms(ROOMS_A).cells();
            Learner[] learners = {
                OptionCritic.of(4, 4).learningRates(0.5, 0.1, 0.1).seed(seed),
                Tabular.qLearning(4).learningRate(0.5).discount(0.99).epsilon(Schedule.constant(0.05)).seed(seed),
                Ppo.a2c(Nets.mlp(cells, 4, seed, 64), Nets.mlp(cells, 1, seed + 1, 64),
                        rooms(ROOMS_A).actionSpace()).seed(seed),
            };
            for (int i = 0; i < learners.length; i++) {
                learners[i].fit(rooms(ROOMS_A), steps);
                RlResult after = learners[i].fit(rooms(ROOMS_B), steps);
                List<Episode> eps = after.episodes();
                double len = 0;
                int n = Math.min(adaptEpisodes, eps.size());
                for (int e = 0; e < n; e++) {
                    len += eps.get(e).steps();
                }
                sums[i][0] += n == 0 ? 200 : len / n;
                sums[i][1] += greedyLength(learners[i], rooms(ROOMS_B), seed);
                sums[i][2] += learners[i] instanceof OptionCritic oc ? oc.meanOptionDuration() : Double.NaN;
            }
        }
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (int i = 0; i < names.length; i++) {
            out.put(names[i], new Outcome(sums[i][0] / seeds, sums[i][1] / seeds, sums[i][2] / seeds));
        }
        return out;
    }

    private static double greedyLength(Agent agent, GridWorld env, long seed) {
        float[] o = env.reset(new java.util.SplittableRandom(seed));
        agent.startEpisode();
        int steps = 0;
        io.github.zzuegg.brainiac.rl.Environment.Step st;
        do {
            st = env.step(agent.act(o, false));
            o = st.observation();
            steps++;
        } while (!st.done());
        return st.terminated() ? steps : 200;
    }
}
