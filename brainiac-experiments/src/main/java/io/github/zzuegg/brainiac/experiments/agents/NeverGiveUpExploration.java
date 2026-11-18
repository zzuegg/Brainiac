package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.Agent57Learner;
import io.github.zzuegg.brainiac.experimental.rl.CuriosityEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.GoExplore;
import io.github.zzuegg.brainiac.experimental.rl.NeverGiveUp;
import io.github.zzuegg.brainiac.experimental.rl.RandomNetworkDistillation;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * W6 3.6: Never Give Up and a tabular Agent57 in the sparse-reward serpentine maze of
 * {@link CuriosityExploration}, at equal environment steps. The same core Q-learner gets no
 * bonus, RND, ICM ({@link CuriosityExploration}) or NGU's episodic-times-lifelong bonus;
 * Agent57 runs a bandit over eight (beta, gamma) policies with the NGU bonus; Go-Explore
 * gets about the same step budget. Metrics: distinct cells visited and goals reached during
 * training, averaged over seeds.
 *
 * <p>Papers: Badia et al., <a href="https://arxiv.org/abs/2002.06038">arXiv:2002.06038</a> (NGU) and
 * <a href="https://arxiv.org/abs/2003.13350">arXiv:2003.13350</a> (Agent57).
 */
public final class NeverGiveUpExploration {

    /**
     * Mean exploration over seeds.
     *
     * @param cells distinct cells visited
     * @param goals goals reached
     */
    public record Outcome(double cells, double goals) {
    }

    private NeverGiveUpExploration() {
    }

    /**
     * @param steps environment steps per method
     * @param seeds seeds to average
     * @return method name to outcome
     */
    public static Map<String, Outcome> run(int steps, int seeds) {
        String[] names = {"none", "RND", "ICM", "NGU", "Agent57 (NGU)", "Go-Explore"};
        double[][] sums = new double[names.length][2];
        for (int s = 0; s < seeds; s++) {
            long seed = 1 + s;
            CuriosityExploration.Bonus[] base = {CuriosityExploration.Bonus.NONE, CuriosityExploration.Bonus.RND,
                CuriosityExploration.Bonus.ICM};
            for (int i = 0; i < base.length; i++) {
                CuriosityExploration.Result r = CuriosityExploration.run(base[i], steps, seed);
                sums[i][0] += r.distinctCells();
                sums[i][1] += r.goalsReached();
            }
            GridWorld grid = maze();
            NeverGiveUp ngu = NeverGiveUp.of(RandomNetworkDistillation.of(grid.cells(), 16, 32, seed));
            CuriosityEnvironment env = CuriosityEnvironment.of(ngu.episodic(grid), ngu, 0.2f).centered(true);
            Tabular.qLearning(4).learningRate(0.5).discount(0.95).epsilon(Schedule.constant(0.05)).seed(seed)
                    .fit(env, steps);
            sums[3][0] += env.distinctObservations();
            sums[3][1] += env.positiveRewards();

            GridWorld grid57 = maze();
            CuriosityEnvironment counted = CuriosityEnvironment.of(grid57, null, 0);
            Agent57Learner.of(4, NeverGiveUp.of(RandomNetworkDistillation.of(grid57.cells(), 16, 32, seed)), 8, 0.3)
                    .seed(seed).fit(counted, steps);
            sums[4][0] += counted.distinctObservations();
            sums[4][1] += counted.positiveRewards();

            GoExplore ge = GoExplore.of(Arrays::toString, seed).explorationSteps(20).explore(maze(), steps / 20);
            sums[5][0] += ge.cells();
            sums[5][1] += ge.solved() ? 1 : 0;
        }
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (int i = 0; i < names.length; i++) {
            out.put(names[i], new Outcome(sums[i][0] / seeds, sums[i][1] / seeds));
        }
        return out;
    }

    private static GridWorld maze() {
        return GridWorld.of(CuriosityExploration.MAZE).stepReward(0).maxSteps(200);
    }
}
