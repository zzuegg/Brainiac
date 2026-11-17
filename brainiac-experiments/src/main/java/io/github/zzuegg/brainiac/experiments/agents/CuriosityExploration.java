package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.CuriosityEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.IntrinsicCuriosityModule;
import io.github.zzuegg.brainiac.experimental.rl.IntrinsicReward;
import io.github.zzuegg.brainiac.experimental.rl.RandomNetworkDistillation;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;

/**
 * Experiment X27, curiosity: the same Q-learner explores a sparse-reward maze (reward only
 * at the far goal) with no bonus, with Random Network Distillation and with the Intrinsic
 * Curiosity Module. The bonus is added by {@link CuriosityEnvironment}, so the learner is
 * the core {@link Tabular} unchanged.
 */
public final class CuriosityExploration {

    /** Which bonus. */
    public enum Bonus {
        /** No intrinsic reward. */
        NONE,
        /** Random Network Distillation. */
        RND,
        /** Intrinsic Curiosity Module. */
        ICM
    }

    /**
     * The outcome.
     *
     * @param distinctCells cells visited during training
     * @param goalsReached how often the goal was reached
     */
    public record Result(int distinctCells, int goalsReached) {
    }

    /** The maze: a long serpentine corridor with the goal at its far end. */
    public static final String[] MAZE = {
        "S...........",
        "###########.",
        "............",
        ".###########",
        "............",
        "###########.",
        "G...........",
    };

    private CuriosityExploration() {
    }

    /**
     * @param bonus the bonus
     * @param steps environment steps
     * @param seed the seed
     * @return what was explored
     */
    public static Result run(Bonus bonus, int steps, long seed) {
        GridWorld grid = GridWorld.of(MAZE).stepReward(0).maxSteps(200);
        IntrinsicReward r = switch (bonus) {
            case NONE -> null;
            case RND -> RandomNetworkDistillation.of(grid.cells(), 16, 32, seed);
            case ICM -> IntrinsicCuriosityModule.of(grid.cells(), 4, 16, 32, seed).forwardWeight(0.8f);
        };
        CuriosityEnvironment env = CuriosityEnvironment.of(grid, r, 0.2f).centered(true);
        Tabular.qLearning(4).learningRate(0.5).discount(0.95).epsilon(Schedule.constant(0.05)).seed(seed)
                .fit(env, steps);
        return new Result(env.distinctObservations(), env.positiveRewards());
    }
}
