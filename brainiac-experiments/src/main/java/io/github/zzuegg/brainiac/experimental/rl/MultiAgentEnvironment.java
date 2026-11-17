package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * A cooperative multi-agent world: several agents act at once, each on its own (partial)
 * observation, and share one team reward. A global state, seen only during centralized
 * training (as in QMIX and MAPPO), may hold more than any agent observes. The multi-agent
 * counterpart of {@link io.github.zzuegg.brainiac.rl.Environment}.
 */
public interface MultiAgentEnvironment {

    /** @return the number of agents */
    int agents();

    /** @return values per agent observation */
    int observationSize();

    /** @return values of the global state */
    int stateSize();

    /** @return discrete actions per agent */
    int actions();

    /**
     * Starts an episode.
     *
     * @param random the episode's randomness
     * @return every agent's first observation
     */
    float[][] reset(SplittableRandom random);

    /** @return the current global state */
    float[] state();

    /**
     * Applies a joint action.
     *
     * @param actions one action per agent
     * @return what happened
     */
    Step step(int[] actions);

    /**
     * The result of a joint step.
     *
     * @param observations every agent's next observation
     * @param reward the shared team reward
     * @param terminated whether a terminal state was reached
     * @param truncated whether the episode was cut off
     */
    record Step(float[][] observations, float reward, boolean terminated, boolean truncated) {
        /** @return whether the episode is over */
        public boolean done() {
            return terminated || truncated;
        }
    }
}
