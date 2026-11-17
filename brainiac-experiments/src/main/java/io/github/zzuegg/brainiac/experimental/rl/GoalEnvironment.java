package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;

/**
 * A goal-conditioned environment: every observation is the state followed by the desired
 * goal, and the reward depends only on whether the goal achieved by a state matches the
 * desired one. That structure is what lets {@link HindsightReplay} pretend, after the
 * fact, that the agent was asked for whatever it actually reached.
 */
public interface GoalEnvironment extends Environment {

    /** @return how many values of an observation describe the goal (they come last) */
    int goalSize();

    /**
     * @param observation an observation (state then goal)
     * @return the goal that the observation's state achieves
     */
    float[] achievedGoal(float[] observation);

    /**
     * @param achieved an achieved goal
     * @param desired a desired goal
     * @return the reward for achieving {@code achieved} when asked for {@code desired}
     */
    float reward(float[] achieved, float[] desired);

    /**
     * @param achieved an achieved goal
     * @param desired a desired goal
     * @return whether that ends the episode as a success
     */
    boolean success(float[] achieved, float[] desired);
}
