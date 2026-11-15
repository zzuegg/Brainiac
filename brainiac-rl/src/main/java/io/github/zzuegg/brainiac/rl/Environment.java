package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/**
 * A world an agent acts in, one step at a time: it sees an observation, picks an action
 * and gets a reward and the next observation, until the episode ends.
 *
 * <pre>{@code
 * float[] obs = env.reset(new SplittableRandom(1));
 * Environment.Step s;
 * do {
 *     s = env.step(agent.act(obs, false));
 *     obs = s.observation();
 * } while (!s.done());
 * }</pre>
 *
 * <p>Implement this interface to train agents on your own problem. Discrete actions are
 * passed as a one-value array holding the action index.
 */
public interface Environment {

    /** @return the shape and bounds of observations */
    Space observationSpace();

    /** @return the valid actions */
    Space actionSpace();

    /**
     * Starts a new episode.
     *
     * @param random the source of the episode's randomness
     * @return the first observation
     */
    float[] reset(SplittableRandom random);

    /**
     * Applies an action.
     *
     * @param action the action: {@code {index}} for a discrete space, a vector for a box
     *     (values outside the box are clipped)
     * @return what happened
     */
    Step step(float[] action);

    /**
     * Applies a discrete action.
     *
     * @param action the action index
     * @return what happened
     */
    default Step step(int action) {
        return step(new float[] {action});
    }

    /** @return a short text picture of the current state, for watching an agent */
    default String render() {
        return toString();
    }

    /**
     * Ends episodes after a number of steps, marking them truncated rather than
     * terminated (so value estimates still bootstrap from the last state).
     *
     * @param maxSteps the most steps per episode
     * @return the limited environment
     */
    default Environment limit(int maxSteps) {
        return new TimeLimit(this, maxSteps);
    }

    /**
     * The result of one step.
     *
     * @param observation the next observation
     * @param reward the reward for the step
     * @param terminated whether the episode ended in a terminal state (nothing follows)
     * @param truncated whether the episode was cut off, for example by a time limit
     */
    record Step(float[] observation, float reward, boolean terminated, boolean truncated) {
        /** @return whether the episode is over */
        public boolean done() {
            return terminated || truncated;
        }
    }
}
