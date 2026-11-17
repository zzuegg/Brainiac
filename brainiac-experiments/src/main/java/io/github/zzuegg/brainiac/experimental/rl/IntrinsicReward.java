package io.github.zzuegg.brainiac.experimental.rl;

/**
 * A source of intrinsic reward: a bonus for a transition that is computed by the agent
 * itself, not given by the environment, such as curiosity (prediction error) or novelty.
 * Implementations learn online: every call both scores the transition and trains on it, so
 * familiar transitions stop paying out. Wrap an environment with
 * {@link CuriosityEnvironment} to give any learner the bonus.
 */
public interface IntrinsicReward {

    /**
     * Scores a transition and learns from it.
     *
     * @param observation the observation before the step
     * @param action the action taken ({@code {index}} for discrete actions)
     * @param next the observation after the step
     * @return the bonus, usually non-negative and normalized to about unit scale
     */
    float observe(float[] observation, float[] action, float[] next);
}
