package io.github.zzuegg.brainiac.rl;

/** An agent that learns by acting in an environment. */
public interface Learner extends Agent {

    /**
     * Trains by interacting with the environment.
     *
     * @param env the environment
     * @param steps the number of environment steps to take
     * @param listener told about every finished episode; may be {@link RlListener#NONE}
     * @return every finished episode
     */
    RlResult fit(Environment env, long steps, RlListener listener);

    /**
     * Trains quietly.
     *
     * @param env the environment
     * @param steps the number of environment steps to take
     * @return every finished episode
     */
    default RlResult fit(Environment env, long steps) {
        return fit(env, steps, RlListener.NONE);
    }
}
