package io.github.zzuegg.brainiac.rl;

/** Something that picks actions: a trained policy, a value-based agent, or a fixed rule. */
@FunctionalInterface
public interface Agent {

    /**
     * Picks an action.
     *
     * @param observation what the agent sees
     * @param explore whether to explore (sample, or act randomly sometimes) as during
     *     training, or act greedily
     * @return the action: {@code {index}} for discrete actions, a vector for continuous ones
     */
    float[] act(float[] observation, boolean explore);

    /**
     * An agent that acts uniformly at random, as a baseline.
     *
     * @param actions the action space
     * @param seed the random seed
     * @return the agent
     */
    static Agent random(Space actions, long seed) {
        java.util.SplittableRandom random = new java.util.SplittableRandom(seed);
        return (observation, explore) -> actions.sample(random);
    }
}
