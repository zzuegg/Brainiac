package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEnvironment;

/**
 * A multi-agent environment where every agent has its own reward and a neighbourhood, as
 * mean-field learners need. The team reward of {@link #step} is the mean of the agents' rewards.
 */
public interface LocalRewardEnvironment extends MultiAgentEnvironment {

    /**
     * @param agent an agent
     * @return the agents whose actions enter its mean action
     */
    int[] neighbours(int agent);

    /** @return every agent's reward from the last step */
    float[] agentRewards();
}
