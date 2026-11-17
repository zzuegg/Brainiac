package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;

/**
 * An environment with a safety cost besides its reward, for constrained RL
 * ({@link ConstrainedPolicyOptimization}): the agent maximizes return while keeping the
 * expected total cost per episode under a limit. The cost of the latest step is read after
 * {@link #step}.
 */
public interface CostedEnvironment extends Environment {

    /** @return the cost of the latest step, usually 0 or 1 */
    float lastCost();
}
