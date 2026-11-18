package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * Greedy evaluation of joint policies on a {@link MultiAgentEnvironment}: the mean
 * undiscounted team return, and for {@link CooperativeNavigation} the landmark coverage at
 * the end of the episode. The multi-agent counterpart of {@code Rl.evaluate}.
 */
public final class MultiAgentEvaluation {

    /** Picks every agent's discrete action from every agent's observation. */
    @FunctionalInterface
    public interface JointPolicy {
        /**
         * @param observations one observation per agent
         * @param explore whether to explore
         * @return one action per agent
         */
        int[] act(float[][] observations, boolean explore);
    }

    private MultiAgentEvaluation() {
    }

    /**
     * @param env the environment
     * @param policy the joint policy, run greedily
     * @param episodes episodes
     * @param seed the seed
     * @return the mean team return
     */
    public static double teamReturn(MultiAgentEnvironment env, JointPolicy policy, int episodes, long seed) {
        return run(env, policy, episodes, seed)[0];
    }

    /**
     * @param env the environment
     * @param policy the joint policy, run greedily
     * @param episodes episodes
     * @param seed the seed
     * @return the mean final coverage
     */
    public static double coverage(CooperativeNavigation env, JointPolicy policy, int episodes, long seed) {
        return run(env, policy, episodes, seed)[1];
    }

    private static double[] run(MultiAgentEnvironment env, JointPolicy policy, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double total = 0;
        double covered = 0;
        for (int e = 0; e < episodes; e++) {
            float[][] obs = env.reset(r.split());
            MultiAgentEnvironment.Step s;
            do {
                s = env.step(policy.act(obs, false));
                obs = s.observations();
                total += s.reward();
            } while (!s.done());
            if (env instanceof CooperativeNavigation nav) {
                covered += nav.coverage();
            }
        }
        return new double[] {total / episodes, covered / episodes};
    }
}
