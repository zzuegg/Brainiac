package io.github.zzuegg.brainiac.experimental.bandit;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.SplittableRandom;

/**
 * A multi-armed bandit policy: pick an arm, observe its reward, repeat. The common interface
 * of {@link StochasticBandits}, {@link AdversarialBandits} and the baselines, so every regret
 * table runs them through one loop ({@link BanditRuns}). {@link #learner} turns any policy into
 * a core {@link Learner} for single-state environments such as {@link io.github.zzuegg.brainiac.rl.Bandit}.
 */
public interface BanditPolicy {

    /** @return the number of arms */
    int arms();

    /** @return the arm to pull next */
    int select();

    /**
     * Records the reward of a pull.
     *
     * @param arm the arm pulled
     * @param reward its reward
     */
    void update(int arm, double reward);

    /**
     * The policy as a core learner: every environment step is one pull of the action's arm,
     * and greedy acting pulls the arm with the most pulls so far.
     *
     * @param seed the seed of the environment resets
     * @return the learner
     */
    default Learner learner(long seed) {
        BanditPolicy policy = this;
        int[] counts = new int[arms()];
        SplittableRandom random = new SplittableRandom(seed);
        return new Learner() {
            @Override
            public float[] act(float[] observation, boolean explore) {
                if (explore) {
                    return new float[] {policy.select()};
                }
                int best = 0;
                for (int i = 1; i < counts.length; i++) {
                    if (counts[i] > counts[best]) {
                        best = i;
                    }
                }
                return new float[] {best};
            }

            @Override
            public RlResult fit(Environment env, long steps, RlListener listener) {
                if (!(env.actionSpace() instanceof Space.Discrete d) || d.n() != policy.arms()) {
                    throw new IllegalArgumentException("needs " + policy.arms() + " discrete actions");
                }
                Tracker t = new Tracker(listener);
                env.reset(random.split());
                while (t.steps() < steps) {
                    int a = policy.select();
                    Environment.Step s = env.step(a);
                    policy.update(a, s.reward());
                    counts[a]++;
                    if (t.step(s)) {
                        env.reset(random.split());
                    }
                }
                return t.result();
            }

            @Override
            public String toString() {
                return policy.toString();
            }
        };
    }
}
