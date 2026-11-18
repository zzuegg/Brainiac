package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;

import java.util.Objects;

/**
 * The mean-field actor-critic (Yang et al., ICML 2018,
 * <a href="https://arxiv.org/abs/1802.05438">arXiv:1802.05438</a>): a shared policy
 * {@code pi(a | o_i)} trained against {@link MeanFieldQ}'s critic {@code Q(o_i, a_i, abar_i)},
 * by the expected policy gradient {@code grad sum_a pi(a | o_i) Q(o_i, a, abar_i)} with a small
 * entropy bonus; the critic's target averages over the actor's next-state policy instead of a
 * Boltzmann policy.
 */
public final class MeanFieldActorCritic {

    private final MeanFieldQ learner;

    private MeanFieldActorCritic(MeanFieldQ learner) {
        this.learner = learner;
    }

    /**
     * @param env the environment (for sizes and neighbourhoods)
     * @param networks builds the critic and the actor
     * @param seed the seed
     * @return the learner
     */
    public static MeanFieldActorCritic of(LocalRewardEnvironment env, RlNetworkFactory networks, long seed) {
        return new MeanFieldActorCritic(new MeanFieldQ(Objects.requireNonNull(env), Objects.requireNonNull(networks),
                true, seed));
    }

    /** @param discount gamma @return this learner */
    public MeanFieldActorCritic discount(float discount) {
        learner.discount(discount);
        return this;
    }

    /** @param seed the exploration seed @return this learner */
    public MeanFieldActorCritic seed(long seed) {
        learner.seed(seed);
        return this;
    }

    /** Forgets the last step's mean actions; call at the start of an episode. */
    public void beginEpisode() {
        learner.beginEpisode();
    }

    /**
     * @param observations one per agent
     * @param explore sample from the policy rather than take its most likely action
     * @return the joint action
     */
    public int[] act(float[][] observations, boolean explore) {
        return learner.act(observations, explore);
    }

    /**
     * @param env the environment
     * @param totalSteps joint steps
     * @return the mean team return per episode over training
     */
    public double fit(LocalRewardEnvironment env, long totalSteps) {
        return learner.fit(env, totalSteps);
    }

    /**
     * @param env the environment
     * @param seed the episode seed
     * @return the greedy team return
     */
    public double evaluate(LocalRewardEnvironment env, long seed) {
        return learner.evaluate(env, seed);
    }

    /** @return gradient updates taken */
    public long updates() {
        return learner.updates();
    }

    @Override
    public String toString() {
        return "MF-AC";
    }
}
