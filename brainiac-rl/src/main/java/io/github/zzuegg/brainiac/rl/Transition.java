package io.github.zzuegg.brainiac.rl;

/**
 * One step of experience, as stored in a {@link ReplayBuffer}.
 *
 * @param observation what the agent saw
 * @param action what it did
 * @param reward the reward (for n-step transitions, the discounted sum of n rewards)
 * @param nextObservation what it saw next
 * @param terminated whether the next state is terminal (no value follows)
 * @param discount the factor for the value of the next state: {@code gamma}, or
 *     {@code gamma^n} for n-step transitions
 */
public record Transition(float[] observation, float[] action, float reward, float[] nextObservation,
        boolean terminated, float discount) {
}
