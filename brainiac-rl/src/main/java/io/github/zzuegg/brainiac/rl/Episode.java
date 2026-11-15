package io.github.zzuegg.brainiac.rl;

/**
 * One finished training episode.
 *
 * @param number the episode number, from 1
 * @param steps the steps it lasted
 * @param totalReward the sum of its rewards (the undiscounted return)
 * @param totalSteps the environment steps taken in training so far, including this episode
 */
public record Episode(int number, int steps, double totalReward, long totalSteps) {
}
