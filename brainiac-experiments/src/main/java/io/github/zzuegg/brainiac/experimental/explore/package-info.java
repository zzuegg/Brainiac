/**
 * Experimental exploration and structured-reward RL blocks, written as if they were part of a
 * core package {@code io.github.zzuegg.brainiac.explore}: deep exploration (bootstrapped DQN
 * with randomised priors, RLSVI) on the Deep Sea and RiverSwim chains, count-based bonuses for
 * large state spaces (SimHash, learned-code SimHash, density pseudo-counts), unsupervised skill
 * discovery (DIAYN, METRA, contrastive successor features), goal-conditioned hierarchies (HIRO
 * with off-policy correction, HAC-style hindsight) on a continuous point maze, and
 * preference-based RL from segment comparisons (Christiano et al., PEBBLE, SURF), with a TD3
 * that is driven step by step and can relabel its replay buffer.
 */
package io.github.zzuegg.brainiac.experimental.explore;
