/**
 * Experimental multi-armed bandit algorithms, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.bandit}: asymptotically and minimax optimal stochastic
 * policies (KL-UCB, Thompson sampling, MOSS, IMED), adversarial and best-of-both-worlds
 * policies (EXP3, EXP3-IX, Tsallis-INF), best-arm identification at fixed confidence and fixed
 * budget (successive elimination, LUCB, Track-and-Stop, top-two, Almost Tracking), dueling
 * bandits (RUCB, double Thompson sampling) and ranking bandits for cascade and position-based
 * click models, all behind the {@link io.github.zzuegg.brainiac.experimental.bandit.BanditPolicy}
 * interface or their own small select/update loops, with a regret runner.
 */
package io.github.zzuegg.brainiac.experimental.bandit;
