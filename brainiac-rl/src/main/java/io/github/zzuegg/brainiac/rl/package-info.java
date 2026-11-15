/**
 * Reinforcement learning: {@link io.github.zzuegg.brainiac.rl.Environment environments}
 * that an agent acts in, and agents that learn from the rewards they get.
 *
 * <pre>{@code
 * Environment env = new CartPole();
 * Ppo ppo = Ppo.discrete(env).seed(1);
 * RlResult result = ppo.fit(env, 50_000, RlListener.every(20));
 * Rl.evaluate(ppo, env, 10, 1);       // mean return over 10 greedy episodes
 * }</pre>
 */
package io.github.zzuegg.brainiac.rl;
