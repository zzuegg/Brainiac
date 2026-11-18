/**
 * Experimental post-training building blocks, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.align}: preference losses beyond the DPO family of
 * {@code experimental.train.PreferenceLosses} (chi-squared and squared-log link functions,
 * length-regularized DPO, exploratory online DPO), reward-model ensembles and weight-averaged
 * reward models against over-optimization, best-of-N distillation, general (non-transitive)
 * preference models and self-play preference optimization, and process supervision for
 * step-by-step reasoning: Monte Carlo step values, rollout step labels, progress (advantage)
 * rewards, implicit process rewards and step-level preference pairs. Once one proves itself it
 * moves to core unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.align;
