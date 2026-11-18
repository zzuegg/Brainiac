/**
 * Experimental deep RL recipes of 2023-2025, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.deeprl}: off-policy actor-critics that scale (CrossQ, BRO,
 * SimbaV2 on a shared SAC engine, TD7 and MR.Q with loss-adjusted prioritised replay),
 * value-based recipes for discrete actions (PQN without replay or targets, self-predictive
 * representations and BBF, DrQ augmentation, EfficientZero), plasticity tools (dormant neurons,
 * ReDo, plasticity injection, resets) and scaling studies (multi-skip residual networks, value
 * scaling grids), plus the learning-curve harness with interquartile means and bootstrap
 * intervals that every comparison reports through, a continuous mountain car and pixel
 * renderings of small grids and the pendulum. Once one proves itself it moves to core unchanged
 * apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.deeprl;
