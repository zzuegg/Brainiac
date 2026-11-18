/**
 * Experimental simulation-based inference, written as if it were part of a core package
 * {@code io.github.zzuegg.brainiac.sbi}: conditional density estimators (mixture density
 * networks and coupling flows), neural likelihood and ratio estimation, flow-matching and
 * score-based posterior estimation, sequential and truncated neural posterior estimation, ABC
 * summary statistics, posterior diagnostics (SBC, expected coverage, TARP, local C2ST),
 * misspecification-robust inference and amortised model comparison. Once a block proves itself
 * it moves to core unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.sbi;
