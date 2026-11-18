/**
 * Experimental time-to-event (survival) building blocks, written as if they were part of a core
 * package {@code io.github.zzuegg.brainiac.survival}: Kaplan-Meier, Cox proportional hazards with
 * Breslow's baseline, the DeepSurv partial-likelihood loss, discrete-time neural hazards, DeepHit
 * with competing risks, random survival forests, and the metrics they are judged by (Harrell's and
 * time-dependent C-index, IPCW integrated Brier score). Once one proves itself it moves to core
 * unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.survival;
