/**
 * Experimental robustness tools, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.robust}: input-space adversarial attacks and adversarial
 * training (FGSM, PGD, TRADES, fast FGSM), certification (randomized smoothing, interval and
 * CROWN bound propagation, certified training), 1-Lipschitz layers (GroupSort, Björck and AOL
 * weights) and test-time adaptation to distribution shift (BN statistics, TENT, MEMO). Once one
 * proves itself it moves to core unchanged apart from dropping {@code .experimental} from the
 * package name.
 */
package io.github.zzuegg.brainiac.experimental.robust;
