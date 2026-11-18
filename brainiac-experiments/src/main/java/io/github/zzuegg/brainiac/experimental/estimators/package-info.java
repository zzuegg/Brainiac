/**
 * Experimental statistical estimators, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.estimators}: gradient estimators for discrete random variables
 * (REINFORCE with leave-one-out baselines, REBAR, RELAX, ARM, DisARM and the Stein-operator
 * estimator RODEO), expectation propagation for Gaussian-process classification, nested sampling
 * for Bayesian evidence, mutual-information estimators (k-nearest-neighbour and variational
 * bounds) and goodness-of-fit and two-sample tests (kernel and finite-set Stein discrepancies,
 * classifier two-sample tests). Once one proves itself it moves to core unchanged apart from
 * dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.estimators;
