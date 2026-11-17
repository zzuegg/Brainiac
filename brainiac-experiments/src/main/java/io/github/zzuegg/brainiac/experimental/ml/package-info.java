/**
 * Experimental classic machine learning estimators, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.ml}: kernel machines, Gaussian processes, tree ensembles,
 * clustering, component analysis, neighbour embeddings, hidden Markov models, factor graphs,
 * topic models, self-organising maps and state-space filters. They run on the host in double
 * precision without autograd, next to core {@code GradientBoostedTrees}, and serve as the
 * classic baselines the neural experiments are measured against. Once one proves itself it
 * moves to core unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.ml;
