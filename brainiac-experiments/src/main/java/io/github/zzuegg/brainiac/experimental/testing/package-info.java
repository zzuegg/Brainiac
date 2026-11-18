/**
 * Experimental hypothesis-testing and selective-inference building blocks, written as if they
 * were part of a core package {@code io.github.zzuegg.brainiac.testing}: batch multiple testing
 * (Bonferroni, Holm, Benjamini-Hochberg/Yekutieli, Storey, e-BH), online FDR control (LOND,
 * LORD++, SAFFRON and their e-value versions), model-X knockoffs, stability selection,
 * conformal novelty detection with FDR control, and Gaussian graphical models (graphical lasso,
 * neighbourhood selection). Once one proves itself it moves to core unchanged apart from
 * dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.testing;
