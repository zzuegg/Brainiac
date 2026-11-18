/**
 * Experimental randomised numerical linear algebra, written as if it were part of a core package
 * {@code io.github.zzuegg.brainiac.randla}: randomised SVD with power iterations,
 * sketch-and-precondition least squares (Gaussian, CountSketch and SRHT sketches with
 * preconditioned LSQR) and stochastic Lanczos quadrature for log-determinants, spectral densities
 * and Hessian spectra. Once one proves itself it moves to core unchanged apart from dropping
 * {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.randla;
