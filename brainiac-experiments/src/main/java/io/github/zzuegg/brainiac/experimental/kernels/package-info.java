/**
 * Experimental CPU kernels and kernel tooling, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.kernels}: a timing harness, a schedule space of tiled matrix
 * multiplications with learned cost models and autotuners, fast algorithms (Winograd convolution,
 * Strassen multiplication), a game for searching low-rank matrix-multiplication decompositions,
 * tensor-graph rewriting with equality saturation, and sparse matrix kernels (CSR, block-sparse,
 * {@code n:m}). Each kernel is checked against the core CPU backend; once one proves itself it
 * moves to core unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.kernels;
