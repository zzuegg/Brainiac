/**
 * Experimental model compression, written as if it were part of a core package
 * {@code io.github.zzuegg.brainiac.compress}: pruning schedules and criteria (gradual magnitude,
 * movement, WoodFisher, pruning at initialisation), structured pruning of language models,
 * convolutional networks and mixtures of experts, distillation losses, weight sharing with
 * Huffman coding, Hessian-aware mixed precision, quantisation-aware activations, tensor
 * decompositions of layers, and token merging. The blocks work on core {@code Var} weights and
 * plug into core optimizers, losses and layers; once one proves itself it moves to core
 * unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.compress;
