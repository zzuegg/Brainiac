/**
 * Experimental code-model building blocks, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.code}: line-by-line execution traces of list programs and a
 * trace tokenizer, a language model trained on trace text, a program sampler conditioned on
 * input/output examples, a generate-run-repair loop with budget accounting, and
 * execution-based reranking (behaviour clustering, CodeT dual agreement and a learned
 * execution verifier). Once one proves itself it moves to core unchanged apart from dropping
 * {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.code;
