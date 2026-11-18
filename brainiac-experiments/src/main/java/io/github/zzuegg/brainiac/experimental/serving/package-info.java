/**
 * Experimental LLM serving blocks: approximate inference over the KV cache (draft-guided
 * selection, cross-layer merging, sampled attention), prompt compression, speculative decoding
 * with distilled and multiple drafts, multi-adapter batching, fair scheduling and grammar mask
 * caches. Written as if they were part of a core serving package; once one proves itself it
 * moves there unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.serving;
