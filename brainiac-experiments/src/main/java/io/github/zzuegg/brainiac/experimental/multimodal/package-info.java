/**
 * Multimodal fusion: bridges between a frozen language model and a vision encoder (gated
 * cross-attention with a perceiver resampler, a Q-Former style query transformer, a linear
 * patch projector), one transformer trained with next-token prediction and diffusion
 * (Transfusion), and bottleneck fusion with modality dropout.
 */
package io.github.zzuegg.brainiac.experimental.multimodal;
