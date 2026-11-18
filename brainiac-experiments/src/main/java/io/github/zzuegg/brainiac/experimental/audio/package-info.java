/**
 * Trainable audio generators, written as core code so they can later move to
 * {@code brainiac-audio} (whose pretrained inference models they do not touch): Griffin-Lim
 * phase reconstruction, a HiFi-GAN-style GAN vocoder with multi-period and multi-scale
 * discriminators, DiffWave/WaveGrad diffusion vocoders, monotonic alignment search and
 * flow-matching text-to-speech (Matcha-style durations and F5-style filler padding), latent
 * audio diffusion over a mel VAE, and finite scalar quantization for audio and video
 * tokenizers.
 */
package io.github.zzuegg.brainiac.experimental.audio;
