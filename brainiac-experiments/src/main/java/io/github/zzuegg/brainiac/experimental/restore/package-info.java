/**
 * Image restoration blocks, written as core code: super-resolution networks (SRCNN, ESPCN with
 * core {@code PixelShuffle}, EDSR, SPAN), supervised denoisers (DnCNN, FFDNet) with classical
 * reference filters, self-supervised denoising (Noise2Noise, Noise2Void, Neighbor2Neighbor,
 * zero-shot Noise2Noise, Median2Median) and plug-and-play / unrolled deblurring.
 */
package io.github.zzuegg.brainiac.experimental.restore;
