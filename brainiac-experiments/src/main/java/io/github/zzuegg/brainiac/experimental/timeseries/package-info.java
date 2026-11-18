/**
 * Experimental time-series building blocks, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.timeseries}: classical state-space forecasters (exponential
 * smoothing, ARIMA), deep forecasters (decomposition with auto-correlation or Fourier
 * attention, the inverted transformer, TimesNet, N-HiTS, TiDE, a Chronos-style tokenised
 * forecaster), conformal prediction intervals, series classifiers and representations (random
 * convolution kernels, TS2Vec), and signal front ends (sinc and Gabor filterbanks, PCEN, the
 * wavelet scattering transform, dual-path RNN separation). Once a block proves itself it moves
 * to core unchanged apart from dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.timeseries;
