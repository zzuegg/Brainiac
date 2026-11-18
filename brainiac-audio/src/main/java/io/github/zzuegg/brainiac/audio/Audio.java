package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.nn.Signal;

import java.util.Objects;

/**
 * Mono audio: samples in {@code [-1, 1]} and their sample rate.
 *
 * @param samples the samples
 * @param sampleRate samples per second
 */
public record Audio(float[] samples, int sampleRate) {

    /**
     * Checks the values.
     *
     * @param samples the samples
     * @param sampleRate samples per second, positive
     */
    public Audio {
        Objects.requireNonNull(samples, "samples");
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("the sample rate must be positive, got " + sampleRate);
        }
    }

    /**
     * Returns the length.
     *
     * @return the duration in seconds
     */
    public double seconds() {
        return (double) samples.length / sampleRate;
    }

    /**
     * Returns this audio at another sample rate (band-limited resampling).
     *
     * @param rate the new sample rate
     * @return this audio if it already has that rate, otherwise a resampled copy
     */
    public Audio resample(int rate) {
        if (rate == sampleRate) {
            return this;
        }
        return new Audio(Signal.resample(samples, sampleRate, rate), rate);
    }

    /**
     * Encodes this audio as a 16-bit PCM WAV file.
     *
     * @return the file's bytes
     */
    public byte[] toWav() {
        return Wav.encode(this);
    }
}
