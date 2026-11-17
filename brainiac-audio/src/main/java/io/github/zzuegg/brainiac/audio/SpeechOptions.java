package io.github.zzuegg.brainiac.audio;

import java.util.function.BooleanSupplier;

/**
 * How to speak.
 *
 * @param voice a {@link Voice#id()}, or {@code null} for the model's first voice
 * @param speed the speaking rate, 1 for normal, 2 for twice as fast
 * @param seed the seed of the model's randomness (VITS samples durations and pitch), so the
 *     same text sounds the same
 * @param cancelled polled between sentences: when it returns {@code true}, synthesis stops
 *     with a {@link java.util.concurrent.CancellationException}
 */
public record SpeechOptions(String voice, float speed, long seed, BooleanSupplier cancelled) {

    /**
     * Checks the values.
     *
     * @param voice the voice or {@code null}
     * @param speed the rate, positive
     * @param seed the seed
     * @param cancelled the cancellation check, or {@code null} for never
     */
    public SpeechOptions {
        if (!(speed > 0) || speed > 10) {
            throw new IllegalArgumentException("speed must be above 0 and at most 10, got " + speed);
        }
        if (cancelled == null) {
            cancelled = () -> false;
        }
        if (voice != null && voice.isBlank()) {
            voice = null;
        }
    }

    /**
     * The defaults: the first voice at normal speed.
     *
     * @return the options
     */
    public static SpeechOptions defaults() {
        return new SpeechOptions(null, 1, 0, null);
    }

    /**
     * Sets the voice.
     *
     * @param id a voice id, or {@code null}
     * @return new options
     */
    public SpeechOptions voice(String id) {
        return new SpeechOptions(id, speed, seed, cancelled);
    }

    /**
     * Sets the speed.
     *
     * @param rate 1 for normal
     * @return new options
     */
    public SpeechOptions speed(float rate) {
        return new SpeechOptions(voice, rate, seed, cancelled);
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return new options
     */
    public SpeechOptions seed(long s) {
        return new SpeechOptions(voice, speed, s, cancelled);
    }

    /**
     * Sets the cancellation check.
     *
     * @param check returns {@code true} to stop
     * @return new options
     */
    public SpeechOptions cancelled(BooleanSupplier check) {
        return new SpeechOptions(voice, speed, seed, check);
    }
}
