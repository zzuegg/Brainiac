package io.github.zzuegg.brainiac.audio;

import java.util.function.BooleanSupplier;

/**
 * How to speak.
 *
 * @param voice a {@link Voice#id()}, or {@code null} for the model's first voice. Models that
 *     {@linkplain TextToSpeech#canMixVoices() mix voices} also take a blend such as
 *     {@code "af_heart*0.7+am_adam*0.3"} (weights are normalized) or {@code "af_bella,af_sarah"}
 *     (equal parts)
 * @param speed the speaking rate, 1 for normal, 2 for twice as fast
 * @param seed the seed of the model's randomness (VITS samples durations and pitch), so the
 *     same text sounds the same
 * @param cancelled polled between sentences: when it returns {@code true}, synthesis stops
 *     with a {@link java.util.concurrent.CancellationException}
 * @param reference a few seconds of a voice to imitate, for models that {@linkplain
 *     TextToSpeech#canClone() clone voices}, or {@code null}
 * @param referenceText what is said in {@code reference}, or {@code null} to transcribe it
 * @param instruction how to speak, in words, for models that {@linkplain
 *     TextToSpeech#followsInstructions() follow instructions} (such as {@code "Speak slowly, in
 *     a whisper"}, or for a model that designs voices, the voice itself), or {@code null}
 * @param language the language of the text, for models that {@linkplain TextToSpeech#languages()
 *     name languages}, or {@code null} to let the model tell
 */
public record SpeechOptions(String voice, float speed, long seed, BooleanSupplier cancelled, Audio reference,
        String referenceText, String instruction, String language) {

    /**
     * Checks the values.
     *
     * @param voice the voice or {@code null}
     * @param speed the rate, positive
     * @param seed the seed
     * @param cancelled the cancellation check, or {@code null} for never
     * @param reference a voice sample, or {@code null}
     * @param referenceText its transcript, or {@code null}
     * @param instruction how to speak, or {@code null}
     * @param language the text's language, or {@code null}
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
        if (referenceText != null && referenceText.isBlank()) {
            referenceText = null;
        }
        if (instruction != null && instruction.isBlank()) {
            instruction = null;
        }
        if (language != null && language.isBlank()) {
            language = null;
        }
    }

    /**
     * Options with a voice sample, without an instruction or language.
     *
     * @param voice the voice or {@code null}
     * @param speed the rate, positive
     * @param seed the seed
     * @param cancelled the cancellation check, or {@code null} for never
     * @param reference a voice sample, or {@code null}
     * @param referenceText its transcript, or {@code null}
     */
    public SpeechOptions(String voice, float speed, long seed, BooleanSupplier cancelled, Audio reference,
            String referenceText) {
        this(voice, speed, seed, cancelled, reference, referenceText, null, null);
    }

    /**
     * Options without a voice sample.
     *
     * @param voice the voice or {@code null}
     * @param speed the rate, positive
     * @param seed the seed
     * @param cancelled the cancellation check, or {@code null} for never
     */
    public SpeechOptions(String voice, float speed, long seed, BooleanSupplier cancelled) {
        this(voice, speed, seed, cancelled, null, null, null, null);
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
        return new SpeechOptions(id, speed, seed, cancelled, reference, referenceText, instruction, language);
    }

    /**
     * Sets the speed.
     *
     * @param rate 1 for normal
     * @return new options
     */
    public SpeechOptions speed(float rate) {
        return new SpeechOptions(voice, rate, seed, cancelled, reference, referenceText, instruction, language);
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return new options
     */
    public SpeechOptions seed(long s) {
        return new SpeechOptions(voice, speed, s, cancelled, reference, referenceText, instruction, language);
    }

    /**
     * Sets the cancellation check.
     *
     * @param check returns {@code true} to stop
     * @return new options
     */
    public SpeechOptions cancelled(BooleanSupplier check) {
        return new SpeechOptions(voice, speed, seed, check, reference, referenceText, instruction, language);
    }

    /**
     * Sets the voice to imitate, for models that clone voices.
     *
     * @param sample a few seconds (about 5 to 15) of clean speech, or {@code null}
     * @param transcript what the sample says, or {@code null} to have it transcribed
     * @return new options
     */
    public SpeechOptions reference(Audio sample, String transcript) {
        return new SpeechOptions(voice, speed, seed, cancelled, sample, transcript, instruction, language);
    }

    /**
     * Sets how to speak, for models that follow instructions.
     *
     * @param text such as {@code "Excited, a little out of breath"}, or {@code null}
     * @return new options
     */
    public SpeechOptions instruction(String text) {
        return new SpeechOptions(voice, speed, seed, cancelled, reference, referenceText, text, language);
    }

    /**
     * Sets the language of the text, for models that name languages.
     *
     * @param name one of {@link TextToSpeech#languages()}, or {@code null} to let the model tell
     * @return new options
     */
    public SpeechOptions language(String name) {
        return new SpeechOptions(voice, speed, seed, cancelled, reference, referenceText, instruction, name);
    }
}
