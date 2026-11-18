package io.github.zzuegg.brainiac.audio;

import java.util.List;
import java.util.function.Consumer;

/**
 * Text in, speech out. Implementations keep their model loaded between calls; a call runs
 * on the caller's thread, and one instance serves one call at a time.
 *
 * <pre>{@code
 * try (TextToSpeech tts = SpeechModels.textToSpeech(Path.of("mms-tts-eng"), Device.cpu(), DType.FLOAT32)) {
 *     Audio speech = tts.synthesize("Hello there.", SpeechOptions.defaults(), chunk -> play(chunk));
 *     Wav.write(speech, Path.of("hello.wav"));
 * }
 * }</pre>
 */
public interface TextToSpeech extends AutoCloseable {

    /**
     * The sample rate of the audio this model makes.
     *
     * @return samples per second
     */
    int sampleRate();

    /**
     * The voices this model can speak with; the first is the default.
     *
     * @return at least one voice
     */
    List<Voice> voices();

    /**
     * Speaks text. Long text is spoken a sentence at a time; each sentence's audio goes to
     * {@code onChunk} as soon as it is ready, so playback can start early.
     *
     * @param text the text
     * @param options the voice, speed and cancellation
     * @param onChunk called with each piece of audio in order, or {@code null}
     * @return all the audio
     * @throws java.util.concurrent.CancellationException if {@code options.cancelled()} said so
     * @throws IllegalArgumentException if the voice is unknown or the text has nothing the model can say
     */
    Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk);

    /**
     * Speaks text with the default options.
     *
     * @param text the text
     * @return the audio
     */
    default Audio synthesize(String text) {
        return synthesize(text, SpeechOptions.defaults(), null);
    }

    /**
     * Whether {@link SpeechOptions#voice()} may blend voices, such as
     * {@code "af_heart*0.7+am_adam*0.3"}.
     *
     * @return {@code true} if voices mix
     */
    default boolean canMixVoices() {
        return false;
    }

    /**
     * Whether the model imitates the voice of {@link SpeechOptions#reference()}.
     *
     * @return {@code true} if it clones voices
     */
    default boolean canClone() {
        return false;
    }

    /**
     * The expression tags the model understands inside the text, such as {@code <laugh>}.
     *
     * @return the tags, empty if none
     */
    default List<String> tags() {
        return List.of();
    }

    /** Frees the model's memory. */
    @Override
    void close();
}
