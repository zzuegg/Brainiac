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

    /** Frees the model's memory. */
    @Override
    void close();
}
