package io.github.zzuegg.brainiac.audio;

import java.util.List;
import java.util.function.Consumer;

/**
 * Speech in, text out. Implementations keep their model loaded between calls; a call
 * runs on the caller's thread, and one instance serves one call at a time.
 *
 * <pre>{@code
 * try (SpeechToText stt = SpeechModels.speechToText(Path.of("whisper-base"), Device.cpu(), DType.FLOAT32)) {
 *     Transcription t = stt.transcribe(Wav.read(Path.of("talk.wav")), TranscriptionOptions.defaults(),
 *             segment -> System.out.println(segment.start() + ": " + segment.text()));
 * }
 * }</pre>
 */
public interface SpeechToText extends AutoCloseable {

    /** The sample rate {@link #transcribe(float[], TranscriptionOptions, Consumer)} takes. */
    int SAMPLE_RATE = 16_000;

    /**
     * Transcribes 16 kHz mono audio.
     *
     * @param samples 16 kHz mono samples in {@code [-1, 1]}
     * @param options how to transcribe
     * @param onSegment called with each segment as soon as it is decoded, or {@code null}
     * @return the transcription
     * @throws java.util.concurrent.CancellationException if {@code options.cancelled()} said so
     * @throws IllegalArgumentException if an option cannot be served, such as an unknown language
     */
    Transcription transcribe(float[] samples, TranscriptionOptions options, Consumer<Transcription.Segment> onSegment);

    /**
     * Transcribes audio at any sample rate.
     *
     * @param audio the audio, resampled to 16 kHz if needed
     * @param options how to transcribe
     * @param onSegment called with each segment as soon as it is decoded, or {@code null}
     * @return the transcription
     */
    default Transcription transcribe(Audio audio, TranscriptionOptions options,
            Consumer<Transcription.Segment> onSegment) {
        return transcribe(audio.resample(SAMPLE_RATE).samples(), options, onSegment);
    }

    /**
     * Transcribes audio with the default options.
     *
     * @param audio the audio
     * @return the transcription
     */
    default Transcription transcribe(Audio audio) {
        return transcribe(audio, TranscriptionOptions.defaults(), null);
    }

    /**
     * The languages the model understands.
     *
     * @return codes such as {@code en} and {@code de}
     */
    List<String> languages();

    /**
     * Whether the model can translate speech to English.
     *
     * @return {@code true} if {@link TranscriptionOptions.Task#TRANSLATE} works
     */
    boolean canTranslate();

    /** Frees the model's memory. */
    @Override
    void close();
}
