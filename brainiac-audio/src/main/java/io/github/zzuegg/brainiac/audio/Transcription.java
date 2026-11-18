package io.github.zzuegg.brainiac.audio;

import java.util.List;

/**
 * What speech-to-text heard.
 *
 * @param text all the text, the segments' texts joined
 * @param language the language code of the speech, detected or as asked ({@code en}, {@code de}, ...)
 * @param segments the text in timed pieces
 * @param seconds the length of the audio
 */
public record Transcription(String text, String language, List<Segment> segments, double seconds) {

    /**
     * Copies the segments.
     *
     * @param text all the text
     * @param language the language code
     * @param segments the timed pieces
     * @param seconds the length of the audio
     */
    public Transcription {
        segments = List.copyOf(segments);
    }

    /**
     * A timed piece of text.
     *
     * @param start where it starts, in seconds from the start of the audio
     * @param end where it ends, in seconds
     * @param text what was said
     * @param avgLogProb the mean log probability of its tokens (below -1 is unsure)
     * @param noSpeechProb the model's probability that its window holds no speech
     */
    public record Segment(double start, double end, String text, double avgLogProb, double noSpeechProb) {
    }
}
