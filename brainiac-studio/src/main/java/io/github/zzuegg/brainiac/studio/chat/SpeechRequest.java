package io.github.zzuegg.brainiac.studio.chat;

import java.util.Objects;

/**
 * What to say, and how.
 *
 * @param model the provider's own model id
 * @param text the words
 * @param voice the voice's name, as {@link SpeechOptions#voices()} lists it
 * @param speed 1 for normal pace; models without a pace setting ignore it
 * @param instructions how to say it, such as "calm and slow", or an empty string
 * @param format {@code mp3}, {@code wav} or {@code opus}; models with one format ignore it
 * @param key the name of the API key to use, or null to try the provider's keys in order
 */
public record SpeechRequest(String model, String text, String voice, double speed, String instructions, String format,
        String key) {

    /** Checks the request. */
    public SpeechRequest {
        Objects.requireNonNull(model, "model");
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("there is nothing to say");
        }
        if (voice == null || voice.isBlank()) {
            throw new IllegalArgumentException("choose a voice");
        }
        if (!(speed >= 0.25 && speed <= 4)) {
            throw new IllegalArgumentException("speed is 0.25 to 4");
        }
        instructions = instructions == null ? "" : instructions;
        format = format == null ? "mp3" : format;
        if (!format.equals("mp3") && !format.equals("wav") && !format.equals("opus")) {
            throw new IllegalArgumentException("format is mp3, wav or opus");
        }
    }
}
