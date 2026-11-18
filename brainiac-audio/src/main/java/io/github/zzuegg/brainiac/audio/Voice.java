package io.github.zzuegg.brainiac.audio;

/**
 * A voice a text-to-speech model can speak with.
 *
 * @param id what {@link SpeechOptions#voice()} takes
 * @param name a name to show
 * @param language the language code it speaks ({@code en}, ...), or {@code null} if unknown
 */
public record Voice(String id, String name, String language) {
}
