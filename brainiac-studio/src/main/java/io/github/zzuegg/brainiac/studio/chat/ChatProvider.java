package io.github.zzuegg.brainiac.studio.chat;

import java.io.IOException;
import java.util.List;

/** A source of chat models: this PC, or a remote provider reached with the user's key. */
public interface ChatProvider {

    /**
     * Returns the provider's id, used in model keys.
     *
     * @return the id, such as {@code anthropic}
     */
    String id();

    /**
     * Returns the name to show.
     *
     * @return the name
     */
    String name();

    /**
     * Returns whether the provider can be used now, for example because it has a key.
     *
     * @return true if ready
     */
    boolean ready();

    /**
     * Returns whether models run on this machine, so nothing leaves it.
     *
     * @return true for local providers
     */
    default boolean local() {
        return false;
    }

    /**
     * Lists the models. Remote providers ask their own model-list endpoint.
     *
     * @return the models
     * @throws IOException if the list cannot be read
     * @throws InterruptedException if interrupted while waiting
     */
    List<ModelInfo> models() throws IOException, InterruptedException;

    /**
     * Picks the model {@code auto} should use for chat, from this provider's list.
     *
     * @param models this provider's models
     * @return the model, or null if none can chat
     */
    default ModelInfo preferred(List<ModelInfo> models) {
        return models.stream().filter(ModelInfo::chats).findFirst().orElse(null);
    }

    /**
     * Picks the model {@code auto} should use for a medium, from this provider's list.
     *
     * @param medium {@code text}, {@code image} or {@code speech}
     * @param models this provider's models
     * @return the model, or null if none fits
     */
    default ModelInfo preferredFor(String medium, List<ModelInfo> models) {
        if (medium.equals("text")) {
            return preferred(models);
        }
        return models.stream().filter(m -> m.media().contains(medium)).findFirst().orElse(null);
    }

    /**
     * Writes an answer, passing it to the listener as it arrives.
     *
     * @param request the conversation
     * @param listener receives the answer
     * @return how the answer ended
     * @throws IOException if the provider fails or the listener stops the answer
     * @throws InterruptedException if interrupted while waiting
     */
    ChatResult chat(ChatRequest request, ChatListener listener) throws IOException, InterruptedException;

    /**
     * Returns whether this provider's models can be held to a grammar: those run on a machine of
     * the user's can, a provider's own API cannot.
     *
     * @return whether they can
     */
    default boolean takesGrammar() {
        return false;
    }

    /**
     * Returns whether this provider runs image and speech models on a machine of the user's:
     * this PC or a worker host. Such models are drawn with step by step, speak and write down
     * recordings through {@link #draw}, {@link #speak} and {@link #transcribe}.
     *
     * @return whether it does
     */
    default boolean runsMedia() {
        return false;
    }

    /**
     * Draws pictures with one of this provider's image models.
     *
     * @param model the model's id
     * @param prompt what to draw
     * @param settings size, steps, guidance, seed and sampler
     * @param listener hears each step and can stop the drawing
     * @return the pictures with their seeds
     * @throws IOException if it has no such model or the drawing fails
     * @throws InterruptedException if interrupted while waiting
     */
    default List<LocalImages.Picture> draw(String model, String prompt, LocalImages.Settings settings,
            LocalImages.Listener listener) throws IOException, InterruptedException {
        throw new IOException(name() + " does not draw pictures");
    }

    /**
     * Speaks a text with one of this provider's speech models.
     *
     * @param model the model's id
     * @param text what to say
     * @param voice the voice, or an empty string for the model's own
     * @param speed 1 for the usual speed
     * @param stop asked now and then whether to give up
     * @return the audio, a WAV
     * @throws IOException if it has no such model or the speaking fails
     * @throws InterruptedException if interrupted while waiting
     */
    default Media speak(String model, String text, String voice, double speed, java.util.function.BooleanSupplier stop)
            throws IOException, InterruptedException {
        throw new IOException(name() + " does not speak");
    }

    /**
     * Writes down a recording with one of this provider's transcribing models.
     *
     * @param model the model's id
     * @param wav the recording, a WAV file
     * @param listening the language and what else to do
     * @param onSegment hears each part as it is written down
     * @param stop asked now and then whether to give up
     * @return what was said
     * @throws IOException if it has no such model or the transcribing fails
     * @throws InterruptedException if interrupted while waiting
     */
    default io.github.zzuegg.brainiac.audio.Transcription transcribe(String model, byte[] wav,
            LocalSpeech.Listening listening,
            java.util.function.Consumer<io.github.zzuegg.brainiac.audio.Transcription.Segment> onSegment,
            java.util.function.BooleanSupplier stop) throws IOException, InterruptedException {
        throw new IOException(name() + " does not transcribe");
    }

    /**
     * Returns the voices and other choices of one of this provider's speech models, as the page
     * takes them.
     *
     * @param model the model's id
     * @return them, or null when the model has none to tell
     */
    default java.util.Map<String, Object> voiceOptions(String model) {
        return null;
    }
}
