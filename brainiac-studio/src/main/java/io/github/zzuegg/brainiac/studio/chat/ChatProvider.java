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
     * Returns whether {@link #video} works here: listed video models of other providers (such
     * as a cloud's) are shown but cannot make clips through the studio yet.
     *
     * @return true if this provider makes clips
     */
    default boolean makesVideo() {
        return false;
    }

    /**
     * Makes a video clip with one of this provider's video models, step by step. This PC does
     * ({@link LocalProvider}); a worker host can too.
     *
     * @param model the model's id within this provider
     * @param prompt what the clip shows
     * @param settings the clip's shape, length and sampling
     * @param listener hears each step and can stop the clip
     * @return the clip
     * @throws IOException if the model cannot be loaded or the provider fails
     * @throws InterruptedException if interrupted while waiting
     * @throws java.util.concurrent.CancellationException if the listener stopped it
     */
    default LocalVideos.Clip video(String model, String prompt, LocalVideos.Settings settings,
            LocalVideos.Listener listener) throws IOException, InterruptedException {
        throw new IOException(name() + " does not make videos");
    }
}
