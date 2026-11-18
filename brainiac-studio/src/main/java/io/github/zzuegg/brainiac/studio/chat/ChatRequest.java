package io.github.zzuegg.brainiac.studio.chat;

import java.util.List;
import java.util.Objects;

/**
 * What to send to a model.
 *
 * @param model the provider's own model id, such as {@code claude-opus-5-5}
 * @param system instructions for the whole conversation, or an empty string
 * @param messages the conversation so far, ending with the user's turn
 * @param maxTokens the most tokens the answer may have
 * @param key the name of the API key to use, or null to try the provider's keys in order
 * @param temperature how freely the model chooses its words, from 0 (always the likeliest) to 2,
 *        or null for the model's own default; providers whose API rejects it leave it out
 */
public record ChatRequest(String model, String system, List<ChatMessage> messages, int maxTokens, String key,
        Float temperature) {

    /**
     * A request answered at the model's own temperature.
     *
     * @param model the provider's own model id
     * @param system instructions for the whole conversation, or an empty string
     * @param messages the conversation so far, ending with the user's turn
     * @param maxTokens the most tokens the answer may have
     * @param key the name of the API key to use, or null to try the provider's keys in order
     */
    public ChatRequest(String model, String system, List<ChatMessage> messages, int maxTokens, String key) {
        this(model, system, messages, maxTokens, key, null);
    }

    /** Checks the request. */
    public ChatRequest {
        Objects.requireNonNull(model, "model");
        system = system == null ? "" : system;
        messages = List.copyOf(messages);
        if (messages.isEmpty() || !"user".equals(messages.getLast().role())) {
            throw new IllegalArgumentException("the conversation must end with a user message");
        }
        if (maxTokens < 1) {
            throw new IllegalArgumentException("maxTokens < 1");
        }
        if (temperature != null && !(temperature >= 0 && temperature <= 2)) {
            throw new IllegalArgumentException("temperature must be between 0 and 2");
        }
    }
}
