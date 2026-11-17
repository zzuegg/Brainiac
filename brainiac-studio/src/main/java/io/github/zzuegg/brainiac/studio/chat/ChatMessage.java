package io.github.zzuegg.brainiac.studio.chat;

import java.util.Objects;

/**
 * One turn of a conversation.
 *
 * @param role {@code user} or {@code assistant}
 * @param text what was said
 */
public record ChatMessage(String role, String text) {

    /** Checks the role and text. */
    public ChatMessage {
        if (!"user".equals(role) && !"assistant".equals(role)) {
            throw new IllegalArgumentException("role must be user or assistant, not " + role);
        }
        Objects.requireNonNull(text, "text");
    }
}
