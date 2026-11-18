package io.github.zzuegg.brainiac.studio.chat;

import java.util.Base64;
import java.util.List;
import java.util.Objects;

/**
 * One turn of a conversation.
 *
 * @param role {@code user} or {@code assistant}
 * @param text what was said
 * @param images the pictures shown with it, for a model that can look at them; only the user shows pictures
 */
public record ChatMessage(String role, String text, List<Media> images) {

    /** The most pictures one message may carry. */
    public static final int MAX_IMAGES = 8;

    /** Checks the role, text and pictures. */
    public ChatMessage {
        if (!"user".equals(role) && !"assistant".equals(role)) {
            throw new IllegalArgumentException("role must be user or assistant, not " + role);
        }
        Objects.requireNonNull(text, "text");
        images = List.copyOf(images);
        if (!images.isEmpty() && !"user".equals(role)) {
            throw new IllegalArgumentException("only a user message carries pictures");
        }
        if (images.size() > MAX_IMAGES) {
            throw new IllegalArgumentException("a message carries at most " + MAX_IMAGES + " pictures");
        }
        for (Media m : images) {
            if (!List.of("image/png", "image/jpeg", "image/webp", "image/gif").contains(m.mime())) {
                throw new IllegalArgumentException("a picture must be PNG, JPEG, WebP or GIF, not " + m.mime());
            }
        }
    }

    /**
     * A turn of words only.
     *
     * @param role {@code user} or {@code assistant}
     * @param text what was said
     */
    public ChatMessage(String role, String text) {
        this(role, text, List.of());
    }

    /**
     * Returns a picture as base64, the way every provider takes it.
     *
     * @param image the picture
     * @return its bytes in base64
     */
    static String base64(Media image) {
        return Base64.getEncoder().encodeToString(image.data());
    }
}
