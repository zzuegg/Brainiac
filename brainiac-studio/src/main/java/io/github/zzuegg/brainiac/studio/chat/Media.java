package io.github.zzuegg.brainiac.studio.chat;

import java.util.Locale;
import java.util.Objects;

/**
 * A generated file: an image or audio.
 *
 * @param data the bytes
 * @param mime the type, such as {@code image/png} or {@code audio/mpeg}
 */
public record Media(byte[] data, String mime) {

    /** Checks the fields. */
    public Media {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(mime, "mime");
    }

    /**
     * Returns the usual file extension for the type.
     *
     * @return the extension without a dot
     */
    public String extension() {
        return switch (mime.toLowerCase(Locale.ROOT)) {
            case "image/png" -> "png";
            case "image/jpeg" -> "jpg";
            case "image/webp" -> "webp";
            case "audio/mpeg", "audio/mp3" -> "mp3";
            case "audio/wav", "audio/x-wav", "audio/wave" -> "wav";
            case "audio/ogg", "audio/opus" -> "opus";
            case "audio/flac" -> "flac";
            case "audio/aac" -> "aac";
            default -> "bin";
        };
    }
}
