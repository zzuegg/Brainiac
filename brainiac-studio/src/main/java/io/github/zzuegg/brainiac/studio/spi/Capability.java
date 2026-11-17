package io.github.zzuegg.brainiac.studio.spi;

import java.util.Objects;

/**
 * Something the studio can do, such as {@code text.chat} or {@code image.generate}.
 *
 * @param key a dotted name, unique across all providers, such as {@code "text.chat"}
 * @param medium what the capability works with
 * @param title a short name for people, such as {@code "Chat"}
 * @param description one sentence on what it does
 * @param source where it comes from, such as a module name
 * @param status whether it can be used now
 * @param note an optional remark on limits, or {@code ""}
 */
public record Capability(String key, Medium medium, String title, String description, String source,
        Status status, String note) {

    /** Whether a capability can be used now. */
    public enum Status {
        /** installed and usable */
        AVAILABLE,
        /** designed for, but nothing provides it yet */
        PLANNED
    }

    /**
     * Checks the fields.
     *
     * @param key a dotted name
     * @param medium the medium
     * @param title a short name
     * @param description one sentence
     * @param source where it comes from
     * @param status the status
     * @param note a remark, or {@code ""}
     */
    public Capability {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(medium, "medium");
        Objects.requireNonNull(title, "title");
        Objects.requireNonNull(description, "description");
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(note, "note");
        if (!key.matches("[a-z][a-z0-9_]*(\\.[a-z0-9_*]+)+")) {
            throw new IllegalArgumentException("capability keys look like `text.chat`, got `" + key + "`");
        }
    }

    /**
     * Describes an installed capability.
     *
     * @param key a dotted name
     * @param medium the medium
     * @param title a short name
     * @param description one sentence
     * @param source where it comes from
     * @return the capability
     */
    public static Capability available(String key, Medium medium, String title, String description, String source) {
        return new Capability(key, medium, title, description, source, Status.AVAILABLE, "");
    }

    /**
     * Describes a capability that is designed for but not provided yet.
     *
     * @param key a dotted name
     * @param medium the medium
     * @param title a short name
     * @param description one sentence
     * @return the capability
     */
    public static Capability planned(String key, Medium medium, String title, String description) {
        return new Capability(key, medium, title, description, "planned", Status.PLANNED, "");
    }

    /**
     * Returns a copy with a remark on limits.
     *
     * @param note the remark
     * @return the copy
     */
    public Capability withNote(String note) {
        return new Capability(key, medium, title, description, source, status, note);
    }
}
