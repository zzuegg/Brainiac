package io.github.zzuegg.brainiac.studio.chat;

import java.util.Objects;
import java.util.Set;

/**
 * What to draw.
 *
 * @param model the provider's own model id
 * @param prompt what the picture shows
 * @param aspect {@code 1:1}, {@code 4:3}, {@code 3:4}, {@code 16:9} or {@code 9:16}
 * @param count how many pictures, 1 to 4
 * @param key the name of the API key to use, or null to try the provider's keys in order
 */
public record ImageRequest(String model, String prompt, String aspect, int count, String key) {

    /** The aspect ratios offered. */
    public static final Set<String> ASPECTS = Set.of("1:1", "4:3", "3:4", "16:9", "9:16");

    /** Checks the request. */
    public ImageRequest {
        Objects.requireNonNull(model, "model");
        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("say what the picture should show");
        }
        aspect = aspect == null ? "1:1" : aspect;
        if (!ASPECTS.contains(aspect)) {
            throw new IllegalArgumentException("aspect is one of " + ASPECTS);
        }
        if (count < 1 || count > 4) {
            throw new IllegalArgumentException("count is 1 to 4");
        }
    }

    /**
     * Returns whether the picture is wider than tall.
     *
     * @return true for 4:3 and 16:9
     */
    boolean wide() {
        return aspect.equals("4:3") || aspect.equals("16:9");
    }

    /**
     * Returns whether the picture is taller than wide.
     *
     * @return true for 3:4 and 9:16
     */
    boolean tall() {
        return aspect.equals("3:4") || aspect.equals("9:16");
    }
}
