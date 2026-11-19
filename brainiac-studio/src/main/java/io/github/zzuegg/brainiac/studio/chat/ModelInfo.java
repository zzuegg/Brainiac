package io.github.zzuegg.brainiac.studio.chat;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A model a provider offers.
 *
 * @param provider the provider's id, such as {@code anthropic} or {@code local}
 * @param id the provider's own id for it
 * @param name a name to show
 * @param media what it makes or reads: {@code text}, {@code image}, {@code speech}, {@code vision},
 *     {@code video}, {@code transcribe}
 * @param created when the provider published it, in seconds since 1970, or 0 if unknown
 * @param detail a short note, such as the size on disk
 */
public record ModelInfo(String provider, String id, String name, List<String> media, long created, String detail) {

    /** Copies the media list. */
    public ModelInfo {
        media = List.copyOf(media);
        detail = detail == null ? "" : detail;
    }

    /**
     * Returns the id that names this model across providers: {@code provider/id}.
     *
     * @return the id
     */
    public String key() {
        return provider + "/" + id;
    }

    /**
     * Returns whether it can hold a text conversation.
     *
     * @return true for chat models
     */
    public boolean chats() {
        return media.contains("text");
    }

    /**
     * The context window its detail names, such as {@code 200k context}.
     *
     * @return tokens, or 0 if the detail does not say
     */
    public long contextTokens() {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("(\\d+(?:\\.\\d+)?)([kM]) context").matcher(detail);
        if (!m.find()) {
            return 0;
        }
        return Math.round(Double.parseDouble(m.group(1)) * (m.group(2).equals("M") ? 1_000_000 : 1000));
    }

    /** Writes a context length as {@code 200k context} or {@code 1M context}; empty when unknown. */
    static String context(long tokens) {
        if (tokens <= 0) {
            return "";
        }
        return tokens >= 1_000_000 && tokens % 1_000_000 < 50_000 ? tokens / 1_000_000 + "M context"
                : tokens >= 1_000_000 ? String.format(java.util.Locale.ROOT, "%.1fM context", tokens / 1e6)
                : Math.round(tokens / 1000.0) + "k context";
    }

    /**
     * Describes the model for the page and for other studios.
     *
     * @return {@code {key, provider, id, name, media, detail}}
     */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("key", key());
        m.put("provider", provider);
        m.put("id", id);
        m.put("name", name);
        m.put("media", media);
        m.put("detail", detail);
        return m;
    }
}
