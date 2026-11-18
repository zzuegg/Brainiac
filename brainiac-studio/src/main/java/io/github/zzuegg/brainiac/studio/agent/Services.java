package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Settings for the outside services agent tools use: web search, the code sandbox, GitHub,
 * posting and mail. Plain settings are kept in {@code services.json}; passwords and tokens are kept with the
 * studio's other keys, under the service's id.
 */
public final class Services {

    /** What each section may hold, and which of its keys are secrets kept with the keys. */
    public static final Map<String, List<String>> FIELDS = Map.of(
            "search", List.of("provider", "url"),
            "sandbox", List.of("kind", "image", "cpus", "memory", "network", "timeout"),
            "mastodon", List.of("instance"),
            "bluesky", List.of("handle", "service"),
            "discord", List.of(),
            "webhook", List.of("url"),
            "mail", List.of("imapHost", "imapPort", "smtpHost", "smtpPort", "user", "from"),
            "calendar", List.of("url"),
            "github", List.of("repo", "api"),
            "embeddings", List.of("source", "url", "model", "folder"));

    /** The ids secrets are kept under, per section. */
    public static final Map<String, String> SECRETS = Map.of(
            "search", "search",
            "mastodon", "mastodon",
            "bluesky", "bluesky",
            "discord", "discord",
            "mail", "mail",
            "github", "github",
            "embeddings", "embeddings");

    private final Path file;
    private final ChatService chat;
    private final Map<String, Map<String, Object>> sections = new LinkedHashMap<>();

    /**
     * Opens the settings.
     *
     * @param file where they are kept
     * @param chat where secrets are kept, or null for none
     */
    public Services(Path file, ChatService chat) {
        this.file = file;
        this.chat = chat;
        if (Files.isRegularFile(file)) {
            try {
                for (Map.Entry<String, Object> e : Json.object(Json.parse(Files.readString(file, StandardCharsets.UTF_8)))
                        .entrySet()) {
                    if (FIELDS.containsKey(e.getKey()) && e.getValue() != null) {
                        sections.put(e.getKey(), new LinkedHashMap<>(Json.object(e.getValue())));
                    }
                }
            } catch (IOException | RuntimeException e) {
                System.getLogger(Services.class.getName()).log(System.Logger.Level.WARNING, "cannot read " + file, e);
            }
        }
    }

    /**
     * Reads one setting.
     *
     * @param section the section
     * @param key the setting
     * @return its value, if set
     */
    public synchronized Optional<String> get(String section, String key) {
        Object v = sections.getOrDefault(section, Map.of()).get(key);
        return v == null || String.valueOf(v).isBlank() ? Optional.empty() : Optional.of(String.valueOf(v));
    }

    /**
     * Reads a section's secret.
     *
     * @param section the section
     * @return its token or password, if one is kept
     */
    public Optional<String> secret(String section) {
        String id = SECRETS.get(section);
        return id == null || chat == null ? Optional.empty() : chat.secret(id);
    }

    /**
     * Changes a section.
     *
     * @param section the section
     * @param values the settings to set; null or blank values are removed
     * @param secret a new token or password, or null to keep the one there is
     * @throws IOException if they cannot be kept
     */
    public synchronized void set(String section, Map<String, Object> values, String secret) throws IOException {
        List<String> allowed = FIELDS.get(section);
        if (allowed == null) {
            throw new IllegalArgumentException("no service " + section + ": use " + String.join(", ", FIELDS.keySet()));
        }
        Map<String, Object> s = sections.computeIfAbsent(section, k -> new LinkedHashMap<>());
        for (Map.Entry<String, Object> e : values.entrySet()) {
            if (!allowed.contains(e.getKey())) {
                throw new IllegalArgumentException("the service " + section + " has no setting " + e.getKey());
            }
            if (e.getValue() == null || String.valueOf(e.getValue()).isBlank()) {
                s.remove(e.getKey());
            } else {
                s.put(e.getKey(), e.getValue());
            }
        }
        if (secret != null && !secret.isBlank()) {
            String id = SECRETS.get(section);
            if (id == null || chat == null) {
                throw new IllegalArgumentException("the service " + section + " keeps no token");
            }
            chat.addKey(id, "default", secret.strip());
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, Json.write(sections), StandardCharsets.UTF_8);
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /**
     * Describes every section for the page, without secrets.
     *
     * @return per section its settings and whether a secret is kept
     */
    public synchronized Map<String, Object> toJson() {
        Map<String, Object> out = new LinkedHashMap<>();
        for (String section : List.of("search", "embeddings", "sandbox", "github", "mastodon", "bluesky", "discord", "webhook", "mail",
                "calendar")) {
            Map<String, Object> m = new LinkedHashMap<>(sections.getOrDefault(section, Map.of()));
            if (SECRETS.containsKey(section)) {
                m.put("hasSecret", secret(section).isPresent());
            }
            out.put(section, m);
        }
        out.put("ffmpeg", Ffmpeg.find() != null);
        return out;
    }
}
