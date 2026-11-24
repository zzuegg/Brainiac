package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.Wav;
import io.github.zzuegg.brainiac.io.Json;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A narrator's recorded speech on disk: whole lines, and the pieces ({@linkplain
 * LineTemplate.Unit units}) that lines are spliced from. A folder holds {@code index.json}
 * and a WAV file per clip; {@link ClipMaker} fills it ahead of time, and a game reads it
 * with a {@link Narrator}, which needs no speech model.
 */
public final class ClipLibrary {

    /** A whole line, spoken in one go. */
    public static final String FULL = "full";
    /** A piece of a line, cut from a take of a template. */
    public static final String UNIT = "unit";

    /**
     * A recorded clip.
     *
     * @param kind {@link #FULL} or {@link #UNIT}
     * @param template the template it was spoken from, or the empty string for a line given as text
     * @param unit the unit's index in the template's line, -1 for a whole line
     * @param text the words, as spelled
     * @param position where in a line it was said ({@code mid}, {@code end.}, {@code end?}, {@code end!})
     * @param file the WAV file, relative to the library folder
     * @param asr how closely speech recognition matched the words, 0 to 1
     * @param voice how close the voice is to the reference, -1 to 1
     */
    public record Clip(String kind, String template, int unit, String text, String position, String file,
            double asr, double voice) {

        Map<String, Object> json() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("kind", kind);
            m.put("template", template);
            m.put("unit", (long) unit);
            m.put("text", text);
            m.put("position", position);
            m.put("file", file);
            m.put("asr", asr);
            m.put("voice", voice);
            return m;
        }
    }

    private final Path dir;
    private final String language;
    private final Map<String, Clip> clips = new LinkedHashMap<>();
    private final Map<String, List<Clip>> byText = new LinkedHashMap<>();
    private final Map<String, Audio> loaded = new ConcurrentHashMap<>();

    private ClipLibrary(Path dir, String language) {
        this.dir = dir;
        this.language = language;
    }

    /**
     * Opens a library folder, creating an empty library if it has no index yet.
     *
     * @param dir the folder
     * @param language the language of a new library ({@code de}, {@code en}); an existing
     *     library keeps its own
     * @return the library
     * @throws UncheckedIOException if the index cannot be read
     */
    public static ClipLibrary open(Path dir, String language) {
        Path index = dir.resolve("index.json");
        if (!Files.isRegularFile(index)) {
            return new ClipLibrary(dir, NumberSpeller.of(language).language());
        }
        try {
            Map<String, Object> root = Json.object(Json.read(index));
            ClipLibrary lib = new ClipLibrary(dir, Json.string(root, "language", language));
            for (Object o : Json.array(root.get("clips"))) {
                Map<String, Object> c = Json.object(o);
                lib.index(new Clip(Json.string(c, "kind", UNIT), Json.string(c, "template", ""),
                        (int) Json.integer(c, "unit", -1), Json.string(c, "text", ""),
                        Json.string(c, "position", "mid"), Json.string(c, "file", ""), Json.number(c, "asr", 0),
                        Json.number(c, "voice", 0)));
            }
            return lib;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Opens an existing library folder.
     *
     * @param dir the folder, with {@code index.json}
     * @return the library
     * @throws UncheckedIOException if the index cannot be read
     * @throws IllegalArgumentException if there is no index
     */
    public static ClipLibrary open(Path dir) {
        if (!Files.isRegularFile(dir.resolve("index.json"))) {
            throw new IllegalArgumentException("no clip library (index.json) in " + dir);
        }
        return open(dir, "en");
    }

    /**
     * The library's language.
     *
     * @return {@code de} or {@code en}
     */
    public String language() {
        return language;
    }

    /**
     * The library's folder.
     *
     * @return the folder
     */
    public Path dir() {
        return dir;
    }

    /**
     * All clips.
     *
     * @return the clips, in the order they were added
     */
    public synchronized List<Clip> clips() {
        return List.copyOf(clips.values());
    }

    static String fullKey(String text) {
        return FULL + "\u0000" + text;
    }

    static String unitKey(String template, int unit, String text) {
        return UNIT + "\u0000" + template + "\u0000" + unit + "\u0000" + text;
    }

    private static String key(Clip c) {
        return c.kind().equals(FULL) ? fullKey(c.text()) : unitKey(c.template(), c.unit(), c.text());
    }

    private void index(Clip c) {
        Clip old = clips.put(key(c), c);
        if (old != null) {
            byText.getOrDefault(old.text(), new ArrayList<>()).remove(old);
        }
        byText.computeIfAbsent(c.text(), k -> new ArrayList<>()).add(c);
    }

    /**
     * The whole-line clip of a line.
     *
     * @param text the spelled line
     * @return the clip, or {@code null}
     */
    public synchronized Clip full(String text) {
        return clips.get(fullKey(text));
    }

    /**
     * The clip of a unit of a template.
     *
     * @param template the template's source
     * @param unit the unit's index
     * @param text the unit's words
     * @return the clip, or {@code null}
     */
    public synchronized Clip unit(String template, int unit, String text) {
        return clips.get(unitKey(template, unit, text));
    }

    /**
     * A unit clip with these words from any template, preferring one said in the same position.
     *
     * @param text the words
     * @param position where in the line they go
     * @return the best clip, or {@code null}
     */
    public synchronized Clip anyUnit(String text, String position) {
        Clip best = null;
        for (Clip c : byText.getOrDefault(text, List.of())) {
            if (!c.kind().equals(UNIT)) {
                continue;
            }
            if (best == null || c.position().equals(position) && !best.position().equals(position)) {
                best = c;
            }
        }
        return best;
    }

    /**
     * A clip's audio, read once and kept.
     *
     * @param clip the clip
     * @return its audio
     */
    public Audio audio(Clip clip) {
        return loaded.computeIfAbsent(clip.file(), f -> Wav.read(dir.resolve(f)));
    }

    /**
     * Adds or replaces a clip, writing its audio. Call {@link #save()} to write the index.
     *
     * @param clip the clip; its {@code file} is chosen here
     * @param audio its audio
     * @return the clip as stored
     */
    public synchronized Clip put(Clip clip, Audio audio) {
        Objects.requireNonNull(audio, "audio");
        String file = "clips/" + clip.kind() + "-" + hash(key(clip)) + ".wav";
        Clip stored = new Clip(clip.kind(), clip.template(), clip.unit(), clip.text(), clip.position(), file,
                clip.asr(), clip.voice());
        try {
            Files.createDirectories(dir.resolve("clips"));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        Wav.write(audio, dir.resolve(file));
        loaded.put(file, audio);
        index(stored);
        return stored;
    }

    /**
     * Writes {@code index.json}.
     *
     * @throws UncheckedIOException if it cannot be written
     */
    public synchronized void save() {
        List<Object> list = new ArrayList<>();
        for (Clip c : clips.values()) {
            list.add(c.json());
        }
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("language", language);
        root.put("clips", list);
        try {
            Files.createDirectories(dir);
            Files.writeString(dir.resolve("index.json"), Json.write(root));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static String hash(String key) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-1").digest(key.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(d, 0, 8);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
