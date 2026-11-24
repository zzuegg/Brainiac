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
 * A narrator's recorded speech on disk: whole lines, and takes of templates to splice lines
 * from. A folder holds {@code index.json} and a WAV file per clip; {@link ClipMaker} fills it
 * ahead of time, and a game reads it with a {@link Narrator}, which needs no speech model.
 *
 * <p>A take of a template is a whole spoken line with its <em>join points</em>: one between
 * each two values, placed inside the words between them, where every take of the template
 * says the same sound. A spliced line switches from one take to another only there.
 */
public final class ClipLibrary {

    /** A whole line, spoken in one go. */
    public static final String FULL = "full";
    /** A take of a template, with join points, for splicing. */
    public static final String TAKE = "take";

    /**
     * A recorded clip.
     *
     * @param kind {@link #FULL} or {@link #TAKE}
     * @param template the template it was spoken from, or the empty string
     * @param text the whole line, as spelled
     * @param units for a take, the words of each {@linkplain LineTemplate.Unit unit} of its line
     * @param joins for a take, the sample of each join point (one fewer than the values in the line)
     * @param frame whether this take is the template's frame, the one join points were chosen on
     * @param file the WAV file, relative to the library folder
     * @param asr how closely speech recognition matched the words, 0 to 1
     * @param voice how close the voice is to the reference, -1 to 1
     */
    public record Clip(String kind, String template, String text, List<String> units, int[] joins, boolean frame,
            String file, double asr, double voice) {

        /**
         * Copies the lists.
         *
         * @param kind the kind
         * @param template the template
         * @param text the line
         * @param units the units' words
         * @param joins the join points
         * @param frame whether it is the frame
         * @param file the file
         * @param asr the recognition match
         * @param voice the voice match
         */
        public Clip {
            units = List.copyOf(units);
            joins = joins.clone();
        }

        Map<String, Object> json() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("kind", kind);
            m.put("template", template);
            m.put("text", text);
            m.put("units", new ArrayList<Object>(units));
            m.put("joins", joins);
            m.put("frame", frame);
            m.put("file", file);
            m.put("asr", asr);
            m.put("voice", voice);
            return m;
        }
    }

    private final Path dir;
    private final String language;
    private final Map<String, Clip> clips = new LinkedHashMap<>();
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
                List<String> units = new ArrayList<>();
                if (c.get("units") != null) {
                    for (Object u : Json.array(c.get("units"))) {
                        units.add((String) u);
                    }
                }
                List<Object> j = c.get("joins") == null ? List.of() : Json.array(c.get("joins"));
                int[] joins = new int[j.size()];
                for (int i = 0; i < joins.length; i++) {
                    joins[i] = ((Number) j.get(i)).intValue();
                }
                lib.index(new Clip(Json.string(c, "kind", FULL), Json.string(c, "template", ""),
                        Json.string(c, "text", ""), units, joins, Boolean.TRUE.equals(c.get("frame")),
                        Json.string(c, "file", ""), Json.number(c, "asr", 0), Json.number(c, "voice", 0)));
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

    private static String key(Clip c) {
        return c.kind() + "\u0000" + c.template() + "\u0000" + c.text();
    }

    private void index(Clip c) {
        clips.put(key(c), c);
    }

    /**
     * The whole-line clip of a line.
     *
     * @param text the spelled line
     * @return the clip, or {@code null}
     */
    public synchronized Clip full(String text) {
        for (Clip c : clips.values()) {
            if (c.kind().equals(FULL) && c.text().equals(text)) {
                return c;
            }
        }
        return null;
    }

    /**
     * The takes of a template.
     *
     * @param template the template's source
     * @return its takes, the frame first
     */
    public synchronized List<Clip> takes(String template) {
        List<Clip> out = new ArrayList<>();
        for (Clip c : clips.values()) {
            if (c.kind().equals(TAKE) && c.template().equals(template)) {
                if (c.frame()) {
                    out.add(0, c);
                } else {
                    out.add(c);
                }
            }
        }
        return out;
    }

    /**
     * A template's take of a line.
     *
     * @param template the template's source
     * @param text the spelled line
     * @return the take, or {@code null}
     */
    public synchronized Clip take(String template, String text) {
        return clips.get(TAKE + "\u0000" + template + "\u0000" + text);
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
        Clip stored = new Clip(clip.kind(), clip.template(), clip.text(), clip.units(), clip.joins(), clip.frame(),
                file, clip.asr(), clip.voice());
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
