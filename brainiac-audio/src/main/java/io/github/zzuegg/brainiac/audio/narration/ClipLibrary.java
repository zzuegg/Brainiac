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
 * A narrator's recorded speech on disk: whole lines, and pieces (numbers, operators,
 * phrases) to put lines together from, each with the melody of its place in a line. A folder
 * holds {@code index.json} and a WAV file per clip; {@link ClipMaker} fills it ahead of time,
 * and a game reads it with a {@link Narrator}, which needs no speech model.
 */
public final class ClipLibrary {

    /** A whole line, spoken in one go. */
    public static final String FULL = "full";
    /** A piece of a line, said with one melody, cut out at pauses. */
    public static final String PIECE = "piece";
    /** The template of a piece cut out of a fluent sentence rather than said on its own. */
    public static final String FLUENT = "fluent";
    /** The melody of a piece inside a line: continuing. */
    public static final String MID = "mid";

    /**
     * A recorded clip.
     *
     * @param kind {@link #FULL} or {@link #PIECE}
     * @param template the template a whole line was spoken from, or the empty string
     * @param text the words, as spelled
     * @param melody for a piece, {@link #MID}, {@code end.}, {@code end?} or {@code end!}; for a line, {@code line}
     * @param file the WAV file, relative to the library folder
     * @param asr how closely speech recognition matched the words, 0 to 1
     * @param voice how close the voice is to the reference, -1 to 1
     */
    public record Clip(String kind, String template, String text, String melody, String file, double asr,
            double voice) {

        Map<String, Object> json() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("kind", kind);
            m.put("template", template);
            m.put("text", text);
            m.put("melody", melody);
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
    private final List<Double> pauses = new ArrayList<>();

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
                lib.index(new Clip(Json.string(c, "kind", FULL), Json.string(c, "template", ""),
                        Json.string(c, "text", ""), Json.string(c, "melody", "line"), Json.string(c, "file", ""),
                        Json.number(c, "asr", 0), Json.number(c, "voice", 0)));
            }
            if (root.get("pauses") != null) {
                for (Object v : Json.array(root.get("pauses"))) {
                    lib.pauses.add(((Number) v).doubleValue());
                }
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
        return c.kind().equals(FULL) ? FULL + "\u0000" + c.text() : pieceKey(c.text(), c.melody());
    }

    /**
     * A melody tied to one place in one kind of sentence: the position plus whether the line is
     * a question or a statement and which of its stretches this is, such as {@code mid@q1} for
     * the first number of "Was ist sieben plus acht?". A piece cut from a sentence of the same
     * shape carries the melody that place has; {@link #pieceOrNear} falls back to the plain
     * position.
     *
     * @param position {@code mid}, {@code end.}, {@code end?} or {@code end!}
     * @param line the line
     * @param index which of the line's units
     * @return the melody
     */
    public static String slot(String position, LineTemplate.Line line, int index) {
        return position + "@" + (line.text().endsWith("?") ? "q" : "s") + index;
    }

    /**
     * A melody without its place in a sentence.
     *
     * @param melody such as {@code mid@q1}
     * @return such as {@code mid}
     */
    public static String base(String melody) {
        int at = melody.indexOf('@');
        return at < 0 ? melody : melody.substring(0, at);
    }

    /** A piece's words without the line's closing mark, which its melody already says. */
    private static String pieceKey(String text, String melody) {
        return PIECE + "\u0000" + text.replaceAll("[\\s.?!,;:]+$", "") + "\u0000" + melody;
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
        return clips.get(FULL + "\u0000" + text);
    }

    /**
     * A piece said with a melody.
     *
     * @param text the words
     * @param melody the melody
     * @return the clip, or {@code null}
     */
    public synchronized Clip piece(String text, String melody) {
        return clips.get(pieceKey(text, melody));
    }

    /**
     * A piece, or failing that the same words with the nearest melody: an ending with another
     * mark, then any ending, then continuing.
     *
     * @param text the words
     * @param melody the melody wanted
     * @return the clip, or {@code null} if the words were never recorded
     */
    public synchronized Clip pieceOrNear(String text, String melody) {
        if (melody.indexOf('@') >= 0) {
            Clip exact = piece(text, melody);
            if (exact != null) {
                return exact;
            }
            melody = base(melody);
        }
        List<String> order = new ArrayList<>(List.of(melody));
        if (melody.startsWith("end")) {
            order.addAll(List.of("end.", "end!", "end?", MID));
        } else {
            order.addAll(List.of("end.", "end!", "end?"));
        }
        for (String m : order) {
            Clip c = piece(text, m);
            if (c != null) {
                return c;
            }
        }
        return null;
    }

    /**
     * Notes pauses measured between pieces in carriers.
     *
     * @param seconds their lengths
     */
    public synchronized void pause(List<Double> seconds) {
        pauses.addAll(seconds);
    }

    /**
     * The median pause between pieces in the carriers, the speaker's own.
     *
     * @return seconds, 0.18 if none was measured
     */
    public synchronized double pauseSeconds() {
        if (pauses.isEmpty()) {
            return 0.18;
        }
        List<Double> sorted = new ArrayList<>(pauses);
        sorted.sort(null);
        return sorted.get(sorted.size() / 2);
    }

    /**
     * How much slower the pieces are said than whole lines: the median seconds per letter of
     * the pieces over that of the whole lines. Words said alone in a carrier are drawn out a
     * little against running speech, so a line put together from them is sped up by this much.
     *
     * @return the ratio, or {@code NaN} if the library has no pieces or no whole lines
     */
    public double pieceSlowdown() {
        List<Double> piece = new ArrayList<>();
        List<Double> full = new ArrayList<>();
        for (Clip c : clips()) {
            int letters = (int) c.text().codePoints().filter(Character::isLetter).count();
            if (letters < 3) {
                continue;
            }
            Audio a = audio(c);
            int[] sp = AudioTools.speech(AudioTools.envelope(a.samples(), a.sampleRate()));
            double perLetter = (sp[1] - sp[0] + 1) / (double) AudioTools.FRAME_RATE / letters;
            (c.kind().equals(FULL) ? full : piece).add(perLetter);
        }
        if (piece.isEmpty() || full.isEmpty()) {
            return Double.NaN;
        }
        return median(piece) / median(full);
    }

    private static double median(List<Double> v) {
        List<Double> sorted = new ArrayList<>(v);
        sorted.sort(null);
        return sorted.get(sorted.size() / 2);
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
        Clip stored = new Clip(clip.kind(), clip.template(), clip.text(), clip.melody(), file, clip.asr(),
                clip.voice());
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
        root.put("pauses", new ArrayList<Object>(pauses));
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
