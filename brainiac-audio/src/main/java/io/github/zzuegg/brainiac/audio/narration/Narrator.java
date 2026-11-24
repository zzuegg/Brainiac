package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Speaks a game's lines from a {@link ClipLibrary}, with no speech model at run time. A line
 * recorded whole is played as recorded; any other line is spliced from recorded pieces, each
 * cut from a take of the same template (so its melody fits its place), with short
 * equal-power crossfades at the joins and the result brought to one loudness.
 *
 * <pre>{@code
 * Narrator narrator = Narrator.open(Path.of("voice/de"));
 * Audio a = narrator.say("Was ist {a} plus {b}?", Map.of("a", 7, "b", 8)).audio();
 * Audio b = narrator.sayMath("12 + 7 = 19").audio();
 * }</pre>
 */
public final class Narrator {

    /** How a line was made. */
    public enum Mode {
        /** Played as one recording of the whole line. */
        FULL,
        /** Spliced from pieces. */
        SPLICED
    }

    /**
     * A spoken line.
     *
     * @param text the spelled line
     * @param audio its audio
     * @param mode whether it was recorded whole or spliced
     * @param clips the clips it was made from
     */
    public record Spoken(String text, Audio audio, Mode mode, List<ClipLibrary.Clip> clips) {
    }

    /**
     * How lines are put together.
     *
     * @param crossfadeSeconds the overlap at each join
     * @param targetLufs the loudness of every line
     * @param preferFull whether a line recorded whole is played instead of splicing
     */
    public record Settings(double crossfadeSeconds, double targetLufs, boolean preferFull) {

        /**
         * The defaults: 12 ms crossfades, -16 LUFS, whole lines first.
         *
         * @return the settings
         */
        public static Settings defaults() {
            return new Settings(0.012, -16, true);
        }
    }

    private final ClipLibrary library;
    private final NumberSpeller speller;
    private final Settings settings;

    /**
     * A narrator for a library.
     *
     * @param library the recorded clips
     * @param settings how to put lines together
     */
    public Narrator(ClipLibrary library, Settings settings) {
        this.library = library;
        this.speller = NumberSpeller.of(library.language());
        this.settings = settings;
    }

    /**
     * Opens a library folder with the default settings.
     *
     * @param dir the folder
     * @return the narrator
     */
    public static Narrator open(Path dir) {
        return new Narrator(ClipLibrary.open(dir), Settings.defaults());
    }

    /**
     * The library's number words.
     *
     * @return the speller
     */
    public NumberSpeller speller() {
        return speller;
    }

    /**
     * Speaks a template with values.
     *
     * @param template such as {@code "Du bist auf dem {n:ordinal:en} Platz!"}
     * @param values the values
     * @return the line
     * @throws IllegalStateException if a piece of the line was never recorded
     */
    public Spoken say(String template, Map<String, ?> values) {
        return say(LineTemplate.parse(template).render(values, speller));
    }

    /**
     * Reads an expression, such as {@code "12 + 7 = 19"}.
     *
     * @param expression the expression
     * @return the line
     * @throws IllegalStateException if a piece of the line was never recorded
     */
    public Spoken sayMath(String expression) {
        MathReader.Templated t = MathReader.template(expression, speller);
        return say(t.template().render(t.values(), speller));
    }

    /**
     * Speaks a spelled line: as recorded whole if it was, otherwise spliced.
     *
     * @param line the line
     * @return the spoken line
     * @throws IllegalStateException if a piece of the line was never recorded
     */
    public Spoken say(LineTemplate.Line line) {
        if (settings.preferFull()) {
            ClipLibrary.Clip full = library.full(line.text());
            if (full != null) {
                Audio a = library.audio(full);
                float[] x = AudioTools.trim(a.samples(), a.sampleRate(), 0.05);
                return new Spoken(line.text(), Loudness.normalize(new Audio(x, a.sampleRate()), settings.targetLufs(),
                        -1), Mode.FULL, List.of(full));
            }
        }
        return splice(line);
    }

    /**
     * Splices a line from its pieces, even if it was recorded whole.
     *
     * @param line the line
     * @return the spoken line
     * @throws IllegalStateException if a piece was never recorded
     */
    public Spoken splice(LineTemplate.Line line) {
        List<LineTemplate.Unit> units = line.units();
        List<ClipLibrary.Clip> used = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        for (int i = 0; i < units.size(); i++) {
            LineTemplate.Unit u = units.get(i);
            ClipLibrary.Clip c = library.unit(line.template(), i, u.text());
            if (c == null) {
                c = library.anyUnit(u.text(), u.position());
            }
            if (c == null) {
                missing.add(u.text());
            }
            used.add(c);
        }
        if (!missing.isEmpty()) {
            throw new IllegalStateException("not recorded: " + missing + " (for: " + line.text() + ")");
        }
        int rate = library.audio(used.get(0)).sampleRate();
        List<float[]> parts = new ArrayList<>();
        for (ClipLibrary.Clip c : used) {
            parts.add(library.audio(c).resample(rate).samples());
        }
        float[] x = AudioTools.join(parts, rate, settings.crossfadeSeconds());
        x = AudioTools.trim(x, rate, 0.05);
        Audio a = Loudness.normalize(new Audio(x, rate), settings.targetLufs(), -1);
        return new Spoken(line.text(), a, Mode.SPLICED, used);
    }

    /**
     * The pieces of a line that were never recorded, for checking a game's lines ahead.
     *
     * @param line the line
     * @return the missing pieces' words, empty if the line can be said
     */
    public List<String> missing(LineTemplate.Line line) {
        if (settings.preferFull() && library.full(line.text()) != null) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        List<LineTemplate.Unit> units = line.units();
        for (int i = 0; i < units.size(); i++) {
            LineTemplate.Unit u = units.get(i);
            if (library.unit(line.template(), i, u.text()) == null && library.anyUnit(u.text(), u.position()) == null) {
                out.add(u.text());
            }
        }
        return out;
    }
}
