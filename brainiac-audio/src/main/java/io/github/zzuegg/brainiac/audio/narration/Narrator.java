package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Speaks a game's lines from a {@link ClipLibrary}, with no speech model at run time. A line
 * recorded whole is played as recorded; any other line is put together from shared pieces
 * (numbers, operators, phrases), each said with the melody of its place, brought to the pace of
 * running speech without changing its pitch, joined by short gaps and brought to one loudness.
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
     * @param gapSeconds the silence between pieces, as between the words of running speech
     * @param speed how much faster pieces are played than recorded, keeping their pitch; 0 to
     *     match the pace of the library's whole lines ({@link ClipLibrary#pieceSlowdown()})
     * @param targetLufs the loudness of every line
     * @param preferFull whether a line recorded whole is played instead of putting it together
     */
    public record Settings(double gapSeconds, double speed, double targetLufs, boolean preferFull) {

        /** The speed when the library has no whole lines to measure the pace of. */
        public static final double DEFAULT_SPEED = 1.2;

        /**
         * The defaults: 60 ms gaps, the pace of whole lines, -16 LUFS, whole lines first.
         *
         * @return the settings
         */
        public static Settings defaults() {
            return new Settings(0.06, 0, -16, true);
        }
    }

    private final ClipLibrary library;
    private final NumberSpeller speller;
    private final Settings settings;
    private double speed = Double.NaN;

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
                return whole(line, full);
            }
        }
        return splice(line);
    }

    private Spoken whole(LineTemplate.Line line, ClipLibrary.Clip full) {
        Audio a = library.audio(full);
        float[] x = AudioTools.trim(a.samples(), a.sampleRate(), 0.05);
        return new Spoken(line.text(), Loudness.normalize(new Audio(x, a.sampleRate()), settings.targetLufs(), -1),
                Mode.FULL, List.of(full));
    }

    /**
     * Puts a line together from pieces: each unit (a number, an operator, a phrase) said with
     * the melody of its place (continuing, or the line's ending), brought to the pace of
     * running speech and joined by gaps as short as between its words.
     *
     * @param line the line
     * @return the spoken line
     * @throws IllegalStateException if a piece was never recorded
     */
    public Spoken splice(LineTemplate.Line line) {
        List<ClipLibrary.Clip> used = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        for (LineTemplate.Unit u : line.units()) {
            ClipLibrary.Clip c = library.pieceOrNear(u.text(), u.position());
            if (c == null) {
                missing.add(u.text());
            }
            used.add(c);
        }
        if (!missing.isEmpty()) {
            throw new IllegalStateException("not recorded: " + missing + " (for: " + line.text() + ")");
        }
        int rate = library.audio(used.get(0)).sampleRate();
        int gap = (int) (settings.gapSeconds() * rate);
        double tempo = speed();
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (ClipLibrary.Clip c : used) {
            float[] p = AudioTools.trim(library.audio(c).resample(rate).samples(), rate, 0.01);
            p = AudioTools.stretch(p, rate, tempo);
            parts.add(p);
            total += p.length + gap;
        }
        float[] x = new float[total];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, x, at, p.length);
            at += p.length + gap;
        }
        x = java.util.Arrays.copyOf(x, Math.max(0, at - gap));
        Audio a = Loudness.normalize(new Audio(x, rate), settings.targetLufs(), -1);
        return new Spoken(line.text(), a, Mode.SPLICED, used);
    }

    /**
     * How much faster pieces are played than recorded.
     *
     * @return {@link Settings#speed()}, or if that is 0 the library's
     *     {@link ClipLibrary#pieceSlowdown()} kept between 1 and 1.5, or
     *     {@link Settings#DEFAULT_SPEED} if it cannot be measured
     */
    public synchronized double speed() {
        if (Double.isNaN(speed)) {
            double s = settings.speed();
            if (s <= 0) {
                double m = library.pieceSlowdown();
                s = Double.isNaN(m) ? Settings.DEFAULT_SPEED : Math.max(1, Math.min(1.5, m));
            }
            speed = s;
        }
        return speed;
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
        for (LineTemplate.Unit u : line.units()) {
            if (library.pieceOrNear(u.text(), u.position()) == null) {
                out.add(u.text());
            }
        }
        return out;
    }
}
