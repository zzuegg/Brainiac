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
     * @param melody whether pieces are bent to one sentence melody (words said alone rise or
     *     fall each on their own, a sentence starts high and falls slowly to its last word),
     *     with literal words such as operators said a little lighter
     */
    public record Settings(double gapSeconds, double speed, double targetLufs, boolean preferFull,
            boolean melody) {

        /** The speed when the library has no whole lines to measure the pace of. */
        public static final double DEFAULT_SPEED = 1.2;

        /**
         * The defaults: 30 ms gaps, the pace of whole lines, -16 LUFS, whole lines first, pieces
         * as recorded (no sentence melody).
         *
         * @return the settings
         */
        public static Settings defaults() {
            return new Settings(0.03, 0, -16, true, false);
        }
    }

    private final ClipLibrary library;
    private final NumberSpeller speller;
    private final Settings settings;
    private double speed = Double.NaN;
    private double voiceHz = Double.NaN;

    /** Where a sentence's melody starts and where it has fallen to before its last word, in semitones. */
    static final double MELODY_TOP = 2.5;
    static final double MELODY_LOW = -1.5;
    /** How much of a word's own pitch movement is kept, once its rise or fall is taken out. */
    static final double ACCENT = 0.5;
    /** The most a piece's pitch is moved, in semitones. */
    static final double MAX_SHIFT = 3;
    /** Literal words inside a line (operators, "ist") are said lighter: quieter and quicker. */
    static final double LIGHT_GAIN = 0.75;
    static final double LIGHT_SPEED = 1.08;

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
        List<LineTemplate.Unit> units = line.units();
        int count = used.size();
        float[][] raw = new float[count][];
        double[] speeds = new double[count];
        double total0 = 0;
        for (int i = 0; i < count; i++) {
            raw[i] = AudioTools.trim(library.audio(used.get(i)).resample(rate).samples(), rate, 0.01);
            speeds[i] = tempo * (light(units.get(i), i, count) ? LIGHT_SPEED : 1);
            total0 += raw[i].length / speeds[i] + gap;
        }
        double lineSeconds = Math.max(1e-3, (total0 - gap) / rate);
        double ref = settings.melody() ? voiceHz() : Double.NaN;
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        double t = 0;
        for (int i = 0; i < count; i++) {
            double[] ratio = Double.isNaN(ref) ? null
                    : melody(raw[i], rate, ref, t / lineSeconds, (t + raw[i].length / speeds[i] / rate) / lineSeconds,
                            i == count - 1 ? units.get(i).position() : null);
            float[] p = ratio == null ? Prosody.psola(raw[i], rate, speeds[i], null)
                    : Prosody.psola(raw[i], rate, speeds[i], ratio);
            if (light(units.get(i), i, count)) {
                for (int k = 0; k < p.length; k++) {
                    p[k] *= (float) LIGHT_GAIN;
                }
            }
            AudioTools.fade(p, rate, 0.004, 0.008);
            parts.add(p);
            total += p.length + gap;
            t += (p.length + gap) / (double) rate;
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
     * The pitch factors that bend a piece to the sentence melody: inside the line its own rise
     * or fall is taken out and it follows the line's slow fall, keeping part of its own
     * movement; the last piece keeps its ending (falling, or rising for a question) and is
     * moved to start where the line has fallen to.
     *
     * @param x the piece
     * @param rate the sample rate
     * @param refHz the voice's middle pitch
     * @param from where the piece starts in the line, 0 to 1
     * @param to where it ends
     * @param ending for the last piece of a line its ending ({@code end.}, {@code end?},
     *     {@code end!}), otherwise {@code null}
     * @return a factor per 10 ms frame, or {@code null} to leave it as it is
     */
    static double[] melody(float[] x, int rate, double refHz, double from, double to, String ending) {
        double[] hz = Prosody.pitch(x, rate);
        double[] c = Prosody.contour(hz, refHz);
        if (c == null) {
            return null;
        }
        int n = c.length;
        double[] want = new double[n];
        if (ending != null) {
            double sum = 0;
            int k = 0;
            for (int i = 0; i < n && k < 5; i++) {
                if (hz[i] > 0) {
                    sum += c[i];
                    k++;
                }
            }
            double start = sum / k;
            double shift = MELODY_TOP + (MELODY_LOW - MELODY_TOP) * from - start;
            // a question's last word starts low and rises: never lift it, or its rise goes out of
            // reach and it stops sounding like a question; a statement's is lifted at most 2
            shift = ending.endsWith("?") ? Math.min(0, shift) : Math.min(2, shift);
            shift = clamp(shift);
            for (int i = 0; i < n; i++) {
                want[i] = c[i] + shift;
            }
        } else {
            // the piece's own rise or fall: a straight line through its voiced frames
            double sx = 0;
            double sy = 0;
            double sxx = 0;
            double sxy = 0;
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (hz[i] > 0) {
                    sx += i;
                    sy += c[i];
                    sxx += (double) i * i;
                    sxy += i * c[i];
                    m++;
                }
            }
            double den = m * sxx - sx * sx;
            double slope = m > 1 && den != 0 ? (m * sxy - sx * sy) / den : 0;
            double icpt = m > 0 ? (sy - slope * sx) / m : 0;
            for (int i = 0; i < n; i++) {
                double pos = from + (to - from) * i / Math.max(1, n - 1);
                double line = MELODY_TOP + (MELODY_LOW - MELODY_TOP) * pos;
                want[i] = line + ACCENT * (c[i] - (icpt + slope * i));
            }
        }
        double[] ratio = new double[n];
        for (int i = 0; i < n; i++) {
            ratio[i] = Math.pow(2, clamp(want[i] - c[i]) / 12);
        }
        return ratio;
    }

    private boolean light(LineTemplate.Unit unit, int index, int count) {
        // the first words carry the line's onset and, in a question, its question word
        return settings.melody() && unit.name() == null && index > 0 && index < count - 1;
    }

    private static double clamp(double semitones) {
        return Math.max(-MAX_SHIFT, Math.min(MAX_SHIFT, semitones));
    }

    /**
     * The middle pitch of the voice: the median over the library's whole lines, or over its
     * pieces if it has none.
     *
     * @return Hz, or {@code NaN} if nothing is voiced
     */
    public synchronized double voiceHz() {
        if (Double.isNaN(voiceHz)) {
            List<Double> full = new ArrayList<>();
            List<Double> pieces = new ArrayList<>();
            for (ClipLibrary.Clip c : library.clips()) {
                Audio a = library.audio(c);
                for (double h : Prosody.pitch(a.samples(), a.sampleRate())) {
                    if (h > 0) {
                        (c.kind().equals(ClipLibrary.FULL) ? full : pieces).add(h);
                    }
                }
            }
            List<Double> use = full.size() > 200 ? full : pieces;
            if (!use.isEmpty()) {
                use.sort(null);
                voiceHz = use.get(use.size() / 2);
            }
        }
        return voiceHz;
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
