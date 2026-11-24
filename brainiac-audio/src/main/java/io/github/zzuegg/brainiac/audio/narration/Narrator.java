package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Speaks a game's lines from a {@link ClipLibrary}, with no speech model at run time. A line
 * recorded whole is played as recorded; any other line is spliced from whole takes of its
 * template, switching takes only at join points inside the words between the values, with
 * pitch-aligned crossfades, and brought to one loudness.
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
     * @param crossfadeSeconds the crossfade at each join, inside a word both takes say
     * @param targetLufs the loudness of every line
     * @param preferFull whether a line recorded whole is played instead of splicing
     */
    public record Settings(double crossfadeSeconds, double targetLufs, boolean preferFull) {

        /**
         * The defaults: 25 ms crossfades, -16 LUFS, whole lines first.
         *
         * @return the settings
         */
        public static Settings defaults() {
            return new Settings(0.025, -16, true);
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
     * Splices a line from takes of its template: each value's stretch, from the join point
     * before it to the one after, comes from a take that says that value, and the takes are
     * chosen so the joins are fewest and sound most alike (a take that already says two of
     * the values needs no join between them). Joins are pitch-aligned crossfades inside the
     * words between values.
     *
     * @param line the line
     * @return the spoken line
     * @throws IllegalStateException if a value was never recorded
     */
    public Spoken splice(LineTemplate.Line line) {
        List<LineTemplate.Unit> units = line.units();
        int[] slots = ClipMaker.slots(units);
        List<ClipLibrary.Clip> takes = library.takes(line.template());
        if (slots.length == 0 || takes.isEmpty()) {
            ClipLibrary.Clip full = library.full(line.text());
            if (full == null) {
                throw new IllegalStateException("not recorded: " + line.text());
            }
            return whole(line, full);
        }
        // candidates for each value's stretch
        List<List<ClipLibrary.Clip>> cand = new ArrayList<>();
        List<String> missing = new ArrayList<>();
        for (int slot : slots) {
            List<ClipLibrary.Clip> c = new ArrayList<>();
            for (ClipLibrary.Clip t : takes) {
                if (t.units().size() == units.size() && t.joins().length == slots.length - 1
                        && t.units().get(slot).equals(units.get(slot).text())) {
                    c.add(t);
                }
            }
            if (c.isEmpty()) {
                missing.add(units.get(slot).text());
            }
            cand.add(c);
        }
        if (!missing.isEmpty()) {
            throw new IllegalStateException("not recorded: " + missing + " (for: " + line.text() + ")");
        }
        List<ClipLibrary.Clip> path = cheapest(cand);
        int rate = library.audio(path.get(0)).sampleRate();
        List<AudioTools.Segment> segments = new ArrayList<>();
        int from = 0;
        for (int j = 0; j < path.size(); j++) {
            ClipLibrary.Clip c = path.get(j);
            boolean last = j == path.size() - 1;
            if (!last && path.get(j + 1) == c) {
                continue;   // the same take goes on: no join here
            }
            float[] x = library.audio(c).resample(rate).samples();
            int to = last ? x.length : c.joins()[j];
            segments.add(new AudioTools.Segment(x, from, to));
            if (!last) {
                from = path.get(j + 1).joins()[j];
            }
        }
        float[] x = AudioTools.splice(segments, rate, settings.crossfadeSeconds(), 0.008);
        x = AudioTools.trim(x, rate, 0.05);
        Audio a = Loudness.normalize(new Audio(x, rate), settings.targetLufs(), -1);
        return new Spoken(line.text(), a, Mode.SPLICED, java.util.List.copyOf(new java.util.LinkedHashSet<>(path)));
    }

    /** The takes, one per value, with the least join cost (Viterbi over the values). */
    private List<ClipLibrary.Clip> cheapest(List<List<ClipLibrary.Clip>> cand) {
        int n = cand.size();
        double[][] cost = new double[n][];
        int[][] back = new int[n][];
        cost[0] = new double[cand.get(0).size()];
        for (int i = 0; i < cost[0].length; i++) {
            cost[0][i] = cand.get(0).get(i).frame() ? 0 : 0.01;
        }
        for (int j = 1; j < n; j++) {
            List<ClipLibrary.Clip> prev = cand.get(j - 1);
            List<ClipLibrary.Clip> cur = cand.get(j);
            cost[j] = new double[cur.size()];
            back[j] = new int[cur.size()];
            for (int k = 0; k < cur.size(); k++) {
                double best = Double.POSITIVE_INFINITY;
                for (int p = 0; p < prev.size(); p++) {
                    double c = cost[j - 1][p] + joinCost(prev.get(p), cur.get(k), j - 1);
                    if (c < best) {
                        best = c;
                        back[j][k] = p;
                    }
                }
                cost[j][k] = best;
            }
        }
        int k = 0;
        for (int i = 1; i < cost[n - 1].length; i++) {
            if (cost[n - 1][i] < cost[n - 1][k]) {
                k = i;
            }
        }
        ClipLibrary.Clip[] out = new ClipLibrary.Clip[n];
        for (int j = n - 1; j >= 0; j--) {
            out[j] = cand.get(j).get(k);
            if (j > 0) {
                k = back[j][k];
            }
        }
        return List.of(out);
    }

    /** How audible a join from one take to another at join point {@code b} would be: 0 for the same take. */
    private double joinCost(ClipLibrary.Clip a, ClipLibrary.Clip b, int join) {
        if (a == b) {
            return 0;
        }
        Audio xa = library.audio(a);
        Audio xb = library.audio(b).resample(xa.sampleRate());
        int rate = xa.sampleRate();
        int ja = a.joins()[join];
        int jb = b.joins()[join];
        double spectral = AudioTools.distance(AudioTools.spectrumAt(xa.samples(), rate, ja),
                AudioTools.spectrumAt(xb.samples(), rate, jb));
        double corr = AudioTools.bestShift(xa.samples(), ja, xb.samples(), jb, (int) (0.025 * rate),
                (int) (0.008 * rate))[1];
        return 1 + spectral + 5 * (1 - corr);
    }

    /**
     * The values of a line that were never recorded, for checking a game's lines ahead.
     *
     * @param line the line
     * @return the missing values' words, empty if the line can be said
     */
    public List<String> missing(LineTemplate.Line line) {
        if (library.full(line.text()) != null) {
            return List.of();
        }
        List<LineTemplate.Unit> units = line.units();
        int[] slots = ClipMaker.slots(units);
        List<ClipLibrary.Clip> takes = library.takes(line.template());
        List<String> out = new ArrayList<>();
        if (slots.length == 0) {
            return List.of(line.text());
        }
        for (int slot : slots) {
            boolean found = false;
            for (ClipLibrary.Clip t : takes) {
                if (t.units().size() == units.size() && t.units().get(slot).equals(units.get(slot).text())) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                out.add(units.get(slot).text());
            }
        }
        return out;
    }
}
