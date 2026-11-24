package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.SpeechOptions;
import io.github.zzuegg.brainiac.audio.SpeechToText;
import io.github.zzuegg.brainiac.audio.TextToSpeech;
import io.github.zzuegg.brainiac.audio.TranscriptionOptions;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Records a narrator's {@link ClipLibrary} ahead of time with a speech model, checking every
 * take with speech recognition.
 *
 * <p>For each line it speaks several takes (seeds {@code seed}, {@code seed + 1}, ...),
 * brings each to the same loudness, transcribes it, and keeps the take whose words match
 * best and whose voice is closest to the reference. Then either
 * <ul>
 * <li>{@link #full}: the take is the line, or
 * <li>{@link #record}: pieces (a number, an operator, a phrase), each with the melody of its
 *     place, are said in short chains with commas ("sieben, plus, acht, ist, fünfzehn.") and
 *     cut out at the pauses. Every template and calculation shares them, and a {@link
 *     Narrator} joins them with natural pauses.
 * </ul>
 * {@link #prepare} picks between them by how many lines the template can make.
 */
public final class ClipMaker {

    /** A speaker embedding, for ranking takes by how close their voice is to the reference. */
    @FunctionalInterface
    public interface VoiceEmbedder {

        /**
         * Embeds a voice.
         *
         * @param audio speech
         * @return the embedding (compared by cosine)
         */
        float[] embed(Audio audio);
    }

    /**
     * How to record.
     *
     * @param options the voice: for a consistent narrator, a cloning model's reference clip
     *     with its transcript, the language and one speaking style for every line
     * @param takes takes per line, the best kept
     * @param targetLufs the loudness every take is brought to before cutting
     * @param minAsr the least recognition match (0 to 1) a take or piece must reach
     * @param fullBudget {@link #prepare} records every line whole when a template makes at most this many
     */
    public record Settings(SpeechOptions options, int takes, double targetLufs, double minAsr, int fullBudget) {

        /**
         * Checks the values.
         *
         * @param options the voice
         * @param takes takes per line, 1 or more
         * @param targetLufs the loudness
         * @param minAsr the least match
         * @param fullBudget the most lines recorded whole per template
         */
        public Settings {
            Objects.requireNonNull(options, "options");
            if (takes < 1) {
                throw new IllegalArgumentException("takes must be 1 or more, got " + takes);
            }
        }

        /**
         * Three takes, -16 LUFS, a match of 0.8, whole lines up to 60 per template.
         *
         * @param options the voice
         * @return the settings
         */
        public static Settings defaults(SpeechOptions options) {
            return new Settings(options, 3, -16, 0.8, 60);
        }
    }

    /**
     * A take of a line.
     *
     * @param audio the take, at the target loudness
     * @param heard what recognition heard
     * @param asr how closely that matches the line, 0 to 1
     * @param voice how close the voice is to the reference, -1 to 1
     * @param seed the seed it was spoken with
     */
    public record Take(Audio audio, String heard, double asr, double voice, long seed) {
    }

    private final TextToSpeech tts;
    private final SpeechToText stt;
    private final ClipLibrary library;
    private final Settings settings;
    private final NumberSpeller speller;
    private final VoiceEmbedder embedder;
    private final Consumer<String> log;
    private float[] referencePrint;

    /**
     * A recorder.
     *
     * @param tts the speech model
     * @param stt the recognizer that checks takes (Whisper large for German)
     * @param library where clips go
     * @param settings how to record
     * @param embedder the speaker embedding to rank voices by, or {@code null} for a rough
     *     spectral one
     * @param log progress messages, or {@code null}
     */
    public ClipMaker(TextToSpeech tts, SpeechToText stt, ClipLibrary library, Settings settings,
            VoiceEmbedder embedder, Consumer<String> log) {
        this.tts = tts;
        this.stt = stt;
        this.library = library;
        this.settings = settings;
        this.speller = NumberSpeller.of(library.language());
        this.embedder = embedder != null ? embedder : AudioTools::voicePrint;
        this.log = log != null ? log : s -> { };
        Audio ref = settings.options().reference();
        if (ref != null) {
            referencePrint = this.embedder.embed(ref);
        }
    }

    /**
     * Sets the voice takes are ranked against, when the options name a saved voice instead
     * of carrying the reference clip.
     *
     * @param reference the reference clip
     */
    public void referenceVoice(Audio reference) {
        referencePrint = embedder.embed(reference);
    }

    /**
     * The language's number words.
     *
     * @return the speller
     */
    public NumberSpeller speller() {
        return speller;
    }

    // ---------------------------------------------------------------- takes

    /**
     * Speaks a text several times and ranks the takes, best first: takes whose words match
     * come before those that do not, then by match, voice and a length near the others'.
     *
     * @param text the text
     * @return the takes, best first
     */
    public List<Take> takes(String text) {
        List<Take> out = new ArrayList<>();
        String want = AudioTools.normalize(text, speller);
        for (int k = 0; k < settings.takes(); k++) {
            out.add(take(text, want, settings.options().seed() + k));
        }
        // no take heard right: try up to two more seeds before giving up
        for (int k = settings.takes(); k < settings.takes() + 2
                && out.stream().noneMatch(t -> t.asr() >= settings.minAsr()); k++) {
            out.add(take(text, want, settings.options().seed() + k));
        }
        double[] lengths = out.stream().mapToDouble(t -> t.audio().seconds()).sorted().toArray();
        double median = lengths[lengths.length / 2];
        double minAsr = settings.minAsr();
        out.sort(Comparator.comparingDouble((Take t) -> t.asr() >= minAsr ? 0 : 1)
                .thenComparingDouble(t -> -(2 * t.asr() + t.voice()
                        - 0.5 * Math.abs(t.audio().seconds() - median) / median)));
        Take best = out.get(0);
        log.accept(String.format("  %-60s asr %.2f voice %.2f %s", text, best.asr(), best.voice(),
                best.asr() < minAsr ? "(no take matched: heard \"" + best.heard() + "\")" : ""));
        return out;
    }

    private Take take(String text, String want, long seed) {
        Audio a = tts.synthesize(text, settings.options().seed(seed), null);
        a = Loudness.normalize(a, settings.targetLufs(), -1);
        String heard = hear(a.samples(), a.sampleRate());
        double asr = AudioTools.similarity(want, AudioTools.normalize(heard, speller));
        float[] print = embedder.embed(a);
        if (referencePrint == null) {
            referencePrint = print;
        }
        return new Take(a, heard, asr, AudioTools.cosine(print, referencePrint), seed);
    }

    private String hear(float[] x, int rate) {
        int pad = (int) (0.25 * rate);
        float[] padded = new float[x.length + 2 * pad];
        System.arraycopy(x, 0, padded, pad, x.length);
        TranscriptionOptions o = TranscriptionOptions.defaults().language(speller.language()).timestamps(false)
                .conditionOnPreviousText(false);
        return stt.transcribe(new Audio(padded, rate), o, null).text().trim();
    }

    // ---------------------------------------------------------------- whole lines

    /**
     * Records a line whole.
     *
     * @param line the spelled line
     * @return the clip
     */
    public ClipLibrary.Clip full(LineTemplate.Line line) {
        String text = line.text();
        ClipLibrary.Clip have = library.full(text);
        if (have != null) {
            return have;
        }
        Take t = takes(text).get(0);
        Audio a = new Audio(AudioTools.trim(t.audio().samples(), t.audio().sampleRate(), 0.05),
                t.audio().sampleRate());
        return library.put(new ClipLibrary.Clip(ClipLibrary.FULL, line.template(), text, "line", "", t.asr(),
                t.voice()), a);
    }

    /**
     * Records each of a template's lines whole.
     *
     * @param template the template
     * @param valueSets the values of each line
     */
    public void full(LineTemplate template, List<? extends Map<String, ?>> valueSets) {
        for (Map<String, ?> v : valueSets) {
            full(template.render(v, speller));
        }
        library.save();
    }

    // ---------------------------------------------------------------- pieces

    /**
     * A piece to record: words said with one melody.
     *
     * @param text the words
     * @param melody {@link ClipLibrary#MID}, {@code end.}, {@code end?} or {@code end!}
     */
    public record Piece(String text, String melody) {
    }

    /** Pieces said before the last one in a carrier (its other pieces continue the melody). */
    static final int CHAIN = 4;

    /**
     * The pieces a line is said with: its units, each with the melody of its place.
     *
     * @param line the line
     * @return its pieces, in order
     */
    public static List<Piece> pieces(LineTemplate.Line line) {
        List<Piece> out = new ArrayList<>();
        for (LineTemplate.Unit u : line.units()) {
            out.add(new Piece(u.text(), u.position()));
        }
        return out;
    }

    /**
     * Records pieces that the library does not have yet. They are said in carriers: short
     * chains such as "sieben, plus, acht, ist, fünfzehn." where the commas make the speaker
     * pause, so each piece keeps the melody of its place (continuing inside the chain, falling
     * or rising at its end) and is cut out of real silence, never out of running speech. A
     * take counts only if recognition hears the whole chain and the pauses split it into
     * exactly its pieces; chains that do not split are said again in smaller chains.
     *
     * @param wanted the pieces
     */
    public void record(java.util.Collection<Piece> wanted) {
        List<Piece> mids = new ArrayList<>();
        Map<String, List<Piece>> ends = new LinkedHashMap<>();
        java.util.Set<Piece> seen = new java.util.HashSet<>();
        for (Piece p : wanted) {
            if (!seen.add(p) || library.piece(p.text(), p.melody()) != null) {
                continue;
            }
            if (p.melody().equals(ClipLibrary.MID)) {
                mids.add(p);
            } else {
                ends.computeIfAbsent(p.melody(), k -> new ArrayList<>()).add(p);
            }
        }
        List<List<Piece>> chains = new ArrayList<>();
        java.util.ArrayDeque<Piece> midQueue = new java.util.ArrayDeque<>(mids);
        for (List<Piece> group : ends.values()) {
            for (Piece end : group) {
                List<Piece> chain = new ArrayList<>();
                while (chain.size() < CHAIN && !midQueue.isEmpty()) {
                    chain.add(midQueue.poll());
                }
                chain.add(end);
                chains.add(chain);
            }
        }
        while (!midQueue.isEmpty()) {
            // more continuing pieces than endings: end the chain on a piece said falling, kept as a bonus
            List<Piece> chain = new ArrayList<>();
            while (chain.size() < CHAIN && !midQueue.isEmpty()) {
                chain.add(midQueue.poll());
            }
            chain.add(new Piece(chain.get(chain.size() - 1).text(), "end."));
            chains.add(chain);
        }
        int done = 0;
        for (List<Piece> chain : chains) {
            recordChain(chain);
            if (++done % 5 == 0) {
                library.save();
            }
        }
        library.save();
    }

    /** The text a chain is said as: its pieces between commas, ending on the last one's mark. */
    static String carrier(List<Piece> chain) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < chain.size(); i++) {
            String t = chain.get(i).text().replaceAll("[.!?,:;]+$", "");
            if (i == 0 && !t.isEmpty()) {
                t = Character.toUpperCase(t.charAt(0)) + t.substring(1);
            }
            b.append(t);
            b.append(i < chain.size() - 1 ? ", " : chain.get(i).melody().substring(3));
        }
        return b.toString();
    }

    private void recordChain(List<Piece> chain) {
        String text = carrier(chain);
        for (Take t : takes(text)) {
            if (t.asr() < settings.minAsr()) {
                break;   // ranked: no later take matches either
            }
            List<float[]> parts = splitAtPauses(t.audio(), chain);
            if (parts != null) {
                for (int i = 0; i < chain.size(); i++) {
                    Piece p = chain.get(i);
                    {
                        library.put(new ClipLibrary.Clip(ClipLibrary.PIECE, "", p.text(), p.melody(), "", t.asr(),
                                t.voice()), new Audio(parts.get(i), t.audio().sampleRate()));
                    }
                }
                library.pause(pauses(t.audio(), chain.size()));
                return;
            }
        }
        if (chain.size() == 1) {
            log.accept("  ! could not record " + chain.get(0).text() + " (" + chain.get(0).melody() + ")");
            return;
        }
        log.accept("  pauses did not split \"" + text + "\"; saying it in smaller chains");
        int half = chain.size() / 2;
        List<Piece> left = new ArrayList<>(chain.subList(0, half));
        List<Piece> right = new ArrayList<>(chain.subList(half, chain.size()));
        if (!left.isEmpty()) {
            Piece last = left.get(left.size() - 1);
            if (last.melody().equals(ClipLibrary.MID) && left.size() > 1) {
                recordChain(left);
            } else {
                for (Piece p : left) {
                    recordChain(List.of(p, new Piece(p.text(), "end.")));
                }
            }
        }
        recordChain(right);
    }

    /**
     * Splits a take at its pauses: the {@code n - 1} longest silences of at least 40 ms
     * inside the speech, if the pieces between them have lengths that fit their words.
     *
     * @param take the take
     * @param chain the pieces it says
     * @return each piece's samples, with 30 ms of the pause on each side, or {@code null}
     */
    static List<float[]> splitAtPauses(Audio take, List<Piece> chain) {
        float[] x = take.samples();
        int rate = take.sampleRate();
        int hop = rate / AudioTools.FRAME_RATE;
        double[] env = AudioTools.envelope(x, rate);
        int[] sp = AudioTools.speech(env);
        List<int[]> runs = AudioTools.silences(env, sp[0], sp[1], 4);
        int n = chain.size();
        if (runs.size() < n - 1) {
            return null;
        }
        runs.sort(Comparator.comparingInt((int[] r) -> r[1] - r[0]).reversed());
        List<int[]> cuts = new ArrayList<>(runs.subList(0, n - 1));
        cuts.sort(Comparator.comparingInt(r -> r[0]));
        // a fourth of the shortest kept pause must still beat every pause left over, or the split is a guess
        int[] from = new int[n];
        int[] to = new int[n];
        from[0] = sp[0];
        for (int i = 0; i < n - 1; i++) {
            to[i] = cuts.get(i)[0];
            from[i + 1] = cuts.get(i)[1];
        }
        to[n - 1] = sp[1] + 1;
        double lo = Double.POSITIVE_INFINITY;
        double hi = 0;
        for (int i = 0; i < n; i++) {
            int letters = Math.max(2, chain.get(i).text().replaceAll("[^\\p{L}\\d]", "").length());
            double perLetter = (double) (to[i] - from[i]) / letters;
            if (to[i] - from[i] < 10) {
                return null;   // under 100 ms: not a word
            }
            lo = Math.min(lo, perLetter);
            hi = Math.max(hi, perLetter);
        }
        if (hi > 4 * lo) {
            return null;
        }
        int margin = 3;
        List<float[]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int a = Math.max(0, (from[i] - margin) * hop);
            int b = Math.min(x.length, (to[i] + margin) * hop);
            float[] piece = java.util.Arrays.copyOfRange(x, a, b);
            AudioTools.fade(piece, rate, 0.01, 0.01);
            out.add(piece);
        }
        return out;
    }

    /** The lengths of the pauses a take was split at, in seconds. */
    private static List<Double> pauses(Audio take, int n) {
        double[] env = AudioTools.envelope(take.samples(), take.sampleRate());
        int[] sp = AudioTools.speech(env);
        List<int[]> runs = AudioTools.silences(env, sp[0], sp[1], 4);
        runs.sort(Comparator.comparingInt((int[] r) -> r[1] - r[0]).reversed());
        List<Double> out = new ArrayList<>();
        for (int i = 0; i < Math.min(n - 1, runs.size()); i++) {
            out.add((runs.get(i)[1] - runs.get(i)[0]) / (double) AudioTools.FRAME_RATE);
        }
        return out;
    }

    // ---------------------------------------------------------------- choosing

    /**
     * Records a template the best affordable way: every line whole if there are at most
     * {@link Settings#fullBudget()} of them, otherwise the pieces its lines are put together
     * from. Pieces are shared: "acht" said continuing is the same piece in "Was ist acht plus
     * zwei?" and in "acht mal drei ist vierundzwanzig", so a new wording only adds its own
     * words.
     *
     * @param template the template
     * @param example a value for every name
     * @param domains the values each name takes
     * @return {@link Narrator.Mode#FULL} or {@link Narrator.Mode#SPLICED}
     */
    public Narrator.Mode prepare(LineTemplate template, Map<String, ?> example,
            Map<String, ? extends List<?>> domains) {
        long lines = 1;
        for (List<?> d : domains.values()) {
            lines *= Math.max(1, d.size());
        }
        log.accept(template.source() + ": " + lines + " lines");
        if (lines <= settings.fullBudget()) {
            full(template, product(example, domains));
            return Narrator.Mode.FULL;
        }
        List<Piece> wanted = new ArrayList<>(pieces(template.render(example, speller)));
        for (Map.Entry<String, ? extends List<?>> d : domains.entrySet()) {
            for (Object v : d.getValue()) {
                Map<String, Object> values = new LinkedHashMap<>(example);
                values.put(d.getKey(), v);
                LineTemplate.Line line = template.render(values, speller);
                List<LineTemplate.Unit> units = line.units();
                for (LineTemplate.Unit u : units) {
                    if (d.getKey().equals(u.name())) {
                        wanted.add(new Piece(u.text(), u.position()));
                    }
                }
            }
        }
        record(wanted);
        return Narrator.Mode.SPLICED;
    }

    /**
     * Records many lines whole, skipping those already recorded and saving the library every 25
     * lines, so a long run can be stopped and started again where it left off.
     *
     * @param lines the lines
     * @return how many were newly recorded
     */
    public int fullLines(List<LineTemplate.Line> lines) {
        int done = 0;
        int i = 0;
        long start = System.nanoTime();
        for (LineTemplate.Line line : lines) {
            i++;
            if (library.full(line.text()) != null) {
                continue;
            }
            full(line);
            done++;
            if (done % 25 == 0) {
                library.save();
                double perLine = (System.nanoTime() - start) / 1e9 / done;
                log.accept(String.format("  %d/%d lines, %.1f s per line", i, lines.size(), perLine));
            }
        }
        library.save();
        return done;
    }

    /**
     * Records whole every calculation in a range ({@link MathReader#calculations}), read out
     * ("sieben plus acht ist fünfzehn") and, with a question template, asked
     * ("Was ist sieben plus acht?"), so a maths game plays each as one fluent take.
     *
     * @param from the smallest number
     * @param to the largest number
     * @param question a template with an {@code {e:math}} hole, such as
     *     {@code "Was ist {e:math}?"}, or {@code null} for no questions
     * @param operators any of {@code "+", "-", "×", "÷"}
     * @return the lines, recorded or found in the library
     */
    public List<LineTemplate.Line> fullMath(int from, int to, LineTemplate question, String... operators) {
        List<LineTemplate.Line> lines = new ArrayList<>();
        for (String c : MathReader.calculations(from, to, operators)) {
            lines.add(MathReader.read(c, speller));
            if (question != null) {
                lines.add(question.render(Map.of("e", c.substring(0, c.indexOf('=')).trim()), speller));
            }
        }
        log.accept("whole calculations " + from + " to " + to + ": " + lines.size() + " lines");
        fullLines(lines);
        return lines;
    }

    /**
     * Records what calculations need: the numbers {@code from} to {@code to} with each melody
     * (continuing, ending a statement, ending a question) and the words of the operators,
     * so {@link Narrator#sayMath} can read any sum of them.
     *
     * @param from the smallest number
     * @param to the largest number
     * @param operators the symbols, such as {@code "+", "-", "×", "÷", "="}
     */
    public void recordMath(int from, int to, String... operators) {
        List<Piece> wanted = new ArrayList<>();
        for (String op : operators) {
            String w = speller.operator(op);
            if (w != null) {
                wanted.add(new Piece(w, ClipLibrary.MID));
            }
        }
        for (int n = from; n <= to; n++) {
            String w = speller.cardinal(n);
            wanted.add(new Piece(w, ClipLibrary.MID));
            wanted.add(new Piece(w, "end."));
            wanted.add(new Piece(w, "end?"));
        }
        record(wanted);
    }

    // ---------------------------------------------------------------- pieces from fluent sentences

    /**
     * Records pieces out of fluent sentences, so each keeps the pace, melody and linking of
     * running speech. Each sentence is spoken whole with no commas; the pieces already in the
     * library (said on their own) are laid end to end as a guide and aligned to the take by
     * dynamic time warping, which says roughly where each piece ends; each cut is then moved
     * to the quietest frame within 50 ms (a stop's closure, a pause, a fricative's weakest
     * point). The pieces replace those said on their own, marked {@link ClipLibrary#FLUENT}.
     *
     * @param sentences the sentences, each made of pieces the library has
     * @return how many sentences were cut into pieces
     */
    public int recordFluent(List<LineTemplate.Line> sentences) {
        // the guides, read before any piece is replaced
        Map<String, float[]> guide = new LinkedHashMap<>();
        int rate = 0;
        for (LineTemplate.Line line : sentences) {
            for (LineTemplate.Unit u : parts(line)) {
                String k = u.text() + "\u0000" + u.position();
                if (!guide.containsKey(k)) {
                    ClipLibrary.Clip c = library.pieceOrNear(u.text(), u.position());
                    if (c == null) {
                        guide.put(k, null);
                        continue;
                    }
                    Audio a = library.audio(c);
                    rate = a.sampleRate();
                    guide.put(k, AudioTools.trim(a.samples(), a.sampleRate(), 0));
                }
            }
        }
        int done = 0;
        for (LineTemplate.Line line : sentences) {
            List<LineTemplate.Unit> units = parts(line);
            List<float[]> parts = new ArrayList<>();
            boolean ok = units.size() > 1;
            for (LineTemplate.Unit u : units) {
                float[] g = guide.get(u.text() + "\u0000" + u.position());
                if (g == null) {
                    log.accept("  ! never said on its own: " + u.text() + " (" + u.position() + ")");
                    ok = false;
                    break;
                }
                parts.add(g);
            }
            if (!ok) {
                log.accept("  ! so not cut: " + line.text());
                continue;
            }
            boolean cut = false;
            for (Take t : takes(line.text())) {
                if (t.asr() < settings.minAsr()) {
                    break;
                }
                List<float[]> pieces = cutFluent(t.audio().resample(rate), parts);
                if (pieces != null) {
                    for (int i = 0; i < units.size(); i++) {
                        LineTemplate.Unit u = units.get(i);
                        library.put(new ClipLibrary.Clip(ClipLibrary.PIECE, ClipLibrary.FLUENT, u.text(), u.position(),
                                "", t.asr(), t.voice()), new Audio(pieces.get(i), rate));
                    }
                    cut = true;
                    done++;
                    break;
                }
            }
            if (!cut) {
                log.accept("  ! could not cut " + line.text());
            }
            if (done % 10 == 0) {
                library.save();
            }
        }
        library.save();
        return done;
    }

    /**
     * A line's units with each number split into the parts it is built from
     * ({@link NumberSpeller#numberParts()}): "einundfünfzig" becomes "einund", "fünfzig", the
     * last taking the number's place in the line.
     *
     * @param line the line
     * @return the units
     */
    List<LineTemplate.Unit> parts(LineTemplate.Line line) {
        List<LineTemplate.Unit> out = new ArrayList<>();
        for (LineTemplate.Unit u : line.units()) {
            List<String> parts = u.name() == null ? null
                    : NumberSpeller.split(u.text().replaceAll("[\\s.?!,;:]+$", ""), speller.numberParts());
            if (parts == null || parts.size() < 2) {
                out.add(u);
                continue;
            }
            for (int k = 0; k < parts.size(); k++) {
                out.add(new LineTemplate.Unit(parts.get(k), u.name(),
                        k < parts.size() - 1 ? ClipLibrary.MID : u.position()));
            }
        }
        return out;
    }

    /**
     * The calculations to choose fluent sentences from: all of them in a small range, and in a
     * large one (more than 20,000) a fixed sample of 6,000, which still holds every part of
     * every number in each place.
     */
    private static List<String> calculationsToRecord(int from, int to, String... operators) {
        long size = (long) (to - from + 1) * (to - from + 1) * operators.length;
        if (size <= 20000) {
            return MathReader.calculations(from, to, operators);
        }
        java.util.Random random = new java.util.Random(1);
        java.util.Set<String> out = new java.util.LinkedHashSet<>();
        for (int tries = 0; out.size() < 6000 && tries < 200000; tries++) {
            String op = operators[random.nextInt(operators.length)];
            int a = from + random.nextInt(to - from + 1);
            int b = from + random.nextInt(to - from + 1);
            Integer c = switch (op) {
                case "+" -> a + b;
                case "-" -> a - b;
                case "×", "*" -> a * b;
                case "÷", "/" -> b != 0 && a % b == 0 ? a / b : null;
                default -> throw new IllegalArgumentException("not an operator: " + op);
            };
            if (c != null && c >= from && c <= to) {
                out.add(a + " " + op + " " + b + " = " + c);
            }
        }
        return List.copyOf(out);
    }

    /**
     * Cuts a fluent take into its pieces, guided by the same pieces said on their own.
     *
     * @param take the fluent take
     * @param guides each piece said on its own, trimmed, in order
     * @return the pieces, or {@code null} if the take does not line up with the guides
     */
    static List<float[]> cutFluent(Audio take, List<float[]> guides) {
        int rate = take.sampleRate();
        int hop = rate / AudioTools.FRAME_RATE;
        float[] x = AudioTools.trim(take.samples(), rate, 0.02);
        int total = 0;
        for (float[] g : guides) {
            total += g.length;
        }
        float[] joined = new float[total];
        int[] ends = new int[guides.size()];
        int at = 0;
        for (int i = 0; i < guides.size(); i++) {
            System.arraycopy(guides.get(i), 0, joined, at, guides.get(i).length);
            at += guides.get(i).length;
            ends[i] = at / hop;
        }
        float[][] gm = AudioTools.logMel(joined, rate);
        float[][] xm = AudioTools.logMel(x, rate);
        int[] map = AudioTools.warp(gm, xm);
        double[] env = AudioTools.envelope(x, rate);
        int frames = env.length;
        int minFrames = 6;
        List<float[]> out = new ArrayList<>();
        int prev = 0;
        for (int i = 0; i < guides.size(); i++) {
            int end;
            if (i == guides.size() - 1) {
                end = frames;
            } else {
                int aligned = map[Math.min(map.length - 1, Math.max(0, ends[i] - 1))] + 1;
                int lo = prev + minFrames;
                int hi = frames - minFrames * (guides.size() - 1 - i);
                if (lo > hi) {
                    return null;
                }
                end = AudioTools.quietest(env, aligned, 5, lo, hi);
            }
            if (end - prev < minFrames) {
                return null;
            }
            float[] p = AudioTools.slice(x, rate, prev, end);
            AudioTools.fade(p, rate, 0.003, 0.003);
            out.add(p);
            prev = end;
        }
        return out;
    }

    /**
     * Records, out of fluent sentences, every piece that calculations and questions about them
     * need: each number continuing, ending a statement and ending a question, the operators,
     * "ist" and the question's lead-in. The sentences are real calculations ("sieben plus acht
     * ist fünfzehn") and questions ("Was ist sieben plus acht?"), chosen one by one to fill
     * the most pieces still missing. Needs the pieces said on their own first
     * ({@link #recordMath}), as guides for cutting.
     *
     * @param from the smallest number
     * @param to the largest number
     * @param lead the question's lead-in, such as {@code "Was ist"}, or {@code null} for none
     * @param operators any of {@code "+", "-", "×", "÷"}
     * @return the sentences recorded
     */
    public List<LineTemplate.Line> fluentMath(int from, int to, String lead, String... operators) {
        List<LineTemplate.Line> candidates = new ArrayList<>();
        for (String c : calculationsToRecord(from, to, operators)) {
            candidates.add(MathReader.read(c, speller));
            if (lead != null) {
                String[] t = c.split(" ");
                LineTemplate q = LineTemplate.parse(lead + " {a} " + speller.operator(t[1]) + " {b}?");
                candidates.add(q.render(Map.of("a", Integer.parseInt(t[0]), "b", Integer.parseInt(t[2])), speller));
            }
        }
        java.util.Set<String> wanted = new java.util.LinkedHashSet<>();
        List<Piece> guides = new ArrayList<>();
        for (LineTemplate.Line l : candidates) {
            for (LineTemplate.Unit u : parts(l)) {
                if (wanted.add(u.text() + "\u0000" + u.position())) {
                    guides.add(new Piece(u.text().replaceAll("[\\s.?!,;:]+$", ""), u.position()));
                }
            }
        }
        // every piece said on its own first, as a guide for cutting (skips those recorded)
        record(guides);
        List<LineTemplate.Line> chosen = new ArrayList<>();
        while (!wanted.isEmpty()) {
            LineTemplate.Line best = null;
            int gain = 0;
            for (LineTemplate.Line l : candidates) {
                int g = 0;
                for (LineTemplate.Unit u : parts(l)) {
                    if (wanted.contains(u.text() + "\u0000" + u.position())) {
                        g++;
                    }
                }
                if (g > gain) {
                    gain = g;
                    best = l;
                }
            }
            if (best == null) {
                break;
            }
            chosen.add(best);
            for (LineTemplate.Unit u : parts(best)) {
                wanted.remove(u.text() + "\u0000" + u.position());
            }
        }
        log.accept("fluent sentences for " + from + " to " + to + ": " + chosen.size());
        recordFluent(chosen);
        return chosen;
    }

    /**
     * Every combination of values.
     *
     * @param base values for names without a domain
     * @param domains the values of each name
     * @return the combinations
     */
    public static List<Map<String, Object>> product(Map<String, ?> base, Map<String, ? extends List<?>> domains) {
        List<Map<String, Object>> out = new ArrayList<>();
        out.add(new LinkedHashMap<>(base));
        for (Map.Entry<String, ? extends List<?>> d : domains.entrySet()) {
            List<Map<String, Object>> next = new ArrayList<>();
            for (Map<String, Object> m : out) {
                for (Object v : d.getValue()) {
                    Map<String, Object> c = new LinkedHashMap<>(m);
                    c.put(d.getKey(), v);
                    next.add(c);
                }
            }
            out = next;
        }
        return out;
    }
}
