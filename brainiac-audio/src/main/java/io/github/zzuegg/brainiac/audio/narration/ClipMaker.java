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
 * <li>{@link #splice}: the template is spoken with example values (the frame) and once more
 *     per value of each name, all whole lines. Between each two values a join point is
 *     chosen inside the words between them (in "Was ist {a} <b>plus</b> {b}?", inside
 *     "plus") where the takes, aligned by dynamic time warping, sound most alike. A {@link
 *     Narrator} switches takes only there, so no word is ever cut at its edge and each number
 *     keeps the melody of its place.
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
            long seed = settings.options().seed() + k;
            Audio a = tts.synthesize(text, settings.options().seed(seed), null);
            a = Loudness.normalize(a, settings.targetLufs(), -1);
            String heard = hear(a.samples(), a.sampleRate());
            double asr = AudioTools.similarity(want, AudioTools.normalize(heard, speller));
            float[] print = embedder.embed(a);
            if (referencePrint == null) {
                referencePrint = print;
            }
            out.add(new Take(a, heard, asr, AudioTools.cosine(print, referencePrint), seed));
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
        return library.put(new ClipLibrary.Clip(ClipLibrary.FULL, line.template(), text, List.of(), new int[0], false,
                "", t.asr(), t.voice()), a);
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

    // ---------------------------------------------------------------- splicing

    /** Takes recorded before join points are chosen, so the choice sees how the takes differ. */
    static final int JOIN_SAMPLE = 8;

    /**
     * The units of a line that say values.
     *
     * @param units the units
     * @return their indices, in order
     */
    static int[] slots(List<LineTemplate.Unit> units) {
        return java.util.stream.IntStream.range(0, units.size()).filter(i -> units.get(i).name() != null).toArray();
    }

    private static List<String> texts(LineTemplate.Line line) {
        return line.units().stream().map(LineTemplate.Unit::text).toList();
    }

    /**
     * Records a template for splicing. The frame (the template with the example values) and
     * a take for each value of each name are spoken whole. Between each two values a join
     * point is chosen inside the words between them, where the frame and the takes, aligned
     * by dynamic time warping, sound most alike and the sound is steady; every take gets the
     * join points mapped onto it. A spliced line then switches takes only inside a word all
     * of them say, never at the edge of a number.
     *
     * @param template the template
     * @param example a value for every name: the frame's
     * @param domains the values to record for each name
     */
    public void splice(LineTemplate template, Map<String, ?> example, Map<String, ? extends List<?>> domains) {
        String src = template.source();
        LineTemplate.Line frameLine = template.render(example, speller);
        List<LineTemplate.Unit> frameUnits = frameLine.units();
        int[] slots = slots(frameUnits);
        if (slots.length < 2) {
            // one value: every take is a whole line already
            List<Map<String, Object>> all = new ArrayList<>();
            for (Map.Entry<String, ? extends List<?>> d : domains.entrySet()) {
                for (Object v : d.getValue()) {
                    Map<String, Object> values = new LinkedHashMap<>(example);
                    values.put(d.getKey(), v);
                    all.add(values);
                }
            }
            full(template, all);
            return;
        }
        List<LineTemplate.Line> lines = new ArrayList<>();
        java.util.Set<String> seen = new java.util.HashSet<>();
        seen.add(frameLine.text());
        for (Map.Entry<String, ? extends List<?>> d : domains.entrySet()) {
            for (Object v : d.getValue()) {
                Map<String, Object> values = new LinkedHashMap<>(example);
                values.put(d.getKey(), v);
                LineTemplate.Line line = template.render(values, speller);
                if (line.units().size() != frameUnits.size()) {
                    log.accept("  skipped " + line.text() + ": its pieces differ from the frame's");
                } else if (seen.add(line.text()) && library.take(src, line.text()) == null) {
                    lines.add(line);
                }
            }
        }
        ClipLibrary.Clip frame = library.take(src, frameLine.text());
        int first = 0;
        if (frame == null) {
            List<Take> sample = new ArrayList<>();
            for (; first < Math.min(JOIN_SAMPLE, lines.size()); first++) {
                sample.add(takes(lines.get(first).text()).get(0));
            }
            Take ft = null;
            Cut cut = null;
            for (Take t : takes(frameLine.text())) {
                Cut c = cutByRecognition(t.audio(), frameUnits);
                if (cut == null || c.score() > cut.score()) {
                    ft = t;
                    cut = c;
                }
                if (c.score() >= settings.minAsr()) {
                    break;
                }
            }
            log.accept(String.format("  frame of %s cut with match %.2f", src, cut.score()));
            List<Audio> others = sample.stream().map(Take::audio).toList();
            int[] joins = chooseJoins(ft.audio(), cut.frames(), slots, others);
            frame = library.put(new ClipLibrary.Clip(ClipLibrary.TAKE, src, frameLine.text(), texts(frameLine), joins,
                    true, "", ft.asr(), ft.voice()), ft.audio());
            for (int i = 0; i < sample.size(); i++) {
                storeTake(src, frame, lines.get(i), sample.get(i));
            }
            library.save();
        }
        for (int i = first; i < lines.size(); i++) {
            storeTake(src, frame, lines.get(i), takes(lines.get(i).text()).get(0));
            if (i % 10 == 9) {
                library.save();
            }
        }
        library.save();
    }

    private void storeTake(String template, ClipLibrary.Clip frame, LineTemplate.Line line, Take t) {
        int[] joins = mapJoins(library.audio(frame), frame.joins(), t.audio());
        library.put(new ClipLibrary.Clip(ClipLibrary.TAKE, template, line.text(), texts(line), joins, false, "",
                t.asr(), t.voice()), t.audio());
    }

    /**
     * Chooses the join points on the frame: for each two neighbouring values, the 10 ms frame
     * inside the words between them (away from their edges) where the other takes, aligned
     * to the frame, differ least from it and the sound changes least. Values with no words
     * between them join at the cut between them.
     *
     * @param frame the frame
     * @param cuts the frame's unit boundaries, in envelope frames
     * @param slots the units that say values
     * @param others other takes of the template
     * @return a sample of the frame per join
     */
    int[] chooseJoins(Audio frame, int[] cuts, int[] slots, List<Audio> others) {
        int rate = frame.sampleRate();
        int hop = rate / AudioTools.FRAME_RATE;
        float[][] mf = AudioTools.logMel(frame.samples(), rate);
        List<float[][]> mels = new ArrayList<>();
        List<int[]> maps = new ArrayList<>();
        for (Audio o : others) {
            float[][] mo = AudioTools.logMel(o.resample(rate).samples(), rate);
            mels.add(mo);
            maps.add(AudioTools.warp(mf, mo));
        }
        int[] joins = new int[slots.length - 1];
        for (int b = 0; b < joins.length; b++) {
            int from = cuts[slots[b]];            // where the words after value b start
            int to = cuts[slots[b + 1] - 1];      // where value b + 1 starts
            if (to - from < 3) {
                joins[b] = Math.min(mf.length - 1, from) * hop;
                continue;
            }
            int margin = Math.max(1, (int) Math.round(0.2 * (to - from)));
            int best = (from + to) / 2;
            double bestCost = Double.POSITIVE_INFINITY;
            for (int t = from + margin; t <= to - margin && t < mf.length - 1; t++) {
                double cost = 0.5 * AudioTools.distance(mf[t - 1], mf[t + 1]);
                for (int k = 0; k < mels.size(); k++) {
                    float[][] mo = mels.get(k);
                    cost += AudioTools.distance(mf[t], mo[maps.get(k)[t]]) / mels.size();
                }
                if (cost < bestCost) {
                    bestCost = cost;
                    best = t;
                }
            }
            joins[b] = best * hop;
        }
        return joins;
    }

    /**
     * Maps the frame's join points onto another take through the warping path, then moves
     * each up to 30 ms to where the take sounds most like the frame there.
     *
     * @param frame the frame
     * @param joins the frame's join points, in samples
     * @param take another take
     * @return the take's join points, in samples
     */
    static int[] mapJoins(Audio frame, int[] joins, Audio take) {
        int rate = take.sampleRate();
        int hop = rate / AudioTools.FRAME_RATE;
        float[][] mf = AudioTools.logMel(frame.resample(rate).samples(), rate);
        float[][] mt = AudioTools.logMel(take.samples(), rate);
        int[] map = AudioTools.warp(mf, mt);
        int[] out = new int[joins.length];
        for (int b = 0; b < joins.length; b++) {
            int jf = Math.min(mf.length - 1, joins[b] / hop);
            int center = map[jf];
            int best = center;
            double bestCost = Double.POSITIVE_INFINITY;
            for (int t = Math.max(1, center - 3); t <= Math.min(mt.length - 2, center + 3); t++) {
                double cost = AudioTools.distance(mf[jf], mt[t]) + 0.1 * Math.abs(t - center);
                if (cost < bestCost) {
                    bestCost = cost;
                    best = t;
                }
            }
            out[b] = best * hop;
        }
        // keep them in order
        for (int b = 1; b < out.length; b++) {
            out[b] = Math.max(out[b], out[b - 1] + hop);
        }
        return out;
    }

    /**
     * Where a take was cut.
     *
     * @param frames the envelope frame of each boundary between units
     * @param score the lowest recognition match of the checked pieces
     */
    record Cut(int[] frames, double score) {
    }

    /** The words of units {@code from} to {@code to - 1}, normalized. */
    private String words(List<LineTemplate.Unit> units, int from, int to) {
        StringBuilder b = new StringBuilder();
        for (int i = from; i < to; i++) {
            b.append(units.get(i).text()).append(' ');
        }
        return AudioTools.normalize(b.toString(), speller);
    }

    /**
     * How well a cut at {@code frame} splits a take after unit {@code b}: recognition of the
     * audio before it against the units up to {@code b}, and of the audio after it against
     * the rest, the lower of the two. Whole stretches are heard far more reliably than one
     * short word on its own. Only the side at the cut counts: recognition often finishes a
     * stretch cut short with words that are not there, which says nothing about the cut.
     */
    private double boundary(float[] x, int rate, int frames, List<LineTemplate.Unit> units, int b, int frame) {
        double before = AudioTools.prefixSimilarity(words(units, 0, b + 1),
                AudioTools.normalize(hear(AudioTools.slice(x, rate, 0, frame), rate), speller));
        if (before < 0.5) {
            return before;
        }
        double after = AudioTools.suffixSimilarity(words(units, b + 1, units.size()),
                AudioTools.normalize(hear(AudioTools.slice(x, rate, frame, frames), rate), speller));
        return Math.min(before, after);
    }

    /**
     * Cuts a take into its units: each boundary at a quiet dip, tried nearest the place the
     * letters predict first, until recognition hears the right words on both sides of it.
     */
    Cut cutByRecognition(Audio take, List<LineTemplate.Unit> units) {
        float[] x = take.samples();
        int rate = take.sampleRate();
        double[] env = AudioTools.envelope(x, rate);
        int[] sp = AudioTools.speech(env);
        List<AudioTools.Valley> valleys = AudioTools.valleys(env);
        int k = units.size();
        int[] cuts = new int[Math.max(0, k - 1)];
        double[] weight = new double[k];
        double total = 0;
        for (int i = 0; i < k; i++) {
            weight[i] = Math.max(2, AudioTools.normalize(units.get(i).text(), speller).length());
            total += weight[i];
        }
        double score = 1;
        int prev = 0;
        double acc = 0;
        for (int b = 0; b < k - 1; b++) {
            acc += weight[b];
            double expected = sp[0] + (sp[1] - sp[0]) * acc / total;
            int lo = Math.max(prev, sp[0]) + 5;
            int hi = sp[1] - 5;
            List<AudioTools.Valley> cand = new ArrayList<>();
            for (AudioTools.Valley v : valleys) {
                if (v.frame() > lo && v.frame() < hi) {
                    cand.add(v);
                }
            }
            cand.sort(Comparator.comparingDouble(v -> Math.abs(v.frame() - expected) - 0.5 * v.depth()));
            int best = (int) Math.round(expected);
            double bestSim = -1;
            for (int c = 0; c < Math.min(6, cand.size()); c++) {
                int f = cand.get(c).frame();
                double sim = boundary(x, rate, env.length, units, b, f);
                if (sim > bestSim) {
                    bestSim = sim;
                    best = f;
                }
                if (sim >= 0.9) {
                    break;
                }
            }
            cuts[b] = best;
            score = Math.min(score, Math.max(0, bestSim));
            prev = best;
        }
        return new Cut(cuts, score);
    }

    // ---------------------------------------------------------------- choosing

    /**
     * Records a template the best affordable way: every line whole if there are at most
     * {@link Settings#fullBudget()} of them, otherwise pieces for splicing.
     *
     * @param template the template
     * @param example a value for every name (the frame, when splicing)
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
        splice(template, example, domains);
        return Narrator.Mode.SPLICED;
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
