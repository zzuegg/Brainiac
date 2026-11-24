package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.SpeechOptions;
import io.github.zzuegg.brainiac.audio.SpeechToText;
import io.github.zzuegg.brainiac.audio.TextToSpeech;
import io.github.zzuegg.brainiac.audio.TranscriptionOptions;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
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
 * <li>{@link #splice}: the template is spoken once with example values (the frame), cut at
 *     the quiet dips between its pieces where recognition confirms the words on each side,
 *     and for each value a take of the same template is aligned to the frame by dynamic time
 *     warping and the value's words cut out, so the number keeps the melody of its place.
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
    private final Map<String, Take> frames = new HashMap<>();
    private final Map<String, int[]> frameCuts = new HashMap<>();

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
        return library.put(new ClipLibrary.Clip(ClipLibrary.FULL, line.template(), -1, text, "line", "", t.asr(),
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

    // ---------------------------------------------------------------- splicing

    /**
     * Records a template's pieces for splicing: the frame (the template with the example
     * values), then each value of each name in a take of its own.
     *
     * @param template the template
     * @param example a value for every name; the literal words are cut from this take
     * @param domains the values to record for each name
     */
    public void splice(LineTemplate template, Map<String, ?> example, Map<String, ? extends List<?>> domains) {
        LineTemplate.Line frameLine = template.render(example, speller);
        Take frame = frame(template, frameLine);
        int[] cuts = frameCuts.get(template.source());
        for (Map.Entry<String, ? extends List<?>> d : domains.entrySet()) {
            String name = d.getKey();
            for (Object v : d.getValue()) {
                Map<String, Object> values = new LinkedHashMap<>(example);
                values.put(name, v);
                LineTemplate.Line line = template.render(values, speller);
                List<LineTemplate.Unit> units = line.units();
                if (units.size() != frameLine.units().size()) {
                    log.accept("  skipped " + line.text() + ": its pieces differ from the frame's");
                    continue;
                }
                boolean need = false;
                for (int i = 0; i < units.size(); i++) {
                    if (name.equals(units.get(i).name()) && library.unit(template.source(), i, units.get(i).text())
                            == null) {
                        need = true;
                    }
                }
                if (need) {
                    recordValue(template, frame, cuts, line, name);
                }
            }
            library.save();
        }
        library.save();
    }

    private Take frame(LineTemplate template, LineTemplate.Line line) {
        Take have = frames.get(template.source());
        if (have != null) {
            return have;
        }
        List<LineTemplate.Unit> units = line.units();
        Take best = null;
        int[] bestCuts = null;
        double bestScore = -1;
        for (Take t : takes(line.text())) {
            Cut c = cutByRecognition(t.audio(), units);
            if (c.score() > bestScore) {
                best = t;
                bestCuts = c.frames();
                bestScore = c.score();
            }
            if (c.score() >= settings.minAsr()) {
                break;
            }
        }
        log.accept(String.format("  frame of %s cut with match %.2f", template.source(), bestScore));
        frames.put(template.source(), best);
        frameCuts.put(template.source(), bestCuts);
        store(template, line, best, bestCuts, null);
        return best;
    }

    private void recordValue(LineTemplate template, Take frame, int[] frameCut, LineTemplate.Line line, String name) {
        List<LineTemplate.Unit> units = line.units();
        Take best = null;
        int[] bestCuts = null;
        double bestScore = -1;
        for (Take t : takes(line.text())) {
            Cut c = cutByWarp(frame.audio(), frameCut, t.audio(), units, name);
            if (c.score() > bestScore) {
                best = t;
                bestCuts = c.frames();
                bestScore = c.score();
            }
            if (c.score() >= settings.minAsr()) {
                break;
            }
        }
        if (bestScore < settings.minAsr()) {
            log.accept(String.format("  ! %s: best cut matched only %.2f", line.text(), bestScore));
        }
        store(template, line, best, bestCuts, name);
    }

    /** Stores the units of a cut take: all of them, or only those of one name. */
    private void store(LineTemplate template, LineTemplate.Line line, Take take, int[] cuts, String only) {
        List<LineTemplate.Unit> units = line.units();
        Audio a = take.audio();
        int rate = a.sampleRate();
        int hop = rate / AudioTools.FRAME_RATE;
        int half = (int) (0.006 * rate);   // half a crossfade of real audio past each cut
        for (int i = 0; i < units.size(); i++) {
            LineTemplate.Unit u = units.get(i);
            if (only != null && !only.equals(u.name())) {
                continue;
            }
            int from = i == 0 ? 0 : Math.max(0, cuts[i - 1] * hop - half);
            int to = i == units.size() - 1 ? a.samples().length : Math.min(a.samples().length, cuts[i] * hop + half);
            float[] x = java.util.Arrays.copyOfRange(a.samples(), from, Math.max(from, to));
            library.put(new ClipLibrary.Clip(ClipLibrary.UNIT, template.source(), i, u.text(), u.position(), "",
                    take.asr(), take.voice()), new Audio(x, rate));
        }
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

    /**
     * Cuts a take of the template by aligning it to the frame: each of the frame's cuts maps
     * through the warping path and snaps to a nearby dip; the edges of the units of {@code
     * name} are then checked by recognition on both sides, moving to other dips until they match.
     */
    Cut cutByWarp(Audio frame, int[] frameCut, Audio take, List<LineTemplate.Unit> units, String name) {
        int rate = take.sampleRate();
        float[] x = take.samples();
        double[] env = AudioTools.envelope(x, rate);
        List<AudioTools.Valley> valleys = AudioTools.valleys(env);
        int[] map = AudioTools.warp(AudioTools.logMel(frame.resample(rate).samples(), rate),
                AudioTools.logMel(x, rate));
        int[] cuts = new int[frameCut.length];
        for (int b = 0; b < frameCut.length; b++) {
            int mapped = map[Math.min(map.length - 1, frameCut[b])];
            cuts[b] = snap(valleys, mapped, 8, mapped);
        }
        double score = 1;
        for (int b = 0; b < cuts.length; b++) {
            boolean edge = name.equals(units.get(b).name()) || name.equals(units.get(b + 1).name());
            if (!edge) {
                continue;
            }
            double bestSim = -1;
            int best = cuts[b];
            for (int f : options(valleys, cuts[b], 15)) {
                if (b > 0 && f <= cuts[b - 1] + 4 || b + 1 < cuts.length && f >= cuts[b + 1] - 4) {
                    continue;
                }
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
        }
        return new Cut(cuts, score);
    }

    /** The dip nearest a frame (deeper ones preferred) within a reach, or a fallback. */
    private static int snap(List<AudioTools.Valley> valleys, int frame, int reach, int fallback) {
        int best = fallback;
        double bestCost = Double.POSITIVE_INFINITY;
        for (AudioTools.Valley v : valleys) {
            int d = Math.abs(v.frame() - frame);
            if (d <= reach) {
                double cost = d - 0.3 * v.depth();
                if (cost < bestCost) {
                    bestCost = cost;
                    best = v.frame();
                }
            }
        }
        return best;
    }

    /** A frame and the dips near it, nearest first, at most three. */
    private static List<Integer> options(List<AudioTools.Valley> valleys, int frame, int reach) {
        List<Integer> out = new ArrayList<>();
        out.add(frame);
        List<AudioTools.Valley> near = new ArrayList<>();
        for (AudioTools.Valley v : valleys) {
            if (Math.abs(v.frame() - frame) <= reach && v.frame() != frame) {
                near.add(v);
            }
        }
        near.sort(Comparator.comparingInt(v -> Math.abs(v.frame() - frame)));
        for (int i = 0; i < Math.min(2, near.size()); i++) {
            out.add(near.get(i).frame());
        }
        return out;
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
