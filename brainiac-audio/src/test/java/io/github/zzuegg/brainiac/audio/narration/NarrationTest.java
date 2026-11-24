package io.github.zzuegg.brainiac.audio.narration;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.SpeechOptions;
import io.github.zzuegg.brainiac.audio.SpeechToText;
import io.github.zzuegg.brainiac.audio.TextToSpeech;
import io.github.zzuegg.brainiac.audio.Transcription;
import io.github.zzuegg.brainiac.audio.TranscriptionOptions;
import io.github.zzuegg.brainiac.audio.Voice;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NarrationTest {

    private static final NumberSpeller DE = NumberSpeller.of("de");
    private static final NumberSpeller EN = NumberSpeller.of("en");

    @Test
    void germanCardinals() {
        assertEquals("null", DE.cardinal(0));
        assertEquals("eins", DE.cardinal(1));
        assertEquals("siebzehn", DE.cardinal(17));
        assertEquals("einundzwanzig", DE.cardinal(21));
        assertEquals("dreißig", DE.cardinal(30));
        assertEquals("hunderteins", DE.cardinal(101));
        assertEquals("hunderteinundzwanzig", DE.cardinal(121));
        assertEquals("zweitausendeinundzwanzig", DE.cardinal(2021));
        assertEquals("hunderteintausend", DE.cardinal(101_000));
        assertEquals("eine Million", DE.cardinal(1_000_000));
        assertEquals("zwei Millionen eins", DE.cardinal(2_000_001));
        assertEquals("minus fünf", DE.cardinal(-5));
        assertEquals("eine", DE.cardinal(1, "eine"));
        assertEquals("drei", DE.cardinal(3, "eine"));
    }

    @Test
    void ordinals() {
        assertEquals("ersten", DE.ordinal(1, "en"));
        assertEquals("dritte", DE.ordinal(3, null));
        assertEquals("siebter", DE.ordinal(7, "er"));
        assertEquals("achten", DE.ordinal(8, "en"));
        assertEquals("zwanzigsten", DE.ordinal(20, "en"));
        assertEquals("einundzwanzigste", DE.ordinal(21, "e"));
        assertEquals("hundertersten", DE.ordinal(101, "en"));
        assertEquals("first", EN.ordinal(1, null));
        assertEquals("twenty-third", EN.ordinal(23, null));
        assertEquals("twelfth", EN.ordinal(12, null));
    }

    @Test
    void decimalsTimesAndClocks() {
        assertEquals("drei Komma eins vier", DE.decimal(new java.math.BigDecimal("3.14")));
        assertEquals("three point one four", EN.decimal(new java.math.BigDecimal("3.14")));
        assertEquals("eine Minute und dreiundzwanzig Komma vier zwei Sekunden", DE.duration(83.42, 2));
        assertEquals("one minute and twenty-three point four two seconds", EN.duration(83.42, 2));
        assertEquals("eine Sekunde", DE.duration(1, 0));
        assertEquals("vierzehn Uhr fünf", DE.clock(14, 5));
        assertEquals("ein Uhr", DE.clock(1, 0));
        assertEquals("two oh five", EN.clock(14, 5));
    }

    @Test
    void spellsFreeText() {
        assertEquals("zwölf plus sieben ist neunzehn", DE.spell("12 + 7 = 19"));
        assertEquals("zwei Komma fünf und vierzehn Uhr fünf", DE.spell("2,5 und 14:05"));
        assertEquals("fifty percent", EN.spell("50%"));
    }

    @Test
    void readsCalculations() {
        assertEquals("zwölf plus sieben ist neunzehn", MathReader.read("12 + 7 = 19", DE).text());
        assertEquals("twelve plus seven equals nineteen", MathReader.read("12 + 7 = 19", EN).text());
        assertEquals("Klammer auf zwei plus drei Klammer zu mal vier ist zwanzig",
                MathReader.read("(2 + 3) × 4 = 20", DE).text());
        assertEquals("minus five minus three equals minus eight", MathReader.read("-5 - 3 = -8", EN).text());
        assertEquals("zwei Komma fünf mal vier ist zehn", MathReader.read("2,5 * 4 = 10", DE).text());
        assertEquals("three plus what equals ten", MathReader.read("3 + ? = 10", EN).text());
        // every sum of the same shape shares one template, so its recordings serve all of them
        assertEquals(MathReader.template("12 + 7 = 19", DE).template().source(),
                MathReader.template("3 + 4 = 7", DE).template().source());
        assertThrows(IllegalArgumentException.class, () -> MathReader.read("2 & 3", DE));
    }

    @Test
    void templatesSplitIntoUnits() {
        LineTemplate t = LineTemplate.parse("Noch {n:cardinal:eine} {n|Runde|Runden}!");
        assertEquals("Noch eine Runde!", t.render(Map.of("n", 1), DE).text());
        assertEquals("Noch drei Runden!", t.render(Map.of("n", 3), DE).text());
        List<LineTemplate.Unit> u = t.render(Map.of("n", 3), DE).units();
        assertEquals(2, u.size());
        assertEquals("drei Runden!", u.get(1).text());
        assertEquals("n", u.get(1).name());
        assertEquals("end!", u.get(1).position());

        LineTemplate q = LineTemplate.parse("Was ist {a} plus {b}?");
        List<LineTemplate.Unit> qu = q.render(Map.of("a", 7, "b", 8), DE).units();
        assertEquals(List.of("Was ist", "sieben", "plus", "acht?"), qu.stream().map(LineTemplate.Unit::text).toList());
        assertEquals("mid", qu.get(1).position());
        assertEquals("end?", qu.get(3).position());

        assertEquals("Du bist auf dem dritten Platz!",
                LineTemplate.parse("Du bist auf dem {n:ordinal:en} Platz!").render(Map.of("n", 3), DE).text());
        assertEquals("Es ist vierzehn Uhr fünf.",
                LineTemplate.parse("Es ist {t:clock}.").render(Map.of("t", LocalTime.of(14, 5)), DE).text());
        assertThrows(IllegalArgumentException.class, () -> q.render(Map.of("a", 1), DE));
    }

    @Test
    void cutChecksIgnoreWordsBeyondTheCut() {
        assertEquals(1.0, AudioTools.prefixSimilarity("wasist", "wasistdas"), 1e-9);
        assertEquals(1.0, AudioTools.suffixSimilarity("plusacht", "sprechtplusacht"), 1e-9);
        assertTrue(AudioTools.prefixSimilarity("wasistdrei", "wasist") < 0.7);
        assertTrue(AudioTools.suffixSimilarity("siebenplus", "benplus") < 0.8);
    }

    @Test
    void joinsLineUpWaveforms() {
        // two takes of the same vowel, out of phase and at other levels: the join must not jump
        int rate = 24000;
        float[] a = new float[rate];
        float[] b = new float[rate];
        for (int i = 0; i < rate; i++) {
            a[i] = (float) (0.5 * Math.sin(2 * Math.PI * 200 * i / rate));
            b[i] = (float) (0.3 * Math.sin(2 * Math.PI * 200 * i / rate + 2.0));
        }
        float[] x = AudioTools.splice(List.of(new AudioTools.Segment(a, 0, 12000),
                new AudioTools.Segment(b, 9000, rate)), rate, 0.025, 0.008);
        float jump = 0;
        for (int i = 1; i < x.length; i++) {
            jump = Math.max(jump, Math.abs(x[i] - x[i - 1]));
        }
        // a 200 Hz sine of 0.5 moves at most 0.5 * 2 pi * 200 / 24000 = 0.026 a sample
        assertTrue(jump < 0.03, "jump " + jump);
        // the level is matched at the join, then eases back to the second take's own
        assertEquals(0.5, AudioTools.rms(x, 12100, 600) * Math.sqrt(2), 0.06);
        assertEquals(0.3, AudioTools.rms(x, x.length - 1000, 600) * Math.sqrt(2), 0.03);
    }

    @Test
    void loudnessNormalizes() {
        int rate = 24000;
        float[] x = new float[rate * 2];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (0.05 * Math.sin(2 * Math.PI * 1000 * i / rate));
        }
        Audio quiet = new Audio(x, rate);
        Audio loud = Loudness.normalize(quiet, -16, -1);
        assertEquals(-16, Loudness.lufs(loud), 0.2);
        // a full-scale 1 kHz sine is about -3 LUFS
        float[] y = new float[rate * 2];
        for (int i = 0; i < y.length; i++) {
            y[i] = (float) Math.sin(2 * Math.PI * 1000 * i / rate);
        }
        assertEquals(-3.0, Loudness.lufs(new Audio(y, rate)), 0.3);
    }

    @Test
    void splicesAndPrefersWholeLines(@TempDir Path dir) {
        ClipLibrary lib = ClipLibrary.open(dir, "de");
        ToneSpeech tts = new ToneSpeech();
        ClipMaker maker = new ClipMaker(tts, new ToneHearing(tts), lib,
                new ClipMaker.Settings(SpeechOptions.defaults(), 2, -16, 0.8, 5), null, null);
        LineTemplate t = LineTemplate.parse("Was ist {a} plus {b}?");
        List<Integer> range = IntStream.rangeClosed(1, 6).boxed().toList();
        Map<String, List<Integer>> domains = new LinkedHashMap<>();
        domains.put("a", range);
        domains.put("b", range);
        assertEquals(Narrator.Mode.SPLICED, maker.prepare(t, Map.of("a", 3, "b", 5), domains));
        maker.full(t.render(Map.of("a", 2, "b", 6), DE));
        lib.save();

        Narrator n = Narrator.open(dir);
        Narrator.Spoken spliced = n.say(t.source(), Map.of("a", 4, "b", 2));
        assertEquals(Narrator.Mode.SPLICED, spliced.mode());
        assertEquals("Was ist vier plus zwei?", spliced.text());
        // the spliced audio says the right words, in order
        assertEquals("was ist vier plus zwei", new ToneHearing(tts).words(spliced.audio()));
        assertEquals(2, spliced.clips().size());   // one join, inside "plus"
        // a line one take already says needs no join
        assertEquals(1, n.say(t.source(), Map.of("a", 4, "b", 5)).clips().size());
        assertEquals(Narrator.Mode.FULL, n.say(t.source(), Map.of("a", 2, "b", 6)).mode());
        assertEquals(List.of("neun?"), n.missing(t.render(Map.of("a", 1, "b", 9), DE)));
        assertThrows(IllegalStateException.class, () -> n.say(t.source(), Map.of("a", 1, "b", 9)));

        ClipMaker small = new ClipMaker(tts, new ToneHearing(tts), lib,
                new ClipMaker.Settings(SpeechOptions.defaults(), 1, -16, 0.8, 5), null, null);
        LineTemplate laps = LineTemplate.parse("Noch {n:cardinal:eine} {n|Runde|Runden}!");
        assertEquals(Narrator.Mode.FULL, small.prepare(laps, Map.of("n", 1), Map.of("n", List.of(1, 2, 3))));
        assertEquals(Narrator.Mode.FULL, Narrator.open(dir).say(laps.source(), Map.of("n", 1)).mode());
    }

    // ---------------------------------------------------------------- a speech model made of tones

    /** Speaks each word as a tone of its own pitch, its length set by its letters, with dips between words. */
    static final class ToneSpeech implements TextToSpeech {
        static final int RATE = 16000;
        final List<String> vocabulary = new ArrayList<>();

        double pitch(String word) {
            int i = vocabulary.indexOf(word);
            if (i < 0) {
                vocabulary.add(word);
                i = vocabulary.size() - 1;
            }
            return 300 + 97 * i;
        }

        static List<String> words(String text) {
            List<String> out = new ArrayList<>();
            for (String w : text.toLowerCase(java.util.Locale.ROOT).split("[^\\p{L}\\d]+")) {
                if (!w.isEmpty()) {
                    out.add(w);
                }
            }
            return out;
        }

        @Override
        public int sampleRate() {
            return RATE;
        }

        @Override
        public List<Voice> voices() {
            return List.of(new Voice("tone", "Tone", "de"));
        }

        @Override
        public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
            List<Float> out = new ArrayList<>();
            java.util.Random r = new java.util.Random(options.seed());
            for (int k = 0; k < 2000; k++) {
                out.add(0f);
            }
            for (String w : words(text)) {
                double f = pitch(w);
                int len = (int) (RATE * (0.06 + 0.04 * w.length()) * (0.95 + 0.1 * r.nextDouble()));
                for (int i = 0; i < len; i++) {
                    double env = Math.sin(Math.PI * i / len);
                    out.add((float) (0.3 * Math.sqrt(env) * Math.sin(2 * Math.PI * f * i / RATE)));
                }
                for (int k = 0; k < 500; k++) {
                    out.add(0.002f * (float) r.nextGaussian());
                }
            }
            for (int k = 0; k < 2000; k++) {
                out.add(0f);
            }
            float[] x = new float[out.size()];
            for (int i = 0; i < x.length; i++) {
                x[i] = out.get(i);
            }
            return new Audio(x, RATE);
        }

        @Override
        public void close() {
        }
    }

    /** Hears {@link ToneSpeech}: each burst's pitch names its word; bursts under half their length are not heard. */
    static final class ToneHearing implements SpeechToText {
        final ToneSpeech tts;

        ToneHearing(ToneSpeech tts) {
            this.tts = tts;
        }

        String words(Audio audio) {
            float[] x = audio.resample(ToneSpeech.RATE).samples();
            List<String> out = new ArrayList<>();
            double[] env = AudioTools.envelope(x, ToneSpeech.RATE);
            double peak = java.util.Arrays.stream(env).max().orElse(0);
            int f = 0;
            while (f < env.length) {
                if (env[f] < peak - 20) {
                    f++;
                    continue;
                }
                int g = f;
                while (g < env.length && env[g] >= peak - 20) {
                    g++;
                }
                float[] burst = AudioTools.slice(x, ToneSpeech.RATE, f, g);
                String w = word(burst);
                if (w != null) {
                    out.add(w);
                }
                f = g;
            }
            return String.join(" ", out);
        }

        private String word(float[] burst) {
            // the strongest frequency, by counting zero crossings
            // the tone's pitch, by counting zero crossings where the tone (not the noise between words) is
            float top = 0;
            for (float v : burst) {
                top = Math.max(top, Math.abs(v));
            }
            int crossings = 0;
            int first = -1;
            int last = -1;
            for (int i = 1; i < burst.length; i++) {
                boolean sign = (burst[i - 1] < 0) != (burst[i] < 0);
                if (sign && Math.max(Math.abs(burst[i - 1]), Math.abs(burst[i])) > 0.05 * top) {
                    crossings++;
                    if (first < 0) {
                        first = i;
                    }
                    last = i;
                }
            }
            if (crossings < 4) {
                return null;
            }
            double hz = (crossings - 1) / 2.0 * ToneSpeech.RATE / (last - first);
            String best = null;
            double bestErr = 40;
            for (String w : tts.vocabulary) {
                double err = Math.abs(tts.pitch(w) - hz);
                if (err < bestErr) {
                    bestErr = err;
                    best = w;
                }
            }
            if (best == null) {
                return null;
            }
            double expected = ToneSpeech.RATE * (0.06 + 0.04 * best.length());
            return last - first > 0.5 * expected ? best : null;
        }

        @Override
        public Transcription transcribe(float[] samples, TranscriptionOptions options,
                Consumer<Transcription.Segment> onSegment) {
            String text = words(new Audio(samples, SAMPLE_RATE));
            return new Transcription(text, "de", List.of(), samples.length / (double) SAMPLE_RATE);
        }

        @Override
        public List<String> languages() {
            return List.of("de");
        }

        @Override
        public boolean canTranslate() {
            return false;
        }

        @Override
        public void close() {
        }
    }
}
