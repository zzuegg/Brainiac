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
    void recordsEveryCalculationWhole(@TempDir Path dir) {
        assertEquals(655, MathReader.calculations(0, 20, "+", "-", "×", "÷").size());
        assertEquals(List.of("6 ÷ 1 = 6", "6 ÷ 2 = 3", "6 ÷ 3 = 2", "6 ÷ 6 = 1"),
                MathReader.calculations(1, 6, "÷").stream().filter(c -> c.startsWith("6 ")).toList());
        ClipLibrary lib = ClipLibrary.open(dir, "de");
        ToneSpeech tts = new ToneSpeech();
        ClipMaker maker = new ClipMaker(tts, new ToneHearing(tts), lib,
                new ClipMaker.Settings(SpeechOptions.defaults(), 1, -16, 0.8, 5), null, null);
        List<LineTemplate.Line> lines = maker.fullMath(1, 4, LineTemplate.parse("Was ist {e:math}?"), "+");
        assertEquals(12, lines.size());
        assertEquals("Was ist zwei plus eins?", lines.get(7).text());
        assertEquals(0, maker.fullLines(lines));   // all recorded already
        Narrator n = Narrator.open(dir);
        assertEquals(Narrator.Mode.FULL, n.sayMath("2 + 1 = 3").mode());
        assertEquals(Narrator.Mode.FULL, n.say("Was ist {e:math}?", Map.of("e", "2 + 1")).mode());
    }

    @Test
    void bendsPiecesToOneSentenceMelody() {
        int rate = 24000;
        float[] glide = new float[rate / 2];
        double phase = 0;
        for (int i = 0; i < glide.length; i++) {
            phase += 2 * Math.PI * (250 + 150.0 * i / glide.length) / rate;   // a word rising like a list item
            glide[i] = (float) (0.5 * Math.sin(phase));
        }
        double[] hz = Prosody.pitch(glide, rate);
        assertEquals(250 + 150 * 0.055 / 0.5, hz[5], 6);
        assertEquals(250 + 150 * (hz.length - 5.5) / 100 / 0.5, hz[hz.length - 6], 10);
        // inside a line the rise is taken out: lowered more at the end than at the start
        double[] mid = Narrator.melody(glide, rate, 300, 0, 0.3, false);
        assertTrue(mid[mid.length - 6] < mid[5] * 0.8, mid[5] + " " + mid[mid.length - 6]);
        float[] bent = Prosody.reshape(glide, rate, 1, mid);
        assertEquals(glide.length, bent.length, 2);
        double[] after = Prosody.pitch(bent, rate);
        assertTrue(after[after.length - 6] - after[5] < 60, after[5] + " " + after[after.length - 6]);
        // the last word keeps its own movement, only moved
        double[] last = Narrator.melody(glide, rate, 300, 0.7, 1, true);
        assertEquals(last[5], last[last.length - 6], 0.02);
    }

    @Test
    void stretchKeepsPitch() {
        int rate = 24000;
        float[] x = new float[rate];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (0.5 * Math.sin(2 * Math.PI * 220 * i / rate));
        }
        float[] y = AudioTools.stretch(x, rate, 1.25);
        assertEquals(x.length / 1.25, y.length, 2);
        int crossings = 0;
        for (int i = 1 + rate / 20; i < y.length - rate / 20; i++) {
            if ((y[i - 1] < 0) != (y[i] < 0)) {
                crossings++;
            }
        }
        double hz = crossings / 2.0 / ((y.length - rate / 10.0) / rate);
        assertTrue(Math.abs(hz - 220) < 6, "pitch kept: " + hz);
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
    void carriersSayPiecesWithTheirMelody() {
        List<ClipMaker.Piece> chain = List.of(new ClipMaker.Piece("sieben", "mid"), new ClipMaker.Piece("plus", "mid"),
                new ClipMaker.Piece("acht?", "end?"));
        assertEquals("Sieben, plus, acht?", ClipMaker.carrier(chain));
        ToneSpeech tts = new ToneSpeech();
        Audio take = tts.synthesize(ClipMaker.carrier(chain), SpeechOptions.defaults(), null);
        List<float[]> parts = ClipMaker.splitAtPauses(take, chain);
        assertEquals(3, parts.size());
        assertEquals("plus", new ToneHearing(tts).words(new Audio(parts.get(1), ToneSpeech.RATE)));
        // a chain whose pauses do not split it into its pieces is not cut at a guess
        assertEquals(null, ClipMaker.splitAtPauses(tts.synthesize("Sieben plus acht?", SpeechOptions.defaults(), null),
                chain));
    }

    @Test
    void putsLinesTogetherFromSharedPieces(@TempDir Path dir) {
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
        maker.recordMath(0, 6, "+", "=");
        maker.full(t.render(Map.of("a", 2, "b", 6), DE));
        lib.save();
        assertTrue(lib.pauseSeconds() > 0.1);

        double slowdown = Narrator.open(dir).speed();
        assertTrue(slowdown >= 1 && slowdown <= 1.5, "speed " + slowdown);
        // the tone "recognizer" knows words by their pitch and the dips between them, so keep
        // both as recorded here
        Narrator n = new Narrator(ClipLibrary.open(dir), new Narrator.Settings(0.03, 1, -16, true, false));
        Narrator.Spoken spoken = n.say(t.source(), Map.of("a", 4, "b", 2));
        assertEquals(Narrator.Mode.SPLICED, spoken.mode());
        assertEquals("was ist vier plus zwei", new ToneHearing(tts).words(spoken.audio()));
        assertEquals(List.of("mid", "mid", "mid", "end?"), spoken.clips().stream().map(ClipLibrary.Clip::melody)
                .toList());
        // a calculation shares the numbers and operators, whatever its wording
        Narrator.Spoken sum = n.sayMath("2 + 3 = 5");
        assertEquals("zwei plus drei ist fünf", new ToneHearing(tts).words(sum.audio()));
        assertEquals("end.", sum.clips().get(4).melody());
        assertEquals(Narrator.Mode.FULL, n.say(t.source(), Map.of("a", 2, "b", 6)).mode());
        assertEquals(List.of("neun?"), n.missing(t.render(Map.of("a", 1, "b", 9), DE)));
        assertThrows(IllegalStateException.class, () -> n.say(t.source(), Map.of("a", 1, "b", 9)));

        LineTemplate laps = LineTemplate.parse("Noch {n:cardinal:eine} {n|Runde|Runden}!");
        ClipMaker small = new ClipMaker(tts, new ToneHearing(tts), lib,
                new ClipMaker.Settings(SpeechOptions.defaults(), 1, -16, 0.8, 5), null, null);
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
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("([\\p{L}\\d]+)([^\\p{L}\\d]*)")
                    .matcher(text.toLowerCase(java.util.Locale.ROOT));
            while (m.find()) {
                String w = m.group(1);
                double f = pitch(w);
                int len = (int) (RATE * (0.06 + 0.04 * w.length()) * (0.95 + 0.1 * r.nextDouble()));
                for (int i = 0; i < len; i++) {
                    double env = Math.sin(Math.PI * i / len);
                    out.add((float) (0.3 * Math.sqrt(env) * Math.sin(2 * Math.PI * f * i / RATE)));
                }
                int gap = m.group(2).matches(".*[,.?!].*") ? 3200 : 500;   // a pause at a comma
                for (int k = 0; k < gap; k++) {
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
