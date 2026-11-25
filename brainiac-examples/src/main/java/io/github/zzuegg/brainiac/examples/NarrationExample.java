package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.audio.Audio;
import io.github.zzuegg.brainiac.audio.Qwen3Speech;
import io.github.zzuegg.brainiac.audio.SpeechModels;
import io.github.zzuegg.brainiac.audio.SpeechOptions;
import io.github.zzuegg.brainiac.audio.SpeechToText;
import io.github.zzuegg.brainiac.audio.TextToSpeech;
import io.github.zzuegg.brainiac.audio.TranscriptionOptions;
import io.github.zzuegg.brainiac.audio.Wav;
import io.github.zzuegg.brainiac.audio.narration.ClipLibrary;
import io.github.zzuegg.brainiac.audio.narration.ClipMaker;
import io.github.zzuegg.brainiac.audio.narration.LineTemplate;
import io.github.zzuegg.brainiac.audio.narration.Loudness;
import io.github.zzuegg.brainiac.audio.narration.MathReader;
import io.github.zzuegg.brainiac.audio.narration.Narrator;
import io.github.zzuegg.brainiac.audio.narration.NumberSpeller;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Builds a narrator's number library with Qwen3-TTS and compares three ways of saying lines
 * with numbers in them: words spoken alone and joined (as the narrator lines of the kids'
 * games were cut), shared pieces (each number once per melody, each operator once, cut from
 * comma carriers at real pauses) put together at the speaker's own pauses, and whole lines.
 *
 * <pre>
 * ./gradlew :brainiac-examples:run --args="narration --base models/Qwen3-TTS-12Hz-1.7B-Base
 *     --whisper models/whisper-large-v3 --design models/Qwen3-TTS-12Hz-1.7B-VoiceDesign --out narration"
 * </pre>
 *
 * <p>The voice is designed once from a description with the VoiceDesign model (or given with
 * {@code --reference file.wav --reference-text "..."}), then every line is cloned from that one
 * clip by the Base model at a low temperature. Options: {@code --language de|en},
 * {@code --device name}, {@code --takes N}, {@code --temperature T}, {@code --quick} (smaller ranges),
 * {@code --whole-math [top]} (also record whole every calculation from 0 to top, 20 by default,
 * read out and asked; about 1,300 lines for 20, and a stopped run picks up where it left off),
 * {@code --parts [top]} (numbers built from parts such as zweihundert|einund|fünfzig, cut from
 * fluent sums: alone, just the parts of a few test lines with 21, 51, 100, 121 and 251; with
 * top, every part up to top), {@code --male} (design a calm male
 * teacher's voice instead of the female narrator), {@code --calm} (a calmer female voice); give
 * each voice its own {@code --out} folder.
 * Writes {@code reference.wav}, {@code library-<lang>/} and {@code samples/} with a {@code report.md}.
 */
public final class NarrationExample {

    private NarrationExample() {
    }

    /** The largest number built from parts, 0 if none. */
    private static int partsTop;

    private static final String VOICE_DE = "Eine warme, freundliche Frauenstimme, Mitte dreißig, klar und ruhig,"
            + " ein wenig langsamer als normal, wie eine Erzählerin in einem Lernspiel für Kinder.";
    private static final String VOICE_EN = "A warm, friendly female voice in her mid thirties, clear and calm,"
            + " a little slower than normal, like the narrator of a learning game for children.";
    private static final String REF_DE = "Hallo! Ich bin heute deine Begleiterin. Wir zählen, rechnen und spielen"
            + " zusammen, und am Ende bekommst du einen goldenen Stern.";
    private static final String REF_EN = "Hello! I'll be your guide today. We'll count, add and play together,"
            + " and at the end you'll get a golden star.";
    // VoiceDesign follows English instructions best and needs the accent said outright, or a German
    // line may come out with an English accent
    private static final String CALM_DE = "A warm, gentle and very calm female voice in her mid thirties, patient and"
            + " friendly like a kind teacher, relaxed and even, a little slower than normal, for a learning game"
            + " for children. She is a native German speaker with a standard German (Hochdeutsch) accent,"
            + " no English accent.";
    private static final String CALM_EN = "A warm, gentle and very calm female voice in her mid thirties, patient and"
            + " friendly like a kind teacher, relaxed and even, a little slower than normal, for a learning game"
            + " for children.";
    private static final String MALE_DE = "A warm, calm male voice in his late thirties, friendly and patient like a"
            + " kind teacher, clear and a little slower than normal, for a learning game for children. He is a"
            + " native German speaker with a standard German (Hochdeutsch) accent, no English accent.";
    private static final String MALE_EN = "A warm, calm male voice in his late thirties, friendly and patient like a"
            + " kind teacher, clear and a little slower than normal, for a learning game for children.";
    // calm voices read a calm reference: an excited one ("Hallo! ... goldenen Stern") is copied into every line
    private static final String CALM_REF_DE = "Schön, dass du da bist. Wir rechnen heute zusammen, ganz in Ruhe,"
            + " Schritt für Schritt. Wenn du einmal nicht weiterweißt, ist das gar nicht schlimm, dann überlegen"
            + " wir gemeinsam.";
    private static final String CALM_REF_EN = "Nice to have you here. Today we'll do some sums together, nice and"
            + " slowly, one step at a time.";

    /**
     * Runs the example.
     *
     * @param args the options
     * @throws IOException if a file cannot be written
     */
    public static void main(String[] args) throws IOException {
        Map<String, String> o = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--")) {
                boolean flag = i + 1 >= args.length || args[i + 1].startsWith("--");
                o.put(args[i].substring(2), flag ? "true" : args[++i]);
            }
        }
        if (!o.containsKey("base") || !o.containsKey("whisper")) {
            System.out.println("Usage: narration --base <Qwen3-TTS Base folder> --whisper <whisper folder>"
                    + " [--design <Qwen3-TTS VoiceDesign folder> | --reference ref.wav --reference-text text]"
                    + " [--out dir] [--language de|en] [--device name] [--takes N] [--temperature T] [--quick]"
                    + " [--whole-math [top]] [--parts [top]] [--male | --calm]");
            return;
        }
        NumberSpeller sp = NumberSpeller.of(o.getOrDefault("language", "de"));
        boolean de = sp.language().equals("de");
        Path out = Path.of(o.getOrDefault("out", "narration"));
        Files.createDirectories(out.resolve("samples"));
        Device device = o.containsKey("device") ? Device.available().stream()
                .filter(d -> d.name().toLowerCase(Locale.ROOT).contains(o.get("device").toLowerCase(Locale.ROOT)))
                .findFirst().orElseThrow() : Device.best();
        DType ttsType = device.kind() == Device.Kind.GPU && device.supports(DType.BFLOAT16) ? DType.BFLOAT16
                : DType.FLOAT32;
        DType sttType = device.kind() == Device.Kind.GPU && device.supports(DType.FLOAT16) ? DType.FLOAT16
                : DType.FLOAT32;
        int takes = Integer.parseInt(o.getOrDefault("takes", "3"));
        float temperature = Float.parseFloat(o.getOrDefault("temperature", "0.6"));
        boolean quick = o.containsKey("quick");
        long start = System.nanoTime();
        System.out.println("device " + device.name() + ", " + ttsType + "; " + takes + " takes at temperature "
                + temperature);

        try (SpeechToText stt = trimming(SpeechModels.speechToText(Path.of(o.get("whisper")), device, sttType),
                device)) {
            // 1. the voice: one reference clip, designed once
            Path refFile = o.containsKey("reference") ? Path.of(o.get("reference")) : out.resolve("reference.wav");
            boolean male = o.containsKey("male");
            boolean calm = male || o.containsKey("calm");
            String refText = o.getOrDefault("reference-text", calm ? (de ? CALM_REF_DE : CALM_REF_EN)
                    : de ? REF_DE : REF_EN);
            if (!Files.isRegularFile(refFile)) {
                if (!o.containsKey("design")) {
                    throw new IllegalArgumentException("give --design <VoiceDesign folder> or --reference <wav>");
                }
                design(Path.of(o.get("design")), device, ttsType, stt, sp, o.getOrDefault("voice", male
                        ? (de ? MALE_DE : MALE_EN) : o.containsKey("calm") ? (de ? CALM_DE : CALM_EN)
                        : de ? VOICE_DE : VOICE_EN), refText, refFile);
            }
            Audio reference = Wav.read(refFile);
            System.out.printf("reference voice %s (%.1f s)%n", refFile, reference.seconds());

            // 2. the narrator: the Base model clones the reference for every line
            try (TextToSpeech model = SpeechModels.textToSpeech(Path.of(o.get("base")), device, ttsType);
                    TextToSpeech tts = trimming(model, device)) {
                SpeechOptions options = SpeechOptions.defaults().language(sp.languageName()).seed(1000);
                ClipMaker.VoiceEmbedder embedder = null;
                if (model instanceof Qwen3Speech q) {
                    q.sampling(new Qwen3Speech.Sampling(temperature, 50, 1, 1.05f, temperature, 50, 1, 8192));
                    q.addVoice("narrator", reference, refText);
                    options = options.voice("narrator");
                    embedder = q::speakerEmbedding;
                } else {
                    options = options.reference(reference, refText);
                }
                ClipLibrary lib = ClipLibrary.open(out.resolve("library-" + sp.language()), sp.language());
                ClipMaker maker = new ClipMaker(tts, stt, lib, new ClipMaker.Settings(options, takes, -16, 0.8, 60),
                        embedder, System.out::println);
                maker.referenceVoice(reference);
                prepare(maker, sp, quick);
                if (o.containsKey("parts")) {
                    // bigger numbers built from parts (zweihundert|einund|fünfzig), cut from fluent sums:
                    // --parts alone records just what the test lines need, --parts N every part up to N
                    String lead = de ? "Was ist" : "What is";
                    if (o.get("parts").equals("true")) {
                        partsTop = 300;
                        maker.fluentFor(bigLines(sp), 0, partsTop, lead, "+", "-");
                    } else {
                        partsTop = Integer.parseInt(o.get("parts"));
                        maker.fluentMath(0, partsTop, lead, "+", "-");
                    }
                }
                if (o.containsKey("whole-math")) {
                    // every calculation up to the top, read out and asked, each as one fluent take
                    int top = Integer.parseInt(o.get("whole-math").equals("true") ? "20" : o.get("whole-math"));
                    maker.fullMath(0, top, LineTemplate.parse(de ? "Was ist {e:math}?" : "What is {e:math}?"),
                            "+", "-", "×", "÷");
                }
                System.out.printf("library ready: %d clips (%.0f s so far)%n", lib.clips().size(),
                        (System.nanoTime() - start) / 1e9);

                // 3. compare: words said alone, shared pieces, and whole lines
                compare(maker, lib, stt, sp, out.resolve("samples"), quick);
            }
        }
        System.out.printf("done in %.0f s%n", (System.nanoTime() - start) / 1e9);
    }

    /**
     * The backend pools released memory by tensor size, and every line has other sizes, so a
     * long recording session spills to system memory and runs out: give it back after each call.
     */
    private static TextToSpeech trimming(TextToSpeech tts, Device device) {
        return new TextToSpeech() {
            @Override
            public int sampleRate() {
                return tts.sampleRate();
            }

            @Override
            public java.util.List<io.github.zzuegg.brainiac.audio.Voice> voices() {
                return tts.voices();
            }

            @Override
            public Audio synthesize(String text, SpeechOptions options, java.util.function.Consumer<Audio> onChunk) {
                try {
                    return tts.synthesize(text, options, onChunk);
                } finally {
                    device.trim();
                }
            }

            @Override
            public boolean canClone() {
                return tts.canClone();
            }

            @Override
            public void close() {
                // the model is closed by its owner
            }
        };
    }

    private static SpeechToText trimming(SpeechToText stt, Device device) {
        return new SpeechToText() {
            @Override
            public io.github.zzuegg.brainiac.audio.Transcription transcribe(float[] samples, TranscriptionOptions options,
                    java.util.function.Consumer<io.github.zzuegg.brainiac.audio.Transcription.Segment> onSegment) {
                try {
                    return stt.transcribe(samples, options, onSegment);
                } finally {
                    device.trim();
                }
            }

            @Override
            public java.util.List<String> languages() {
                return stt.languages();
            }

            @Override
            public boolean canTranslate() {
                return stt.canTranslate();
            }

            @Override
            public void close() {
                stt.close();
            }
        };
    }

    /** Designs the narrator's voice: a few takes of the reference text, the clearest one kept. */
    private static void design(Path dir, Device device, DType type, SpeechToText stt, NumberSpeller sp,
            String description, String text, Path file) {
        System.out.println("designing the voice: " + description);
        try (TextToSpeech model = SpeechModels.textToSpeech(dir, device, type)) {
            TextToSpeech tts = trimming(model, device);
            Audio best = null;
            double bestScore = -1;
            for (int seed = 0; seed < 8; seed++) {
                Audio a = tts.synthesize(text, SpeechOptions.defaults().instruction(description)
                        .language(sp.languageName()).seed(seed), null);
                String heard = stt.transcribe(a, TranscriptionOptions.defaults().language(sp.language()), null).text();
                // a take Whisper does not take for the language at all has a strong foreign accent
                String said = stt.transcribe(a, TranscriptionOptions.defaults(), null).language();
                boolean nativeLanguage = sp.language().equals(said);
                double score = similarity(sp, text, heard) - 0.02 * Math.abs(a.seconds() - 10)
                        - (nativeLanguage ? 0 : 1);
                System.out.printf("  take %d: %.1f s, match %.2f, heard as %s%n", seed, a.seconds(),
                        similarity(sp, text, heard), said);
                // every take kept, to listen to and pick another with --reference
                String name = file.getFileName().toString().replaceAll("\\.wav$", "");
                Wav.write(Loudness.normalize(a, -16, -1), file.resolveSibling(name + "-take" + seed + ".wav"));
                if (score > bestScore) {
                    best = a;
                    bestScore = score;
                }
            }
            Wav.write(Loudness.normalize(best, -16, -1), file);
        }
    }

    private static double similarity(NumberSpeller sp, String a, String b) {
        String x = norm(sp, a);
        String y = norm(sp, b);
        int[][] d = new int[x.length() + 1][y.length() + 1];
        for (int i = 0; i <= x.length(); i++) {
            for (int j = 0; j <= y.length(); j++) {
                d[i][j] = i == 0 ? j : j == 0 ? i : Math.min(d[i - 1][j - 1] + (x.charAt(i - 1) == y.charAt(j - 1)
                        ? 0 : 1), Math.min(d[i - 1][j], d[i][j - 1]) + 1);
            }
        }
        return 1 - (double) d[x.length()][y.length()] / Math.max(1, Math.max(x.length(), y.length()));
    }

    private static String norm(NumberSpeller sp, String s) {
        StringBuilder b = new StringBuilder();
        sp.spell(s).toLowerCase(Locale.ROOT).codePoints().filter(Character::isLetterOrDigit).forEach(b::appendCodePoint);
        return b.toString().replace("ß", "ss");
    }

    private static List<Integer> range(int from, int to) {
        return IntStream.rangeClosed(from, to).boxed().toList();
    }

    /** The templates of a racing game and a maths game. */
    private static void prepare(ClipMaker maker, NumberSpeller sp, boolean quick) {
        boolean de = sp.language().equals("de");
        int top = quick ? 6 : 10;
        // every calculation up to the top: numbers once per melody, operators once
        maker.recordMath(0, 2 * top, "+", "-", "×", "÷", "=");
        // then the same pieces again, cut out of fluent calculations and questions
        maker.fluentMath(0, 2 * top, de ? "Was ist" : "What is", "+", "-", "×", "÷");
        // a question with two numbers: 100 lines, so put together from pieces
        LineTemplate ask = LineTemplate.parse(de ? "Was ist {a} plus {b}?" : "What is {a} plus {b}?");
        Map<String, List<?>> d = new LinkedHashMap<>();
        d.put("a", range(1, top));
        d.put("b", range(1, top));
        maker.prepare(ask, Map.of("a", 3, "b", 5), d);
        // a calculation read out, the template MathReader makes for every sum
        MathReader.Templated sum = MathReader.template("3 + 4 = 7", sp);
        Map<String, List<?>> s = new LinkedHashMap<>();
        s.put("n0", range(1, top));
        s.put("n1", range(1, top));
        s.put("n2", range(2, 2 * top));
        maker.prepare(sum.template(), sum.values(), s);
        // small sets: every line whole
        LineTemplate place = LineTemplate.parse(de ? "Du bist auf dem {n:ordinal:en} Platz!" : "You're in {n:ordinal} place!");
        maker.prepare(place, Map.of("n", 1), Map.of("n", range(1, quick ? 4 : 8)));
        LineTemplate laps = LineTemplate.parse(de ? "Noch {n:cardinal:eine} {n|Runde|Runden}!" : "{n} {n|lap|laps} to go!");
        maker.prepare(laps, Map.of("n", 1), Map.of("n", range(1, quick ? 3 : 5)));
    }

    private record Case(String label, LineTemplate.Line line) {
    }

    /** Lines with numbers built from parts: 21, 51, 100, 121, 251. */
    private static List<LineTemplate.Line> bigLines(NumberSpeller sp) {
        boolean de = sp.language().equals("de");
        LineTemplate ask = LineTemplate.parse(de ? "Was ist {a} plus {b}?" : "What is {a} plus {b}?");
        List<LineTemplate.Line> out = new ArrayList<>();
        out.add(ask.render(Map.of("a", 251, "b", 21), sp));
        out.add(ask.render(Map.of("a", 100, "b", 51), sp));
        for (String e : new String[] {"21 + 30 = 51", "200 + 51 = 251", "143 - 22 = 121"}) {
            out.add(MathReader.read(e, sp));
        }
        return out;
    }

    private static void compare(ClipMaker maker, ClipLibrary lib, SpeechToText stt, NumberSpeller sp, Path dir,
            boolean quick) throws IOException {
        boolean de = sp.language().equals("de");
        Narrator narrator = new Narrator(lib, Narrator.Settings.defaults());
        LineTemplate ask = LineTemplate.parse(de ? "Was ist {a} plus {b}?" : "What is {a} plus {b}?");
        List<Case> cases = new ArrayList<>();
        int[][] asks = quick ? new int[][] {{4, 6}, {2, 5}} : new int[][] {{7, 8}, {9, 4}, {6, 10}};
        for (int[] p : asks) {
            cases.add(new Case("question", ask.render(Map.of("a", p[0], "b", p[1]), sp)));
        }
        String[] sums = quick ? new String[] {"4 + 2 = 6", "11 - 5 = 6"} : new String[] {"6 + 7 = 13", "17 - 9 = 8",
            "4 × 3 = 12", "12 ÷ 4 = 3"};
        if (partsTop >= 300) {
            for (LineTemplate.Line l : bigLines(sp)) {
                cases.add(new Case("big number", l));
            }
        }
        for (String e : sums) {
            cases.add(new Case("calculation", MathReader.read(e, sp)));
        }
        StringBuilder report = new StringBuilder("# Words vs pieces vs whole lines\n\n| # | line | how | length | heard | match |\n"
                + "|---|---|---|---|---|---|\n");
        TranscriptionOptions to = TranscriptionOptions.defaults().language(sp.language());
        int n = 0;
        for (Case c : cases) {
            n++;
            String id = String.format("%02d", n);
            Map<String, Audio> made = new LinkedHashMap<>();
            made.put("words", wordByWord(maker, c.line()));
            made.put("pieces", narrator.splice(c.line()).audio());
            maker.full(c.line());
            lib.save();
            made.put("whole", narrator.say(c.line()).audio());
            for (Map.Entry<String, Audio> m : made.entrySet()) {
                Wav.write(m.getValue(), dir.resolve(id + "-" + m.getKey() + ".wav"));
                String heard = stt.transcribe(m.getValue(), to, null).text().trim();
                report.append(String.format(Locale.ROOT, "| %s | %s | %s | %.2f s | %s | %.2f |%n", id, c.line().text(),
                        m.getKey(), m.getValue().seconds(), heard, similarity(sp, c.line().text(), heard)));
            }
            System.out.println("sample " + id + ": " + c.line().text());
        }
        Files.writeString(dir.resolve("report.md"), report.toString());
        // a few more questions, as a game would ask them
        int top = quick ? 6 : 10;
        for (int i = 0; i < 4; i++) {
            int a = 1 + (i * 3 + 1) % top;
            int b = 1 + (i * 7 + 2) % top;
            Narrator.Spoken s = narrator.say(ask.source(), Map.of("a", a, "b", b));
            Wav.write(s.audio(), dir.resolve("game-" + i + "-" + s.mode().name().toLowerCase(Locale.ROOT) + ".wav"));
        }
    }

    /** The old way: each piece spoken on its own, trimmed and joined with a short gap. */
    private static Audio wordByWord(ClipMaker maker, LineTemplate.Line line) {
        List<float[]> parts = new ArrayList<>();
        int rate = 0;
        for (LineTemplate.Unit u : line.units()) {
            Audio a = maker.takes(u.text()).get(0).audio();
            rate = a.sampleRate();
            parts.add(trim(a));
        }
        int gap = rate * 80 / 1000;
        int total = 0;
        for (float[] p : parts) {
            total += p.length + gap;
        }
        float[] x = new float[total];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, x, at, p.length);
            at += p.length + gap;
        }
        return Loudness.normalize(new Audio(x, rate), -16, -1);
    }

    private static float[] trim(Audio a) {
        float[] x = a.samples();
        float peak = 0;
        for (float v : x) {
            peak = Math.max(peak, Math.abs(v));
        }
        float t = peak * 0.02f;
        int s = 0;
        while (s < x.length && Math.abs(x[s]) < t) {
            s++;
        }
        int e = x.length;
        while (e > s && Math.abs(x[e - 1]) < t) {
            e--;
        }
        return java.util.Arrays.copyOfRange(x, Math.max(0, s - a.sampleRate() / 100),
                Math.min(x.length, e + a.sampleRate() / 50));
    }
}
