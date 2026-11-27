import io.github.zzuegg.brainiac.audio.*;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.nio.file.*;
import java.util.*;

/** Two Qwen3 tricks for bare letter sounds: (1) instruction models, (2) in-context priming of the Base clone. */
public class QwenTricks {
    static final Path M = Path.of("D:/_Programming/Brainiac-models-large");
    static final Path K = Path.of("C:/Users/micha/AppData/Local/Temp/kalimo-test");
    static final String INSTR_EN = "Like a phonics teacher for young children, say only the sound of the letter, never its name, "
            + "then the word. Warm, calm, clear.";
    static final String INSTR_DE = "Wie eine Lehrerin beim Lesenlernen: sprich nur den Laut des Buchstabens, niemals seinen Namen, "
            + "dann das Wort. Warm, ruhig, deutlich.";
    static final String VOICE_DE = "A warm, calm, friendly young woman. She is a native German speaker with a standard German "
            + "(Hochdeutsch) accent, no English accent. ";
    static final String VOICE_EN = "A warm, calm, friendly young woman. She is a native British English speaker with a clear "
            + "standard Southern British accent. ";
    // name, language, text
    static final String[][] LINES = {{"de-m", "german", "Mmm, wie Maus."}, {"de-f", "german", "Fff, wie Fisch."},
            {"de-t", "german", "T, wie Tiger."}, {"de-k", "german", "K, wie Kuh."}, {"en-t", "english", "T, like tiger."}};
    static StringBuilder tsv = new StringBuilder();

    static void take(TextToSpeech tts, Path out, String name, String lang, String text, SpeechOptions o) throws Exception {
        for (int s = 0; s < 5; s++) {
            String f = name + "-seed" + s + ".wav";
            Audio a;
            if (Files.exists(out.resolve(f))) a = Wav.read(out.resolve(f));
            else {
                a = tts.synthesize(text, o.seed(777 + s), null);
                Wav.write(a, out.resolve(f));
                Device.best().trim();
            }
            tsv.append(f).append('\t').append(lang).append('\t').append(text).append('\t')
               .append(String.format(Locale.ROOT, "%.2f", a.seconds())).append('\n');
            System.out.println(f);
        }
    }

    static Audio concat(Path... files) throws Exception {
        List<float[]> parts = new ArrayList<>();
        int sr = 0, n = 0;
        for (Path p : files) {
            Audio a = Wav.read(p);
            sr = a.sampleRate();
            parts.add(a.samples());
            parts.add(new float[sr / 2]);
            n += a.samples().length + sr / 2;
        }
        float[] all = new float[n];
        int at = 0;
        for (float[] p : parts) { System.arraycopy(p, 0, all, at, p.length); at += p.length; }
        return new Audio(all, sr);
    }

    public static void main(String[] a) throws Exception {
        Path out = Path.of(a[0]);
        Files.createDirectories(out);
        Device gpu = Device.best();
        String phase = a[1];
        if (phase.equals("design")) {
            try (Qwen3Speech tts = Qwen3Speech.load(M.resolve("Qwen3-TTS-12Hz-1.7B-VoiceDesign"), gpu, DType.BFLOAT16)) {
                for (String[] l : LINES) {
                    boolean de = l[1].equals("german");
                    String voice = de ? VOICE_DE : VOICE_EN;
                    take(tts, out, l[0] + "-design-instr-en", l[1], l[2], SpeechOptions.defaults().language(l[1]).instruction(voice + INSTR_EN));
                    take(tts, out, l[0] + "-design-instr-de", l[1], l[2], SpeechOptions.defaults().language(l[1]).instruction(voice + INSTR_DE));
                }
            }
        } else if (phase.equals("custom")) {
            try (Qwen3Speech tts = Qwen3Speech.load(M.resolve("Qwen3-TTS-12Hz-1.7B-CustomVoice"), gpu, DType.BFLOAT16)) {
                for (String[] l : LINES) {
                    take(tts, out, l[0] + "-custom-serena-instr-en", l[1], l[2], SpeechOptions.defaults().language(l[1]).voice("serena").instruction(INSTR_EN));
                    take(tts, out, l[0] + "-custom-serena-instr-de", l[1], l[2], SpeechOptions.defaults().language(l[1]).voice("serena").instruction(INSTR_DE));
                }
            }
        } else {
            try (Qwen3Speech tts = Qwen3Speech.load(M.resolve("Qwen3-TTS-12Hz-1.7B-Base"), gpu, DType.BFLOAT16)) {
                Audio deRef = concat(K.resolve("de-all/de-s-triple-seed0.wav"), K.resolve("cut2v/de-m-v2.wav"), K.resolve("cut2v/de-f-v2.wav"));
                Audio enRef = concat(K.resolve("en-voice/en-S-Sss-seed0.wav"), K.resolve("cut2/en-m.wav"));
                Wav.write(deRef, out.resolve("primed-reference-de.wav"));
                Wav.write(enRef, out.resolve("primed-reference-en.wav"));
                tts.addVoice("de-primed", deRef, "Ssss, wie Sofa. Mmm, wie Maus. Fff, wie Fisch.");
                tts.addVoice("en-primed", enRef, "Ssss, like snake. Mmm, like mouse.");
                String[][] lines = {{"de-m", "german", "Mmm, wie Maus."}, {"de-m-cap", "german", "M, wie Maus."},
                        {"de-f", "german", "Fff, wie Fisch."}, {"de-t", "german", "T, wie Tiger."}, {"de-k", "german", "K, wie Kuh."},
                        {"en-t", "english", "T, like tiger."}, {"en-m-cap", "english", "M, like mouse."}};
                for (String[] l : lines) {
                    String v = l[1].equals("german") ? "de-primed" : "en-primed";
                    take(tts, out, l[0] + "-primed", l[1], l[2], SpeechOptions.defaults().language(l[1]).voice(v));
                }
            }
        }
        Files.writeString(out.resolve("gen-" + phase + ".tsv"), tsv.toString());
    }
}
