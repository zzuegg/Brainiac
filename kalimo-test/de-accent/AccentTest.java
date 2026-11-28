package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.nio.file.*;
import java.util.*;

/**
 * German accent check. phase "design": 4 new VoiceDesign references with an explicit native-German accent.
 * phase "lines": the same test sentences cloned from the old reference, the #230 calm reference and each new one.
 */
public class AccentTest {
    static final Path M = Path.of("D:/_Programming/Brainiac-models-large");
    static final String DESC = "A warm, gentle and calm young woman in her early thirties, friendly and patient like a kind "
            + "kindergarten teacher, speaking clearly and a little slower than normal to children aged 3 to 6. She is a native "
            + "German speaker from Germany with a standard German (Hochdeutsch) accent, no English accent, no American accent.";
    static final String CALM_REF_DE = "Schön, dass du da bist. Wir rechnen heute zusammen, ganz in Ruhe, Schritt für Schritt. "
            + "Wenn du einmal nicht weiterweißt, ist das gar nicht schlimm, dann überlegen wir gemeinsam.";
    static final String NEW_REF_DE = "Schön, dass du da bist. Wir spielen heute zusammen, ganz in Ruhe. Hör gut zu: Gans, "
            + "Geschenk und Giraffe fangen alle gleich an. Wenn du einmal nicht weiterweißt, ist das gar nicht schlimm.";
    static final String OLD_REF_DE = "Hallo! Ich bin deine Begleitung beim Lernen. Wir zählen, lesen und rätseln "
            + "zusammen, und wenn etwas schwierig ist, probieren wir es einfach noch einmal.";

    public static void main(String[] a) throws Exception {
        Path out = Path.of(a[0]);
        Files.createDirectories(out);
        if (a[1].equals("design")) {
            try (Qwen3Speech tts = Qwen3Speech.load(M.resolve("Qwen3-TTS-12Hz-1.7B-VoiceDesign"), Device.best(), DType.BFLOAT16)) {
                Qwen3Speech.Sampling d = tts.sampling();
                tts.sampling(new Qwen3Speech.Sampling(0.6f, d.topK(), d.topP(), d.repetitionPenalty(), 0.6f, d.subTopK(), d.subTopP(), d.maxFrames()));
                for (int s = 0; s < 4; s++) {
                    Audio au = tts.synthesize(NEW_REF_DE, SpeechOptions.defaults().seed(2000 + s).language("german").instruction(DESC), null);
                    Wav.write(au, out.resolve("ref-new" + s + ".wav"));
                    Device.best().trim();
                }
            }
            return;
        }
        String[][] refs = {
            {"old", "D:/_Programming/competitive-learn/voice-samples/qwen/female/00-reference.wav", OLD_REF_DE},
            {"calm230", "D:/_Programming/Brainiac-models-large/narration-calm/reference.wav", CALM_REF_DE},
            {"new0", out.resolve("ref-new0.wav").toString(), NEW_REF_DE}, {"new1", out.resolve("ref-new1.wav").toString(), NEW_REF_DE},
            {"new2", out.resolve("ref-new2.wav").toString(), NEW_REF_DE}, {"new3", out.resolve("ref-new3.wav").toString(), NEW_REF_DE}};
        String[][] lines = {{"ask-g", "Hör gut zu: Gans. Was fängt genauso an?"}, {"match-geschenk", "Geschenk, wie Gans!"},
                {"match-giraffe", "Giraffe, wie Gans!"}, {"match-gitarre", "Gitarre, wie Gans!"}};
        StringBuilder tsv = new StringBuilder();
        try (Qwen3Speech tts = Qwen3Speech.load(M.resolve("Qwen3-TTS-12Hz-1.7B-Base"), Device.best(), DType.BFLOAT16)) {
            Qwen3Speech.Sampling d = tts.sampling();
            tts.sampling(new Qwen3Speech.Sampling(0.6f, d.topK(), d.topP(), d.repetitionPenalty(), 0.6f, d.subTopK(), d.subTopP(), d.maxFrames()));
            for (String[] r : refs) {
                Audio ref = Wav.read(Path.of(r[1]));
                tts.addVoice(r[0], ref, r[2]);
                float[] emb = tts.speakerEncoder().embed(ref.samples());
                Wav.write(ref, out.resolve("reference-" + r[0] + ".wav"));
                for (String[] l : lines) {
                    for (int s = 0; s < 2; s++) {
                        Audio au = tts.synthesize(l[1] + " …", SpeechOptions.defaults().seed(777 + s).language("german").voice(r[0]), null);
                        String f = r[0] + "-" + l[0] + "-seed" + s + ".wav";
                        Wav.write(au, out.resolve(f));
                        tsv.append(f).append("\tgerman\t").append(l[1]).append('\t')
                           .append(String.format(Locale.ROOT, "%.2f", au.seconds())).append('\t')
                           .append(String.format(Locale.ROOT, "%.3f", WordTest.cosine(emb, tts.speakerEncoder().embed(au.samples())))).append('\n');
                        Device.best().trim();
                    }
                }
            }
        }
        Files.writeString(out.resolve("gen.tsv"), tsv.toString());
    }
}
