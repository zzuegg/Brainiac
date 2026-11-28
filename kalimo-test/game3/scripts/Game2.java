package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.nio.file.*;
import java.util.*;

/**
 * Takes for game2/lines.tsv (stem, language, text) with the cloned narrator, temperature 0.6.
 * args: dir, list file (lines.tsv or retry.tsv), first seed, takes, dots ("de" = add " …" to German text, "all", "none").
 * Writes takes/<stem>-seedN.wav and appends file, language, text, seconds, speaker cosine, spoken text to takes/gen.tsv.
 */
public class Game2 {
    public static void main(String[] a) throws Exception {
        Path dir = Path.of(a[0]);
        Path takes = dir.resolve("takes");
        Files.createDirectories(takes);
        int first = Integer.parseInt(a[2]), n = Integer.parseInt(a[3]);
        String dots = a[4];
        List<String[]> rows = new ArrayList<>();
        for (String l : Files.readAllLines(dir.resolve(a[1]))) if (!l.isBlank()) rows.add(l.split("\t"));
        Path genFile = takes.resolve("gen.tsv");
        Set<String> done = new HashSet<>();
        if (Files.exists(genFile)) for (String l : Files.readAllLines(genFile)) done.add(l.split("\t")[0]);
        for (String lang : new String[]{"german", "english"}) {
            if (rows.stream().noneMatch(r -> r[1].equals(lang))) continue;
            boolean de = lang.equals("german");
            Path ref = de ? Path.of(System.getProperty("ref.de", "D:/_Programming/competitive-learn/voice-samples/qwen/female/00-reference.wav"))
                    : Path.of("C:/Users/micha/AppData/Local/Temp/kalimo-test/en-voice/reference.wav");
            String refText = de && System.getProperty("reftext.de") != null ? Files.readString(Path.of(System.getProperty("reftext.de"))).strip() : de ? "Hallo! Ich bin deine Begleitung beim Lernen. Wir zählen, lesen und rätseln "
                    + "zusammen, und wenn etwas schwierig ist, probieren wir es einfach noch einmal."
                    : "Hello there! I'm your friend for learning. We'll count, read and solve little "
                    + "puzzles together, and if something feels tricky, we'll simply try it again.";
            try (Qwen3Speech tts = Qwen3Speech.load(Path.of("D:/_Programming/Brainiac-models-large/Qwen3-TTS-12Hz-1.7B-Base"),
                    Device.best(), DType.BFLOAT16)) {
                Audio refAudio = Wav.read(ref);
                tts.addVoice("narrator", refAudio, refText);
                float[] refEmb = tts.speakerEncoder().embed(refAudio.samples());
                Qwen3Speech.Sampling d = tts.sampling();
                tts.sampling(new Qwen3Speech.Sampling(0.6f, d.topK(), d.topP(), d.repetitionPenalty(), 0.6f, d.subTopK(),
                        d.subTopP(), d.maxFrames()));
                for (String[] r : rows) {
                    if (!r[1].equals(lang)) continue;
                    boolean addDots = dots.equals("all") || dots.equals("de") && de;
                    String spoken = addDots ? r[2] + " …" : r[2];
                    for (int s = first; s < first + n; s++) {
                        String f = r[0].replace('/', '_') + "-seed" + s + ".wav";
                        if (done.contains(f)) continue;
                        Audio au = tts.synthesize(spoken, SpeechOptions.defaults().seed(777 + s).language(lang).voice("narrator"), null);
                        Wav.write(au, takes.resolve(f));
                        float cos = WordTest.cosine(refEmb, tts.speakerEncoder().embed(au.samples()));
                        Files.writeString(genFile, f + "\t" + lang + "\t" + r[2] + "\t"
                                + String.format(Locale.ROOT, "%.2f", au.seconds()) + "\t"
                                + String.format(Locale.ROOT, "%.3f", cos) + "\t" + spoken + "\n",
                                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                        Device.best().trim();
                    }
                }
            }
        }
    }
}
