import io.github.zzuegg.brainiac.audio.*;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.nio.file.*;
import java.util.*;

/** Whisper for gen.tsv rows not yet in judged.tsv; appends row + transcript + detected language. Reuses transcripts from judged7.tsv. */
public class JudgeLang {
    public static void main(String[] a) throws Exception {
        Path dir = Path.of(a[0]);
        List<String> rows = Files.readAllLines(dir.resolve("gen.tsv"));
        Path outFile = dir.resolve("judged.tsv");
        Set<String> seen = new HashSet<>();
        if (Files.exists(outFile)) {
            for (String l : Files.readAllLines(outFile)) if (!l.isBlank()) seen.add(l.split("\t")[0]);
        }
        Map<String, String> old = new HashMap<>();
        Path oldFile = dir.resolve("judged7.tsv");
        if (Files.exists(oldFile)) {
            for (String l : Files.readAllLines(oldFile)) {
                String[] c = l.split("\t");
                if (c.length >= 7) old.put(c[0], c[6]);
            }
        }
        try (WhisperTranscriber stt = WhisperTranscriber.load(Path.of("D:/_Programming/Brainiac-models-large/whisper-large-v3"), Device.best(), DType.BFLOAT16)) {
            for (String r : rows) {
                if (r.isBlank()) continue;
                String[] c = r.split("\t");
                if (seen.contains(c[0])) continue;
                Audio au = Wav.read(dir.resolve(c[0]));
                String lang = c[1].equals("german") ? "de" : "en";
                String t = old.containsKey(c[0]) ? old.get(c[0])
                        : stt.transcribe(au, TranscriptionOptions.defaults().language(lang), null).text().strip();
                String det = stt.transcribe(au, TranscriptionOptions.defaults(), null).language();
                Files.writeString(outFile, r + "\t" + t + "\t" + det + "\n", StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                Device.best().trim();
            }
        }
    }
}
