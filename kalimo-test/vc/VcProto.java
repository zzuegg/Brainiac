package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import java.nio.file.*;
import java.util.*;

/**
 * Scratch prototype (outside the repo): phonetically exact source from Kokoro driven by raw IPA,
 * then Chatterbox S3Gen voice conversion into the narrator's voice.
 * phase "source": Kokoro IPA -> source-*.wav ; phase "convert": every source-*.wav -> vc-*.wav for its language's target.
 */
public class VcProto {
    static final Path KOKORO = Path.of("D:/_Programming/Brainiac/models/kokoro-82m");
    static final Path CHATTERBOX = Path.of("D:/_Programming/Brainiac-models-large/chatterbox");
    static final Path DE_REF = Path.of("D:/_Programming/competitive-learn/voice-samples/qwen/female/00-reference.wav");
    static final Path EN_REF = Path.of("C:/Users/micha/AppData/Local/Temp/kalimo-test/en-voice/reference.wav");

    public static void main(String[] a) throws Exception {
        Path out = Path.of(a[0]);
        Files.createDirectories(out);
        Device gpu = Device.best();
        if (a[1].equals("source")) {
            // name, phonemes, speed, kokoro voice
            String[][] src = {
                {"de-m-mlong", "mːːː", "1.0", "af_heart"}, {"de-m-mlong-slow", "mːːː", "0.6", "af_heart"}, {"de-m-m", "mː", "0.6", "af_heart"},
                {"de-f-flong", "fːːː", "1.0", "af_heart"}, {"de-f-flong-slow", "fːːː", "0.6", "af_heart"}, {"de-f-f", "fː", "0.6", "af_heart"},
                {"de-t-th", "tʰ", "1.0", "af_heart"}, {"de-t-t", "t", "1.0", "af_heart"}, {"de-t-th-slow", "tʰ", "0.7", "af_heart"},
                {"en-t-th", "tʰ", "1.0", "bf_emma"}, {"en-t-t", "t", "1.0", "bf_emma"}, {"en-t-th-slow", "tʰ", "0.7", "bf_emma"},
            };
            try (KokoroSpeech k = KokoroSpeech.load(KOKORO, gpu, DType.FLOAT32)) {
                StringBuilder log = new StringBuilder();
                for (String[] s : src) {
                    float[][] pack = k.voicePack(s[3]);
                    int[] ids = k.model().tokenize(s[1]);
                    float[] style = pack[Math.min(s[1].length(), pack.length) - 1];
                    for (int seed = 0; seed < 3; seed++) {
                        Kokoro.Output o = k.model().synthesize(ids, style, Float.parseFloat(s[2]), new SplittableRandom(seed));
                        Audio au = new Audio(o.audio(), k.sampleRate());
                        String f = "source-" + s[0] + "-seed" + seed + ".wav";
                        Wav.write(au, out.resolve(f));
                        log.append(f).append('\t').append(s[1]).append('\t').append(s[2]).append('\t').append(s[3]).append('\t')
                           .append(String.format(Locale.ROOT, "%.3f", au.seconds())).append('\t')
                           .append(Arrays.toString(ids)).append('\t').append(Arrays.toString(o.durations())).append('\n');
                        System.out.println(f + " " + au.seconds() + "s ids " + Arrays.toString(ids) + " dur " + Arrays.toString(o.durations()));
                    }
                }
                Files.writeString(out.resolve("source.tsv"), log.toString());
            }
        } else {
            try (ChatterboxSpeech cb = ChatterboxSpeech.load(CHATTERBOX, gpu, DType.BFLOAT16)) {
                ChatterboxSpeech.Conditionals de = cb.prepare(Wav.read(DE_REF));
                ChatterboxSpeech.Conditionals en = cb.prepare(Wav.read(EN_REF));
                List<Path> files;
                try (var st = Files.list(out)) {
                    files = st.filter(p -> p.getFileName().toString().startsWith("source-")
                            && p.getFileName().toString().endsWith(".wav")).sorted().toList();
                }
                for (Path p : files) {
                    String name = p.getFileName().toString();
                    ChatterboxSpeech.Conditionals c = name.startsWith("source-de-") ? de : en;
                    Audio s = Wav.read(p);
                    float[] x16 = Signal.resample(s.samples(), s.sampleRate(), S3SpeechTokenizer.SAMPLE_RATE);
                    // 240 ms of silence on each side, so even a short burst gets several codes
                    int pad = (int) (0.24 * S3SpeechTokenizer.SAMPLE_RATE);
                    float[] padded = new float[x16.length + 2 * pad];
                    System.arraycopy(x16, 0, padded, pad, x16.length);
                    int[] tokens = cb.s3().encode(padded, 0);
                    SplittableRandom r = new SplittableRandom(1);
                    float[] noise = new float[S3Flow.MELS * 2 * (c.genTokens().length + tokens.length)];
                    for (int i = 0; i < noise.length; i++) noise[i] = (float) r.nextGaussian();
                    int[] all = new int[c.genTokens().length + tokens.length];
                    System.arraycopy(c.genTokens(), 0, all, 0, c.genTokens().length);
                    System.arraycopy(tokens, 0, all, c.genTokens().length, tokens.length);
                    float[] wave = Var.noGrad(() -> {
                        Var mu = cb.flow().encode(all);
                        Var mel = cb.flow().decode(mu, c.xvector(), c.genMel(), c.genFrames(), noise, 10);
                        return cb.hift().synthesize(mel, new SplittableRandom(1));
                    });
                    Wav.write(new Audio(wave, HiftVocoder.SAMPLE_RATE), out.resolve(name.replace("source-", "vc-")));
                    System.out.println(name + " tokens " + tokens.length + " -> " + wave.length / (float) HiftVocoder.SAMPLE_RATE + "s");
                    Device.best().trim();
                }
            }
        }
    }
}
