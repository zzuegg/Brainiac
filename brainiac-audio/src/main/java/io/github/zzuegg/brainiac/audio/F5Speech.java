package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MelSpectrogram;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.random.RandomGenerator;
import java.util.regex.Pattern;

/**
 * Zero-shot voice cloning with F5-TTS and the Vocos vocoder: give a few seconds of someone
 * speaking and what they say, and new text comes out in that voice, as the reference
 * implementation's {@code infer_process} does it (the reference clipped to 12 seconds and
 * trimmed of silence, quiet references raised to an RMS of 0.1, the text cut into pieces
 * that fit the model's 22 second window, each piece's length estimated from the
 * reference's speaking rate, 32 flow steps, pieces cross-faded over 0.15 seconds).
 *
 * <p>The voice comes from {@link SpeechOptions#reference}, or from the {@code voices}
 * folder next to the checkpoint: {@code name.wav} with its transcript in {@code name.txt}.
 * Without a transcript the reference is transcribed by the {@link #transcriber} if one is
 * set.
 *
 * <pre>{@code
 * try (F5Speech tts = F5Speech.load(Path.of("models/f5-tts"), Device.cpu(), DType.FLOAT32)) {
 *     Audio me = Wav.read(Path.of("me.wav"));
 *     Audio a = tts.synthesize("Hello from my clone.",
 *             SpeechOptions.defaults().reference(me, "What I say in me.wav."), null);
 * }
 * }</pre>
 */
public final class F5Speech implements TextToSpeech {

    /** Reference recordings are raised to this RMS (and the output lowered back). */
    static final float TARGET_RMS = 0.1f;
    /** The longest reference used, in seconds. */
    static final double MAX_REFERENCE_SECONDS = 12;
    /** The model's window: reference and new speech together, in seconds. */
    static final double WINDOW_SECONDS = 22;
    /** The cross-fade between pieces, in seconds. */
    static final double CROSS_FADE = 0.15;

    /** A reference voice: audio and what it says. */
    private record Reference(String name, Audio audio, String transcript) {
    }

    /**
     * How sampling runs.
     *
     * @param steps the flow's Euler steps (32)
     * @param guidance the classifier-free guidance strength (2)
     * @param sway the sway sampling coefficient (-1)
     */
    public record Sampling(int steps, float guidance, float sway) {
        /** The reference implementation's settings. */
        public static final Sampling DEFAULT = new Sampling(32, 2f, -1f);
    }

    private final F5Tts model;
    private final Vocos vocoder;
    private final MelSpectrogram features;
    private final Map<String, Reference> voices = new LinkedHashMap<>();
    private final Path voiceDir;
    private volatile SpeechToText transcriber;
    private volatile Sampling sampling = Sampling.DEFAULT;

    private F5Speech(F5Tts model, Vocos vocoder, Path voiceDir) {
        this.model = model;
        this.vocoder = vocoder;
        this.features = vocoder.features();
        if (vocoder.config().mels() != model.config().mels()) {
            throw new IllegalArgumentException("the vocoder reads " + vocoder.config().mels() + " mel bands, F5-TTS"
                    + " makes " + model.config().mels());
        }
        if (voiceDir != null && Files.isDirectory(voiceDir)) {
            try (var files = Files.list(voiceDir)) {
                for (Path f : files.filter(f -> f.getFileName().toString().toLowerCase().endsWith(".wav")).sorted()
                        .toList()) {
                    String id = f.getFileName().toString().replaceFirst("(?i)\\.wav$", "");
                    Path txt = voiceDir.resolve(id + ".txt");
                    String transcript = Files.isRegularFile(txt) ? Files.readString(txt, StandardCharsets.UTF_8).strip()
                            : null;
                    voices.put(id, new Reference(id, null, transcript));
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        this.voiceDir = voiceDir;
    }

    /**
     * Loads F5-TTS and finds its vocoder: a Vocos folder inside {@code dir}, next to it, or
     * next to its parent (such as {@code models/f5-tts} and {@code models/vocos-mel-24khz}).
     *
     * @param dir the F5-TTS folder (with the checkpoint and {@code vocab.txt}, directly or in
     *     one subfolder such as {@code F5TTS_v1_Base})
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the text-to-speech model
     */
    public static F5Speech load(Path dir, Device device, DType dtype) {
        Path ckpt = F5Tts.find(dir);
        if (ckpt == null) {
            throw new IllegalArgumentException(dir + " has no F5-TTS checkpoint with vocab.txt");
        }
        Path vocos = findVocos(dir);
        if (vocos == null) {
            throw new IllegalArgumentException("no Vocos vocoder (charactr/vocos-mel-24khz: config.yaml and"
                    + " pytorch_model.bin) in or next to " + dir);
        }
        return load(ckpt, vocos, device, dtype);
    }

    /**
     * Loads F5-TTS with a given vocoder.
     *
     * @param f5 the folder with the checkpoint and {@code vocab.txt}
     * @param vocos the Vocos folder
     * @param device where they run
     * @param dtype the type of their weights
     * @return the text-to-speech model
     */
    public static F5Speech load(Path f5, Path vocos, Device device, DType dtype) {
        F5Tts model = F5Tts.load(f5, device, dtype);
        Vocos v = null;
        try {
            v = Vocos.load(vocos, device, DType.FLOAT32);
            Path voiceDir = Files.isDirectory(f5.resolve("voices")) ? f5.resolve("voices")
                    : f5.getParent() != null ? f5.getParent().resolve("voices") : null;
            return new F5Speech(model, v, voiceDir);
        } catch (RuntimeException e) {
            model.close();
            if (v != null) {
                v.close();
            }
            throw e;
        }
    }

    /**
     * Whether a folder holds F5-TTS (directly or in one subfolder) with a Vocos vocoder in
     * reach.
     *
     * @param dir a folder
     * @return whether {@link #load(Path, Device, DType)} can load it
     */
    public static boolean isF5(Path dir) {
        return Files.isDirectory(dir) && F5Tts.find(dir) != null && findVocos(dir) != null;
    }

    static Path findVocos(Path dir) {
        List<Path> places = new ArrayList<>();
        Path p = dir.toAbsolutePath().normalize();
        for (int up = 0; up < 3 && p != null; up++, p = p.getParent()) {
            places.add(p);
        }
        for (Path place : places) {
            if (Vocos.isVocos(place)) {
                return place;
            }
            try (var subs = Files.list(place)) {
                Path found = subs.filter(Files::isDirectory).sorted().filter(Vocos::isVocos).findFirst().orElse(null);
                if (found != null) {
                    return found;
                }
            } catch (IOException e) {
                // unreadable: look further up
            }
        }
        return null;
    }

    /** @return the flow-matching network */
    public F5Tts model() {
        return model;
    }

    /** @return the vocoder */
    public Vocos vocoder() {
        return vocoder;
    }

    /**
     * Sets what transcribes references given without a transcript.
     *
     * @param stt a speech-to-text model (such as Whisper), or {@code null}
     */
    public void transcriber(SpeechToText stt) {
        this.transcriber = stt;
    }

    /**
     * Sets the sampling: fewer steps are faster and rougher.
     *
     * @param s the settings
     */
    public void sampling(Sampling s) {
        this.sampling = java.util.Objects.requireNonNull(s, "sampling");
    }

    @Override
    public int sampleRate() {
        return vocoder.sampleRate();
    }

    @Override
    public List<Voice> voices() {
        return voices.values().stream().map(r -> new Voice(r.name(), r.name(), "en")).toList();
    }

    @Override
    public boolean canClone() {
        return true;
    }

    /**
     * Adds a voice to clone by id.
     *
     * @param id its id
     * @param reference a few seconds (at most 12 are used) of the voice
     * @param transcript what the reference says, or {@code null} to transcribe it
     */
    public void addVoice(String id, Audio reference, String transcript) {
        voices.put(id, new Reference(id, reference, transcript));
    }

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        Audio ref;
        String transcript;
        if (options.reference() != null) {
            ref = options.reference();
            transcript = options.referenceText();
        } else {
            Reference r = options.voice() != null ? voices.get(options.voice())
                    : voices.isEmpty() ? null : voices.values().iterator().next();
            if (r == null) {
                throw new IllegalArgumentException(options.voice() != null
                        ? "unknown voice '" + options.voice() + "'; this model has " + voices.keySet()
                        : "F5-TTS clones a voice: give a reference recording and its transcript"
                                + " (SpeechOptions.reference), or put name.wav and name.txt in " + voiceDir);
            }
            ref = r.audio() != null ? r.audio() : Wav.read(voiceDir.resolve(r.name() + ".wav"));
            transcript = r.transcript();
        }
        Audio prepared = prepareReference(ref);
        if (transcript == null) {
            SpeechToText stt = transcriber;
            if (stt == null) {
                throw new IllegalArgumentException("the reference recording needs its transcript (or set a"
                        + " transcriber to write it)");
            }
            transcript = stt.transcribe(prepared).text().strip();
        }
        long seed = options.seed();
        return generate(prepared, referenceText(transcript), text, options.speed(), sampling,
                i -> new SplittableRandom(seed + i), options.cancelled(), onChunk);
    }

    // ---- the reference -------------------------------------------------------------------

    /**
     * Prepares a reference as the reference implementation does: resampled, clipped to 12 seconds
     * (at a pause after 6 seconds when there is one), silence below -42 dBFS trimmed from both
     * ends, 50 ms of silence added.
     *
     * @param audio the recording
     * @return the prepared reference
     */
    Audio prepareReference(Audio audio) {
        int sr = sampleRate();
        float[] x = audio.resample(sr).samples();
        int max = (int) (MAX_REFERENCE_SECONDS * sr);
        if (x.length > max) {
            x = java.util.Arrays.copyOf(x, clipPoint(x, sr, max));
        }
        int chunk = sr / 100;
        int start = 0;
        while (start < x.length && dbfs(x, start, Math.min(x.length, start + chunk)) < -42) {
            start += chunk;
        }
        int end = x.length;
        while (end > start && dbfs(x, Math.max(start, end - chunk), end) < -42) {
            end -= chunk;
        }
        float[] out = new float[Math.max(0, end - start) + sr / 20];
        System.arraycopy(x, start, out, 0, Math.max(0, end - start));
        return new Audio(out, sr);
    }

    /** Where to clip a long reference: the start of the last pause of 0.1 s or more between 6 and 12 s. */
    private static int clipPoint(float[] x, int sr, int max) {
        int chunk = sr / 100;
        int quietRun = 0;
        int best = -1;
        for (int at = 0; at + chunk <= max; at += chunk) {
            if (dbfs(x, at, at + chunk) < -40) {
                quietRun += chunk;
                if (quietRun >= sr / 10 && at + chunk - quietRun >= 6 * sr) {
                    best = at + chunk - quietRun + Math.min(quietRun, sr) / 2;
                }
            } else {
                quietRun = 0;
            }
        }
        return best > 0 ? Math.min(best, max) : max;
    }

    private static double dbfs(float[] x, int from, int to) {
        double sum = 0;
        for (int i = from; i < to; i++) {
            sum += (double) x[i] * x[i];
        }
        double rms = Math.sqrt(sum / Math.max(1, to - from));
        return rms == 0 ? Double.NEGATIVE_INFINITY : 20 * Math.log10(rms);
    }

    /** The transcript ending in ". " as the model was prompted in training. */
    static String referenceText(String t) {
        if (t.endsWith(". ") || t.endsWith("。")) {
            return t;
        }
        return t.endsWith(".") ? t + " " : t + ". ";
    }

    // ---- generation ----------------------------------------------------------------------

    private static final Pattern SENTENCE = Pattern.compile("(?<=[;:,.!?])\\s+|(?<=[；：，。！？])");

    /**
     * Cuts text into pieces of at most {@code maxBytes} UTF-8 bytes at sentence and clause
     * breaks, as F5-TTS's {@code chunk_text}.
     */
    static List<String> chunk(String text, int maxBytes) {
        List<String> chunks = new ArrayList<>();
        String cur = "";
        for (String s : SENTENCE.split(text, -1)) {
            if (s.isEmpty()) {
                continue;
            }
            String piece = bytes(s.substring(s.length() - 1)) == 1 ? s + " " : s;
            if (bytes(cur) + bytes(s) <= maxBytes) {
                cur += piece;
            } else {
                if (!cur.isEmpty()) {
                    chunks.add(cur.strip());
                }
                cur = piece;
            }
        }
        if (!cur.isEmpty()) {
            chunks.add(cur.strip());
        }
        return chunks;
    }

    private static int bytes(String s) {
        return s.getBytes(StandardCharsets.UTF_8).length;
    }

    /**
     * The reference implementation's {@code infer_process} on a prepared reference.
     *
     * @param ref the reference (any rate; resampled)
     * @param refText its transcript, ending in ". "
     * @param text what to say
     * @param speed the speaking rate
     * @param s the sampling settings
     * @param noise the start noise for each piece
     * @param cancelled checked between steps
     * @param onChunk told audio as it becomes final, or {@code null}
     * @return the speech
     */
    Audio generate(Audio ref, String refText, String text, float speed, Sampling s, IntFunction<RandomGenerator> noise,
            java.util.function.BooleanSupplier cancelled, Consumer<Audio> onChunk) {
        int sr = sampleRate();
        int hop = vocoder.config().hop();
        float[] audio = ref.resample(sr).samples();
        double refSeconds = (double) audio.length / sr;
        int maxBytes = (int) (bytes(refText) / refSeconds * (WINDOW_SECONDS - refSeconds) * speed);
        List<String> pieces = chunk(text, maxBytes);
        if (pieces.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text is empty");
        }
        double sum = 0;
        for (float v : audio) {
            sum += (double) v * v;
        }
        float rms = (float) Math.sqrt(sum / audio.length);
        float gain = 1;
        if (rms < TARGET_RMS && rms > 0) {
            gain = TARGET_RMS / rms;
            float[] boosted = new float[audio.length];
            for (int i = 0; i < audio.length; i++) {
                boosted[i] = audio[i] * gain;
            }
            audio = boosted;
        }
        String prompt = bytes(refText.substring(refText.length() - 1)) == 1 ? refText + " " : refText;
        float[][] cond = mel(audio);
        int refFrames = audio.length / hop;
        int fade = (int) (CROSS_FADE * sr);
        float[] done = new float[0];
        int emitted = 0;
        for (int i = 0; i < pieces.size(); i++) {
            String piece = pieces.get(i);
            double localSpeed = bytes(piece) < 10 ? 0.3 : speed;
            int[] ids = model.encode(prompt + piece);
            int length = refFrames + (int) ((double) refFrames / bytes(prompt) * bytes(piece) / localSpeed);
            int step = i;
            float[][] mel = model.sample(cond, ids, length, s.steps(), s.guidance(), s.sway(), noise.apply(step),
                    k -> {
                        if (cancelled.getAsBoolean()) {
                            throw new CancellationException("speech synthesis cancelled");
                        }
                    });
            float[][] gen = java.util.Arrays.copyOfRange(mel, refFrames, mel.length);
            float[][] byBand = new float[gen[0].length][gen.length];
            for (int f = 0; f < gen.length; f++) {
                for (int m = 0; m < gen[f].length; m++) {
                    byBand[m][f] = gen[f][m];
                }
            }
            float[] wave = vocoder.decode(byBand);
            if (gain != 1) {
                for (int k = 0; k < wave.length; k++) {
                    wave[k] /= gain;
                }
            }
            done = crossFade(done, wave, fade);
            if (onChunk != null) {
                int finalUpTo = i == pieces.size() - 1 ? done.length : Math.max(emitted, done.length - fade);
                if (finalUpTo > emitted) {
                    onChunk.accept(new Audio(java.util.Arrays.copyOfRange(done, emitted, finalUpTo), sr));
                    emitted = finalUpTo;
                }
            }
        }
        return new Audio(done, sr);
    }

    static float[] crossFade(float[] prev, float[] next, int samples) {
        if (prev.length == 0) {
            return next;
        }
        int n = Math.min(samples, Math.min(prev.length, next.length));
        float[] out = new float[prev.length + next.length - n];
        System.arraycopy(prev, 0, out, 0, prev.length - n);
        for (int k = 0; k < n; k++) {
            double in = n == 1 ? 0 : (double) k / (n - 1);
            out[prev.length - n + k] = (float) (prev[prev.length - n + k] * (1 - in) + next[k] * in);
        }
        System.arraycopy(next, n, out, prev.length, next.length - n);
        return out;
    }

    /** The reference's mel frames {@code [frames][mels]}. */
    float[][] mel(float[] audio) {
        Var m = features.forward(Var.of(Tensor.of(Shape.of(1, audio.length), audio, Device.cpu())));
        int mels = m.shape().dim(1);
        int frames = m.shape().dim(2);
        float[] v = m.value().toArray();
        float[][] out = new float[frames][mels];
        for (int b = 0; b < mels; b++) {
            for (int f = 0; f < frames; f++) {
                out[f][b] = v[b * frames + f];
            }
        }
        return out;
    }

    @Override
    public void close() {
        model.close();
        vocoder.close();
    }
}
