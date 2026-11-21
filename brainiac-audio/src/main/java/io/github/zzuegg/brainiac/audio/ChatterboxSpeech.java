package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

/**
 * Resemble AI's Chatterbox text to speech ({@code ResembleAI/chatterbox}): T3, a 0.5B Llama,
 * writes 25 Hz S3 speech tokens for the text in a voice, then S3Gen turns them into 24 kHz
 * audio (a flow-matching mel decoder and the HiFT vocoder). A voice is cloned from a few
 * seconds of speech; the folder's {@code conds.pt} is the voice it speaks in otherwise.
 *
 * <pre>{@code
 * try (ChatterboxSpeech tts = ChatterboxSpeech.load(Path.of("chatterbox"), gpu, DType.FLOAT32)) {
 *     Audio me = Wav.read(Path.of("me.wav"));
 *     Audio a = tts.synthesize("This is not really me.", SpeechOptions.defaults().reference(me, null), null);
 * }
 * }</pre>
 *
 * <p>The reference marks its audio with an imperceptible watermark (Perth); this port does not.
 */
public final class ChatterboxSpeech implements TextToSpeech {

    /** The voice id of the folder's built-in voice ({@code conds.pt}). */
    public static final String DEFAULT_VOICE = "default";
    private static final int MAX_CHARS = 300;
    private static final int PROMPT_TOKENS = 150;
    private static final int T3_REFERENCE_SECONDS = 6;
    private static final int S3GEN_REFERENCE_SECONDS = 10;
    private static final int FLOW_STEPS = 10;
    private static final int TOKEN_VOCABULARY = 6561;

    /**
     * A voice, prepared: what T3 and S3Gen read of it.
     *
     * @param speaker the voice encoder's embedding
     * @param promptTokens the reference's first speech tokens, for T3
     * @param genTokens the reference's speech tokens, for S3Gen
     * @param genMel the reference's mel {@code [frames * 80]}, for S3Gen
     * @param genFrames the mel's frames, twice the tokens
     * @param xvector the CAM++ embedding, for S3Gen
     */
    record Conditionals(float[] speaker, int[] promptTokens, int[] genTokens, float[] genMel, int genFrames,
            float[] xvector) {
    }

    private final Device device;
    private final DType dtype;
    private final ChatterboxT3 t3;
    private final ChatterboxVoiceEncoder voiceEncoder;
    private final S3Flow flow;
    private final HiftVocoder hift;
    private final S3SpeechTokenizer s3;
    private final CamPlusEncoder camPlus;
    private final PretrainedTokenizer tokenizer;
    private final Weights[] weights;
    private final Tensor mel24Basis;
    private final int startText;
    private final int stopText;
    private final Map<String, Conditionals> voices = new LinkedHashMap<>();
    private final Map<String, Object[]> pending = new ConcurrentHashMap<>();
    private volatile ChatterboxT3.Sampling sampling = ChatterboxT3.Sampling.defaults();
    private volatile float exaggeration = 0.5f;

    private ChatterboxSpeech(Path dir, Device device, DType dtype) {
        this.device = device;
        this.dtype = dtype;
        tokenizer = PretrainedTokenizer.load(dir.resolve("tokenizer.json"));
        startText = tokenizer.id("[START]").orElse(255);
        stopText = tokenizer.id("[STOP]").orElse(0);
        WeightSource ve = WeightSource.open(file(dir, "ve"));
        WeightSource t3s = WeightSource.open(dir.resolve("t3_cfg.safetensors"));
        WeightSource gen = WeightSource.open(file(dir, "s3gen"));
        Weights wv = new Weights(ve, device, dtype);
        Weights wt = new Weights(t3s, device, dtype);
        // the vocoder and the mel decoder are small and checked in float32
        Weights wg = new Weights(gen, device, dtype);
        weights = new Weights[] {wv, wt, wg};
        try {
            voiceEncoder = new ChatterboxVoiceEncoder(wv);
            t3 = new ChatterboxT3(t3s, wt);
            flow = new S3Flow(wg, "flow.");
            hift = new HiftVocoder(wg, "mel2wav.");
            s3 = new S3SpeechTokenizer(wg, "tokenizer.");
            camPlus = new CamPlusEncoder(wg, "speaker_encoder.");
        } catch (RuntimeException e) {
            for (Weights w : weights) {
                w.close();
            }
            throw e;
        } finally {
            ve.close();
            t3s.close();
            gen.close();
        }
        mel24Basis = Signal.melFilterbank(HiftVocoder.SAMPLE_RATE, 1920, S3Flow.MELS, 0, 8000, Signal.MelScale.SLANEY,
                true).to(device);
        Path conds = dir.resolve("conds.pt");
        if (Files.isRegularFile(conds)) {
            voices.put(DEFAULT_VOICE, readConds(conds));
        }
    }

    private static Path file(Path dir, String name) {
        Path s = dir.resolve(name + ".safetensors");
        return Files.isRegularFile(s) ? s : dir.resolve(name + ".pt");
    }

    /**
     * Loads a Chatterbox folder.
     *
     * @param dir a folder with {@code t3_cfg.safetensors}, {@code s3gen.safetensors} (or
     *     {@code .pt}), {@code ve.safetensors} (or {@code .pt}), {@code tokenizer.json} and,
     *     for a voice without cloning, {@code conds.pt}
     * @param device where it runs
     * @param dtype the weights' type
     * @return the model, loaded until closed
     */
    public static ChatterboxSpeech load(Path dir, Device device, DType dtype) {
        if (!isChatterbox(dir)) {
            throw new IllegalArgumentException(dir + " is not a Chatterbox folder (t3_cfg.safetensors, s3gen.safetensors,"
                    + " ve.safetensors and tokenizer.json, as ResembleAI/chatterbox)");
        }
        return new ChatterboxSpeech(dir, device, dtype);
    }

    /**
     * Whether a folder holds the English Chatterbox model.
     *
     * @param dir a folder
     * @return whether {@link #load} reads it
     */
    public static boolean isChatterbox(Path dir) {
        return Files.isRegularFile(dir.resolve("t3_cfg.safetensors")) && Files.isRegularFile(dir.resolve("tokenizer.json"))
                && Files.isRegularFile(file(dir, "s3gen")) && Files.isRegularFile(file(dir, "ve"));
    }

    private Conditionals readConds(Path file) {
        try (WeightSource c = WeightSource.open(file)) {
            float[] speaker = floats(c, "t3.speaker_emb");
            int[] prompt = c.contains("t3.cond_prompt_speech_tokens") ? ints(c, "t3.cond_prompt_speech_tokens") : null;
            int[] tokens = ints(c, "gen.prompt_token");
            float[] feat = floats(c, "gen.prompt_feat");
            return new Conditionals(speaker, prompt, tokens, feat, feat.length / S3Flow.MELS, floats(c, "gen.embedding"));
        }
    }

    private static float[] floats(WeightSource s, String name) {
        Tensor t = s.tensor(name, Device.cpu(), DType.FLOAT32);
        float[] f = t.toArray();
        t.release();
        return f;
    }

    private static int[] ints(WeightSource s, String name) {
        float[] f = floats(s, name);
        int[] out = new int[f.length];
        for (int i = 0; i < f.length; i++) {
            out[i] = Math.round(f[i]);
        }
        return out;
    }

    // ---------------------------------------------------------------- settings

    /**
     * Sets how T3 writes tokens (the reference's defaults: temperature 0.8, min-p 0.05,
     * repetition penalty 1.2, guidance 0.5).
     *
     * @param temperature the softmax temperature, 0 for the most likely token
     * @param cfgWeight how strongly the text guides the voice, 0 to 1
     * @param exaggeration the emotion exaggeration, 0.5 neutral, higher more expressive
     */
    public void settings(float temperature, float cfgWeight, float exaggeration) {
        ChatterboxT3.Sampling s = sampling;
        sampling = new ChatterboxT3.Sampling(temperature, s.minP(), s.topP(), s.repetitionPenalty(), cfgWeight,
                s.maxTokens());
        this.exaggeration = exaggeration;
    }

    void sampling(ChatterboxT3.Sampling s) {
        sampling = s;
    }

    @Override
    public int sampleRate() {
        return HiftVocoder.SAMPLE_RATE;
    }

    @Override
    public List<Voice> voices() {
        List<Voice> out = new ArrayList<>();
        synchronized (voices) {
            for (String v : voices.keySet()) {
                out.add(new Voice(v, v.equals(DEFAULT_VOICE) ? "Default" : v, "en"));
            }
        }
        for (String v : pending.keySet()) {
            if (out.stream().noneMatch(x -> x.id().equals(v))) {
                out.add(new Voice(v, v, "en"));
            }
        }
        return out;
    }

    @Override
    public boolean canClone() {
        return true;
    }

    @Override
    public List<String> languages() {
        return List.of("english");
    }

    /**
     * Adds a voice to clone by id, prepared when first used.
     *
     * @param id its id
     * @param reference a few seconds of the voice (up to 10 are used)
     */
    public void addVoice(String id, Audio reference) {
        pending.put(id, new Object[] {reference});
        synchronized (voices) {
            voices.remove(id);
        }
    }

    /**
     * Forgets a voice added with {@link #addVoice}.
     *
     * @param id its id
     */
    public void removeVoice(String id) {
        pending.remove(id);
        synchronized (voices) {
            voices.remove(id);
        }
    }

    // ---------------------------------------------------------------- synthesis

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        Conditionals voice = options.reference() != null ? prepare(options.reference()) : voice(options.voice());
        List<String> pieces = OrpheusSpeech.pieces(text, MAX_CHARS);
        if (pieces.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text is empty");
        }
        ChatterboxT3.Sampling s = sampling;
        Var cond = t3.cond(voice.speaker(), voice.promptTokens(), exaggeration);
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (int i = 0; i < pieces.size(); i++) {
            if (options.cancelled().getAsBoolean()) {
                throw new CancellationException("speech synthesis cancelled");
            }
            SplittableRandom random = new SplittableRandom(options.seed() + i);
            int[] speech = t3.generate(textTokens(pieces.get(i)), cond, s, random, options.cancelled());
            int[] tokens = speechTokens(speech);
            if (tokens.length == 0) {
                continue;
            }
            float[] wave = vocode(tokens, voice, random);
            parts.add(wave);
            total += wave.length;
            if (onChunk != null) {
                onChunk.accept(new Audio(wave, sampleRate()));
            }
        }
        if (parts.isEmpty()) {
            throw new IllegalStateException("the model wrote no speech");
        }
        float[] all = new float[total];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, all, at, p.length);
            at += p.length;
        }
        return new Audio(all, sampleRate());
    }

    private Conditionals voice(String id) {
        String name = id;
        synchronized (voices) {
            if (name == null || !voices.containsKey(name) && !pending.containsKey(name)) {
                if (name != null && !name.isBlank() && !name.equals(DEFAULT_VOICE)) {
                    throw new IllegalArgumentException("unknown voice " + id + "; this model has "
                            + voices().stream().map(Voice::id).toList());
                }
                name = voices.containsKey(DEFAULT_VOICE) ? DEFAULT_VOICE : voices.isEmpty() ? null
                        : voices.keySet().iterator().next();
            }
            if (name != null && voices.containsKey(name)) {
                return voices.get(name);
            }
        }
        if (name == null && !pending.isEmpty()) {
            name = pending.keySet().iterator().next();
        }
        if (name == null) {
            throw new IllegalArgumentException("this Chatterbox folder has no conds.pt: give a few seconds of a voice"
                    + " to clone as the reference");
        }
        Conditionals c = prepare((Audio) pending.get(name)[0]);
        synchronized (voices) {
            voices.put(name, c);
        }
        return c;
    }

    /** The reference's {@code punc_norm}: capitalized, spaces collapsed, rare punctuation replaced, an ending. */
    static String normalize(String text) {
        if (text.isEmpty()) {
            return "You need to add some text for me to talk.";
        }
        if (Character.isLowerCase(text.charAt(0))) {
            text = Character.toUpperCase(text.charAt(0)) + text.substring(1);
        }
        text = String.join(" ", text.trim().split("\\s+"));
        String[][] swaps = {{"...", ", "}, {"…", ", "}, {":", ","}, {" - ", ", "}, {";", ", "}, {"—", "-"},
            {"–", "-"}, {" ,", ","}, {"“", "\""}, {"”", "\""}, {"‘", "'"}, {"’", "'"}};
        for (String[] s : swaps) {
            text = text.replace(s[0], s[1]);
        }
        text = text.stripTrailing();
        if (!(text.endsWith(".") || text.endsWith("!") || text.endsWith("?") || text.endsWith("-")
                || text.endsWith(","))) {
            text += ".";
        }
        return text;
    }

    /** The text tokens T3 reads: start, the graphemes (spaces as {@code [SPACE]}), stop. */
    int[] textTokens(String text) {
        int[] ids = tokenizer.encode(normalize(text).replace(" ", "[SPACE]"), false);
        int[] out = new int[ids.length + 2];
        out[0] = startText;
        System.arraycopy(ids, 0, out, 1, ids.length);
        out[out.length - 1] = stopText;
        return out;
    }

    /** The speech tokens S3Gen reads: up to the stop token, codes only. */
    static int[] speechTokens(int[] written) {
        List<Integer> out = new ArrayList<>();
        for (int t : written) {
            if (t == ChatterboxT3.STOP_SPEECH) {
                break;
            }
            if (t < TOKEN_VOCABULARY) {
                out.add(t);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /** S3Gen: tokens to mel (after the voice's) to audio, with the reference's fade-in. */
    float[] vocode(int[] tokens, Conditionals voice, SplittableRandom random) {
        return Var.noGrad(() -> {
            Var mel = mel(tokens, voice, gaussian(S3Flow.MELS * 2 * (voice.genTokens().length + tokens.length), random));
            float[] wave = hift.synthesize(mel, random);
            HiftVocoder.fadeIn(wave);
            return wave;
        });
    }

    /** The flow's mel for tokens after the voice's, from given noise. */
    Var mel(int[] tokens, Conditionals voice, float[] noise) {
        int[] all = new int[voice.genTokens().length + tokens.length];
        System.arraycopy(voice.genTokens(), 0, all, 0, voice.genTokens().length);
        System.arraycopy(tokens, 0, all, voice.genTokens().length, tokens.length);
        Var mu = flow.encode(all);
        return flow.decode(mu, voice.xvector(), voice.genMel(), voice.genFrames(), noise, FLOW_STEPS);
    }

    private static float[] gaussian(int n, SplittableRandom random) {
        float[] z = new float[n];
        for (int i = 0; i < n; i++) {
            z[i] = (float) random.nextGaussian();
        }
        return z;
    }

    /**
     * Prepares a voice to clone, as the reference's {@code prepare_conditionals}.
     *
     * @param reference a few seconds of speech
     * @return what T3 and S3Gen read of it
     */
    Conditionals prepare(Audio reference) {
        float[] x24 = reference.sampleRate() == HiftVocoder.SAMPLE_RATE ? reference.samples()
                : Signal.resample(reference.samples(), reference.sampleRate(), HiftVocoder.SAMPLE_RATE);
        float[] x16 = Signal.resample(x24, HiftVocoder.SAMPLE_RATE, S3SpeechTokenizer.SAMPLE_RATE);
        float[] gen24 = java.util.Arrays.copyOf(x24, Math.min(x24.length, S3GEN_REFERENCE_SECONDS * HiftVocoder.SAMPLE_RATE));
        float[] gen16 = Signal.resample(gen24, HiftVocoder.SAMPLE_RATE, S3SpeechTokenizer.SAMPLE_RATE);
        if (gen24.length < HiftVocoder.SAMPLE_RATE / 2) {
            throw new IllegalArgumentException("the reference is too short: give 3 to 10 seconds of speech");
        }
        float[] mel = mel24(gen24);
        int frames = mel.length / S3Flow.MELS;
        int[] tokens = s3.encode(gen16, 0);
        if (frames != 2 * tokens.length) {
            tokens = java.util.Arrays.copyOf(tokens, Math.min(tokens.length, frames / 2));
        }
        // the flow continues the reference's mel: as many frames as two per token
        int keep = 2 * tokens.length;
        float[] genMel = java.util.Arrays.copyOf(mel, keep * S3Flow.MELS);
        float[] xvector = camPlus.embed(gen16);
        float[] t3ref = java.util.Arrays.copyOf(x16, Math.min(x16.length, T3_REFERENCE_SECONDS * S3SpeechTokenizer.SAMPLE_RATE));
        int[] prompt = s3.encode(t3ref, PROMPT_TOKENS);
        float[] speaker = voiceEncoder.embed(x16);
        return new Conditionals(speaker, prompt, tokens, genMel, keep, xvector);
    }

    /**
     * S3Gen's 24 kHz mel ({@code matcha}'s): reflect-padded by 720, uncentred 1920-sample Hann
     * frames every 480, magnitudes, 80 Slaney bands up to 8 kHz, natural log of at least 1e-5.
     *
     * @return {@code [frames * 80]}, frame-major
     */
    float[] mel24(float[] audio) {
        return Var.noGrad(() -> {
            int nFft = 1920;
            int hop = 480;
            int pad = (nFft - hop) / 2;
            Var x = Var.of(Tensor.of(Shape.of(1, 1, audio.length), audio, device));
            Var padded = Functional.padAxis(x, 2, pad, pad, PadMode.REFLECT, 0).reshape(audio.length + 2 * pad);
            int frames = (audio.length + 2 * pad - nFft) / hop + 1;
            float[] idx = new float[frames * nFft];
            for (int f = 0; f < frames; f++) {
                for (int j = 0; j < nFft; j++) {
                    idx[f * nFft + j] = f * hop + j;
                }
            }
            Var spec = padded.gather(0, Tensor.of(Shape.of(frames, nFft), idx, device))
                    .mul(Var.of(Signal.hann(nFft, device))).rfft();
            Var mag = spec.square().sum(2, false).add(1e-9f).sqrt();
            return mag.matmul(Var.of(mel24Basis)).clamp(1e-5f, Float.MAX_VALUE).log().value().toArray();
        });
    }

    // ---------------------------------------------------------------- parts, for tests

    ChatterboxT3 t3() {
        return t3;
    }

    S3Flow flow() {
        return flow;
    }

    HiftVocoder hift() {
        return hift;
    }

    S3SpeechTokenizer s3() {
        return s3;
    }

    CamPlusEncoder camPlus() {
        return camPlus;
    }

    ChatterboxVoiceEncoder voiceEncoder() {
        return voiceEncoder;
    }

    Conditionals builtIn() {
        synchronized (voices) {
            return voices.get(DEFAULT_VOICE);
        }
    }

    @Override
    public void close() {
        for (Weights w : weights) {
            w.close();
        }
    }
}
