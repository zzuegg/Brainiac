package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.llm.Rope;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Fish Audio S2-Pro ({@code fishaudio/s2-pro}): a dual autoregressive text-to-speech model. A
 * Qwen3 transformer reads a chat-style prompt (a system message with an optional reference
 * voice as transcript and codec frames, then the text) and writes one semantic token per frame;
 * a small transformer writes the frame's other nine codes from its hidden state; {@link
 * FishCodec} turns the frames into 44.1 kHz audio. Cloning takes a few seconds of a voice and
 * what it says.
 *
 * <p>Both transformers run on {@link Llama}: their fused projections are split, and the query
 * and key rows reordered from Fish's interleaved rotary pairs to halves (with the rotary tables
 * rounded to bfloat16, as Fish's are). Sampling is the reference's: top-k, then top-p on the
 * untempered distribution, then the temperature, with repetition-aware resampling of semantic
 * tokens that repeat within the last 10.
 *
 * <p>This model's weights are under the Fish Audio Research License, which limits commercial use.
 */
public final class FishSpeech implements TextToSpeech {

    /** The id under which a model with no reference speaks: a voice the model picks. */
    public static final String DEFAULT_VOICE = "default";
    private static final int RAS_WINDOW = 10;
    private static final float RAS_TEMPERATURE = 1.0f;
    private static final float RAS_TOP_P = 0.9f;
    private static final int MAX_CHARS = 200;

    /**
     * How tokens are drawn (the reference's defaults: temperature 0.7, top-p 0.7, top-k 30,
     * at most 1024 frames per piece of text).
     *
     * @param temperature the softmax temperature
     * @param topP the nucleus, taken before the temperature
     * @param topK the most likely tokens kept
     * @param maxFrames frames per piece of text at most
     */
    public record Sampling(float temperature, float topP, int topK, int maxFrames) {
        /** @return the reference's defaults */
        public static Sampling defaults() {
            return new Sampling(0.7f, 0.7f, 30, 1024);
        }
    }

    private final Device device;
    private final DType dtype;
    private final Weights weights;
    private final FishCodec codec;
    private final PretrainedTokenizer tokenizer;
    private final Llama slow;
    private final Llama fast;
    private final Var textEmbedding;
    private final Var codebookEmbedding;
    private final int codebooks;
    private final int codebookSize;
    private final int semanticBegin;
    private final int semanticEnd;
    private final int imEnd;
    private final int vocab;
    private final Map<String, Object[]> saved = new LinkedHashMap<>();
    private final Map<String, Reference> prepared = new HashMap<>();
    private volatile Sampling sampling = Sampling.defaults();

    /** A voice to clone: its transcript and codec frames {@code [codebooks][frames]}. */
    record Reference(String text, int[][] codes) {
    }

    private FishSpeech(Map<String, Object> config, WeightSource source, FishCodec codec, PretrainedTokenizer tokenizer,
            Device device, DType dtype) {
        this.device = device;
        this.dtype = dtype;
        this.codec = codec;
        this.tokenizer = tokenizer;
        Map<String, Object> tc = Json.object(config.get("text_config"));
        Map<String, Object> ac = Json.object(config.get("audio_decoder_config"));
        semanticBegin = (int) Json.integer(config, "semantic_start_token_id", 151678);
        semanticEnd = (int) Json.integer(config, "semantic_end_token_id", 155773);
        imEnd = tokenizer.id("<|im_end|>").orElse((int) Json.integer(config, "eos_token_id", 151645));
        codebooks = (int) Json.integer(ac, "num_codebooks", 10);
        codebookSize = (int) Json.integer(ac, "vocab_size", 4096);
        weights = new Weights(source, device, dtype);
        try {
            slow = Llama.load(new Fused(source, "text_model.model.", "embeddings", null,
                    (int) Json.integer(tc, "head_dim", 128)), config(tc, true), device, dtype);
            fast = Llama.load(new Fused(source, "audio_decoder.", "embeddings", "audio_decoder.output.weight",
                    (int) Json.integer(ac, "head_dim", 128)), config(ac, false), device, dtype);
            textEmbedding = slow.namedParameters().get("model.embed_tokens.weight");
            codebookEmbedding = weights.read("audio_decoder.codebook_embeddings.weight");
            vocab = slow.vocabSize();
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    /** Fish's transformer settings as a Llama's: Qwen3 with query and key norms, or plain. */
    private static LlamaConfig config(Map<String, Object> c, boolean slow) {
        Map<String, Object> m = new HashMap<>();
        m.put("model_type", Boolean.TRUE.equals(c.get("attention_qk_norm")) ? "qwen3" : "llama");
        m.put("vocab_size", c.get("vocab_size"));
        m.put("hidden_size", c.get("dim"));
        m.put("intermediate_size", c.get("intermediate_size"));
        m.put("num_hidden_layers", c.get("n_layer"));
        m.put("num_attention_heads", c.get("n_head"));
        m.put("num_key_value_heads", c.get("n_local_heads"));
        m.put("head_dim", c.get("head_dim"));
        m.put("rms_norm_eps", c.getOrDefault("norm_eps", 1e-6));
        m.put("rope_theta", c.getOrDefault("rope_base", 1_000_000));
        m.put("max_position_embeddings", c.getOrDefault("max_seq_len", 32768));
        m.put("hidden_act", "silu");
        m.put("tie_word_embeddings", slow);
        m.put("attention_bias", false);
        LlamaConfig config = LlamaConfig.fromHuggingFace(m);
        Rope.Settings r = config.rope();
        return config.withRope(new Rope.Settings(r.dim(), r.theta(), "bf16", 1, 1, 4, r.originalMaxPositions(), null));
    }

    /**
     * Loads an S2-Pro folder: {@code config.json} ({@code model_type fish_qwen3_omni}), the
     * model's safetensors, {@code tokenizer.json} and {@code codec.pth}.
     *
     * @param dir the folder
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the model
     */
    public static FishSpeech load(Path dir, Device device, DType dtype) {
        if (!isFish(dir)) {
            throw new IllegalArgumentException(dir + " is not a Fish Audio S2-Pro folder (config.json with model_type"
                    + " fish_qwen3_omni, and codec.pth), such as fishaudio/s2-pro");
        }
        Map<String, Object> config;
        try {
            config = Json.object(Json.read(dir.resolve("config.json")));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        PretrainedTokenizer tokenizer = PretrainedTokenizer.load(dir);
        FishCodec codec = FishCodec.load(dir.resolve("codec.pth"), device, dtype);
        Path weights = Files.isRegularFile(dir.resolve("model.safetensors.index.json"))
                ? dir.resolve("model.safetensors.index.json") : dir.resolve("model.safetensors");
        try (WeightSource source = WeightSource.open(weights)) {
            return new FishSpeech(config, source, codec, tokenizer, device, dtype);
        } catch (RuntimeException e) {
            codec.close();
            throw e;
        }
    }

    /**
     * Whether a folder holds a Fish Audio S2-Pro model with its codec.
     *
     * @param dir a folder
     * @return whether {@link #load} reads it
     */
    public static boolean isFish(Path dir) {
        Path c = dir.resolve("config.json");
        try {
            return Files.isRegularFile(c) && Files.isRegularFile(dir.resolve("codec.pth"))
                    && "fish_qwen3_omni".equals(Json.object(Json.read(c)).get("model_type"));
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    // ---------------------------------------------------------------- settings and voices

    /**
     * Sets how tokens are drawn.
     *
     * @param s the settings
     */
    public void sampling(Sampling s) {
        sampling = s;
    }

    /** @return how tokens are drawn */
    public Sampling sampling() {
        return sampling;
    }

    @Override
    public int sampleRate() {
        return FishCodec.SAMPLE_RATE;
    }

    @Override
    public List<Voice> voices() {
        List<Voice> out = new ArrayList<>();
        out.add(new Voice(DEFAULT_VOICE, "Default", null));
        synchronized (saved) {
            saved.keySet().stream().sorted().forEach(n -> out.add(new Voice(n, n, null)));
        }
        return out;
    }

    @Override
    public boolean canClone() {
        return true;
    }

    /**
     * Adds a voice to clone by id.
     *
     * @param id its id, as {@link SpeechOptions#voice} names it
     * @param reference a few seconds of the voice (up to 30 are used)
     * @param transcript what the recording says
     */
    public void addVoice(String id, Audio reference, String transcript) {
        synchronized (saved) {
            saved.put(id, new Object[] {reference, transcript});
            prepared.remove(id);
        }
    }

    /**
     * Forgets a voice added with {@link #addVoice}.
     *
     * @param id its id
     */
    public void removeVoice(String id) {
        synchronized (saved) {
            saved.remove(id);
            prepared.remove(id);
        }
    }

    private Reference voice(SpeechOptions options) {
        if (options.reference() != null) {
            if (options.referenceText() == null || options.referenceText().isBlank()) {
                throw new IllegalArgumentException("Fish Audio clones a voice from a recording and what it says:"
                        + " give the transcript too");
            }
            return prepare(options.reference(), options.referenceText());
        }
        String id = options.voice();
        if (id == null || id.isEmpty() || id.equals(DEFAULT_VOICE)) {
            return null;
        }
        Object[] s;
        synchronized (saved) {
            Reference r = prepared.get(id);
            if (r != null) {
                return r;
            }
            s = saved.get(id);
        }
        if (s == null) {
            throw new IllegalArgumentException("no voice " + id + "; add it with addVoice");
        }
        Reference r = prepare((Audio) s[0], (String) s[1]);
        synchronized (saved) {
            prepared.put(id, r);
        }
        return r;
    }

    /** A reference voice's codec frames: the recording at 44.1 kHz, at most 30 seconds. */
    Reference prepare(Audio reference, String transcript) {
        float[] x = reference.sampleRate() == FishCodec.SAMPLE_RATE ? reference.samples()
                : Signal.resample(reference.samples(), reference.sampleRate(), FishCodec.SAMPLE_RATE);
        x = java.util.Arrays.copyOf(x, Math.min(x.length, 30 * FishCodec.SAMPLE_RATE));
        return new Reference(transcript.strip(), codec.encode(x));
    }

    // ---------------------------------------------------------------- synthesis

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        Reference ref = voice(options);
        List<String> pieces = OrpheusSpeech.pieces(text, MAX_CHARS);
        if (pieces.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text is empty");
        }
        BooleanSupplier cancelled = options.cancelled();
        SplittableRandom random = new SplittableRandom(options.seed());
        Sampling s = sampling;
        List<String> said = new ArrayList<>();
        List<int[][]> answers = new ArrayList<>();
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (String piece : pieces) {
            if (cancelled.getAsBoolean()) {
                throw new CancellationException("speech synthesis cancelled");
            }
            List<String> texts = new ArrayList<>(said);
            texts.add(piece);
            int[][] prompt = prompt(ref, texts, answers);
            int[][] frames = generate(prompt, s, random, cancelled);
            if (frames.length == 0) {
                continue;
            }
            int[][] codes = transpose(frames);
            said.add(piece);
            answers.add(codes);
            float[] wave = codec.decode(codes);
            if (options.speed() != 1) {
                wave = Signal.timeStretch(wave, sampleRate(), options.speed());
            }
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

    private static int[][] transpose(int[][] frames) {
        int[][] out = new int[frames[0].length][frames.length];
        for (int f = 0; f < frames.length; f++) {
            for (int q = 0; q < frames[f].length; q++) {
                out[q][f] = frames[f][q];
            }
        }
        return out;
    }

    /**
     * The prompt as the reference's {@code generate_long} builds it: the system message (with the
     * reference's transcript and frames), then each piece of text as a user message, those
     * already spoken answered by their frames, and the assistant's voice message opened.
     *
     * @param ref the voice, or {@code null}
     * @param texts the pieces of text, the last to speak now
     * @param answers the frames {@code [codebooks][frames]} of all but the last
     * @return {@code [1 + codebooks][steps]}: the tokens, then the codes under the frames
     */
    int[][] prompt(Reference ref, List<String> texts, List<int[][]> answers) {
        List<int[]> tokens = new ArrayList<>();
        List<int[][]> codes = new ArrayList<>();
        Consumer<String> text = t -> {
            tokens.add(tokenizer.encode(t));
            codes.add(null);
        };
        Consumer<int[][]> vq = c -> {
            int[] ids = new int[c[0].length];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = c[0][i] + semanticBegin;
            }
            tokens.add(ids);
            codes.add(c);
        };
        text.accept("<|im_start|>system\n");
        if (ref != null) {
            text.accept("convert the provided text to speech reference to the following:\n\nText:\n");
            text.accept("<|speaker:0|>" + ref.text());
            text.accept("\n\nSpeech:\n");
            vq.accept(ref.codes());
        } else {
            text.accept("convert the provided text to speech");
        }
        text.accept("<|im_end|>\n");
        for (int i = 0; i < texts.size(); i++) {
            text.accept("<|im_start|>user\n");
            text.accept(texts.get(i));
            text.accept("<|im_end|>\n");
            if (i < answers.size()) {
                text.accept("<|im_start|>assistant\n<|voice|>");
                vq.accept(answers.get(i));
                text.accept("<|im_end|>\n");
            }
        }
        text.accept("<|im_start|>assistant\n<|voice|>");
        int steps = tokens.stream().mapToInt(t -> t.length).sum();
        int[][] out = new int[codebooks + 1][steps];
        int at = 0;
        for (int p = 0; p < tokens.size(); p++) {
            int[] t = tokens.get(p);
            System.arraycopy(t, 0, out[0], at, t.length);
            int[][] c = codes.get(p);
            if (c != null) {
                for (int q = 0; q < codebooks; q++) {
                    System.arraycopy(c[q], 0, out[q + 1], at, t.length);
                }
            }
            at += t.length;
        }
        return out;
    }

    /** The slow transformer's input: token embeddings, plus the codebook embeddings of frames, scaled. */
    private Var embed(int[][] values) {
        int t = values[0].length;
        float[] ids = new float[t];
        float[] mask = new float[t];
        float[] scale = new float[t];
        float[] cb = new float[codebooks * t];
        float s = (float) (1 / Math.sqrt(codebooks + 1));
        for (int i = 0; i < t; i++) {
            int id = values[0][i];
            ids[i] = id;
            boolean semantic = id >= semanticBegin && id <= semanticEnd;
            mask[i] = semantic ? 1 : 0;
            scale[i] = semantic ? s : 1;
            for (int q = 0; q < codebooks; q++) {
                cb[i * codebooks + q] = values[q + 1][i] + q * codebookSize;
            }
        }
        Var x = textEmbedding.gather(0, Tensor.of(Shape.of(t), ids, device));                       // [t, d]
        int d = x.shape().dim(1);
        Var sum = codebookEmbedding.gather(0, Tensor.of(Shape.of(t * codebooks), cb, device))
                .reshape(t, codebooks, d).sum(1, false);
        Var m = Var.of(Tensor.of(Shape.of(t, 1), mask, device).to(x.dtype()));
        Var sc = Var.of(Tensor.of(Shape.of(t, 1), scale, device).to(x.dtype()));
        return Vocos.settle(x.add(sum.mul(m)).mul(sc));
    }

    /**
     * Writes frames for a prompt until the end token or {@link Sampling#maxFrames}.
     *
     * @return {@code [frames][codebooks]}
     */
    int[][] generate(int[][] prompt, Sampling s, SplittableRandom random, BooleanSupplier cancelled) {
        return Var.noGrad(() -> {
            List<int[]> frames = new ArrayList<>();
            int[] window = new int[RAS_WINDOW];
            Llama.State state = slow.newState(1);
            try {
                Var x = embed(prompt);
                Tensor[] out = slow.nextWithHidden(state, new int[1][x.shape().dim(0)], 0, x.value());
                for (int step = 0; step < s.maxFrames(); step++) {
                    if (cancelled.getAsBoolean()) {
                        throw new CancellationException("speech synthesis cancelled");
                    }
                    float[] logits = out[0].to(DType.FLOAT32).toArray();
                    for (int v = 0; v < logits.length; v++) {
                        if (v != imEnd && (v < semanticBegin || v > semanticEnd)) {
                            logits[v] = Float.NEGATIVE_INFINITY;
                        }
                    }
                    int main = sample(probs(logits, s.temperature(), s.topP(), s.topK()), random);
                    if (step > 0) {
                        int high = sample(probs(logits, RAS_TEMPERATURE, RAS_TOP_P, s.topK()), random);
                        boolean repeated = false;
                        for (int w : window) {
                            repeated |= w == main;
                        }
                        if (repeated && main >= semanticBegin && main <= semanticEnd) {
                            main = high;
                        }
                    }
                    System.arraycopy(window, 1, window, 0, RAS_WINDOW - 1);
                    window[RAS_WINDOW - 1] = main;
                    if (main == imEnd) {
                        break;
                    }
                    Tensor h = out[1];
                    int steps = h.shape().dim(1);
                    Var hidden = Var.of(h.reshape(steps, h.shape().dim(2)).slice(0, steps - 1, steps).to(dtype));
                    int[] frame = codes(hidden, main, s, random);
                    frames.add(frame);
                    int[][] column = new int[codebooks + 1][1];
                    column[0][0] = main;
                    for (int q = 0; q < codebooks; q++) {
                        column[q + 1][0] = frame[q];
                    }
                    out = slow.nextWithHidden(state, new int[1][1], 0, embed(column).value());
                }
            } finally {
                state.close();
            }
            return frames.toArray(int[][]::new);
        });
    }

    /** The fast transformer: a frame's codes from the slow one's hidden state and its semantic token. */
    private int[] codes(Var hidden, int main, Sampling s, SplittableRandom random) {
        int[] frame = new int[codebooks];
        frame[0] = Math.min(Math.max(main - semanticBegin, 0), codebookSize - 1);
        Llama.State state = fast.newState(1);
        try {
            fast.next(state, new int[1][1], 0, hidden.value());
            for (int q = 1; q < codebooks; q++) {
                Tensor logits = fast.next(state, new int[][] {{frame[q - 1]}}, null);
                frame[q] = sample(probs(logits.to(DType.FLOAT32).toArray(), s.temperature(), s.topP(), s.topK()),
                        random);
            }
        } finally {
            state.close();
        }
        return frame;
    }

    /**
     * The reference's {@code logits_to_probs}: the top-k and the nucleus of the untempered
     * distribution (the most likely always kept), then the softmax at the temperature.
     *
     * @param logits the scores, {@code -inf} for impossible tokens
     * @param temperature the temperature
     * @param topP the nucleus
     * @param topK the tokens kept at most
     * @return the probabilities
     */
    static float[] probs(float[] logits, float temperature, float topP, int topK) {
        int n = logits.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Float.compare(logits[b], logits[a]));
        float top = logits[order[0]];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += Math.exp((double) logits[order[i]] - top);
        }
        boolean[] keep = new boolean[n];
        double cum = 0;
        for (int i = 0; i < n; i++) {
            cum += Math.exp((double) logits[order[i]] - top) / sum;
            keep[order[i]] = i == 0 || (cum <= topP && i < topK);
        }
        float t = Math.max(temperature, 1e-5f);
        double z = 0;
        double[] e = new double[n];
        for (int i = 0; i < n; i++) {
            e[i] = keep[i] && logits[i] != Float.NEGATIVE_INFINITY ? Math.exp((logits[i] - top) / t) : 0;
            z += e[i];
        }
        float[] p = new float[n];
        for (int i = 0; i < n; i++) {
            p[i] = (float) (e[i] / z);
        }
        return p;
    }

    private static int sample(float[] p, SplittableRandom random) {
        double r = random.nextDouble();
        int last = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) {
                last = i;
                r -= p[i];
                if (r < 0) {
                    return i;
                }
            }
        }
        return last;
    }

    // ---------------------------------------------------------------- rotary helpers

    /**
     * Fish's rotary tables for positions {@code 0..t-1}: cos and sin of each pair's angle,
     * computed in float32 and rounded to bfloat16.
     *
     * @return cos and sin, each {@code [1, 1, t, dim / 2]}
     */
    static Tensor[] rotaryTable(int t, int dim, double theta, Device device, DType dtype) {
        int half = dim / 2;
        float[] cos = new float[t * half];
        float[] sin = new float[t * half];
        for (int i = 0; i < half; i++) {
            float inv = 1f / (float) Math.pow(theta, (float) (2 * i) / dim);
            for (int p = 0; p < t; p++) {
                float a = (float) p * inv;
                cos[p * half + i] = bf16((float) Math.cos(a));
                sin[p * half + i] = bf16((float) Math.sin(a));
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(1, 1, t, half), cos, device).to(dtype),
                Tensor.of(Shape.of(1, 1, t, half), sin, device).to(dtype)};
    }

    /** Rotates {@code [b, heads, t, dim]} whose pairs are halves ({@link #halves} reordered). */
    static Var rotate(Var x, Tensor[] table) {
        int d = x.shape().dim(3);
        Var a = x.slice(3, 0, d / 2);
        Var b = x.slice(3, d / 2, d);
        Var c = Var.of(table[0]);
        Var s = Var.of(table[1]);
        return Var.concat(3, a.mul(c).sub(b.mul(s)), b.mul(c).add(a.mul(s)));
    }

    /**
     * Reorders the rows of each head from interleaved rotary pairs ({@code 0, 1}, {@code 2, 3},
     * ...) to halves ({@code i, i + dim / 2}), which leaves attention unchanged.
     *
     * @param rows {@code [heads * headDim, width]}
     */
    static float[] halves(float[] rows, int heads, int headDim, int width) {
        float[] out = new float[rows.length];
        int half = headDim / 2;
        for (int h = 0; h < heads; h++) {
            for (int j = 0; j < headDim; j++) {
                int from = j < half ? 2 * j : 2 * (j - half) + 1;
                System.arraycopy(rows, (h * headDim + from) * width, out, (h * headDim + j) * width, width);
            }
        }
        return out;
    }

    private static float bf16(float v) {
        int bits = Float.floatToRawIntBits(v);
        bits += 0x7FFF + ((bits >>> 16) & 1);
        return Float.intBitsToFloat(bits & 0xFFFF0000);
    }

    /**
     * Fish's transformer weights under Llama's names: the fused query, key and value split, query
     * and key rows (and their norms) reordered to rotate halves, everything else as stored.
     */
    private static final class Fused implements WeightSource {
        private final WeightSource base;
        private final Map<String, String> plain = new LinkedHashMap<>();
        private final Map<String, Supplier<float[]>> derived = new LinkedHashMap<>();
        private final Map<String, Shape> shapes = new HashMap<>();
        private final Map<String, String> natural = new HashMap<>();

        Fused(WeightSource base, String prefix, String embeddings, String head, int headDim) {
            this.base = base;
            plain.put("model.embed_tokens.weight", prefix + embeddings + ".weight");
            plain.put("model.norm.weight", prefix + "norm.weight");
            if (head != null) {
                plain.put("lm_head.weight", head);
            }
            for (int l = 0; base.contains(prefix + "layers." + l + ".attention.wqkv.weight"); l++) {
                String f = prefix + "layers." + l + ".";
                String t = "model.layers." + l + ".";
                plain.put(t + "self_attn.o_proj.weight", f + "attention.wo.weight");
                plain.put(t + "mlp.gate_proj.weight", f + "feed_forward.w1.weight");
                plain.put(t + "mlp.up_proj.weight", f + "feed_forward.w3.weight");
                plain.put(t + "mlp.down_proj.weight", f + "feed_forward.w2.weight");
                plain.put(t + "input_layernorm.weight", f + "attention_norm.weight");
                plain.put(t + "post_attention_layernorm.weight", f + "ffn_norm.weight");
                String qkv = f + "attention.wqkv.weight";
                Shape s = base.shape(qkv);
                int width = s.dim(1);
                int heads = base.shape(f + "attention.wo.weight").dim(1) / headDim;
                int kv = (s.dim(0) / headDim - heads) / 2;
                int[] rows = {heads * headDim, kv * headDim, kv * headDim};
                String[] names = {"q_proj", "k_proj", "v_proj"};
                int start = 0;
                for (int p = 0; p < 3; p++) {
                    int from = start;
                    int count = rows[p];
                    int h = p == 0 ? heads : kv;
                    boolean rotate = p < 2;
                    String n = t + "self_attn." + names[p] + ".weight";
                    shapes.put(n, Shape.of(count, width));
                    natural.put(n, qkv);
                    derived.put(n, () -> {
                        float[] all = floats(base, qkv);
                        float[] part = java.util.Arrays.copyOfRange(all, from * width, (from + count) * width);
                        return rotate ? halves(part, h, headDim, width) : part;
                    });
                    start += count;
                }
                for (String norm : new String[] {"q_norm", "k_norm"}) {
                    String src = f + "attention." + norm + ".weight";
                    if (base.contains(src)) {
                        String n = t + "self_attn." + norm + ".weight";
                        shapes.put(n, base.shape(src));
                        natural.put(n, src);
                        derived.put(n, () -> halves(floats(base, src), 1, headDim, 1));
                    }
                }
            }
        }

        private static float[] floats(WeightSource s, String name) {
            Tensor t = s.tensor(name, Device.cpu(), DType.FLOAT32);
            float[] f = t.toArray();
            t.release();
            return f;
        }

        @Override
        public List<String> names() {
            List<String> out = new ArrayList<>(plain.keySet());
            out.addAll(derived.keySet());
            return out;
        }

        @Override
        public boolean contains(String name) {
            return plain.containsKey(name) || derived.containsKey(name);
        }

        @Override
        public Shape shape(String name) {
            return derived.containsKey(name) ? shapes.get(name) : base.shape(of(name));
        }

        @Override
        public String format(String name) {
            return base.format(derived.containsKey(name) ? natural.get(name) : of(name));
        }

        @Override
        public DType naturalType(String name) {
            return base.naturalType(derived.containsKey(name) ? natural.get(name) : of(name));
        }

        private String of(String name) {
            String n = plain.get(name);
            if (n == null) {
                throw new IllegalArgumentException("no tensor " + name);
            }
            return n;
        }

        @Override
        public Tensor tensor(String name, Device device, DType dtype) {
            Supplier<float[]> d = derived.get(name);
            if (d != null) {
                return Tensor.of(shapes.get(name), d.get(), device).to(dtype);
            }
            return base.tensor(of(name), device, dtype);
        }

        @Override
        public void close() {
        }
    }

    @Override
    public void close() {
        weights.close();
        codec.close();
    }

    // ---------------------------------------------------------------- for tests

    FishCodec codec() {
        return codec;
    }

    int imEnd() {
        return imEnd;
    }
}
