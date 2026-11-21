package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
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
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.function.Consumer;

/**
 * Qwen3-TTS (Alibaba Qwen, 2026): a Qwen3 language model, the talker, that writes speech as
 * the codes of the {@link Qwen3TtsCodec 12 Hz speech tokenizer}. Each step it reads the sum of
 * the last frame's sixteen code embeddings and the next text token's embedding, and predicts
 * the frame's first code; a small transformer, the code predictor, then writes the other
 * fifteen. Ten languages, voices cloned from three seconds of audio, and speaking styles
 * given in words.
 *
 * <p>Loads the three kinds of {@code Qwen/Qwen3-TTS-12Hz-*} folders (with {@code
 * speech_tokenizer} inside, as they are published):
 * <ul>
 * <li>{@code -Base}: clones the voice of {@link SpeechOptions#reference}. With the reference's
 * transcript it continues the reference's own codes (in-context cloning, the closest match);
 * without one it uses only the speaker encoder's embedding of the voice.
 * <li>{@code -CustomVoice}: nine named {@linkplain #voices() voices}, styled by {@link
 * SpeechOptions#instruction} on the 1.7B model.
 * <li>{@code -VoiceDesign}: a voice described by {@link SpeechOptions#instruction}, such as
 * {@code "A warm, low female voice, speaking slowly"}.
 * </ul>
 *
 * <pre>{@code
 * try (Qwen3Speech tts = Qwen3Speech.load(Path.of("models/Qwen3-TTS-12Hz-1.7B-CustomVoice"), gpu, DType.BFLOAT16)) {
 *     Audio a = tts.synthesize("Guten Morgen! Wie geht's?",
 *             SpeechOptions.defaults().voice("ryan").language("german").instruction("Cheerful"), null);
 * }
 * }</pre>
 */
public final class Qwen3Speech implements TextToSpeech {

    /** What a folder's model does, from its {@code tts_model_type}. */
    public enum Kind {
        /** Clones a reference voice. */
        BASE,
        /** Speaks with named voices. */
        CUSTOM_VOICE,
        /** Speaks with a voice described in words. */
        VOICE_DESIGN
    }

    /** Text characters per generation: whole sentences up to this many go together. */
    static final int MAX_CHARS = 400;
    /** The voice id a VoiceDesign model speaks with: the instruction's. */
    static final String DESIGN_VOICE = "designed";

    /**
     * How codes are drawn, as {@code generate_config.json} says (by default temperature 0.9,
     * top-k 50 and repetition penalty 1.05 for the first code, the same without the penalty
     * for the code predictor's).
     *
     * @param temperature the first code's temperature, 0 for the most likely
     * @param topK the first code's top-k, 0 for all
     * @param topP the first code's top-p, 1 for all
     * @param repetitionPenalty divides the scores of first codes already written
     * @param subTemperature the other codes' temperature, 0 for the most likely
     * @param subTopK the other codes' top-k
     * @param subTopP the other codes' top-p
     * @param maxFrames the most frames per piece of text (12.5 a second)
     */
    public record Sampling(float temperature, int topK, float topP, float repetitionPenalty, float subTemperature,
            int subTopK, float subTopP, int maxFrames) {

        /** @return the reference's defaults */
        public static Sampling defaults() {
            return new Sampling(0.9f, 50, 1, 1.05f, 0.9f, 50, 1, 8192);
        }

        /**
         * The most likely codes every time (keeping the repetition penalty), for tests.
         *
         * @param maxFrames the most frames
         * @return the settings
         */
        public static Sampling greedy(int maxFrames) {
            return new Sampling(0, 0, 1, 1.05f, 0, 0, 1, maxFrames);
        }

        static Sampling read(Path file) {
            Sampling d = defaults();
            if (!Files.isRegularFile(file)) {
                return d;
            }
            try {
                Map<String, Object> c = Json.object(Json.read(file));
                boolean sample = !Boolean.FALSE.equals(c.get("do_sample"));
                boolean subSample = !Boolean.FALSE.equals(c.get("subtalker_dosample"));
                return new Sampling(sample ? (float) Json.number(c, "temperature", d.temperature()) : 0,
                        (int) Json.integer(c, "top_k", d.topK()), (float) Json.number(c, "top_p", d.topP()),
                        (float) Json.number(c, "repetition_penalty", d.repetitionPenalty()),
                        subSample ? (float) Json.number(c, "subtalker_temperature", d.subTemperature()) : 0,
                        (int) Json.integer(c, "subtalker_top_k", d.subTopK()),
                        (float) Json.number(c, "subtalker_top_p", d.subTopP()),
                        (int) Json.integer(c, "max_new_tokens", d.maxFrames()));
            } catch (IOException | RuntimeException e) {
                return d;
            }
        }
    }

    /** One generation's inputs, as token ids (see {@link #prompt}). */
    record Prompt(int[] text, int[] instruct, String speaker, String language, float[] speakerEmbedding, int[] refText,
            int[][] refCodes, boolean nonStreaming) {
    }

    private final Kind kind;
    private final Llama talker;
    private final Llama predictor;
    private final PretrainedTokenizer tokenizer;
    private final Qwen3TtsCodec codec;
    private final Qwen3SpeakerEncoder speakerEncoder;
    private final Weights weights;
    private final Device device;
    private final DType dtype;
    private final int groups;
    private final int vocab;
    private final Map<String, Integer> codecIds = new HashMap<>();
    private final Map<String, Integer> speakers = new LinkedHashMap<>();
    private final Map<String, Object> dialects = new HashMap<>();
    private final Map<String, Integer> languageIds = new LinkedHashMap<>();
    // voices to clone by id (Base models): the recording and transcript, prepared on first use
    private final Map<String, Object[]> saved = new java.util.concurrent.ConcurrentHashMap<>();
    private final Map<String, Clone> prepared = new java.util.concurrent.ConcurrentHashMap<>();
    private final int ttsPad;
    private final int ttsBos;
    private final int ttsEos;
    private final String size;
    private final Var codecEmbedding;
    private final Var textEmbedding;
    private final Var fc1;
    private final Var fc1b;
    private final Var fc2;
    private final Var fc2b;
    private final Var[] predictorEmbeddings;
    private final Var[] predictorHeads;
    private final Var mtp;
    private final Var mtpB;
    private volatile Sampling sampling;

    private Qwen3Speech(Path dir, Map<String, Object> config, WeightSource source, Qwen3TtsCodec codec,
            PretrainedTokenizer tokenizer, Device device, DType dtype) {
        this.device = device;
        this.dtype = dtype;
        this.codec = codec;
        this.tokenizer = tokenizer;
        this.kind = kind(config);
        this.size = Json.string(config, "tts_model_size", "");
        Map<String, Object> tc = Json.object(config.get("talker_config"));
        Map<String, Object> cc = Json.object(tc.get("code_predictor_config"));
        groups = (int) Json.integer(tc, "num_code_groups", 16);
        vocab = (int) Json.integer(tc, "vocab_size");
        for (String k : List.of("codec_eos_token_id", "codec_think_id", "codec_nothink_id", "codec_think_bos_id",
                "codec_think_eos_id", "codec_pad_id", "codec_bos_id")) {
            codecIds.put(k, (int) Json.integer(tc, k));
        }
        if (tc.get("spk_id") instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                speakers.put(String.valueOf(e.getKey()), ((Number) e.getValue()).intValue());
            }
        }
        if (tc.get("spk_is_dialect") instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                dialects.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        if (tc.get("codec_language_id") instanceof Map<?, ?> m) {
            for (Map.Entry<?, ?> e : m.entrySet()) {
                languageIds.put(String.valueOf(e.getKey()), ((Number) e.getValue()).intValue());
            }
        }
        ttsPad = (int) Json.integer(config, "tts_pad_token_id", 151671);
        ttsBos = (int) Json.integer(config, "tts_bos_token_id", 151672);
        ttsEos = (int) Json.integer(config, "tts_eos_token_id", 151673);
        sampling = Sampling.read(dir.resolve("generation_config.json"));

        weights = new Weights(source, device, dtype);
        Llama t = null;
        Llama p = null;
        try {
            t = Llama.load(source.renamed(Qwen3Speech::talkerName), llamaConfig(tc), device, dtype);
            Map<String, String> alias = new HashMap<>();
            for (String n : source.names()) {
                String prefix = "talker.code_predictor.model.";
                if (n.startsWith(prefix) && !n.startsWith(prefix + "codec_embedding.")) {
                    alias.put("model." + n.substring(prefix.length()), n);
                }
            }
            // the code predictor reads only embeddings and uses one output layer per codebook:
            // its first stands in for both the embedding table and the output layer Llama expects
            alias.put("model.embed_tokens.weight", "talker.code_predictor.lm_head.0.weight");
            alias.put("lm_head.weight", "talker.code_predictor.lm_head.0.weight");
            p = Llama.load(new Alias(source, alias), llamaConfig(cc), device, dtype);
            talker = t;
            predictor = p;
            codecEmbedding = t.namedParameters().get("model.embed_tokens.weight");
            textEmbedding = weights.read("talker.model.text_embedding.weight");
            fc1 = weights.linear("talker.text_projection.linear_fc1.weight");
            fc1b = weights.read("talker.text_projection.linear_fc1.bias");
            fc2 = weights.linear("talker.text_projection.linear_fc2.weight");
            fc2b = weights.read("talker.text_projection.linear_fc2.bias");
            predictorEmbeddings = new Var[groups - 1];
            predictorHeads = new Var[groups - 1];
            for (int i = 0; i < groups - 1; i++) {
                predictorEmbeddings[i] = weights.read("talker.code_predictor.model.codec_embedding." + i + ".weight");
                predictorHeads[i] = weights.linear("talker.code_predictor.lm_head." + i + ".weight");
            }
            String proj = "talker.code_predictor.small_to_mtp_projection.";
            mtp = weights.contains(proj + "weight") ? weights.linear(proj + "weight") : null;
            mtpB = weights.contains(proj + "bias") ? weights.read(proj + "bias") : null;
            speakerEncoder = kind == Kind.BASE && weights.contains("speaker_encoder.fc.weight")
                    ? new Qwen3SpeakerEncoder(weights, Json.object(config.get("speaker_encoder_config"))) : null;
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    private static String talkerName(String n) {
        if (n.equals("talker.model.codec_embedding.weight")) {
            return "model.embed_tokens.weight";
        }
        if (n.equals("talker.codec_head.weight")) {
            return "lm_head.weight";
        }
        if (n.startsWith("talker.model.") && !n.startsWith("talker.model.text_embedding")) {
            return "model." + n.substring("talker.model.".length());
        }
        return null;
    }

    /** A Qwen3 configuration for the talker or code predictor (text positions: plain rotary). */
    private static LlamaConfig llamaConfig(Map<String, Object> c) {
        Map<String, Object> m = new HashMap<>();
        for (String k : List.of("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta",
                "max_position_embeddings", "hidden_act")) {
            if (c.containsKey(k)) {
                m.put(k, c.get(k));
            }
        }
        m.put("model_type", "qwen3");
        m.put("tie_word_embeddings", false);
        m.put("attention_bias", Boolean.TRUE.equals(c.get("attention_bias")));
        return LlamaConfig.fromHuggingFace(m);
    }

    /** Weights under other names: some of a source's tensors, each under one or more names. */
    record Alias(WeightSource base, Map<String, String> map) implements WeightSource {
        @Override
        public List<String> names() {
            return List.copyOf(map.keySet());
        }

        @Override
        public boolean contains(String name) {
            return map.containsKey(name);
        }

        private String of(String name) {
            String n = map.get(name);
            if (n == null) {
                throw new IllegalArgumentException("no tensor " + name);
            }
            return n;
        }

        @Override
        public Shape shape(String name) {
            return base.shape(of(name));
        }

        @Override
        public String format(String name) {
            return base.format(of(name));
        }

        @Override
        public DType naturalType(String name) {
            return base.naturalType(of(name));
        }

        @Override
        public Tensor tensor(String name, Device device, DType dtype) {
            return base.tensor(of(name), device, dtype);
        }

        @Override
        public void close() {
        }
    }

    /**
     * Loads a Qwen3-TTS-12Hz folder: {@code config.json}, the weights, the text tokenizer and
     * {@code speech_tokenizer}.
     *
     * @param dir the folder
     * @param device where it runs
     * @param dtype the type of the weights (the codec and speaker encoder run in float32)
     * @return the model
     */
    public static Qwen3Speech load(Path dir, Device device, DType dtype) {
        if (!isQwen3Tts(dir)) {
            throw new IllegalArgumentException(dir + " is not a Qwen3-TTS-12Hz folder (config.json with model_type"
                    + " qwen3_tts and a speech_tokenizer folder), such as Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice");
        }
        Map<String, Object> config;
        try {
            config = Json.object(Json.read(dir.resolve("config.json")));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        PretrainedTokenizer tokenizer = PretrainedTokenizer.load(dir);
        Qwen3TtsCodec codec = Qwen3TtsCodec.load(dir.resolve("speech_tokenizer"), device, DType.FLOAT32);
        try (WeightSource source = WeightSource.open(dir)) {
            return new Qwen3Speech(dir, config, source, codec, tokenizer, device, dtype);
        } catch (RuntimeException e) {
            codec.close();
            throw e;
        }
    }

    /**
     * Whether a folder holds a Qwen3-TTS-12Hz model with its speech tokenizer.
     *
     * @param dir a folder
     * @return whether {@link #load} reads it
     */
    public static boolean isQwen3Tts(Path dir) {
        return kind(dir).isPresent();
    }

    /**
     * What a folder's Qwen3-TTS model does, from its {@code config.json}, without loading it.
     *
     * @param dir a folder
     * @return its kind, or empty if it holds no Qwen3-TTS model with its speech tokenizer
     */
    public static java.util.Optional<Kind> kind(Path dir) {
        Path c = dir.resolve("config.json");
        try {
            if (!Files.isRegularFile(c) || !Qwen3TtsCodec.isCodec(dir.resolve("speech_tokenizer"))) {
                return java.util.Optional.empty();
            }
            Map<String, Object> config = Json.object(Json.read(c));
            return "qwen3_tts".equals(config.get("model_type")) ? java.util.Optional.of(kind(config))
                    : java.util.Optional.empty();
        } catch (IOException | RuntimeException e) {
            return java.util.Optional.empty();
        }
    }

    private static Kind kind(Map<String, Object> config) {
        return switch (Json.string(config, "tts_model_type", "base")) {
            case "custom_voice" -> Kind.CUSTOM_VOICE;
            case "voice_design" -> Kind.VOICE_DESIGN;
            default -> Kind.BASE;
        };
    }

    /** The speaker encoder of a Base model, else null. */
    Qwen3SpeakerEncoder speakerEncoder() {
        return speakerEncoder;
    }

    /** @return what the model does */
    public Kind kind() {
        return kind;
    }

    /** @return the speech tokenizer */
    public Qwen3TtsCodec codec() {
        return codec;
    }

    /**
     * Sets how codes are drawn.
     *
     * @param s the settings
     */
    public void sampling(Sampling s) {
        this.sampling = Objects.requireNonNull(s, "sampling");
    }

    /** @return how codes are drawn */
    public Sampling sampling() {
        return sampling;
    }

    @Override
    public int sampleRate() {
        return codec.sampleRate();
    }

    @Override
    public List<Voice> voices() {
        return switch (kind) {
            case BASE -> saved.keySet().stream().sorted().map(n -> new Voice(n, n, null)).toList();
            case VOICE_DESIGN -> List.of(new Voice(DESIGN_VOICE, "Designed voice", null));
            case CUSTOM_VOICE -> speakers.keySet().stream().map(s -> new Voice(s,
                    s.substring(0, 1).toUpperCase(Locale.ROOT) + s.substring(1), language(s))).toList();
        };
    }

    private String language(String speaker) {
        Object d = dialects.get(speaker);
        return d instanceof String s ? s : null;
    }

    @Override
    public boolean canClone() {
        return kind == Kind.BASE;
    }

    /**
     * Adds a voice to clone by id (Base models).
     *
     * @param id its id, as {@link SpeechOptions#voice} names it
     * @param reference a few seconds of the voice
     * @param transcript what the reference says, or {@code null} to clone from the voice's
     *     sound alone (a looser match)
     */
    public void addVoice(String id, Audio reference, String transcript) {
        if (kind != Kind.BASE) {
            throw new IllegalStateException("only a Qwen3-TTS Base model clones voices");
        }
        saved.put(id, new Object[] {reference, transcript});
        prepared.remove(id);
    }

    /**
     * Forgets a voice added with {@link #addVoice}.
     *
     * @param id its id
     */
    public void removeVoice(String id) {
        saved.remove(id);
        prepared.remove(id);
    }

    @Override
    public boolean followsInstructions() {
        // the 0.6B CustomVoice model ignores instructions, as the reference drops them
        return kind == Kind.VOICE_DESIGN || kind == Kind.CUSTOM_VOICE && !size.equals("0b6");
    }

    @Override
    public List<String> languages() {
        List<String> out = new ArrayList<>();
        for (String l : languageIds.keySet()) {
            if (!l.contains("dialect")) {
                out.add(l);
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- synthesis

    /** A voice to clone, prepared once: its codes, transcript and speaker embedding. */
    private record Clone(int[][] codes, int[] refText, float[] embedding) {
    }

    @Override
    public Audio synthesize(String text, SpeechOptions options, Consumer<Audio> onChunk) {
        List<String> pieces = OrpheusSpeech.pieces(text, MAX_CHARS);
        if (pieces.isEmpty()) {
            throw new IllegalArgumentException("nothing to say: the text is empty");
        }
        String language = options.language() == null ? "auto" : options.language().toLowerCase(Locale.ROOT);
        if (!language.equals("auto") && !languageIds.containsKey(language)) {
            throw new IllegalArgumentException("unknown language " + options.language() + "; this model knows "
                    + languages());
        }
        String speaker = null;
        Clone clone = null;
        int[] instruct = null;
        switch (kind) {
            case CUSTOM_VOICE -> {
                speaker = options.voice() == null ? speakers.keySet().iterator().next()
                        : options.voice().toLowerCase(Locale.ROOT);
                if (!speakers.containsKey(speaker)) {
                    throw new IllegalArgumentException("unknown voice " + options.voice() + "; this model has "
                            + speakers.keySet());
                }
                if (followsInstructions() && options.instruction() != null) {
                    instruct = instruction(options.instruction());
                }
            }
            case VOICE_DESIGN -> {
                if (options.instruction() != null) {
                    instruct = instruction(options.instruction());
                }
            }
            case BASE -> {
                if (options.reference() != null) {
                    clone = prepare(options.reference(), options.referenceText());
                } else {
                    String id = options.voice() != null && saved.containsKey(options.voice()) ? options.voice()
                            : saved.isEmpty() ? null : saved.keySet().stream().sorted().findFirst().orElseThrow();
                    if (id == null) {
                        throw new IllegalArgumentException("a Qwen3-TTS Base model clones a voice: give a few seconds"
                                + " of it as the reference (and what it says, for the closest match)");
                    }
                    clone = prepared.computeIfAbsent(id, k -> {
                        Object[] r = saved.get(k);
                        return prepare((Audio) r[0], (String) r[1]);
                    });
                }
            }
        }
        boolean nonStreaming = kind != Kind.BASE;
        List<float[]> parts = new ArrayList<>();
        int total = 0;
        for (int i = 0; i < pieces.size(); i++) {
            if (options.cancelled().getAsBoolean()) {
                throw new CancellationException("speech synthesis cancelled");
            }
            Prompt p = new Prompt(textIds(pieces.get(i)), instruct, speaker, language,
                    clone == null ? null : clone.embedding(), clone == null ? null : clone.refText(),
                    clone == null ? null : clone.codes(), nonStreaming);
            int[][] frames = generate(p, sampling, new SplittableRandom(options.seed() + i), options.cancelled());
            if (frames.length == 0) {
                continue;
            }
            float[] wave = decode(frames, p.refCodes());
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

    /** The token ids of {@code "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n"}. */
    int[] textIds(String text) {
        return tokenizer.encode("<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n", false);
    }

    /** The token ids of {@code "<|im_start|>user\n" + instruction + "<|im_end|>\n"}. */
    int[] instruction(String instruction) {
        return tokenizer.encode("<|im_start|>user\n" + instruction + "<|im_end|>\n", false);
    }

    private Clone prepare(Audio reference, String transcript) {
        float[] samples = reference.sampleRate() == codec.sampleRate() ? reference.samples()
                : Signal.resample(reference.samples(), reference.sampleRate(), codec.sampleRate());
        if (speakerEncoder == null) {
            throw new IllegalStateException("this model has no speaker encoder");
        }
        float[] embedding = speakerEncoder.embed(samples);
        if (transcript == null || !codec.canEncode()) {
            return new Clone(null, null, embedding);
        }
        int[] ref = tokenizer.encode("<|im_start|>assistant\n" + transcript + "<|im_end|>\n", false);
        return new Clone(codec.encode(samples), ref, embedding);
    }

    /**
     * Decodes frames, after the reference's codes when cloning in context, and cuts the
     * reference's share of the audio off again, as the reference does.
     */
    float[] decode(int[][] frames, int[][] refCodes) {
        if (refCodes == null) {
            return codec.decode(frames);
        }
        int[][] all = new int[refCodes.length + frames.length][];
        System.arraycopy(refCodes, 0, all, 0, refCodes.length);
        System.arraycopy(frames, 0, all, refCodes.length, frames.length);
        float[] wave = codec.decode(all);
        int cut = (int) ((double) refCodes.length / Math.max(all.length, 1) * wave.length);
        return java.util.Arrays.copyOfRange(wave, cut, wave.length);
    }

    // ---------------------------------------------------------------- generation

    private Var rows(Var table, int... ids) {
        float[] idx = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            idx[i] = ids[i];
        }
        return table.gather(0, Tensor.of(Shape.of(ids.length), idx, device));
    }

    /** Text tokens through the text embedding and the projection MLP: {@code [n, hidden]}. */
    private Var project(int... ids) {
        return rows(textEmbedding, ids).matmul(fc1).add(fc1b).silu().matmul(fc2).add(fc2b);
    }

    private Var codec(int... ids) {
        return rows(codecEmbedding, ids);
    }

    private static int[] range(int[] a, int from, int to) {
        return java.util.Arrays.copyOfRange(a, from, to);
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = java.util.Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /** The sum of a frame's code embeddings: the talker's for the first code, the predictor's for the rest. */
    private Var frameEmbedding(int[] frame) {
        Var e = codec(frame[0]);
        for (int g = 1; g < groups; g++) {
            e = e.add(rows(predictorEmbeddings[g - 1], frame[g]));
        }
        return e;
    }

    /**
     * The talker's input embeddings for a prompt, and the text embeddings it reads one per
     * step afterwards (as {@code Qwen3TTSForConditionalGeneration.generate} builds them).
     */
    Var[] prompt(Prompt p) {
        List<Var> parts = new ArrayList<>();
        if (p.instruct() != null) {
            parts.add(project(p.instruct()));
        }
        Var special = project(ttsBos, ttsEos, ttsPad);
        Var bosE = special.slice(0, 0, 1);
        Var eosE = special.slice(0, 1, 2);
        Var padE = special.slice(0, 2, 3);
        Integer lang = p.language().equals("auto") ? null : languageIds.get(p.language());
        if ((p.language().equals("chinese") || p.language().equals("auto")) && p.speaker() != null
                && dialects.get(p.speaker()) instanceof String dialect) {
            lang = languageIds.get(dialect);
        }
        int[] think = lang == null
                ? new int[] {codecIds.get("codec_nothink_id"), codecIds.get("codec_think_bos_id"),
                        codecIds.get("codec_think_eos_id")}
                : new int[] {codecIds.get("codec_think_id"), codecIds.get("codec_think_bos_id"), lang,
                        codecIds.get("codec_think_eos_id")};
        List<Var> codecParts = new ArrayList<>();
        codecParts.add(codec(think));
        if (p.speakerEmbedding() != null) {
            codecParts.add(Var.of(Tensor.of(Shape.of(1, p.speakerEmbedding().length), p.speakerEmbedding(), device)
                    .to(dtype)));
        } else if (p.speaker() != null) {
            codecParts.add(codec(speakers.get(p.speaker())));
        }
        codecParts.add(codec(codecIds.get("codec_pad_id"), codecIds.get("codec_bos_id")));
        Var codecE = Var.concat(0, codecParts.toArray(Var[]::new));
        int n = codecE.shape().dim(0);
        int hidden = codecE.shape().dim(1);
        Var role = project(range(p.text(), 0, 3));
        Var body = (n > 2 ? Var.concat(0, padE.expand(Shape.of(n - 2, hidden)), bosE) : bosE)
                .add(codecE.slice(0, 0, n - 1));
        List<Var> seq = new ArrayList<>(List.of(role, body));
        int[] text = range(p.text(), 3, p.text().length - 5);
        Var trailing;
        if (p.refCodes() != null) {
            int[] ref = range(p.refText(), 3, p.refText().length - 2);
            Var textE = Var.concat(0, project(concat(ref, text)), eosE);
            List<Var> frames = new ArrayList<>();
            frames.add(codec(codecIds.get("codec_bos_id")));
            for (int[] f : p.refCodes()) {
                frames.add(frameEmbedding(f));
            }
            Var refE = Var.concat(0, frames.toArray(Var[]::new));
            int tl = textE.shape().dim(0);
            int cl = refE.shape().dim(0);
            if (p.nonStreaming()) {
                int[] pads = new int[tl];
                java.util.Arrays.fill(pads, codecIds.get("codec_pad_id"));
                seq.add(textE.add(codec(pads)));
                seq.add(refE.add(padE));
                trailing = padE;
            } else if (tl > cl) {
                seq.add(textE.slice(0, 0, cl).add(refE));
                trailing = textE.slice(0, cl, tl);
            } else {
                Var padded = tl == cl ? textE : Var.concat(0, textE, padE.expand(Shape.of(cl - tl, hidden)));
                seq.add(padded.add(refE));
                trailing = padE;
            }
        } else if (p.nonStreaming()) {
            int[] pads = new int[text.length + 1];
            java.util.Arrays.fill(pads, codecIds.get("codec_pad_id"));
            seq.add(Var.concat(0, project(text), eosE).add(codec(pads)));
            seq.add(padE.add(codec(codecIds.get("codec_bos_id"))));
            trailing = padE;
        } else {
            seq.add(project(range(text, 0, 1)).add(codecE.slice(0, n - 1, n)));
            trailing = text.length > 1 ? Var.concat(0, project(range(text, 1, text.length)), eosE) : eosE;
        }
        parts.add(Var.concat(0, seq.toArray(Var[]::new)));
        return new Var[] {Var.concat(0, parts.toArray(Var[]::new)), trailing, padE};
    }

    /**
     * Writes a prompt's frames: the talker's first code per step, the code predictor's
     * others, until the end code or {@link Sampling#maxFrames}.
     *
     * @return {@code [frames][groups]}
     */
    int[][] generate(Prompt p, Sampling s, SplittableRandom random, java.util.function.BooleanSupplier cancelled) {
        return Var.noGrad(() -> {
            Var[] pr = prompt(p);
            Var embeds = pr[0];
            Var trailing = pr[1];
            Var padE = pr[2];
            int eos = codecIds.get("codec_eos_token_id");
            List<int[]> frames = new ArrayList<>();
            List<Integer> firsts = new ArrayList<>();
            Llama.State state = talker.newState(1);
            try {
                Tensor[] out = talker.nextWithHidden(state, new int[1][embeds.shape().dim(0)], 0, embeds.value());
                for (int step = 0; ; step++) {
                    if (cancelled.getAsBoolean()) {
                        throw new CancellationException("speech synthesis cancelled");
                    }
                    float[] logits = out[0].toArray();
                    int c0 = firstCode(logits, firsts, s, random, eos);
                    firsts.add(c0);
                    if (c0 == eos || firsts.size() >= s.maxFrames()) {
                        break;
                    }
                    Tensor h = out[1];
                    Var last = Var.of(h.reshape(h.shape().dim(1), h.shape().dim(2)).slice(0, h.shape().dim(1) - 1,
                            h.shape().dim(1)).to(dtype));
                    int[] frame = predict(last, c0, s, random);
                    frames.add(frame);
                    Var next = frameEmbedding(frame);
                    next = next.add(step < trailing.shape().dim(0) ? trailing.slice(0, step, step + 1) : padE);
                    out = talker.nextWithHidden(state, new int[1][1], 0, next.value());
                }
            } finally {
                state.close();
            }
            return frames.toArray(int[][]::new);
        });
    }

    /** The code predictor: codes 1.. of a frame from the talker's hidden state and the first code. */
    private int[] predict(Var talkerHidden, int first, Sampling s, SplittableRandom random) {
        int[] frame = new int[groups];
        frame[0] = first;
        Llama.State state = predictor.newState(1);
        try {
            Var in = Var.concat(0, talkerHidden, codec(first));
            for (int g = 1; g < groups; g++) {
                Var x = mtp == null ? in : in.matmul(mtp).add(mtpB);
                Tensor[] out = predictor.nextWithHidden(state, new int[1][x.shape().dim(0)], 0, x.value());
                Tensor h = out[1];
                int steps = h.shape().dim(1);
                Var last = Var.of(h.reshape(steps, h.shape().dim(2)).slice(0, steps - 1, steps).to(dtype));
                float[] logits = last.matmul(predictorHeads[g - 1]).to(DType.FLOAT32).value().toArray();
                frame[g] = sample(logits, s.subTemperature(), s.subTopK(), s.subTopP(), random);
                in = rows(predictorEmbeddings[g - 1], frame[g]);
            }
        } finally {
            state.close();
        }
        return frame;
    }

    /**
     * The first code of a frame, with the talker's logits processors: the repetition penalty
     * on first codes already written, the control tokens (the top 1024 of the vocabulary but
     * the end code) suppressed, and no end before the second frame.
     */
    private int firstCode(float[] logits, List<Integer> previous, Sampling s, SplittableRandom random, int eos) {
        float penalty = s.repetitionPenalty();
        if (penalty != 1) {
            for (int t : new java.util.HashSet<>(previous)) {
                logits[t] = logits[t] < 0 ? logits[t] * penalty : logits[t] / penalty;
            }
        }
        for (int i = Math.max(0, vocab - 1024); i < vocab; i++) {
            if (i != eos) {
                logits[i] = Float.NEGATIVE_INFINITY;
            }
        }
        if (previous.size() < 2) {
            logits[eos] = Float.NEGATIVE_INFINITY;
        }
        return sample(logits, s.temperature(), s.topK(), s.topP(), random);
    }

    /** Draws an index: the largest at temperature 0, else from the top-k, top-p softmax. */
    static int sample(float[] logits, float temperature, int topK, float topP, SplittableRandom random) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        if (temperature <= 0) {
            return best;
        }
        Integer[] order = new Integer[logits.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> Float.compare(logits[b], logits[a]));
        int keep = topK > 0 ? Math.min(topK, order.length) : order.length;
        double[] p = new double[keep];
        double sum = 0;
        for (int i = 0; i < keep; i++) {
            float l = logits[order[i]];
            p[i] = Float.isInfinite(l) ? 0 : Math.exp((l - logits[best]) / temperature);
            sum += p[i];
        }
        if (topP < 1) {
            double acc = 0;
            for (int i = 0; i < keep; i++) {
                acc += p[i] / sum;
                if (acc >= topP) {
                    keep = i + 1;
                    break;
                }
            }
            sum = 0;
            for (int i = 0; i < keep; i++) {
                sum += p[i];
            }
        }
        double r = random.nextDouble() * sum;
        for (int i = 0; i < keep; i++) {
            r -= p[i];
            if (r <= 0) {
                return order[i];
            }
        }
        return order[keep - 1];
    }

    @Override
    public void close() {
        Weights.release(talker, predictor);
        weights.close();
        codec.close();
    }
}
