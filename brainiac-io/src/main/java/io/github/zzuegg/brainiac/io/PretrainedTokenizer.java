package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.data.Tokenizer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.UnaryOperator;

/**
 * A tokenizer published with a model: a Hugging Face {@code tokenizer.json}, a
 * SentencePiece {@code .model}, or the vocabulary inside a GGUF file. It produces the same
 * ids as the library the model was trained with, so pretrained weights see the tokens
 * they expect.
 *
 * <p>Supported: byte-level BPE (GPT-2, Llama 3, Qwen, Mistral Nemo and most newer models),
 * SentencePiece-style BPE with byte fallback (Llama 2, Mistral, Gemma), Unigram (T5, ALBERT,
 * XLM-R) and WordPiece (BERT), with their normalizers, pre-tokenizers, decoders, special
 * tokens and templates.
 *
 * <pre>{@code
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("Qwen2.5-0.5B-Instruct"));
 * int[] ids = tok.encode("Hello world");                // text only
 * int[] withBos = tok.encode("Hello world", true);      // plus the model's special tokens
 * String text = tok.decode(ids);
 * int end = tok.id("<|im_end|>").orElseThrow();
 * }</pre>
 */
public final class PretrainedTokenizer implements Tokenizer {

    /** A token matched as a whole before the rest of the text is tokenized. */
    private record Added(String content, int id, boolean special, boolean lstrip, boolean rstrip) {
    }

    private final List<String> tokens;
    private final Map<String, Integer> ids;
    private final List<Added> added;
    private final Set<Integer> special;
    private final Engine engine;
    private final Object config;
    private final Integer bos;
    private final Integer eos;
    private final String chatTemplate;
    private volatile ChatTemplate template;

    /** How text becomes ids and back: the Hugging Face pipeline or SentencePiece's. */
    private sealed interface Engine permits Hf, Spm {
    }

    private record Hf(UnaryOperator<String> normalizer, TokenizerParts.PreTokenizer preTokenizer,
            TokenizerParts.Model model, TokenizerParts.Decoder decoder, List<Object> single) implements Engine {
    }

    private record Spm(TokenizerParts.Model model, boolean dummyPrefix, boolean removeExtraSpaces, boolean escape,
            int unk, int[] types, boolean addBos) implements Engine {
    }

    private PretrainedTokenizer(List<String> tokens, List<Added> added, Engine engine, Object config, Integer bos,
            Integer eos, String chatTemplate) {
        this.tokens = tokens;
        this.ids = new HashMap<>();
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens.get(i) != null) {
                ids.putIfAbsent(tokens.get(i), i);
            }
        }
        List<Added> sorted = new ArrayList<>(added);
        sorted.sort(Comparator.comparingInt((Added a) -> a.content.length()).reversed());
        this.added = List.copyOf(sorted);
        Set<Integer> sp = new HashSet<>();
        for (Added a : added) {
            if (a.special) {
                sp.add(a.id);
            }
        }
        this.special = Collections.unmodifiableSet(sp);
        this.engine = engine;
        this.config = config;
        this.bos = bos;
        this.eos = eos;
        this.chatTemplate = chatTemplate;
    }

    // ---------------------------------------------------------------- loading

    /**
     * Loads a tokenizer: a model directory (its {@code tokenizer.json}, else {@code
     * tokenizer.model}, plus {@code tokenizer_config.json} when present), a {@code .json},
     * a SentencePiece {@code .model}, or a {@code .gguf} file.
     *
     * @param path the file or directory
     * @return the tokenizer
     * @throws UncheckedIOException if it cannot be read
     */
    public static PretrainedTokenizer load(Path path) {
        try {
            if (Files.isDirectory(path)) {
                Path config = path.resolve("tokenizer_config.json");
                Map<String, Object> cfg = Files.isRegularFile(config) ? Json.object(Json.read(config)) : Map.of();
                Path json = path.resolve("tokenizer.json");
                if (Files.isRegularFile(json)) {
                    return fromJson(Files.readString(json, StandardCharsets.UTF_8), cfg);
                }
                Path model = path.resolve("tokenizer.model");
                if (Files.isRegularFile(model)) {
                    return fromSentencePiece(Files.readAllBytes(model), cfg);
                }
                throw new IllegalArgumentException("no tokenizer.json or tokenizer.model in " + path);
            }
            String name = path.getFileName().toString();
            if (name.endsWith(".gguf")) {
                try (Gguf g = Gguf.open(path)) {
                    return fromGguf(g);
                }
            }
            if (name.endsWith(".model")) {
                return fromSentencePiece(Files.readAllBytes(path), Map.of());
            }
            return fromJson(Files.readString(path, StandardCharsets.UTF_8), Map.of());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Reads a Hugging Face {@code tokenizer.json}.
     *
     * @param json its text
     * @param tokenizerConfig the parsed {@code tokenizer_config.json} (for the chat template
     *        and the names of the begin and end tokens), or an empty map
     * @return the tokenizer
     */
    public static PretrainedTokenizer fromJson(String json, Map<String, Object> tokenizerConfig) {
        Map<String, Object> root = Json.object(Json.parse(json));
        if (root.containsKey("sentencepiece")) {
            return fromSpmJson(Json.object(root.get("sentencepiece")));
        }
        Map<String, Object> model = Json.object(root.get("model"));
        String type = Json.string(model, "type", null);
        if (type == null) {
            type = model.containsKey("merges") ? "BPE" : model.get("vocab") instanceof List ? "Unigram" : "WordPiece";
        }
        List<String> tokens = new ArrayList<>();
        TokenizerParts.Model m;
        switch (type) {
            case "BPE", "WordPiece", "WordLevel" -> {
                Map<String, Object> vocabNode = Json.object(model.get("vocab"));
                Map<String, Integer> vocab = new HashMap<>();
                for (Map.Entry<String, Object> e : vocabNode.entrySet()) {
                    int id = ((Number) e.getValue()).intValue();
                    vocab.put(e.getKey(), id);
                    set(tokens, id, e.getKey());
                }
                if (type.equals("BPE")) {
                    List<String[]> merges = new ArrayList<>();
                    for (Object o : Json.array(model.get("merges"))) {
                        if (o instanceof String s) {
                            int sp = s.indexOf(' ', 1);
                            merges.add(new String[] {s.substring(0, sp), s.substring(sp + 1)});
                        } else {
                            List<Object> pair = Json.array(o);
                            merges.add(new String[] {(String) pair.get(0), (String) pair.get(1)});
                        }
                    }
                    m = new TokenizerParts.Bpe(vocab, merges, Json.string(model, "unk_token", null),
                            Json.string(model, "continuing_subword_prefix", null),
                            Json.string(model, "end_of_word_suffix", null), Boolean.TRUE.equals(model.get("fuse_unk")),
                            Boolean.TRUE.equals(model.get("byte_fallback")),
                            Boolean.TRUE.equals(model.get("ignore_merges")));
                } else if (type.equals("WordPiece")) {
                    m = new TokenizerParts.WordPiece(vocab, Json.string(model, "unk_token", "[UNK]"),
                            Json.string(model, "continuing_subword_prefix", "##"),
                            (int) Json.integer(model, "max_input_chars_per_word", 100));
                } else {
                    Integer unk = vocab.get(Json.string(model, "unk_token", "<unk>"));
                    m = new TokenizerParts.WordLevel(vocab, unk == null ? 0 : unk);
                }
            }
            case "Unigram" -> {
                List<Object> vocab = Json.array(model.get("vocab"));
                double[] scores = new double[vocab.size()];
                for (int i = 0; i < vocab.size(); i++) {
                    List<Object> e = Json.array(vocab.get(i));
                    tokens.add((String) e.get(0));
                    scores[i] = ((Number) e.get(1)).doubleValue();
                }
                m = new TokenizerParts.Unigram(tokens, scores, (int) Json.integer(model, "unk_id", 0),
                        Boolean.TRUE.equals(model.get("byte_fallback")), true);
            }
            default -> throw new IllegalArgumentException("unsupported tokenizer model " + type);
        }
        List<Added> added = new ArrayList<>();
        if (root.get("added_tokens") instanceof List<?> list) {
            for (Object o : list) {
                Map<String, Object> a = Json.object(o);
                int id = (int) Json.integer(a, "id");
                String content = Json.string(a, "content", "");
                set(tokens, id, content);
                added.add(new Added(content, id, Boolean.TRUE.equals(a.get("special")),
                        Boolean.TRUE.equals(a.get("lstrip")), Boolean.TRUE.equals(a.get("rstrip"))));
            }
        }
        Object normalizer = root.get("normalizer");
        Object preTokenizer = root.get("pre_tokenizer");
        if (LLAMA_CLASSES.contains(String.valueOf(tokenizerConfig.get("tokenizer_class")))
                && normalizer instanceof Map<?, ?> nm && prependsMetaspace(nm)) {
            // transformers builds Llama tokenizers' pipeline itself rather than reading it from
            // tokenizer.json: a Metaspace pre-tokenizer that, unless "legacy" is set, puts "▁"
            // only before the text's first piece, not again after each special token
            List<Object> rest = new ArrayList<>();
            for (Object o : Json.array(nm.get("normalizers"))) {
                if (!isPrepend(o) && !isSpaceToMetaspace(o)) {
                    rest.add(o);
                }
            }
            normalizer = rest.isEmpty() ? null : Map.of("type", "Sequence", "normalizers", rest);
            boolean legacy = !Boolean.FALSE.equals(tokenizerConfig.get("legacy"));
            preTokenizer = Map.of("type", "Metaspace", "replacement", "\u2581", "prepend_scheme",
                    legacy ? "always" : "first", "split", false);
        }
        Hf engine = new Hf(TokenizerParts.normalizer(normalizer), TokenizerParts.preTokenizer(preTokenizer), m,
                TokenizerParts.decoder(root.get("decoder")), template(root.get("post_processor")));
        Map<String, Integer> lookup = new HashMap<>();
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens.get(i) != null) {
                lookup.putIfAbsent(tokens.get(i), i);
            }
        }
        Integer bos = lookup.get(tokenName(tokenizerConfig.get("bos_token")));
        Integer eos = lookup.get(tokenName(tokenizerConfig.get("eos_token")));
        String template = tokenizerConfig.get("chat_template") instanceof String s ? s : null;
        return new PretrainedTokenizer(tokens, added, engine, root, bos, eos, template);
    }

    private static String tokenName(Object node) {
        if (node instanceof String s) {
            return s;
        }
        if (node instanceof Map<?, ?> m && m.get("content") instanceof String s) {
            return s;
        }
        return null;
    }

    private static void set(List<String> tokens, int id, String token) {
        while (tokens.size() <= id) {
            tokens.add(null);
        }
        tokens.set(id, token);
    }

    /**
     * The post-processor's template for one sequence, as a list of items: {@code "$A"} for
     * the text, or an {@code int[]} of ids to insert.
     */
    private static List<Object> template(Object node) {
        if (node == null) {
            return List.of("$A");
        }
        Map<String, Object> n = Json.object(node);
        String type = Json.string(n, "type", "");
        switch (type) {
            case "Sequence" -> {
                List<Object> result = List.of("$A");
                for (Object p : Json.array(n.get("processors"))) {
                    List<Object> t = template(p);
                    if (!t.equals(List.of("$A"))) {
                        result = t;
                    }
                }
                return result;
            }
            case "TemplateProcessing" -> {
                Map<String, Object> specials = Json.object(n.get("special_tokens"));
                List<Object> out = new ArrayList<>();
                for (Object item : Json.array(n.get("single"))) {
                    Map<String, Object> it = Json.object(item);
                    if (it.containsKey("Sequence")) {
                        out.add("$A");
                    } else {
                        String id = Json.string(Json.object(it.get("SpecialToken")), "id", "");
                        List<Object> idList = Json.array(Json.object(specials.get(id)).get("ids"));
                        out.add(idList.stream().mapToInt(o -> ((Number) o).intValue()).toArray());
                    }
                }
                return out;
            }
            case "BertProcessing", "RobertaProcessing" -> {
                int cls = ((Number) Json.array(n.get("cls")).get(1)).intValue();
                int sep = ((Number) Json.array(n.get("sep")).get(1)).intValue();
                return List.of(new int[] {cls}, "$A", new int[] {sep});
            }
            default -> {
                return List.of("$A");
            }
        }
    }

    /** Tokenizer classes whose pipeline transformers builds itself. */
    private static final Set<String> LLAMA_CLASSES = Set.of("LlamaTokenizer", "LlamaTokenizerFast",
            "CodeLlamaTokenizer", "CodeLlamaTokenizerFast");

    /** Whether a normalizer is the old Llama one: a Sequence that prepends "▁". */
    private static boolean prependsMetaspace(Map<?, ?> n) {
        return "Sequence".equals(n.get("type")) && n.get("normalizers") instanceof List<?> list
                && list.stream().anyMatch(PretrainedTokenizer::isPrepend);
    }

    private static boolean isPrepend(Object o) {
        return o instanceof Map<?, ?> m && "Prepend".equals(m.get("type")) && "\u2581".equals(m.get("prepend"));
    }

    private static boolean isSpaceToMetaspace(Object o) {
        return o instanceof Map<?, ?> m && "Replace".equals(m.get("type")) && "\u2581".equals(m.get("content"))
                && m.get("pattern") instanceof Map<?, ?> pat && " ".equals(pat.get("String"));
    }

    /**
     * Reads a SentencePiece {@code .model} file.
     *
     * @param model the file's bytes
     * @param tokenizerConfig the parsed {@code tokenizer_config.json}, or an empty map
     * @return the tokenizer
     */
    public static PretrainedTokenizer fromSentencePiece(byte[] model, Map<String, Object> tokenizerConfig) {
        SentencePieceModel sp = SentencePieceModel.parse(model);
        Map<String, Object> spec = new LinkedHashMap<>();
        spec.put("pieces", sp.pieces());
        spec.put("scores", sp.scores());
        spec.put("types", Arrays.stream(sp.types()).boxed().toList());
        spec.put("bpe", sp.bpe());
        spec.put("byte_fallback", sp.byteFallback());
        spec.put("unk_id", (long) sp.unkId());
        spec.put("bos_id", (long) sp.bosId());
        spec.put("eos_id", (long) sp.eosId());
        spec.put("add_dummy_prefix", sp.addDummyPrefix());
        spec.put("remove_extra_whitespaces", sp.removeExtraWhitespaces());
        spec.put("escape_whitespaces", sp.escapeWhitespaces());
        spec.put("add_bos", !Boolean.FALSE.equals(tokenizerConfig.get("add_bos_token")));
        spec.put("parse_control", false);
        if (tokenizerConfig.get("chat_template") instanceof String s) {
            spec.put("chat_template", s);
        }
        return fromSpmJson(spec);
    }

    private static PretrainedTokenizer fromSpmJson(Map<String, Object> spec) {
        List<String> pieces = new ArrayList<>();
        for (Object o : Json.array(spec.get("pieces"))) {
            pieces.add((String) o);
        }
        List<Object> scoreList = spec.get("scores") instanceof double[] d
                ? Arrays.stream(d).boxed().map(Object.class::cast).toList() : Json.array(spec.get("scores"));
        double[] scores = new double[pieces.size()];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = ((Number) scoreList.get(i)).doubleValue();
        }
        List<Object> typeList = Json.array(spec.get("types"));
        int[] types = new int[pieces.size()];
        for (int i = 0; i < types.length; i++) {
            types[i] = ((Number) typeList.get(i)).intValue();
        }
        boolean parseControl = Boolean.TRUE.equals(spec.get("parse_control"));
        int unk = (int) Json.integer(spec, "unk_id", 0);
        Map<String, Integer> vocab = new HashMap<>();
        List<Added> added = new ArrayList<>();
        for (int i = 0; i < pieces.size(); i++) {
            int t = types[i];
            if (t == SentencePieceModel.NORMAL || t == SentencePieceModel.BYTE) {
                vocab.putIfAbsent(pieces.get(i), i);
            } else if (t == SentencePieceModel.USER_DEFINED) {
                added.add(new Added(pieces.get(i), i, false, false, false));
            } else if (t == SentencePieceModel.CONTROL && parseControl && !pieces.get(i).isEmpty()) {
                added.add(new Added(pieces.get(i), i, true, false, false));
            }
        }
        boolean byteFallback = Boolean.TRUE.equals(spec.get("byte_fallback"));
        TokenizerParts.Model model;
        if (Boolean.TRUE.equals(spec.get("bpe"))) {
            model = new TokenizerParts.ScoreBpe(vocab, scores, unk, byteFallback);
        } else {
            Set<Integer> excluded = new HashSet<>();
            for (int i = 0; i < types.length; i++) {
                if (types[i] != SentencePieceModel.NORMAL) {
                    excluded.add(i);
                }
            }
            // byte, control and user-defined pieces never come out of the lattice
            model = new TokenizerParts.Unigram(pieces, scores, unk, byteFallback, false).excluding(excluded);
        }
        Spm engine = new Spm(model, !Boolean.FALSE.equals(spec.get("add_dummy_prefix")),
                !Boolean.FALSE.equals(spec.get("remove_extra_whitespaces")),
                !Boolean.FALSE.equals(spec.get("escape_whitespaces")), unk, types,
                !Boolean.FALSE.equals(spec.get("add_bos")));
        int bos = (int) Json.integer(spec, "bos_id", -1);
        int eos = (int) Json.integer(spec, "eos_id", -1);
        for (int i = 0; i < types.length; i++) {
            if (types[i] == SentencePieceModel.CONTROL && !parseControl) {
                added.add(new Added("", i, true, false, false));   // special, but never matched in text
            }
        }
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("sentencepiece", spec);
        return new PretrainedTokenizer(pieces, added, engine, config, bos >= 0 && bos < pieces.size() ? bos : null,
                eos >= 0 && eos < pieces.size() ? eos : null, Json.string(spec, "chat_template", null));
    }

    /**
     * Reads the tokenizer stored in a GGUF file's metadata ({@code tokenizer.ggml.*}):
     * SentencePiece-style models ({@code "llama"}) and byte-level BPE ({@code "gpt2"}).
     *
     * @param gguf the opened file
     * @return the tokenizer
     */
    @SuppressWarnings("unchecked")
    public static PretrainedTokenizer fromGguf(Gguf gguf) {
        Map<String, Object> meta = gguf.metadata();
        String kind = String.valueOf(meta.get("tokenizer.ggml.model"));
        List<Object> tokenList = (List<Object>) meta.get("tokenizer.ggml.tokens");
        if (tokenList == null) {
            throw new IllegalArgumentException(gguf.path() + " has no tokenizer");
        }
        List<Object> typeList = (List<Object>) meta.getOrDefault("tokenizer.ggml.token_type", List.of());
        String template = meta.get("tokenizer.chat_template") instanceof String s ? s : null;
        long bos = meta.get("tokenizer.ggml.bos_token_id") instanceof Long l ? l : -1;
        long eos = meta.get("tokenizer.ggml.eos_token_id") instanceof Long l ? l : -1;
        switch (kind) {
            case "llama" -> {
                Map<String, Object> spec = new LinkedHashMap<>();
                spec.put("pieces", tokenList);
                List<Object> scores = (List<Object>) meta.get("tokenizer.ggml.scores");
                spec.put("scores", scores != null ? scores : Collections.nCopies(tokenList.size(), 0.0));
                spec.put("types", typeList.isEmpty() ? Collections.nCopies(tokenList.size(), 1L) : typeList);
                spec.put("bpe", true);
                spec.put("byte_fallback", typeList.contains(6L));
                spec.put("unk_id", meta.getOrDefault("tokenizer.ggml.unknown_token_id", 0L));
                spec.put("bos_id", bos);
                spec.put("eos_id", eos);
                spec.put("add_dummy_prefix", !Boolean.FALSE.equals(meta.get("tokenizer.ggml.add_space_prefix")));
                spec.put("remove_extra_whitespaces", false);
                spec.put("escape_whitespaces", true);
                spec.put("add_bos", !Boolean.FALSE.equals(meta.get("tokenizer.ggml.add_bos_token")));
                spec.put("parse_control", true);
                if (template != null) {
                    spec.put("chat_template", template);
                }
                return fromSpmJson(spec);
            }
            case "gpt2" -> {
                Map<String, Object> root = new LinkedHashMap<>();
                List<Object> addedTokens = new ArrayList<>();
                Map<String, Object> vocab = new LinkedHashMap<>();
                for (int i = 0; i < tokenList.size(); i++) {
                    long type = i < typeList.size() ? (Long) typeList.get(i) : 1;
                    String t = (String) tokenList.get(i);
                    vocab.put(t, (long) i);
                    if (type == 3 || type == 4) {
                        Map<String, Object> a = new LinkedHashMap<>();
                        a.put("id", (long) i);
                        a.put("content", t);
                        a.put("special", type == 3);
                        addedTokens.add(a);
                    }
                }
                String pre = String.valueOf(meta.get("tokenizer.ggml.pre"));
                String regex = switch (pre) {
                    case "qwen2", "deepseek-r1-qwen", "qwen35" -> TokenizerParts.QWEN2_PATTERN;
                    case "gpt-2", "gpt2", "starcoder", "refact", "olmo", "phi-2", "stablelm2" ->
                            TokenizerParts.GPT2_PATTERN;
                    default -> TokenizerParts.LLAMA3_PATTERN;
                };
                Map<String, Object> model = new LinkedHashMap<>();
                model.put("type", "BPE");
                model.put("vocab", vocab);
                model.put("merges", meta.getOrDefault("tokenizer.ggml.merges", List.of()));
                model.put("ignore_merges", pre.equals("llama-bpe") || pre.equals("llama3"));
                root.put("added_tokens", addedTokens);
                root.put("pre_tokenizer", Map.of("type", "Sequence", "pretokenizers", List.of(
                        Map.of("type", "Split", "pattern", Map.of("Regex", regex), "behavior", "Isolated"),
                        Map.of("type", "ByteLevel", "add_prefix_space", false, "use_regex", false))));
                root.put("decoder", Map.of("type", "ByteLevel"));
                if (bos >= 0 && Boolean.TRUE.equals(meta.get("tokenizer.ggml.add_bos_token"))) {
                    String name = (String) tokenList.get((int) bos);
                    root.put("post_processor", Map.of("type", "TemplateProcessing",
                            "single", List.of(Map.of("SpecialToken", Map.of("id", name)), Map.of("Sequence",
                                    Map.of("id", "A"))),
                            "special_tokens", Map.of(name, Map.of("id", name, "ids", List.of(bos)))));
                }
                root.put("model", model);
                Map<String, Object> cfg = new HashMap<>();
                if (bos >= 0) {
                    cfg.put("bos_token", tokenList.get((int) bos));
                }
                if (eos >= 0) {
                    cfg.put("eos_token", tokenList.get((int) eos));
                }
                if (template != null) {
                    cfg.put("chat_template", template);
                }
                return fromJson(Json.write(root), cfg);
            }
            default -> throw new IllegalArgumentException("GGUF tokenizer model " + kind + " is not supported");
        }
    }

    /**
     * The {@code tokenizer.ggml.*} metadata that describes this tokenizer in a GGUF file, so
     * llama.cpp (and {@link #fromGguf}) can rebuild it: SentencePiece models as
     * {@code "llama"}, byte-level BPE from {@code tokenizer.json} as {@code "gpt2"}. GGUF has no
     * place for a few rarer settings: a byte-level {@code add_prefix_space}, and SentencePiece's
     * removal of extra whitespace, so texts that depend on them encode differently there.
     *
     * @return the keys and values, in order
     * @throws IllegalArgumentException for WordPiece and Unigram {@code tokenizer.json} models
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> ggufMetadata() {
        Map<String, Object> m = new LinkedHashMap<>();
        Map<String, Object> cfg = config instanceof Map<?, ?> c ? (Map<String, Object>) c : Map.of();
        List<Integer> types = new ArrayList<>();
        if (engine instanceof Spm) {
            cfg = Json.object(cfg.get("sentencepiece"));
            m.put("tokenizer.ggml.model", "llama");
            List<Object> specTypes = Json.array(cfg.get("types"));
            for (Object t : specTypes) {
                types.add(((Number) t).intValue());
            }
            List<Float> scores = new ArrayList<>();
            List<Object> scoreList = cfg.get("scores") instanceof double[] d
                    ? Arrays.stream(d).boxed().map(Object.class::cast).toList() : Json.array(cfg.get("scores"));
            for (Object o : scoreList) {
                scores.add(((Number) o).floatValue());
            }
            m.put("tokenizer.ggml.tokens", tokens);
            m.put("tokenizer.ggml.scores", scores);
            m.put("tokenizer.ggml.token_type", types);
            m.put("tokenizer.ggml.add_space_prefix", !Boolean.FALSE.equals(cfg.get("add_dummy_prefix")));
        } else {
            Map<String, Object> model = Json.object(cfg.get("model"));
            if (!model.containsKey("merges")) {
                throw new IllegalArgumentException("only byte-level BPE and SentencePiece tokenizers go into GGUF");
            }
            m.put("tokenizer.ggml.model", "gpt2");
            String regex = findRegex(cfg.get("pre_tokenizer"));
            String pre = TokenizerParts.QWEN2_PATTERN.equals(regex) ? "qwen2"
                    : TokenizerParts.LLAMA3_PATTERN.equals(regex) ? "llama-bpe" : "gpt2";
            m.put("tokenizer.ggml.pre", pre);
            Set<Integer> control = new HashSet<>();
            Set<Integer> user = new HashSet<>();
            for (Added a : added) {
                (a.special ? control : user).add(a.id);
            }
            for (int i = 0; i < tokens.size(); i++) {
                types.add(control.contains(i) ? 3 : user.contains(i) ? 4 : 1);
            }
            List<String> merges = new ArrayList<>();
            for (Object o : Json.array(model.get("merges"))) {
                merges.add(o instanceof String str ? str : Json.array(o).get(0) + " " + Json.array(o).get(1));
            }
            List<String> names = new ArrayList<>(tokens.size());
            for (int i = 0; i < tokens.size(); i++) {
                names.add(tokens.get(i) == null ? "[PAD" + i + "]" : tokens.get(i));
            }
            m.put("tokenizer.ggml.tokens", names);
            m.put("tokenizer.ggml.token_type", types);
            m.put("tokenizer.ggml.merges", merges);
        }
        if (bos != null) {
            m.put("tokenizer.ggml.bos_token_id", (long) bos);
        }
        if (eos != null) {
            m.put("tokenizer.ggml.eos_token_id", (long) eos);
        }
        int[] with = encode("a", true);
        int[] without = encode("a", false);
        m.put("tokenizer.ggml.add_bos_token", with.length > without.length && bos != null && with[0] == bos);
        if (chatTemplate != null) {
            m.put("tokenizer.chat_template", chatTemplate);
        }
        return m;
    }

    /** The first split pattern in a pre-tokenizer description, or null. */
    private static String findRegex(Object node) {
        if (node instanceof Map<?, ?> map) {
            if (map.get("Regex") instanceof String r) {
                return r;
            }
            for (Object v : map.values()) {
                String r = findRegex(v);
                if (r != null) {
                    return r;
                }
            }
        } else if (node instanceof List<?> list) {
            for (Object v : list) {
                String r = findRegex(v);
                if (r != null) {
                    return r;
                }
            }
        }
        return null;
    }

    // ---------------------------------------------------------------- encoding

    /**
     * Encodes text without adding special tokens: what a model sees for plain text.
     * Special tokens written in the text, such as {@code <|im_start|>}, are recognized.
     *
     * @param text the text
     * @return the ids
     */
    @Override
    public int[] encode(String text) {
        return encode(text, false);
    }

    /**
     * Encodes text, optionally with the special tokens the model adds around a sequence
     * (such as Llama's begin token, or BERT's {@code [CLS]} and {@code [SEP]}).
     *
     * @param text the text
     * @param addSpecialTokens whether to add them
     * @return the ids
     */
    public int[] encode(String text, boolean addSpecialTokens) {
        Objects.requireNonNull(text, "text");
        List<Integer> out = new ArrayList<>();
        if (engine instanceof Spm spm) {
            String norm = normalizeSpm(spm, text);
            for (Object piece : splitAdded(norm, true)) {
                if (piece instanceof Integer id) {
                    out.add(id);
                } else if (!((String) piece).isEmpty()) {
                    spm.model.encode((String) piece, out);
                }
            }
            if (addSpecialTokens && spm.addBos && bos != null) {
                out.addFirst(bos);
            }
        } else {
            Hf hf = (Hf) engine;
            boolean first = true;
            for (Object piece : splitAdded(text, false)) {
                if (piece instanceof Integer id) {
                    out.add(id);
                } else {
                    String s = hf.normalizer.apply((String) piece);
                    if (!s.isEmpty()) {
                        for (String word : hf.preTokenizer.split(s, first)) {
                            if (!word.isEmpty()) {
                                hf.model.encode(word, out);
                            }
                        }
                    }
                }
                first = false;
            }
            if (addSpecialTokens) {
                List<Integer> wrapped = new ArrayList<>();
                for (Object item : hf.single) {
                    if (item instanceof int[] special) {
                        for (int id : special) {
                            wrapped.add(id);
                        }
                    } else {
                        wrapped.addAll(out);
                    }
                }
                out = wrapped;
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static String normalizeSpm(Spm spm, String text) {
        String s = text;
        if (spm.removeExtraSpaces) {
            // SentencePiece trims and collapses spaces only, not other whitespace
            int start = 0;
            while (start < text.length() && text.charAt(start) == ' ') {
                start++;
            }
            int end = text.length();
            while (end > start && text.charAt(end - 1) == ' ') {
                end--;
            }
            s = text.substring(start, end).replaceAll(" {2,}", " ");
        }
        if (s.isEmpty()) {
            return s;
        }
        if (spm.dummyPrefix) {
            s = " " + s;
        }
        return spm.escape ? s.replace(' ', '▁') : s;
    }

    /** Splits text into added tokens (as Integer ids) and the text between them. */
    private List<Object> splitAdded(String text, boolean spmStyle) {
        List<Object> out = new ArrayList<>();
        if (added.isEmpty()) {
            out.add(text);
            return out;
        }
        int at = 0;
        int segment = 0;
        while (at < text.length()) {
            Added match = null;
            for (Added a : added) {
                if (!a.content.isEmpty() && text.startsWith(a.content, at)) {
                    match = a;
                    break;
                }
            }
            if (match == null) {
                at++;
                continue;
            }
            String before = text.substring(segment, at);
            if (match.lstrip) {
                before = spmStyle ? before : before.stripTrailing();
            }
            out.add(before);
            out.add(match.id);
            at += match.content.length();
            if (match.rstrip) {
                while (at < text.length() && Character.isWhitespace(text.charAt(at))) {
                    at++;
                }
            }
            segment = at;
        }
        out.add(text.substring(segment));
        out.removeIf(o -> o instanceof String s && s.isEmpty());
        return out;
    }

    // ---------------------------------------------------------------- decoding

    @Override
    public String decode(int... ids) {
        return decode(ids, false);
    }

    /**
     * Decodes ids back into text.
     *
     * @param ids the ids
     * @param skipSpecialTokens whether to leave special tokens out
     * @return the text
     */
    public String decode(int[] ids, boolean skipSpecialTokens) {
        if (engine instanceof Spm spm) {
            StringBuilder sb = new StringBuilder();
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            for (int id : ids) {
                int type = id >= 0 && id < spm.types.length ? spm.types[id] : SentencePieceModel.NORMAL;
                if (type == SentencePieceModel.CONTROL) {
                    continue;
                }
                if (type == SentencePieceModel.BYTE) {
                    String p = tokens.get(id);
                    bytes.write(Integer.parseInt(p.substring(3, 5), 16));
                    continue;
                }
                if (bytes.size() > 0) {
                    sb.append(TokenizerParts.utf8(bytes.toByteArray()));
                    bytes.reset();
                }
                sb.append(type == SentencePieceModel.UNKNOWN ? " ⁇ " : token(id));
            }
            sb.append(TokenizerParts.utf8(bytes.toByteArray()));
            String s = spm.escape ? sb.toString().replace('▁', ' ') : sb.toString();
            return spm.dummyPrefix && s.startsWith(" ") ? s.substring(1) : s;
        }
        Hf hf = (Hf) engine;
        List<String> pieces = new ArrayList<>();
        for (int id : ids) {
            if (skipSpecialTokens && special.contains(id)) {
                continue;
            }
            pieces.add(token(id));
        }
        return String.join("", hf.decoder.decode(pieces));
    }

    // ---------------------------------------------------------------- vocabulary

    /**
     * Returns the token of an id, as the vocabulary writes it.
     *
     * @param id the id
     * @return the token
     * @throws IllegalArgumentException if no token has this id
     */
    public String token(int id) {
        if (id < 0 || id >= tokens.size() || tokens.get(id) == null) {
            throw new IllegalArgumentException("no token with id " + id);
        }
        return tokens.get(id);
    }

    /**
     * Returns the id of a token, as the vocabulary writes it.
     *
     * @param token the token, such as {@code "<|im_end|>"}
     * @return its id, if it has one
     */
    public OptionalInt id(String token) {
        Integer id = ids.get(token);
        return id == null ? OptionalInt.empty() : OptionalInt.of(id);
    }

    /** @return the id of the begin-of-sequence token, if the model names one */
    public OptionalInt bos() {
        return bos == null ? OptionalInt.empty() : OptionalInt.of(bos);
    }

    /** @return the id of the end-of-sequence token, if the model names one */
    public OptionalInt eos() {
        return eos == null ? OptionalInt.empty() : OptionalInt.of(eos);
    }

    /** @return the ids of the special tokens (control tokens, which decoding can skip) */
    public Set<Integer> specialIds() {
        return special;
    }

    /**
     * Returns the chat template (a Jinja template) published with the model, if any.
     *
     * @return the template text, or {@code null}
     */
    public String chatTemplate() {
        return chatTemplate;
    }

    /**
     * Renders a conversation with the model's {@linkplain #chatTemplate chat template},
     * with {@code bos_token} and {@code eos_token} set from the vocabulary.
     *
     * @param messages the conversation; see {@link ChatTemplate#message}
     * @param addGenerationPrompt whether to end with the start of an assistant turn
     * @return the prompt text, ready for {@link #encode(String)}
     * @throws IllegalStateException if the model has no chat template
     */
    public String applyChatTemplate(List<? extends Map<String, ?>> messages, boolean addGenerationPrompt) {
        if (chatTemplate == null) {
            throw new IllegalStateException("this tokenizer has no chat template");
        }
        ChatTemplate t = template;
        if (t == null) {
            t = ChatTemplate.of(chatTemplate);
            template = t;
        }
        Map<String, Object> extra = new HashMap<>();
        extra.put("bos_token", bos == null ? "" : token(bos));
        extra.put("eos_token", eos == null ? "" : token(eos));
        return t.render(messages, addGenerationPrompt, extra);
    }

    @Override
    public int size() {
        return tokens.size();
    }

    @Override
    public String kind() {
        return "pretrained";
    }

    /**
     * Saves the tokenizer as JSON: a Hugging Face {@code tokenizer.json} for tokenizers
     * read from one (or from a byte-level GGUF vocabulary), Brainiac's own form for
     * SentencePiece ones. {@link #load} reads both.
     *
     * @param path the file
     * @throws IOException if it cannot be written
     */
    @Override
    public void save(Path path) throws IOException {
        Files.writeString(path, Json.write(config), StandardCharsets.UTF_8);
    }

    @Override
    public String toString() {
        return "PretrainedTokenizer[" + (engine instanceof Spm ? "sentencepiece" : "tokenizer.json") + ", "
                + tokens.size() + " tokens]";
    }
}
