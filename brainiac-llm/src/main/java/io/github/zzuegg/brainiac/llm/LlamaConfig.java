package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.Json;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The shape of a Llama-family model: Llama 2 and 3, Mistral, Qwen 2 and the many models that
 * share their design (RMSNorm before each sublayer, rotary positions, grouped-query
 * attention, a SwiGLU feed-forward network).
 *
 * @param architecture the family, such as {@code "llama"} or {@code "qwen2"}
 * @param vocabSize the vocabulary size
 * @param hidden the hidden size
 * @param intermediate the feed-forward hidden size
 * @param layers the number of blocks
 * @param heads the query heads
 * @param kvHeads the key/value heads
 * @param headDim the size of each head
 * @param rmsEps the RMSNorm epsilon
 * @param rope the rotary position settings
 * @param tieEmbeddings whether the output projection reuses the token embeddings
 * @param attentionBias whether the query, key and value projections have biases (Qwen 2)
 * @param maxPositions the context length the model was trained for
 */
public record LlamaConfig(String architecture, int vocabSize, int hidden, int intermediate, int layers, int heads,
        int kvHeads, int headDim, float rmsEps, Rope.Settings rope, boolean tieEmbeddings, boolean attentionBias,
        int maxPositions) {

    /**
     * Reads a Hugging Face {@code config.json} (transformers 4 or 5 layout).
     *
     * @param config the parsed file
     * @return the configuration
     */
    public static LlamaConfig fromHuggingFace(Map<String, Object> config) {
        String type = Json.string(config, "model_type", "llama");
        int hidden = (int) Json.integer(config, "hidden_size");
        int heads = (int) Json.integer(config, "num_attention_heads");
        int headDim = (int) Json.integer(config, "head_dim", hidden / heads);
        Map<String, Object> ropeNode = config.get("rope_parameters") instanceof Map<?, ?> m
                ? Json.object(m) : config.get("rope_scaling") instanceof Map<?, ?> m2 ? Json.object(m2) : Map.of();
        double theta = Json.number(ropeNode, "rope_theta", Json.number(config, "rope_theta", 10000));
        String ropeType = Json.string(ropeNode, "rope_type", Json.string(ropeNode, "type", "default"));
        Rope.Settings rope = new Rope.Settings(headDim, theta, ropeType,
                Json.number(ropeNode, "factor", 1), Json.number(ropeNode, "low_freq_factor", 1),
                Json.number(ropeNode, "high_freq_factor", 4),
                (int) Json.integer(ropeNode, "original_max_position_embeddings",
                        Json.integer(config, "max_position_embeddings", 2048)), null);
        boolean bias = config.containsKey("attention_bias") ? Boolean.TRUE.equals(config.get("attention_bias"))
                : type.equals("qwen2");
        return new LlamaConfig(type, (int) Json.integer(config, "vocab_size"), hidden,
                (int) Json.integer(config, "intermediate_size"), (int) Json.integer(config, "num_hidden_layers"),
                heads, (int) Json.integer(config, "num_key_value_heads", heads), headDim,
                (float) Json.number(config, "rms_norm_eps", 1e-6), rope,
                Boolean.TRUE.equals(config.get("tie_word_embeddings")), bias,
                (int) Json.integer(config, "max_position_embeddings", 2048));
    }

    /**
     * Reads the model settings of a GGUF file.
     *
     * @param gguf the opened file
     * @return the configuration
     */
    public static LlamaConfig fromGguf(Gguf gguf) {
        Map<String, Object> m = gguf.metadata();
        String a = gguf.architecture();
        if (!a.equals("llama") && !a.equals("qwen2")) {
            throw new IllegalArgumentException("GGUF architecture " + a + " is not supported; llama and qwen2 are");
        }
        int hidden = (int) number(m, a + ".embedding_length", -1);
        int heads = (int) number(m, a + ".attention.head_count", -1);
        int headDim = (int) number(m, a + ".rope.dimension_count", hidden / heads);
        int vocab = m.get(a + ".vocab_size") instanceof Long v ? v.intValue()
                : m.get("tokenizer.ggml.tokens") instanceof List<?> t ? t.size()
                : gguf.shape("token_embd.weight").dim(0);
        double theta = number(m, a + ".rope.freq_base", 10000);
        String scaling = m.get(a + ".rope.scaling.type") instanceof String s ? s : "default";
        Rope.Settings rope = new Rope.Settings(headDim, theta, scaling.equals("none") ? "default" : scaling,
                number(m, a + ".rope.scaling.factor", 1), 1, 4,
                (int) number(m, a + ".rope.scaling.original_context_length", number(m, a + ".context_length", 2048)),
                null);
        return new LlamaConfig(a, vocab, hidden, (int) number(m, a + ".feed_forward_length", -1),
                (int) number(m, a + ".block_count", -1), heads,
                (int) number(m, a + ".attention.head_count_kv", heads), headDim,
                (float) number(m, a + ".attention.layer_norm_rms_epsilon", 1e-6), rope,
                !gguf.contains("output.weight"), a.equals("qwen2"), (int) number(m, a + ".context_length", 2048));
    }

    private static double number(Map<String, Object> m, String key, double otherwise) {
        Object v = m.get(key);
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        if (otherwise < 0) {
            throw new IllegalArgumentException("GGUF metadata has no " + key);
        }
        return otherwise;
    }

    /**
     * Writes this configuration as a Hugging Face {@code config.json}.
     *
     * @return the JSON object
     */
    public Map<String, Object> toHuggingFace() {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("architectures", List.of(architecture.equals("qwen2") ? "Qwen2ForCausalLM" : "LlamaForCausalLM"));
        c.put("model_type", architecture);
        c.put("vocab_size", (long) vocabSize);
        c.put("hidden_size", (long) hidden);
        c.put("intermediate_size", (long) intermediate);
        c.put("num_hidden_layers", (long) layers);
        c.put("num_attention_heads", (long) heads);
        c.put("num_key_value_heads", (long) kvHeads);
        c.put("head_dim", (long) headDim);
        c.put("rms_norm_eps", (double) rmsEps);
        c.put("rope_theta", rope.theta());
        if (!rope.type().equals("default")) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("rope_type", rope.type());
            s.put("factor", rope.factor());
            s.put("low_freq_factor", rope.lowFreqFactor());
            s.put("high_freq_factor", rope.highFreqFactor());
            s.put("original_max_position_embeddings", (long) rope.originalMaxPositions());
            c.put("rope_scaling", s);
        }
        c.put("tie_word_embeddings", tieEmbeddings);
        c.put("attention_bias", attentionBias);
        c.put("max_position_embeddings", (long) maxPositions);
        c.put("hidden_act", "silu");
        return c;
    }

    /**
     * A small model for tests and training from scratch.
     *
     * @param vocabSize the vocabulary size
     * @param hidden the hidden size
     * @param layers the number of blocks
     * @param heads the query heads
     * @param kvHeads the key/value heads
     * @return the configuration, with a feed-forward size of {@code 8/3 * hidden} rounded to 32
     */
    public static LlamaConfig small(int vocabSize, int hidden, int layers, int heads, int kvHeads) {
        int ffn = (hidden * 8 / 3 + 31) / 32 * 32;
        return new LlamaConfig("llama", vocabSize, hidden, ffn, layers, heads, kvHeads, hidden / heads, 1e-5f,
                new Rope.Settings(hidden / heads, 10000, "default", 1, 1, 4, 2048, null), false, false, 2048);
    }
}
