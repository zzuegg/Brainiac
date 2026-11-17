package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.Json;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The shape of a Llama-family model: Llama 2 and 3, Mistral, Qwen 2 and 3, Phi-3, Gemma 1, 2
 * and 3 and the many models that share their design (RMSNorm before each sublayer, rotary
 * positions, grouped-query attention, a gated feed-forward network). What sets the families
 * apart is in {@link Extras}.
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
 * @param extras what the other families add
 */
public record LlamaConfig(String architecture, int vocabSize, int hidden, int intermediate, int layers, int heads,
        int kvHeads, int headDim, float rmsEps, Rope.Settings rope, boolean tieEmbeddings, boolean attentionBias,
        int maxPositions, Extras extras) {

    /**
     * What the families beyond Llama add.
     *
     * @param qkNorm whether queries and keys are RMS-normalized per head before the rotation
     *     (Qwen 3, Gemma 3)
     * @param slidingWindow how many steps local layers see, themselves included, or 0 for none
     *     (Mistral, Phi-3, Gemma 2 and 3)
     * @param globalEvery with a window: every this many layers one sees everything (layer
     *     {@code i} when {@code (i + 1) % globalEvery == 0}; Gemma 2: 2, Gemma 3: 6), or 0
     *     when every layer is local
     * @param localTheta the rotary base of local layers, without scaling, or 0 for the same
     *     rotation as global layers (Gemma 3: 10000)
     * @param activation the gate's activation: {@code SILU}, or {@code GELU_TANH} for Gemma
     * @param scaleEmbeddings whether embeddings are multiplied by {@code sqrt(hidden)} (Gemma)
     * @param postNorms whether the attention and feed-forward outputs are normalized again
     *     before the residual (Gemma 2 and 3)
     * @param normOffset whether Hugging Face files store norm weights less one, the model
     *     scaling by {@code 1 + w} (Gemma); the weights here always include the one
     * @param attentionSoftcap the attention score cap, or 0 for none (Gemma 2: 50)
     * @param finalSoftcap the logit cap, or 0 for none (Gemma 2: 30)
     * @param queryScale the attention score scale, or 0 for {@code 1 / sqrt(headDim)}
     * @param fused whether the weights join q, k and v in one projection and gate and up in
     *     another (Phi-3)
     * @param moe the mixture of experts that replaces the feed-forward network, or
     *     {@link Moe#NONE}
     */
    public record Extras(boolean qkNorm, int slidingWindow, int globalEvery, double localTheta,
            ActivationOp activation, boolean scaleEmbeddings, boolean postNorms, boolean normOffset,
            float attentionSoftcap, float finalSoftcap, float queryScale, boolean fused, Moe moe) {

        /** Plain Llama. */
        public static final Extras NONE = new Extras(false, 0, 0, 0, ActivationOp.SILU, false, false, false, 0, 0, 0,
                false, Moe.NONE);

        /** Checks the experts are there. */
        public Extras {
            java.util.Objects.requireNonNull(moe, "moe");
        }

        /**
         * @param layer a layer
         * @return whether it sees only the window
         */
        public boolean local(int layer) {
            return slidingWindow > 0 && !(globalEvery > 0 && (layer + 1) % globalEvery == 0);
        }
    }

    /**
     * A sparse mixture of experts in every block (Mixtral, Qwen 2 and 3 MoE): a router picks
     * {@code perToken} of {@code experts} gated feed-forward networks for each token, whose
     * outputs are summed by the router's softmax weights.
     *
     * @param experts the experts per block, or 0 for a dense feed-forward network
     * @param perToken the experts each token uses
     * @param ffn each expert's hidden size
     * @param normalize whether the chosen experts' weights are scaled to sum to one (Mixtral,
     *     Qwen 3) or kept as the softmax gave them (Qwen 2)
     * @param sharedFfn the hidden size of an expert every token uses, gated by a sigmoid
     *     (Qwen 2), or 0 for none
     */
    public record Moe(int experts, int perToken, int ffn, boolean normalize, int sharedFfn) {

        /** No experts. */
        public static final Moe NONE = new Moe(0, 0, 0, false, 0);

        /** Checks the sizes. */
        public Moe {
            if (experts < 0 || experts > 0 && (perToken < 1 || perToken > experts || ffn < 1)) {
                throw new IllegalArgumentException("bad mixture of experts: " + perToken + " of " + experts
                        + " experts of " + ffn);
            }
        }

        /** @return whether there are experts */
        public boolean present() {
            return experts > 0;
        }
    }

    /**
     * A plain Llama-family configuration, with no {@link Extras}.
     *
     * @param architecture the family
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
     * @param attentionBias whether q, k and v have biases
     * @param maxPositions the trained context length
     */
    public LlamaConfig(String architecture, int vocabSize, int hidden, int intermediate, int layers, int heads,
            int kvHeads, int headDim, float rmsEps, Rope.Settings rope, boolean tieEmbeddings, boolean attentionBias,
            int maxPositions) {
        this(architecture, vocabSize, hidden, intermediate, layers, heads, kvHeads, headDim, rmsEps, rope,
                tieEmbeddings, attentionBias, maxPositions, Extras.NONE);
    }

    /** The compact form checks the extras are there. */
    public LlamaConfig {
        java.util.Objects.requireNonNull(extras, "extras");
    }

    /**
     * @param rope new rotary settings
     * @return this configuration with them
     */
    public LlamaConfig withRope(Rope.Settings rope) {
        return new LlamaConfig(architecture, vocabSize, hidden, intermediate, layers, heads, kvHeads, headDim, rmsEps,
                rope, tieEmbeddings, attentionBias, maxPositions, extras);
    }

    /**
     * @param extras new extras
     * @return this configuration with them
     */
    public LlamaConfig withExtras(Extras extras) {
        return new LlamaConfig(architecture, vocabSize, hidden, intermediate, layers, heads, kvHeads, headDim, rmsEps,
                rope, tieEmbeddings, attentionBias, maxPositions, extras);
    }

    private static boolean gemma(String type) {
        return type.startsWith("gemma");
    }

    /**
     * The Hugging Face {@code model_type} values whose {@code config.json} this reads and whose
     * weights {@link Llama} loads from a model folder.
     */
    public static final java.util.Set<String> MODEL_TYPES = java.util.Set.of("llama", "mistral", "qwen2", "qwen3",
            "phi3", "gemma", "gemma2", "gemma3_text", "mixtral", "qwen2_moe", "qwen3_moe");

    /** The GGUF {@code general.architecture} values {@link #fromGguf} reads. */
    public static final java.util.Set<String> GGUF_ARCHITECTURES = java.util.Set.of("llama", "qwen2", "qwen3", "phi3",
            "gemma", "gemma2", "gemma3", "qwen2moe", "qwen3moe");

    /**
     * Reads a Hugging Face {@code config.json} (transformers 4 or 5 layout) of Llama, Mistral,
     * Qwen 2 or 3, Phi-3 or Gemma 1, 2 or 3 ({@code gemma3_text}).
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
        // transformers 5 keeps a rotation per layer type (Gemma 3)
        Map<String, Object> localNode = null;
        if (ropeNode.get("full_attention") instanceof Map<?, ?> full) {
            localNode = ropeNode.get("sliding_attention") instanceof Map<?, ?> l ? Json.object(l) : null;
            ropeNode = Json.object(full);
        }
        double theta = Json.number(ropeNode, "rope_theta", Json.number(config, "rope_theta", 10000));
        double partial = Json.number(ropeNode, "partial_rotary_factor", Json.number(config, "partial_rotary_factor", 1));
        int rotary = (int) Math.round(headDim * partial);
        String ropeType = Json.string(ropeNode, "rope_type", Json.string(ropeNode, "type", "default"));
        int maxPositions = (int) Json.integer(config, "max_position_embeddings", 2048);
        int original = (int) Json.integer(ropeNode, "original_max_position_embeddings",
                Json.integer(config, "original_max_position_embeddings", maxPositions));
        double factor = Json.number(ropeNode, "factor", 1);
        float[] freqFactors = null;
        if (ropeType.equals("longrope")) {
            // the short factors: the model's own context; past it Phi-3 would switch to the long ones
            freqFactors = floats(ropeNode.get("short_factor"));
            factor = Json.number(ropeNode, "factor", (double) maxPositions / original);
        }
        Rope.Settings rope = new Rope.Settings(rotary, theta, ropeType, factor,
                Json.number(ropeNode, "low_freq_factor", 1), Json.number(ropeNode, "high_freq_factor", 4), original,
                freqFactors).withYarn(Json.number(ropeNode, "beta_fast", 32), Json.number(ropeNode, "beta_slow", 1),
                        Json.number(ropeNode, "attention_factor", 0));
        boolean bias = config.containsKey("attention_bias") ? Boolean.TRUE.equals(config.get("attention_bias"))
                : config.containsKey("qkv_bias") ? Boolean.TRUE.equals(config.get("qkv_bias"))
                : type.equals("qwen2") || type.equals("qwen2_moe");

        boolean gemma = gemma(type);
        boolean gemma23 = type.equals("gemma2") || type.startsWith("gemma3");
        int layers = (int) Json.integer(config, "num_hidden_layers");
        int window = 0;
        int every = 0;
        if (config.get("sliding_window") instanceof Number w && !Boolean.FALSE.equals(config.get("use_sliding_window"))) {
            window = w.intValue();
            if (config.get("layer_types") instanceof List<?> types) {
                every = globalEvery(types, layers);
                if (every < 0) {
                    window = 0;
                }
            } else if (type.equals("gemma2")) {
                every = 2;
            } else if (type.startsWith("gemma3")) {
                every = (int) Json.integer(config, "sliding_window_pattern", 6);
            }
        }
        double localTheta = localNode != null ? Json.number(localNode, "rope_theta", 10000)
                : Json.number(config, "rope_local_base_freq", 0);
        String act = Json.string(config, "hidden_activation", Json.string(config, "hidden_act", "silu"));
        ActivationOp activation = switch (act) {
            case "silu", "swish" -> ActivationOp.SILU;
            case "gelu_pytorch_tanh", "gelu_new", "gelu_fast" -> ActivationOp.GELU_TANH;
            case "gelu" -> gemma ? ActivationOp.GELU_TANH : ActivationOp.GELU;   // Gemma's gelu is tanh's
            default -> throw new IllegalArgumentException("activation " + act + " is not supported");
        };
        float queryScale = config.get("query_pre_attn_scalar") instanceof Number s
                ? (float) (1 / Math.sqrt(s.doubleValue())) : 0;
        Extras extras = new Extras(type.equals("qwen3") || type.equals("qwen3_moe") || type.startsWith("gemma3"),
                Math.max(window, 0),
                Math.max(every, 0), localTheta, activation, gemma, gemma23, gemma,
                (float) Json.number(config, "attn_logit_softcapping", 0),
                (float) Json.number(config, "final_logit_softcapping", 0), queryScale, type.equals("phi3"),
                moe(config, type));
        boolean tied = config.containsKey("tie_word_embeddings") ? Boolean.TRUE.equals(config.get("tie_word_embeddings"))
                : gemma;
        return new LlamaConfig(type, (int) Json.integer(config, "vocab_size"), hidden,
                (int) Json.integer(config, "intermediate_size"), layers, heads,
                (int) Json.integer(config, "num_key_value_heads", heads), headDim,
                (float) Json.number(config, "rms_norm_eps", 1e-6), rope, tied, bias, maxPositions, extras);
    }

    private static Moe moe(Map<String, Object> config, String type) {
        long experts = Json.integer(config, "num_local_experts", Json.integer(config, "num_experts", 0));
        if (experts == 0) {
            return Moe.NONE;
        }
        if (Json.integer(config, "decoder_sparse_step", 1) != 1
                || config.get("mlp_only_layers") instanceof List<?> dense && !dense.isEmpty()) {
            throw new IllegalArgumentException("mixtures of experts in only some layers are not supported");
        }
        boolean mixtral = type.equals("mixtral");
        return new Moe((int) experts, (int) Json.integer(config, "num_experts_per_tok", 2),
                (int) (mixtral ? Json.integer(config, "intermediate_size") : Json.integer(config, "moe_intermediate_size")),
                mixtral || Boolean.TRUE.equals(config.get("norm_topk_prob")),
                (int) Json.integer(config, "shared_expert_intermediate_size", 0));
    }

    /** The period of the full-attention layers among {@code layer_types}: 0 none, -1 no window at all. */
    private static int globalEvery(List<?> types, int layers) {
        int first = types.indexOf("full_attention");
        if (!types.contains("sliding_attention")) {
            return -1;
        }
        if (first < 0) {
            return 0;
        }
        int every = first + 1;
        for (int i = 0; i < Math.min(types.size(), layers); i++) {
            boolean full = "full_attention".equals(types.get(i));
            if (full != ((i + 1) % every == 0)) {
                throw new IllegalArgumentException("layer_types " + types + " are not one full layer in every "
                        + every + "; that pattern is not supported");
            }
        }
        return every;
    }

    private static float[] floats(Object list) {
        if (!(list instanceof List<?> l)) {
            return null;
        }
        float[] f = new float[l.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = ((Number) l.get(i)).floatValue();
        }
        return f;
    }

    /**
     * Reads the model settings of a GGUF file: architectures {@code llama} (also Mistral),
     * {@code qwen2}, {@code qwen3}, {@code phi3}, {@code gemma}, {@code gemma2} and {@code gemma3}.
     *
     * @param gguf the opened file
     * @return the configuration
     */
    public static LlamaConfig fromGguf(Gguf gguf) {
        Map<String, Object> m = gguf.metadata();
        String a = gguf.architecture();
        if (!GGUF_ARCHITECTURES.contains(a)) {
            throw new IllegalArgumentException("GGUF architecture " + a + " is not supported; "
                    + new java.util.TreeSet<>(GGUF_ARCHITECTURES) + " are");
        }
        int hidden = (int) number(m, a + ".embedding_length", -1);
        int heads = (int) number(m, a + ".attention.head_count", -1);
        int layers = (int) number(m, a + ".block_count", -1);
        int headDim = (int) number(m, a + ".attention.key_length", hidden / heads);
        int rotary = (int) number(m, a + ".rope.dimension_count", headDim);
        int vocab = m.get(a + ".vocab_size") instanceof Long v ? v.intValue()
                : m.get("tokenizer.ggml.tokens") instanceof List<?> t ? t.size()
                : gguf.shape("token_embd.weight").dim(0);
        int context = (int) number(m, a + ".context_length", 2048);
        double theta = number(m, a + ".rope.freq_base", 10000);
        String scaling = m.get(a + ".rope.scaling.type") instanceof String s ? s : "default";
        int original = (int) number(m, a + ".rope.scaling.original_context_length", context);
        double factor = number(m, a + ".rope.scaling.factor", 1);
        float[] freqFactors = null;
        double attentionFactor = 0;
        if (gguf.contains("rope_factors_short.weight")) {
            // Phi-3's longrope, with the short factors (see fromHuggingFace)
            scaling = "longrope";
            freqFactors = gguf.tensor("rope_factors_short.weight", io.github.zzuegg.brainiac.tensor.Device.cpu(),
                    io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray();
            factor = (double) context / original;
            attentionFactor = number(m, a + ".rope.scaling.attn_factor", 0);
        }
        Rope.Settings rope = new Rope.Settings(rotary, theta, scaling.equals("none") ? "default" : scaling, factor, 1,
                4, original, freqFactors).withYarn(number(m, a + ".rope.scaling.yarn_beta_fast", 32),
                        number(m, a + ".rope.scaling.yarn_beta_slow", 1), attentionFactor);
        boolean gemma = gemma(a);
        boolean gemma23 = a.equals("gemma2") || a.equals("gemma3");
        int window = (int) number(m, a + ".attention.sliding_window", 0);
        int every = a.equals("gemma2") ? 2 : a.equals("gemma3") ? 6 : 0;
        every = (int) number(m, a + ".attention.sliding_window_pattern", every);
        // as llama.cpp: queries scaled by 1 / sqrt(head size), but for the 27B Gemmas
        boolean big = a.equals("gemma2") && layers == 46 || a.equals("gemma3") && layers == 62;
        float queryScale = (float) number(m, a + ".attention.scale",
                big ? 1 / Math.sqrt((double) hidden / heads) : 0);
        Moe moe = Moe.NONE;
        if (m.get(a + ".expert_count") instanceof Number count && count.intValue() > 0) {
            // Mixtral is written as "llama" with experts
            moe = new Moe(count.intValue(), (int) number(m, a + ".expert_used_count", -1),
                    (int) number(m, a + ".expert_feed_forward_length", number(m, a + ".feed_forward_length", -1)),
                    a.equals("qwen2moe") ? false : !Boolean.FALSE.equals(m.get(a + ".expert_weights_norm")),
                    (int) number(m, a + ".expert_shared_feed_forward_length", 0));
        }
        Extras extras = new Extras(a.equals("qwen3") || a.equals("qwen3moe") || a.equals("gemma3"), window,
                window > 0 ? every : 0,
                a.equals("gemma3") ? number(m, a + ".rope.freq_base_swa", 10000) : 0,
                gemma ? ActivationOp.GELU_TANH : ActivationOp.SILU, gemma, gemma23, gemma,
                (float) number(m, a + ".attn_logit_softcapping", 0), (float) number(m, a + ".final_logit_softcapping", 0),
                queryScale, gguf.contains("blk.0.attn_qkv.weight"), moe);
        String type = switch (a) {
            case "gemma3" -> "gemma3_text";
            case "qwen2moe" -> "qwen2_moe";
            case "qwen3moe" -> "qwen3_moe";
            case "llama" -> moe.present() ? "mixtral" : a;
            default -> a;
        };
        return new LlamaConfig(type, vocab, hidden, (int) number(m, a + ".feed_forward_length",
                moe.present() ? moe.ffn() : -1), layers, heads,
                (int) number(m, a + ".attention.head_count_kv", heads), headDim,
                (float) number(m, a + ".attention.layer_norm_rms_epsilon", 1e-6), rope,
                !gguf.contains("output.weight"), a.equals("qwen2") || gguf.contains("blk.0.attn_q.bias"), context,
                extras);
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
        c.put("architectures", List.of(switch (architecture) {
            case "mistral" -> "MistralForCausalLM";
            case "mixtral" -> "MixtralForCausalLM";
            case "qwen2_moe" -> "Qwen2MoeForCausalLM";
            case "qwen3_moe" -> "Qwen3MoeForCausalLM";
            case "qwen2" -> "Qwen2ForCausalLM";
            case "qwen3" -> "Qwen3ForCausalLM";
            case "phi3" -> "Phi3ForCausalLM";
            case "gemma" -> "GemmaForCausalLM";
            case "gemma2" -> "Gemma2ForCausalLM";
            case "gemma3_text" -> "Gemma3ForCausalLM";
            default -> "LlamaForCausalLM";
        }));
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
        if (rope.dim() != headDim) {
            c.put("partial_rotary_factor", (double) rope.dim() / headDim);
        }
        if (rope.type().equals("longrope")) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("rope_type", "longrope");
            List<Object> f = new ArrayList<>();
            for (float x : rope.freqFactors()) {
                f.add((double) x);
            }
            s.put("short_factor", f);
            s.put("long_factor", f);
            s.put("factor", rope.factor());
            s.put("original_max_position_embeddings", (long) rope.originalMaxPositions());
            c.put("rope_scaling", s);
        } else if (!rope.type().equals("default")) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("rope_type", rope.type());
            s.put("factor", rope.factor());
            s.put("low_freq_factor", rope.lowFreqFactor());
            s.put("high_freq_factor", rope.highFreqFactor());
            s.put("original_max_position_embeddings", (long) rope.originalMaxPositions());
            if (rope.type().equals("yarn")) {
                s.put("beta_fast", rope.betaFast());
                s.put("beta_slow", rope.betaSlow());
                if (rope.attentionFactor() > 0) {
                    s.put("attention_factor", rope.attentionFactor());
                }
            }
            c.put("rope_scaling", s);
        }
        c.put("tie_word_embeddings", tieEmbeddings);
        c.put("attention_bias", attentionBias);
        c.put("max_position_embeddings", (long) maxPositions);
        Extras e = extras;
        String act = e.activation() == ActivationOp.GELU_TANH ? "gelu_pytorch_tanh"
                : e.activation() == ActivationOp.GELU ? "gelu" : "silu";
        c.put(gemma(architecture) ? "hidden_activation" : "hidden_act", act);
        if (e.slidingWindow() > 0) {
            c.put("sliding_window", (long) e.slidingWindow());
            c.put("use_sliding_window", true);
            List<Object> types = new ArrayList<>();
            for (int i = 0; i < layers; i++) {
                types.add(e.local(i) ? "sliding_attention" : "full_attention");
            }
            c.put("layer_types", types);
            if (e.globalEvery() > 0) {
                c.put("sliding_window_pattern", (long) e.globalEvery());
            }
        }
        Moe moe = e.moe();
        if (moe.present()) {
            c.put(architecture.equals("mixtral") ? "num_local_experts" : "num_experts", (long) moe.experts());
            c.put("num_experts_per_tok", (long) moe.perToken());
            if (!architecture.equals("mixtral")) {
                c.put("moe_intermediate_size", (long) moe.ffn());
                c.put("norm_topk_prob", moe.normalize());
                c.put("decoder_sparse_step", 1L);
            }
            if (moe.sharedFfn() > 0) {
                c.put("shared_expert_intermediate_size", (long) moe.sharedFfn());
            }
        }
        if (e.localTheta() > 0) {
            c.put("rope_local_base_freq", e.localTheta());
        }
        if (e.queryScale() > 0) {
            c.put("query_pre_attn_scalar", 1 / ((double) e.queryScale() * e.queryScale()));
        } else if (gemma(architecture)) {
            c.put("query_pre_attn_scalar", (long) headDim);
        }
        if (gemma(architecture)) {
            c.put("attn_logit_softcapping", e.attentionSoftcap() > 0 ? (Object) (double) e.attentionSoftcap() : null);
            c.put("final_logit_softcapping", e.finalSoftcap() > 0 ? (Object) (double) e.finalSoftcap() : null);
        }
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
