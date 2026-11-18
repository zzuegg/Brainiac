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
     * @param mla DeepSeek's multi-head latent attention in place of grouped-query attention, or
     *     {@link Mla#NONE}
     * @param attentionSinks whether each head has a learned sink logit that joins every
     *     softmax over keys without a value, so a query can attend to nothing (gpt-oss)
     */
    public record Extras(boolean qkNorm, int slidingWindow, int globalEvery, double localTheta,
            ActivationOp activation, boolean scaleEmbeddings, boolean postNorms, boolean normOffset,
            float attentionSoftcap, float finalSoftcap, float queryScale, boolean fused, Moe moe, Mla mla,
            boolean attentionSinks) {

        /** Plain Llama. */
        public static final Extras NONE = new Extras(false, 0, 0, 0, ActivationOp.SILU, false, false, false, 0, 0, 0,
                false, Moe.NONE, Mla.NONE, false);

        /** Checks the experts and the latent attention are there. */
        public Extras {
            java.util.Objects.requireNonNull(moe, "moe");
            java.util.Objects.requireNonNull(mla, "mla");
        }

        /**
         * The extras without attention sinks.
         *
         * @param qkNorm whether queries and keys are RMS-normalized per head
         * @param slidingWindow the window of local layers, or 0
         * @param globalEvery with a window, every this many layers one sees everything
         * @param localTheta the rotary base of local layers, or 0
         * @param activation the gate's activation
         * @param scaleEmbeddings whether embeddings are multiplied by {@code sqrt(hidden)}
         * @param postNorms whether sublayer outputs are normalized again
         * @param normOffset whether files store norm weights less one
         * @param attentionSoftcap the attention score cap, or 0
         * @param finalSoftcap the logit cap, or 0
         * @param queryScale the attention score scale, or 0 for the default
         * @param fused whether q, k and v (and gate and up) are joined in the weights
         * @param moe the mixture of experts, or {@link Moe#NONE}
         * @param mla the latent attention, or {@link Mla#NONE}
         */
        public Extras(boolean qkNorm, int slidingWindow, int globalEvery, double localTheta,
                ActivationOp activation, boolean scaleEmbeddings, boolean postNorms, boolean normOffset,
                float attentionSoftcap, float finalSoftcap, float queryScale, boolean fused, Moe moe, Mla mla) {
            this(qkNorm, slidingWindow, globalEvery, localTheta, activation, scaleEmbeddings, postNorms, normOffset,
                    attentionSoftcap, finalSoftcap, queryScale, fused, moe, mla, false);
        }

        /**
         * The extras without latent attention, as the families before DeepSeek had them.
         *
         * @param qkNorm whether queries and keys are RMS-normalized per head
         * @param slidingWindow the window of local layers, or 0
         * @param globalEvery with a window, every this many layers one sees everything
         * @param localTheta the rotary base of local layers, or 0
         * @param activation the gate's activation
         * @param scaleEmbeddings whether embeddings are multiplied by {@code sqrt(hidden)}
         * @param postNorms whether sublayer outputs are normalized again
         * @param normOffset whether files store norm weights less one
         * @param attentionSoftcap the attention score cap, or 0
         * @param finalSoftcap the logit cap, or 0
         * @param queryScale the attention score scale, or 0 for the default
         * @param fused whether q, k and v (and gate and up) are joined in the weights
         * @param moe the mixture of experts, or {@link Moe#NONE}
         */
        public Extras(boolean qkNorm, int slidingWindow, int globalEvery, double localTheta,
                ActivationOp activation, boolean scaleEmbeddings, boolean postNorms, boolean normOffset,
                float attentionSoftcap, float finalSoftcap, float queryScale, boolean fused, Moe moe) {
            this(qkNorm, slidingWindow, globalEvery, localTheta, activation, scaleEmbeddings, postNorms, normOffset,
                    attentionSoftcap, finalSoftcap, queryScale, fused, moe, Mla.NONE, false);
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
     * A sparse mixture of experts (Mixtral, Qwen 2 and 3 MoE, DeepSeek): a router picks
     * {@code perToken} of {@code experts} gated feed-forward networks for each token, whose
     * outputs are summed by the router's weights.
     *
     * @param experts the experts per block, or 0 for a dense feed-forward network
     * @param perToken the experts each token uses
     * @param ffn each expert's hidden size
     * @param normalize whether the chosen experts' weights are scaled to sum to one (Mixtral,
     *     Qwen 3, DeepSeek-V3) or kept as the router gave them (Qwen 2, DeepSeek-V2)
     * @param sharedFfn the hidden size of the expert every token uses (DeepSeek: all its shared
     *     experts as one), or 0 for none
     * @param routing how the router scores and picks experts
     * @param swigluLimit for gpt-oss's clamped SwiGLU, the limit {@code l}: an expert computes
     *     {@code (clamp(up, -l, l) + 1) * g * sigmoid(1.702 g)} with {@code g = min(gate, l)};
     *     0 for the family's plain gated activation
     */
    public record Moe(int experts, int perToken, int ffn, boolean normalize, int sharedFfn, Routing routing,
            float swigluLimit) {

        /** No experts. */
        public static final Moe NONE = new Moe(0, 0, 0, false, 0);

        /** Checks the sizes. */
        public Moe {
            java.util.Objects.requireNonNull(routing, "routing");
            if (experts < 0 || experts > 0 && (perToken < 1 || perToken > experts || ffn < 1)) {
                throw new IllegalArgumentException("bad mixture of experts: " + perToken + " of " + experts
                        + " experts of " + ffn);
            }
            if (experts > 0 && (routing.groups() < 1 || experts % routing.groups() != 0 || routing.topGroups() < 1
                    || routing.topGroups() > routing.groups()
                    || routing.topGroups() * (experts / routing.groups()) < perToken)) {
                throw new IllegalArgumentException("bad expert groups: " + routing.topGroups() + " of "
                        + routing.groups() + " groups of " + experts + " experts, " + perToken + " per token");
            }
        }

        /**
         * Experts with the family's plain gated activation.
         *
         * @param experts the experts per block, or 0
         * @param perToken the experts each token uses
         * @param ffn each expert's hidden size
         * @param normalize whether the chosen weights are scaled to sum to one
         * @param sharedFfn the hidden size of the shared expert, or 0
         * @param routing how the router scores and picks experts
         */
        public Moe(int experts, int perToken, int ffn, boolean normalize, int sharedFfn, Routing routing) {
            this(experts, perToken, ffn, normalize, sharedFfn, routing, 0);
        }

        /**
         * Experts routed by a softmax, as Mixtral and Qwen route them.
         *
         * @param experts the experts per block, or 0
         * @param perToken the experts each token uses
         * @param ffn each expert's hidden size
         * @param normalize whether the chosen weights are scaled to sum to one
         * @param sharedFfn the hidden size of a shared expert gated by a sigmoid (Qwen 2), or 0
         */
        public Moe(int experts, int perToken, int ffn, boolean normalize, int sharedFfn) {
            this(experts, perToken, ffn, normalize, sharedFfn, Routing.SOFTMAX, 0);
        }

        /** @return whether there are experts */
        public boolean present() {
            return experts > 0;
        }

        /**
         * @param layer a block
         * @return whether it has experts rather than a dense feed-forward network
         */
        public boolean in(int layer) {
            return experts > 0 && layer >= routing.denseLayers();
        }
    }

    /**
     * How a router picks experts. Mixtral and Qwen take the largest softmax weights. DeepSeek
     * adds groups: the experts are split in {@code groups} groups and a token picks only among
     * those of its {@code topGroups} best groups (V2 ranks a group by its best softmax weight,
     * V3 by the sum of its two best biased sigmoid scores); V3 also picks by the sigmoid scores
     * plus a per-expert bias learned for balance, weighting by the scores without it.
     *
     * @param sigmoid whether experts are scored by a sigmoid each (DeepSeek-V3, Kimi K2) rather
     *     than a softmax over all
     * @param groups the expert groups, 1 for none
     * @param topGroups the groups each token picks from
     * @param scale the factor on the chosen weights (after normalizing, if they are)
     * @param denseLayers the leading blocks with a dense feed-forward network instead
     * @param sharedGated whether the shared expert is gated by a sigmoid of its own (Qwen 2)
     *     or always added (DeepSeek)
     */
    public record Routing(boolean sigmoid, int groups, int topGroups, float scale, int denseLayers,
            boolean sharedGated) {

        /** Mixtral's and Qwen's: softmax, no groups, every block. */
        public static final Routing SOFTMAX = new Routing(false, 1, 1, 1, 0, true);
    }

    /**
     * DeepSeek's multi-head latent attention (DeepSeek-V2, V3, Kimi K2): keys and values come
     * from one small latent vector per token, which is all the cache keeps, plus a few rotary
     * features every head shares. Queries come from the hidden state, through a latent of their
     * own when {@code qRank > 0}. The configuration's {@code headDim} is then {@code nope + rope},
     * the size of a query and a key; values have {@code valueDim} features per head.
     *
     * @param qRank the query latent's size, or 0 for a direct projection (V2-Lite)
     * @param kvRank the key and value latent's size, 0 for no latent attention
     * @param nope the features of each query and key head that are not rotated
     * @param rope the rotated features, the configuration's rotary dimension
     * @param valueDim the features of each value head
     * @param interleaved whether the rotated features pair up as neighbours ({@code x0, x1}),
     *     as DeepSeek's weights hold them, rather than as halves
     */
    public record Mla(int qRank, int kvRank, int nope, int rope, int valueDim, boolean interleaved) {

        /** No latent attention. */
        public static final Mla NONE = new Mla(0, 0, 0, 0, 0, false);

        /** @return whether the model has latent attention */
        public boolean present() {
            return kvRank > 0;
        }

        /** @return the features of a cached step: the latent and the shared rotary key */
        public int cached() {
            return kvRank + rope;
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
            "phi3", "gemma", "gemma2", "gemma3_text", "mixtral", "qwen2_moe", "qwen3_moe", "deepseek_v2", "deepseek_v3",
            "kimi_k2");

    /** The GGUF {@code general.architecture} values {@link #fromGguf} reads. */
    public static final java.util.Set<String> GGUF_ARCHITECTURES = java.util.Set.of("llama", "qwen2", "qwen3", "phi3",
            "gemma", "gemma2", "gemma3", "qwen2moe", "qwen3moe", "deepseek2",
            "gpt-oss");

    private static boolean deepseek(String type) {
        return type.equals("deepseek_v2") || type.equals("deepseek_v3") || type.equals("kimi_k2");
    }

    /** YaRN's attention scaling for a context {@code scale} times the original. */
    private static double yarnMscale(double scale, double mscale) {
        return scale <= 1 ? 1 : 0.1 * mscale * Math.log(scale) + 1;
    }

    /**
     * Reads a Hugging Face {@code config.json} (transformers 4 or 5 layout) of Llama, Mistral,
     * Qwen 2 or 3, Phi-3 or Gemma 1, 2 or 3 ({@code gemma3_text}).
     *
     * @param config the parsed file
     * @return the configuration
     */
    public static LlamaConfig fromHuggingFace(Map<String, Object> config) {
        String type = Json.string(config, "model_type", "llama");
        if (deepseek(type)) {
            return deepseekFromHuggingFace(config, type);
        }
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
        if (ropeType.equals("mrope")) {
            // Qwen2-VL's older spelling: default frequencies, the sections are read by the VLM
            ropeType = "default";
        }
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

    /**
     * DeepSeek-V2 and V3 and Kimi K2 ({@code DeepseekV2ForCausalLM}, {@code DeepseekV3ForCausalLM}):
     * latent attention, experts after {@code first_k_dense_replace} dense blocks, YaRN.
     */
    private static LlamaConfig deepseekFromHuggingFace(Map<String, Object> config, String type) {
        boolean v2 = type.equals("deepseek_v2");
        int hidden = (int) Json.integer(config, "hidden_size");
        int heads = (int) Json.integer(config, "num_attention_heads");
        int layers = (int) Json.integer(config, "num_hidden_layers");
        int nope = (int) Json.integer(config, "qk_nope_head_dim");
        int ropeDim = (int) Json.integer(config, "qk_rope_head_dim");
        int qRank = config.get("q_lora_rank") instanceof Number n ? n.intValue() : 0;
        Mla mla = new Mla(qRank, (int) Json.integer(config, "kv_lora_rank"), nope, ropeDim,
                (int) Json.integer(config, "v_head_dim"), v2 || !Boolean.FALSE.equals(config.get("rope_interleave")));
        int maxPositions = (int) Json.integer(config, "max_position_embeddings", 4096);
        Map<String, Object> ropeNode = config.get("rope_parameters") instanceof Map<?, ?> m
                ? Json.object(m) : config.get("rope_scaling") instanceof Map<?, ?> m2 ? Json.object(m2) : Map.of();
        double theta = Json.number(ropeNode, "rope_theta", Json.number(config, "rope_theta", 10000));
        String ropeType = Json.string(ropeNode, "rope_type", Json.string(ropeNode, "type", "default"));
        double factor = Json.number(ropeNode, "factor", 1);
        int original = (int) Json.integer(ropeNode, "original_max_position_embeddings", maxPositions);
        // as DeepSeek: with mscale and mscale_all_dim, cos and sin are scaled by their ratio
        // (1 when equal, as in every release) and the scores by mscale_all_dim's squared
        double mscale = Json.number(ropeNode, "mscale", 0);
        double mscaleAll = Json.number(ropeNode, "mscale_all_dim", 0);
        double attentionFactor = Json.number(ropeNode, "attention_factor", 0);
        if (attentionFactor == 0 && mscale != 0 && mscaleAll != 0) {
            attentionFactor = yarnMscale(factor, mscale) / yarnMscale(factor, mscaleAll);
        }
        double queryScale = 1 / Math.sqrt(nope + ropeDim);
        if (!ropeType.equals("default") && mscaleAll != 0) {
            double s = yarnMscale(factor, mscaleAll);
            queryScale *= s * s;
        }
        Rope.Settings rope = new Rope.Settings(ropeDim, theta, ropeType, factor, 1, 4, original, null)
                .withYarn(Json.number(ropeNode, "beta_fast", 32), Json.number(ropeNode, "beta_slow", 1),
                        attentionFactor);
        Moe moe = Moe.NONE;
        if (config.get("n_routed_experts") instanceof Number experts && experts.intValue() > 0) {
            if (Json.integer(config, "moe_layer_freq", 1) != 1) {
                throw new IllegalArgumentException("experts in every " + config.get("moe_layer_freq")
                        + "th block only are not supported");
            }
            String method = Json.string(config, "topk_method", v2 ? "greedy" : "noaux_tc");
            boolean sigmoid = Json.string(config, "scoring_func", v2 ? "softmax" : "sigmoid").equals("sigmoid");
            if (!List.of("greedy", "group_limited_greedy", "noaux_tc").contains(method)) {
                throw new IllegalArgumentException("topk_method " + method + " is not supported");
            }
            boolean grouped = !method.equals("greedy") && config.get("n_group") instanceof Number;
            int groups = grouped ? (int) Json.integer(config, "n_group") : 1;
            int topGroups = grouped ? (int) Json.integer(config, "topk_group", groups) : 1;
            int ffn = (int) Json.integer(config, "moe_intermediate_size");
            moe = new Moe(experts.intValue(), (int) Json.integer(config, "num_experts_per_tok"), ffn,
                    Boolean.TRUE.equals(config.get("norm_topk_prob")),
                    ffn * (int) Json.integer(config, "n_shared_experts", 0),
                    new Routing(sigmoid, groups, topGroups, (float) Json.number(config, "routed_scaling_factor", 1),
                            Math.min(layers, (int) Json.integer(config, "first_k_dense_replace", 0)), false));
        }
        Extras extras = new Extras(false, 0, 0, 0, ActivationOp.SILU, false, false, false, 0, 0, (float) queryScale,
                false, moe, mla);
        return new LlamaConfig(type, (int) Json.integer(config, "vocab_size"), hidden,
                (int) Json.integer(config, "intermediate_size"), layers, heads, heads, nope + ropeDim,
                (float) Json.number(config, "rms_norm_eps", 1e-6), rope,
                Boolean.TRUE.equals(config.get("tie_word_embeddings")), false, maxPositions, extras);
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
        if (a.equals("deepseek2")) {
            return deepseekFromGguf(gguf, m);
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
        boolean oss = a.equals("gpt-oss");
        // gpt-oss: layer 0 sees the window, layer 1 everything, and so on
        int every = a.equals("gemma2") || oss ? 2 : a.equals("gemma3") ? 6 : 0;
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
                    // gpt-oss: a softmax over the chosen experts' logits, the same as normalizing
                    a.equals("qwen2moe") ? false : oss || !Boolean.FALSE.equals(m.get(a + ".expert_weights_norm")),
                    (int) number(m, a + ".expert_shared_feed_forward_length", 0), Routing.SOFTMAX, oss ? 7f : 0f);
        }
        Extras extras = new Extras(a.equals("qwen3") || a.equals("qwen3moe") || a.equals("gemma3"), window,
                window > 0 ? every : 0,
                a.equals("gemma3") ? number(m, a + ".rope.freq_base_swa", 10000) : 0,
                gemma ? ActivationOp.GELU_TANH : ActivationOp.SILU, gemma, gemma23, gemma,
                (float) number(m, a + ".attn_logit_softcapping", 0), (float) number(m, a + ".final_logit_softcapping", 0),
                queryScale, gguf.contains("blk.0.attn_qkv.weight"), moe, Mla.NONE,
                gguf.contains("blk.0.attn_sinks.weight") || gguf.contains("blk.0.attn_sinks"));
        String type = switch (a) {
            case "gpt-oss" -> "gpt_oss";
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

    /**
     * llama.cpp's {@code deepseek2} (DeepSeek-V2, V3, R1, Kimi K2). Newer files keep the query
     * and key sizes of the latent attention under {@code key_length_mla} and {@code
     * value_length_mla}, the plain {@code key_length} then being the latent's.
     */
    private static LlamaConfig deepseekFromGguf(Gguf gguf, Map<String, Object> m) {
        String a = "deepseek2";
        int hidden = (int) number(m, a + ".embedding_length", -1);
        int heads = (int) number(m, a + ".attention.head_count", -1);
        int layers = (int) number(m, a + ".block_count", -1);
        int ropeDim = (int) number(m, a + ".rope.dimension_count", -1);
        int kvRank = (int) number(m, a + ".attention.kv_lora_rank", -1);
        boolean split = m.containsKey(a + ".attention.key_length_mla");
        int qk = (int) number(m, a + (split ? ".attention.key_length_mla" : ".attention.key_length"), -1);
        int valueDim = (int) number(m, a + (split ? ".attention.value_length_mla" : ".attention.value_length"), -1);
        Mla mla = new Mla((int) number(m, a + ".attention.q_lora_rank", 0), kvRank, qk - ropeDim, ropeDim, valueDim,
                true);
        int vocab = m.get(a + ".vocab_size") instanceof Number v ? v.intValue()
                : m.get("tokenizer.ggml.tokens") instanceof List<?> t ? t.size()
                : gguf.shape("token_embd.weight").dim(0);
        int context = (int) number(m, a + ".context_length", 4096);
        String scaling = m.get(a + ".rope.scaling.type") instanceof String s ? s : "default";
        double factor = number(m, a + ".rope.scaling.factor", 1);
        double queryScale = 1 / Math.sqrt(qk);
        double logMul = number(m, a + ".rope.scaling.yarn_log_multiplier", 0);
        if (scaling.equals("yarn") && factor > 1 && logMul > 0) {
            // 0.1 mscale_all_dim, as llama.cpp's converter writes it for DeepSeek
            double s = 1 + logMul * Math.log(factor);
            queryScale *= s * s;
        }
        Rope.Settings rope = new Rope.Settings(ropeDim, number(m, a + ".rope.freq_base", 10000),
                scaling.equals("none") ? "default" : scaling, factor, 1, 4,
                (int) number(m, a + ".rope.scaling.original_context_length", context), null)
                .withYarn(number(m, a + ".rope.scaling.yarn_beta_fast", 32),
                        number(m, a + ".rope.scaling.yarn_beta_slow", 1), scaling.equals("yarn") ? 1 : 0);
        Moe moe = Moe.NONE;
        int dense = (int) number(m, a + ".leading_dense_block_count", 0);
        if (m.get(a + ".expert_count") instanceof Number count && count.intValue() > 0 && dense < layers) {
            int ffn = (int) number(m, a + ".expert_feed_forward_length", -1);
            int gating = (int) number(m, a + ".expert_gating_func", 1);
            if (gating != 1 && gating != 2) {
                throw new IllegalArgumentException("expert gating function " + gating + " is not supported");
            }
            int groups = (int) number(m, a + ".expert_group_count", 1);
            moe = new Moe(count.intValue(), (int) number(m, a + ".expert_used_count", -1), ffn,
                    Boolean.TRUE.equals(m.get(a + ".expert_weights_norm")),
                    ffn * (int) number(m, a + ".expert_shared_count", 0),
                    new Routing(gating == 2, Math.max(groups, 1),
                            Math.max(1, (int) number(m, a + ".expert_group_used_count", groups)),
                            (float) number(m, a + ".expert_weights_scale", 1), dense, false));
        }
        Extras extras = new Extras(false, 0, 0, 0, ActivationOp.SILU, false, false, false, 0, 0, (float) queryScale,
                false, moe, mla);
        return new LlamaConfig("deepseek_v3", vocab, hidden, (int) number(m, a + ".feed_forward_length", -1), layers,
                heads, heads, qk, (float) number(m, a + ".attention.layer_norm_rms_epsilon", 1e-6), rope,
                !gguf.contains("output.weight"), false, context, extras);
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
        if (extras.mla().present()) {
            return deepseekToHuggingFace();
        }
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

    /** DeepSeek-V2 and V3's {@code config.json}: V2 when the experts are routed by a softmax. */
    private Map<String, Object> deepseekToHuggingFace() {
        Mla mla = extras.mla();
        Moe moe = extras.moe();
        boolean v2 = architecture.equals("deepseek_v2") || moe.present() && !moe.routing().sigmoid();
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("architectures", List.of(v2 ? "DeepseekV2ForCausalLM" : "DeepseekV3ForCausalLM"));
        c.put("model_type", v2 ? "deepseek_v2" : architecture.equals("kimi_k2") ? "kimi_k2" : "deepseek_v3");
        c.put("vocab_size", (long) vocabSize);
        c.put("hidden_size", (long) hidden);
        c.put("intermediate_size", (long) intermediate);
        c.put("num_hidden_layers", (long) layers);
        c.put("num_attention_heads", (long) heads);
        c.put("num_key_value_heads", (long) heads);
        c.put("q_lora_rank", mla.qRank() > 0 ? (Object) (long) mla.qRank() : null);
        c.put("kv_lora_rank", (long) mla.kvRank());
        c.put("qk_nope_head_dim", (long) mla.nope());
        c.put("qk_rope_head_dim", (long) mla.rope());
        c.put("v_head_dim", (long) mla.valueDim());
        c.put("rms_norm_eps", (double) rmsEps);
        c.put("max_position_embeddings", (long) maxPositions);
        c.put("tie_word_embeddings", tieEmbeddings);
        c.put("hidden_act", "silu");
        c.put("attention_bias", false);
        if (!v2) {
            c.put("rope_interleave", mla.interleaved());
        }
        Map<String, Object> r = new LinkedHashMap<>();
        r.put("rope_type", rope.type());
        r.put("rope_theta", rope.theta());
        if (!rope.type().equals("default")) {
            r.put("factor", rope.factor());
            r.put("original_max_position_embeddings", (long) rope.originalMaxPositions());
            r.put("beta_fast", rope.betaFast());
            r.put("beta_slow", rope.betaSlow());
            // the score scale as mscale_all_dim, and cos and sin as its ratio to mscale
            double base = 1 / Math.sqrt(headDim);
            double s = Math.sqrt(extras.queryScale() / base);
            double all = rope.factor() > 1 ? (s - 1) / (0.1 * Math.log(rope.factor())) : 0;
            double magnitude = rope.attentionFactor() > 0 ? rope.attentionFactor() : 1;
            r.put("mscale_all_dim", all);
            r.put("mscale", rope.factor() > 1 ? ((magnitude * s) - 1) / (0.1 * Math.log(rope.factor())) : all);
        }
        c.put("rope_parameters", r);
        if (moe.present()) {
            Routing routing = moe.routing();
            c.put("n_routed_experts", (long) moe.experts());
            c.put("num_experts_per_tok", (long) moe.perToken());
            c.put("moe_intermediate_size", (long) moe.ffn());
            c.put("n_shared_experts", (long) (moe.sharedFfn() / moe.ffn()));
            c.put("first_k_dense_replace", (long) routing.denseLayers());
            c.put("norm_topk_prob", moe.normalize());
            c.put("routed_scaling_factor", (double) routing.scale());
            c.put("n_group", (long) routing.groups());
            c.put("topk_group", (long) routing.topGroups());
            c.put("scoring_func", routing.sigmoid() ? "sigmoid" : "softmax");
            c.put("topk_method", routing.sigmoid() ? "noaux_tc" : routing.groups() > 1 ? "group_limited_greedy"
                    : "greedy");
            c.put("moe_layer_freq", 1L);
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
