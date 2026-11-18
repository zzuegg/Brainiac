package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.GgufWriter;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A Llama-family decoder: token embeddings, then blocks of RMSNorm, grouped-query
 * self-attention with rotary positions and a SwiGLU feed-forward network, each added to a
 * residual stream, then a final RMSNorm and the output projection.
 *
 * <p>It loads published weights: a Hugging Face model directory ({@code config.json} and
 * safetensors, Llama 2 and 3, Mistral, Qwen 2 and their fine-tunes) or a GGUF file, where
 * {@code Q8_0} and {@code Q4_0} matrices stay packed. Its outputs match transformers'.
 * Weights loaded as tensors are trainable {@link Var}s, named as in the Hugging Face files.
 *
 * <pre>{@code
 * Llama model = Llama.load(Path.of("Qwen2.5-0.5B-Instruct"), Device.gpu(), DType.BFLOAT16);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("Qwen2.5-0.5B-Instruct"));
 * String reply = Generator.of(model, tok).generate(tok.applyChatTemplate(messages, true),
 *         Sampling.defaults().temperature(0.7f).maxTokens(200));
 * }</pre>
 */
public final class Llama implements LanguageModel {

    /**
     * A projection {@code x W + b} with {@code W} stored {@code [in, out]}: transposed once
     * from Hugging Face's {@code [out, in]} when loading, so no step copies it.
     */
    private static final class Linear {
        private Var weight;
        private Var bias;
        /** Low-rank adapter {@code [in, rank]} and {@code [rank, out]}, or null. */
        private Var loraA;
        private Var loraB;
        private float loraScale;
        /** Whether the weight and bias are constants: only the adapter learns. */
        private boolean frozen;

        Linear(Var weight, Var bias) {
            this.weight = weight;
            this.bias = bias;
        }

        Var weight() {
            return weight;
        }

        Var base() {
            return frozen ? Var.of(weight.value()) : weight;
        }

        Var bias() {
            return bias;
        }

        Var apply(Var x) {
            Var y = x.matmul(frozen ? Var.of(weight.value()) : weight);
            if (bias != null) {
                y = y.add(frozen ? Var.of(bias.value()) : bias);
            }
            if (loraA != null) {
                y = y.add(x.matmul(loraA).matmul(loraB).mul(loraScale));
            }
            return y;
        }
    }

    /** Hugging Face's {@code [out, in]} matrix as {@code [in, out]}, freeing the original. */
    private static Tensor inOut(Tensor outIn) {
        Tensor t = outIn.transpose();
        Tensor.evaluate(List.of(t));
        outIn.release();
        return t;
    }

    /**
     * A block's weights. Q, k, v, gate and up are null where the model keeps them joined
     * ({@link Fused}); the q/k and post norms are null where the family has none.
     */
    private record Block(Var attnNorm, Linear q, Linear k, Linear v, Linear o, Var ffnNorm, Linear gate, Linear up,
            Linear down, Var qNorm, Var kNorm, Var postAttnNorm, Var postFfnNorm, Experts experts) {
    }

    /**
     * A block's mixture of experts: the router, each expert's gate, up and down projections,
     * and an optional shared expert with its sigmoid gate.
     */
    private record Experts(Linear router, Linear[] gate, Linear[] up, Linear[] down, Linear sharedGate,
            Linear sharedUp, Linear sharedDown, Linear sharedScale) {
    }

    /** The module prefix of a block's experts, and its names for gate, up and down. */
    private static String[] expertNames(LlamaConfig c) {
        return c.architecture().equals("mixtral")
                ? new String[] {"block_sparse_moe.", "w1", "w3", "w2"}
                : new String[] {"mlp.", "gate_proj", "up_proj", "down_proj"};
    }

    private static Experts experts(LlamaConfig c, Map<String, Linear> linears, String p) {
        LlamaConfig.Moe moe = c.extras().moe();
        if (!moe.present()) {
            return null;
        }
        String[] n = expertNames(c);
        Linear[] gate = new Linear[moe.experts()];
        Linear[] up = new Linear[moe.experts()];
        Linear[] down = new Linear[moe.experts()];
        for (int e = 0; e < moe.experts(); e++) {
            String x = p + n[0] + "experts." + e + ".";
            gate[e] = linears.get(x + n[1]);
            up[e] = linears.get(x + n[2]);
            down[e] = linears.get(x + n[3]);
        }
        boolean shared = moe.sharedFfn() > 0;
        return new Experts(linears.get(p + n[0] + "gate"), gate, up, down,
                shared ? linears.get(p + "mlp.shared_expert.gate_proj") : null,
                shared ? linears.get(p + "mlp.shared_expert.up_proj") : null,
                shared ? linears.get(p + "mlp.shared_expert.down_proj") : null,
                shared ? linears.get(p + "mlp.shared_expert_gate") : null);
    }

    private final LlamaConfig config;
    private final Device device;
    private final DType dtype;
    private final Rope rope;
    /** The rotation of local (sliding-window) layers when it differs, or null. */
    private final Rope localRope;
    private final Var embed;
    private final Linear lmHead;
    private Var norm;
    private final List<Block> blocks;
    /** Per block, its device and type after {@link #offload}; null while all are on {@code device}. */
    private Device[] layerDevice;
    private DType[] layerType;
    /** Where the final norm and output layer run, and their type. */
    private Device headDevice;
    private DType headType;
    /** A tied output layer's copy of the embeddings on {@code headDevice}, or null. */
    private Var tiedOnHead;
    private final Map<String, Var> named;
    private final Map<String, Linear> linears;
    /** Whether the embeddings and norms are constants (see {@link #addLora}). */
    private boolean frozen;
    /** Per block, the joined projections of {@link #fuseProjections()}, or null. */
    private Fused[] fused;

    /** A block's q, k and v in one product, and gate and up in another; either may be null. */
    private record Fused(Linear qkv, Linear gateUp) {
    }

    private Llama(LlamaConfig config, Device device, DType dtype, Map<String, Var> named, Map<String, Linear> linears) {
        this.config = config;
        this.device = device;
        this.dtype = dtype;
        this.rope = new Rope(config.rope());
        LlamaConfig.Extras e = config.extras();
        this.localRope = e.localTheta() > 0 ? new Rope(new Rope.Settings(config.rope().dim(), e.localTheta(),
                "default", 1, 1, 4, config.rope().originalMaxPositions(), null)) : null;
        this.named = named;
        this.linears = linears;
        this.embed = need(named, "model.embed_tokens.weight");
        this.norm = need(named, "model.norm.weight");
        this.lmHead = config.tieEmbeddings() ? null : linears.get("lm_head");
        this.blocks = new ArrayList<>();
        boolean post = e.postNorms();
        if (e.fused()) {
            fused = new Fused[config.layers()];
        }
        for (int i = 0; i < config.layers(); i++) {
            String p = "model.layers." + i + ".";
            blocks.add(new Block(need(named, p + "input_layernorm.weight"), linears.get(p + "self_attn.q_proj"),
                    linears.get(p + "self_attn.k_proj"), linears.get(p + "self_attn.v_proj"),
                    linears.get(p + "self_attn.o_proj"),
                    need(named, p + (post ? "pre_feedforward_layernorm" : "post_attention_layernorm") + ".weight"),
                    linears.get(p + "mlp.gate_proj"), linears.get(p + "mlp.up_proj"), linears.get(p + "mlp.down_proj"),
                    e.qkNorm() ? need(named, p + "self_attn.q_norm.weight") : null,
                    e.qkNorm() ? need(named, p + "self_attn.k_norm.weight") : null,
                    post ? need(named, p + "post_attention_layernorm.weight") : null,
                    post ? need(named, p + "post_feedforward_layernorm.weight") : null, experts(config, linears, p)));
            if (e.fused()) {
                fused[i] = new Fused(linears.get(p + "self_attn.qkv_proj"), linears.get(p + "mlp.gate_up_proj"));
            }
        }
    }

    /** The norm weights of block {@code i}, by Hugging Face name. */
    private static List<String> normNames(LlamaConfig c, int i) {
        String p = "model.layers." + i + ".";
        List<String> names = new ArrayList<>(List.of(p + "input_layernorm.weight", p + "post_attention_layernorm.weight"));
        if (c.extras().postNorms()) {
            names.add(p + "pre_feedforward_layernorm.weight");
            names.add(p + "post_feedforward_layernorm.weight");
        }
        if (c.extras().qkNorm()) {
            names.add(p + "self_attn.q_norm.weight");
            names.add(p + "self_attn.k_norm.weight");
        }
        return names;
    }

    private static Var need(Map<String, Var> named, String name) {
        Var v = named.get(name);
        if (v == null) {
            throw new IllegalArgumentException("the weights have no " + name);
        }
        return v;
    }

    /** The projections of a model, by Hugging Face module name, with their weight and bias names. */
    private static List<String> linearNames(LlamaConfig c) {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < c.layers(); i++) {
            String p = "model.layers." + i + ".";
            List<String> modules = new ArrayList<>(c.extras().fused()
                    ? List.of("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
                    : List.of("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"));
            LlamaConfig.Moe moe = c.extras().moe();
            if (moe.present()) {
                modules.removeIf(m -> m.startsWith("mlp."));
                String[] n = expertNames(c);
                modules.add(n[0] + "gate");
                for (int e = 0; e < moe.experts(); e++) {
                    for (int k = 1; k <= 3; k++) {
                        modules.add(n[0] + "experts." + e + "." + n[k]);
                    }
                }
                if (moe.sharedFfn() > 0) {
                    modules.addAll(List.of("mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                            "mlp.shared_expert.down_proj", "mlp.shared_expert_gate"));
                }
            }
            for (String s : modules) {
                names.add(p + s);
            }
        }
        if (!c.tieEmbeddings()) {
            names.add("lm_head");
        }
        return names;
    }

    private static boolean hasBias(LlamaConfig c, String linear) {
        return c.attentionBias() && (linear.endsWith("q_proj") || linear.endsWith("k_proj")
                || linear.endsWith("v_proj"));
    }

    // ---------------------------------------------------------------- loading

    /**
     * Loads a model: a Hugging Face model directory ({@code config.json} with safetensors
     * weights) or a {@code .gguf} file.
     *
     * @param path the directory or file
     * @param device where to put the weights
     * @param dtype the type of the weights and activations (packed GGUF matrices stay packed)
     * @return the model
     */
    public static Llama load(Path path, Device device, DType dtype) {
        if (path.getFileName().toString().endsWith(".gguf")) {
            try (Gguf g = Gguf.open(path)) {
                return load(g, device, dtype);
            }
        }
        try {
            LlamaConfig config = LlamaConfig.fromHuggingFace(Json.object(Json.read(path.resolve("config.json"))));
            try (WeightSource w = WeightSource.open(path)) {
                return load(w, config, device, dtype);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Loads a model from weights named as in Hugging Face files.
     *
     * @param weights the weights
     * @param config the model shape
     * @param device where to put them
     * @param dtype their type
     * @return the model
     */
    public static Llama load(WeightSource weights, LlamaConfig config, Device device, DType dtype) {
        Map<String, Var> named = new LinkedHashMap<>();
        Map<String, Linear> linears = new LinkedHashMap<>();
        java.util.function.Function<String, Var> read = n -> {
            Var v = Var.param(weights.tensor(n, device, dtype));
            named.put(n, v);
            return v;
        };
        read.apply("model.embed_tokens.weight");
        List<String> norms = new ArrayList<>(List.of("model.norm.weight"));
        for (int i = 0; i < config.layers(); i++) {
            norms.addAll(normNames(config, i));
        }
        for (String n : norms) {
            Tensor w = weights.tensor(n, device, dtype);
            if (config.extras().normOffset()) {
                w = w.add(1f);   // Gemma stores w and scales by 1 + w
            }
            Var v = Var.param(w);
            named.put(n, v);
        }
        for (String l : linearNames(config)) {
            Var bias = hasBias(config, l) ? read.apply(l + ".bias") : null;
            Var w = Var.param(inOut(weights.tensor(l + ".weight", device, dtype)));
            named.put(l + ".weight", w);
            linears.put(l, new Linear(w, bias));
        }
        return new Llama(config, device, dtype, named, linears);
    }

    /**
     * Loads a model from a GGUF file. {@code Q8_0} and {@code Q4_0} matrices stay packed
     * (and are not trained); every other tensor is decoded into {@code dtype}.
     *
     * @param gguf the opened file
     * @param device where to put the weights
     * @param dtype the type of the other weights and of the activations
     * @return the model
     */
    public static Llama load(Gguf gguf, Device device, DType dtype) {
        LlamaConfig config = LlamaConfig.fromGguf(gguf);
        if (gguf.contains("rope_freqs.weight")) {
            float[] f = gguf.tensor("rope_freqs.weight", Device.cpu(), DType.FLOAT32).toArray();
            config = config.withRope(config.rope().withFreqFactors(f));
        }
        LlamaConfig.Extras extras = config.extras();
        // llama.cpp's converter interleaves the two rotary halves of Llama's query and key rows
        boolean permuted = gguf.architecture().equals("llama");
        Map<String, Var> named = new LinkedHashMap<>();
        Map<String, Linear> linears = new LinkedHashMap<>();
        named.put("model.embed_tokens.weight", Var.param(gguf.tensor("token_embd.weight", device, dtype)));
        named.put("model.norm.weight", Var.param(gguf.tensor("output_norm.weight", device, dtype)));
        Map<String, String> parts = ggufParts(config);
        for (int i = 0; i < config.layers(); i++) {
            String p = "model.layers." + i + ".";
            String g = "blk." + i + ".";
            for (Map.Entry<String, String> n : ggufNorms(config).entrySet()) {
                named.put(p + n.getKey() + ".weight", Var.param(gguf.tensor(g + n.getValue() + ".weight", device, dtype)));
            }
            for (Map.Entry<String, String> e : parts.entrySet()) {
                String l = p + e.getKey();
                int heads = e.getValue().equals("attn_q") ? config.heads()
                        : e.getValue().equals("attn_k") ? config.kvHeads() : 0;
                int[] order = permuted && heads > 0 ? unpermute(heads, config.headDim()) : null;
                Var bias = null;
                if (gguf.contains(g + e.getValue() + ".bias")) {
                    Tensor b = gguf.tensor(g + e.getValue() + ".bias", device, dtype);
                    bias = Var.param(order == null ? b : b.gather(0, indices(order, device)));
                    named.put(l + ".bias", bias);
                }
                linears.put(l, linear(gguf, g + e.getValue() + ".weight", l, order, device, dtype, named, bias));
            }
            LlamaConfig.Moe moe = extras.moe();
            if (moe.present()) {
                String[] n = expertNames(config);
                String[] stored = {"ffn_gate", "ffn_up", "ffn_down"};
                linears.put(p + n[0] + "gate", linear(gguf, g + "ffn_gate_inp.weight", p + n[0] + "gate", null, device,
                        dtype, named, null));
                for (int x = 0; x < moe.experts(); x++) {
                    for (int k = 0; k < 3; k++) {
                        // a stack of all experts, or (older files) one matrix each
                        String stack = g + stored[k] + "_exps.weight";
                        String w = gguf.contains(stack) ? gguf.slice(stack, x) : g + stored[k] + "." + x + ".weight";
                        String hf = p + n[0] + "experts." + x + "." + n[k + 1];
                        linears.put(hf, linear(gguf, w, hf, null, device, dtype, named, null));
                    }
                }
                if (moe.sharedFfn() > 0) {
                    String[] hf = {"mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj",
                            "mlp.shared_expert.down_proj", "mlp.shared_expert_gate"};
                    String[] gg = {"ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp_shexp"};
                    for (int k = 0; k < 4; k++) {
                        Tensor shapeCheck = null;
                        String w = g + gg[k] + ".weight";
                        if (k == 3 && gguf.shape(w).rank() == 1) {
                            // the shared expert's gate stored as a vector: one output
                            shapeCheck = gguf.tensor(w, device, dtype).reshape(1, config.hidden());
                            Var v = Var.param(inOut(shapeCheck));
                            named.put(p + hf[k] + ".weight", v);
                            linears.put(p + hf[k], new Linear(v, null));
                            continue;
                        }
                        linears.put(p + hf[k], linear(gguf, w, p + hf[k], null, device, dtype, named, null));
                    }
                }
            }
        }
        if (!config.tieEmbeddings()) {
            linears.put("lm_head", linear(gguf, "output.weight", "lm_head", null, device, dtype, named, null));
        }
        return new Llama(config, device, dtype, named, linears);
    }

    /** Block projections: Hugging Face module name to llama.cpp's. */
    private static Map<String, String> ggufParts(LlamaConfig c) {
        Map<String, String> parts = new LinkedHashMap<>();
        if (c.extras().fused()) {
            parts.put("self_attn.qkv_proj", "attn_qkv");
        } else {
            parts.put("self_attn.q_proj", "attn_q");
            parts.put("self_attn.k_proj", "attn_k");
            parts.put("self_attn.v_proj", "attn_v");
        }
        parts.put("self_attn.o_proj", "attn_output");
        if (c.extras().fused()) {
            parts.put("mlp.gate_up_proj", "ffn_up");
        } else {
            parts.put("mlp.gate_proj", "ffn_gate");
            parts.put("mlp.up_proj", "ffn_up");
        }
        parts.put("mlp.down_proj", "ffn_down");
        if (c.extras().moe().present()) {
            parts.keySet().removeIf(k -> k.startsWith("mlp."));
        }
        return parts;
    }

    /** Block norms: Hugging Face module name to llama.cpp's. */
    private static Map<String, String> ggufNorms(LlamaConfig c) {
        Map<String, String> norms = new LinkedHashMap<>();
        norms.put("input_layernorm", "attn_norm");
        if (c.extras().postNorms()) {
            norms.put("post_attention_layernorm", "post_attention_norm");
            norms.put("pre_feedforward_layernorm", "ffn_norm");
            norms.put("post_feedforward_layernorm", "post_ffw_norm");
        } else {
            norms.put("post_attention_layernorm", "ffn_norm");
        }
        if (c.extras().qkNorm()) {
            norms.put("self_attn.q_norm", "attn_q_norm");
            norms.put("self_attn.k_norm", "attn_k_norm");
        }
        return norms;
    }

    private static Linear linear(Gguf gguf, String name, String hfName, int[] order, Device device, DType dtype,
            Map<String, Var> named, Var bias) {
        if (gguf.packable(name)) {
            return new Linear(Var.of(gguf.packedTransposed(name, device, order)), bias);
        }
        Tensor w = gguf.tensor(name, device, dtype);
        if (order != null) {
            w = w.gather(0, indices(order, device));
        }
        Var v = Var.param(inOut(w));
        named.put(hfName + ".weight", v);
        return new Linear(v, bias);
    }

    /** For each Hugging Face output row, the row llama.cpp's converter moved it to. */
    private static int[] unpermute(int heads, int headDim) {
        int half = headDim / 2;
        int[] order = new int[heads * headDim];
        for (int h = 0; h < heads; h++) {
            for (int a = 0; a < 2; a++) {
                for (int j = 0; j < half; j++) {
                    order[h * headDim + a * half + j] = h * headDim + j * 2 + a;
                }
            }
        }
        return order;
    }

    private static Tensor indices(int[] ids, Device device) {
        float[] f = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            f[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), f, device);
    }

    /**
     * Creates a model with random weights, for training from scratch.
     *
     * @param config the shape
     * @param seed the random seed
     * @param device where to put the weights
     * @param dtype their type
     * @return the model
     */
    public static Llama random(LlamaConfig config, long seed, Device device, DType dtype) {
        SplittableRandom random = new SplittableRandom(seed);
        Map<String, Var> named = new LinkedHashMap<>();
        Map<String, Linear> linears = new LinkedHashMap<>();
        float std = 0.02f;
        named.put("model.embed_tokens.weight", Var.param(Tensor.normal(Shape.of(config.vocabSize(), config.hidden()),
                0, std, random).to(device).to(dtype)));
        named.put("model.norm.weight", Var.param(Tensor.ones(Shape.of(config.hidden())).to(device).to(dtype)));
        for (int i = 0; i < config.layers(); i++) {
            for (String n : normNames(config, i)) {
                int size = n.contains("self_attn.") ? config.headDim() : config.hidden();
                named.put(n, Var.param(Tensor.ones(Shape.of(size)).to(device).to(dtype)));
            }
        }
        int qOut = config.heads() * config.headDim();
        int kvOut = config.kvHeads() * config.headDim();
        // residual projections scaled down by depth, as GPT-2 does
        float residual = (float) (std / Math.sqrt(2.0 * config.layers()));
        for (String l : linearNames(config)) {
            LlamaConfig.Moe moe = config.extras().moe();
            boolean expert = l.contains(".experts.");
            boolean shared = l.contains(".shared_expert.");
            int ffn = expert ? moe.ffn() : shared ? moe.sharedFfn() : config.intermediate();
            int[] shape = l.endsWith(".gate") ? new int[] {moe.experts(), config.hidden()}
                    : l.endsWith("shared_expert_gate") ? new int[] {1, config.hidden()}
                    : expert || shared ? (l.endsWith("down_proj") || l.endsWith("w2") ? new int[] {config.hidden(), ffn}
                            : new int[] {ffn, config.hidden()})
                    : l.endsWith("qkv_proj") ? new int[] {qOut + 2 * kvOut, config.hidden()}
                    : l.endsWith("gate_up_proj") ? new int[] {2 * config.intermediate(), config.hidden()}
                    : l.endsWith("q_proj") ? new int[] {qOut, config.hidden()}
                    : l.endsWith("k_proj") || l.endsWith("v_proj") ? new int[] {kvOut, config.hidden()}
                    : l.endsWith("o_proj") ? new int[] {config.hidden(), qOut}
                    : l.endsWith("down_proj") ? new int[] {config.hidden(), config.intermediate()}
                    : l.equals("lm_head") ? new int[] {config.vocabSize(), config.hidden()}
                    : new int[] {config.intermediate(), config.hidden()};
            float s = l.endsWith("o_proj") || l.endsWith("down_proj") ? residual : std;
            Var w = Var.param(Tensor.normal(Shape.of(shape[1], shape[0]), 0, s, random).to(device).to(dtype));
            named.put(l + ".weight", w);
            Var b = null;
            if (hasBias(config, l)) {
                b = Var.param(Tensor.zeros(Shape.of(shape[0]), dtype, device));
                named.put(l + ".bias", b);
            }
            linears.put(l, new Linear(w, b));
        }
        return new Llama(config, device, dtype, named, linears);
    }

    // ---------------------------------------------------------------- device split

    /**
     * Moves the last {@code layers} blocks, the final norm and the output layer to another
     * device, as llama.cpp's {@code --n-gpu-layers} does: a model too large for the GPU runs
     * its first blocks on the CPU and the rest on the GPU, the hidden state crossing once per
     * step. Keys and values stay with their block. Packed weights stay packed; other weights
     * take {@code dtype} there. A tied output layer gets its own copy of the embeddings on
     * the device, and the embeddings themselves stay where they are.
     *
     * @param target the device for the last blocks, usually a GPU
     * @param dtype the weights' and activations' type there
     * @param layers how many blocks to move, 0 to all
     * @return this model
     */
    public Llama offload(Device target, DType dtype, int layers) {
        java.util.Objects.requireNonNull(target, "target");
        java.util.Objects.requireNonNull(dtype, "dtype");
        if (layers < 0 || layers > blocks.size()) {
            throw new IllegalArgumentException("can move 0 to " + blocks.size() + " blocks, not " + layers);
        }
        if (layerDevice != null) {
            throw new IllegalStateException("the model is split already");
        }
        layerDevice = new Device[blocks.size()];
        layerType = new DType[blocks.size()];
        java.util.Arrays.fill(layerDevice, device);
        java.util.Arrays.fill(layerType, this.dtype);
        for (int i = blocks.size() - layers; i < blocks.size(); i++) {
            layerDevice[i] = target;
            layerType[i] = dtype;
            String prefix = "model.layers." + i + ".";
            for (Map.Entry<String, Var> e : List.copyOf(named.entrySet())) {
                if (e.getKey().startsWith(prefix) && e.getKey().contains("norm")) {
                    named.put(e.getKey(), moved(e.getValue(), target, dtype));
                }
            }
            for (Map.Entry<String, Linear> e : linears.entrySet()) {
                if (e.getKey().startsWith(prefix)) {
                    move(e.getKey(), e.getValue(), target, dtype);
                }
            }
            Block b = blocks.get(i);
            String post = config.extras().postNorms() ? "pre_feedforward_layernorm" : "post_attention_layernorm";
            blocks.set(i, new Block(named.get(prefix + "input_layernorm.weight"), b.q, b.k, b.v, b.o,
                    named.get(prefix + post + ".weight"), b.gate, b.up, b.down,
                    named.get(prefix + "self_attn.q_norm.weight"), named.get(prefix + "self_attn.k_norm.weight"),
                    config.extras().postNorms() ? named.get(prefix + "post_attention_layernorm.weight") : null,
                    config.extras().postNorms() ? named.get(prefix + "post_feedforward_layernorm.weight") : null,
                    b.experts));
        }
        headDevice = target;
        headType = dtype;
        named.put("model.norm.weight", moved(norm, target, dtype));
        norm = named.get("model.norm.weight");
        if (lmHead != null) {
            move("lm_head", lmHead, target, dtype);
        } else if (!target.equals(device)) {
            tiedOnHead = Var.of(embed.value().to(target).to(dtype));
        }
        return this;
    }

    private static Var moved(Var v, Device target, DType dtype) {
        if (v == null) {
            return null;
        }
        Tensor t = v.value().to(target);
        if (!t.isPacked()) {
            t = t.to(dtype);
        }
        if (t != v.value()) {
            v.value().release();
        }
        return v.requiresGrad() ? Var.param(t) : Var.of(t);
    }

    private void move(String name, Linear l, Device target, DType dtype) {
        boolean listed = named.get(name + ".weight") == l.weight;
        l.weight = moved(l.weight, target, dtype);
        if (listed) {
            named.put(name + ".weight", l.weight);
        }
        if (l.bias != null) {
            l.bias = moved(l.bias, target, dtype);
            named.put(name + ".bias", l.bias);
        }
        if (l.loraA != null) {
            l.loraA = moved(l.loraA, target, dtype);
            l.loraB = moved(l.loraB, target, dtype);
        }
    }

    private Device deviceOf(int layer) {
        return layerDevice == null ? device : layerDevice[layer];
    }

    private DType typeOf(int layer) {
        return layerType == null ? dtype : layerType[layer];
    }

    private Device headDevice() {
        return headDevice == null ? device : headDevice;
    }

    private DType headType() {
        return headType == null ? dtype : headType;
    }

    // ---------------------------------------------------------------- the model

    /** @return the model's shape */
    public LlamaConfig config() {
        return config;
    }

    /** @return where the weights are */
    public Device device() {
        return device;
    }

    /** @return the type of the weights and activations */
    public DType dtype() {
        return dtype;
    }

    @Override
    public int vocabSize() {
        return config.vocabSize();
    }

    /**
     * Returns the trainable weights by their Hugging Face names. Projection matrices are held
     * {@code [in, out]}, the transpose of Hugging Face's layout ({@link #save} writes them
     * back in that layout). Packed GGUF matrices are not among them.
     *
     * @return the weights, in a stable order
     */
    public Map<String, Var> namedParameters() {
        return java.util.Collections.unmodifiableMap(named);
    }

    /** @return the trainable weights */
    public List<Var> parameters() {
        return List.copyOf(named.values());
    }

    /** {@code -Dbrainiac.llm.fusedNorm=false}: RMSNorm from separate operations while decoding too. */
    private static final boolean FUSED_NORM = !"false".equals(System.getProperty("brainiac.llm.fusedNorm"));

    private Var rmsNorm(Var x, Var weight, boolean decoding) {
        if (decoding && FUSED_NORM && x.value().hasRmsNorm()) {
            // no gradients while decoding: one kernel instead of a reduction and its chains
            return Var.of(x.value().rmsNorm(weight.value(), config.rmsEps()));
        }
        weight = base(weight);
        Var x32 = x.to(DType.FLOAT32);
        int last = x.shape().rank() - 1;
        Var y = x32.div(x32.square().mean(last, true).add(config.rmsEps()).sqrt());
        return y.to(x.dtype()).mul(weight);
    }

    /**
     * The keys and values of each block, in buffers that grow by doubling: each step writes
     * only its new keys and values, and attention reads the whole buffer with the unused
     * end masked, so no step copies what earlier steps cached.
     */
    private static final class Cache implements State {
        final Tensor[] keys;
        final Tensor[] values;
        /** Per sequence: whether each cached step is padding. */
        boolean[][] padding;
        /** Per sequence: the position of the next real token. */
        int[] next;
        int length;
        int capacity;
        /** Eviction: the first steps and the latest steps kept, or window 0 to keep all. */
        int sinks;
        int window;
        /** Whether keys and values are kept as 8-bit rows. */
        final boolean quantized;

        Cache(int layers, int batch, boolean quantized) {
            this.quantized = quantized;
            keys = new Tensor[layers];
            values = new Tensor[layers];
            padding = new boolean[batch][0];
            next = new int[batch];
        }

        @Override
        public int length() {
            return length;
        }

        /** Makes room for {@code steps} more, keeping what is cached. */
        void reserve(int steps, int kvHeads, int headDim, java.util.function.IntFunction<DType> types,
                java.util.function.IntFunction<Device> devices) {
            int need = length + steps;
            if (need <= capacity) {
                return;
            }
            int grown = Math.max(need, Math.max(64, 2 * capacity));
            int batch = next.length;
            Shape shape = Shape.of(batch, kvHeads, grown, headDim);
            for (int l = 0; l < keys.length; l++) {
                DType dtype = types.apply(l);
                Device device = devices.apply(l);
                Tensor k = quantized ? Tensor.quantizedRows(shape, dtype, device) : Tensor.zeros(shape, dtype, device);
                Tensor v = quantized ? Tensor.quantizedRows(shape, dtype, device) : Tensor.zeros(shape, dtype, device);
                Tensor.evaluate(List.of(k, v));
                if (length > 0) {
                    k.write(2, 0, keys[l].slice(2, 0, length));
                    v.write(2, 0, values[l].slice(2, 0, length));
                    keys[l].release();
                    values[l].release();
                }
                keys[l] = k;
                values[l] = v;
            }
            capacity = grown;
        }

        /**
         * Drops the middle once it is a quarter window long, so the copy happens every
         * {@code window / 4} steps rather than at every one.
         */
        void evict() {
            int keep = sinks + window;
            if (window == 0 || length <= keep + Math.max(1, window / 4)) {
                return;
            }
            int from = length - window;
            for (int l = 0; l < keys.length; l++) {
                Tensor k = keys[l].slice(2, from, length);
                Tensor v = values[l].slice(2, from, length);
                Tensor.evaluate(List.of(k, v));   // read before the buffers are written
                keys[l].write(2, sinks, k);
                values[l].write(2, sinks, v);
            }
            for (int b = 0; b < padding.length; b++) {
                boolean[] p = java.util.Arrays.copyOf(padding[b], keep);
                System.arraycopy(padding[b], from, p, sinks, window);
                padding[b] = p;
            }
            length = keep;
        }

        @Override
        public State truncate(int keep) {
            if (window > 0) {
                throw new UnsupportedOperationException("a state that forgets cannot go back");
            }
            if (keep < 0 || keep > length) {
                throw new IllegalArgumentException("cannot keep " + keep + " of " + length + " steps");
            }
            for (int b = 0; b < next.length; b++) {
                for (int t = keep; t < length; t++) {
                    if (!padding[b][t]) {
                        next[b]--;
                    }
                }
                padding[b] = java.util.Arrays.copyOf(padding[b], keep);
            }
            length = keep;   // what follows is overwritten and masked until then
            return this;
        }

        @Override
        public State select(int[] rows) {
            Cache c = rows(rows);
            if (capacity > 0) {
                release(this);
            }
            return c;
        }

        @Override
        public State fork() {
            int[] all = new int[next.length];
            for (int i = 0; i < all.length; i++) {
                all[i] = i;
            }
            return rows(all);
        }

        /** A new cache of chosen rows; this one stays as it is. */
        private Cache rows(int[] rows) {
            Cache c = new Cache(keys.length, rows.length, quantized);
            c.sinks = sinks;
            c.window = window;
            if (capacity > 0) {
                for (int l = 0; l < keys.length; l++) {
                    Tensor index = indices(rows, keys[l].device());
                    c.keys[l] = requantized(keys[l].gather(0, index));
                    c.values[l] = requantized(values[l].gather(0, index));
                }
                Tensor.evaluate(java.util.Arrays.asList(c.keys));
                Tensor.evaluate(java.util.Arrays.asList(c.values));
            }
            for (int i = 0; i < rows.length; i++) {
                c.padding[i] = padding[rows[i]].clone();
                c.next[i] = next[rows[i]];
            }
            c.length = length;
            c.capacity = capacity;
            return c;
        }

        @Override
        public State join(State other) {
            if (!(other instanceof Cache o) || o.keys.length != keys.length) {
                throw new IllegalArgumentException("only states of the same model join");
            }
            if (quantized || o.quantized || window > 0 || o.window > 0) {
                throw new UnsupportedOperationException("8-bit and windowed caches do not join");
            }
            int batch = next.length + o.next.length;
            Cache c = new Cache(keys.length, batch, false);
            int len = Math.max(length, o.length);
            Tensor any = capacity > 0 ? keys[0] : o.capacity > 0 ? o.keys[0] : null;
            if (any != null) {
                int cap = Math.max(len, Math.max(capacity, o.capacity));
                for (int l = 0; l < keys.length; l++) {
                    Tensor like = capacity > 0 ? keys[l] : o.keys[l];   // this layer's device and type
                    c.keys[l] = Tensor.concat(0, padded(keys[l], next.length, length, len, cap, like),
                            padded(o.keys[l], o.next.length, o.length, len, cap, like));
                    c.values[l] = Tensor.concat(0, padded(values[l], next.length, length, len, cap, like),
                            padded(o.values[l], o.next.length, o.length, len, cap, like));
                }
                Tensor.evaluate(java.util.Arrays.asList(c.keys));
                Tensor.evaluate(java.util.Arrays.asList(c.values));
                c.capacity = cap;
                release(this);
                release(o);
            }
            for (int b = 0; b < batch; b++) {
                Cache from = b < next.length ? this : o;
                int row = b < next.length ? b : b - next.length;
                boolean[] p = new boolean[len];
                int shift = len - from.length;
                java.util.Arrays.fill(p, 0, shift, true);
                System.arraycopy(from.padding[row], 0, p, shift, from.length);
                c.padding[b] = p;
                c.next[b] = from.next[row];
            }
            c.length = len;
            return c;
        }

        @Override
        public void close() {
            release(this);
            java.util.Arrays.fill(keys, null);
            java.util.Arrays.fill(values, null);
            capacity = 0;
        }

        private static void release(Cache c) {
            for (int l = 0; l < c.keys.length; l++) {
                if (c.keys[l] != null) {
                    c.keys[l].release();
                    c.values[l].release();
                }
            }
        }

        /** {@code rows} sequences of {@code length} cached steps moved to end at {@code len} in a buffer of {@code cap}. */
        private static Tensor padded(Tensor t, int rows, int length, int len, int cap, Tensor like) {
            Shape s = Shape.of(rows, like.shape().dim(1), cap, like.shape().dim(3));
            Tensor out = Tensor.zeros(s, like.dtype(), like.device());
            Tensor.evaluate(List.of(out));
            if (t != null && length > 0) {
                out.write(2, len - length, t.slice(2, 0, length));
            }
            return out;
        }

        /** Gathered values kept as 8-bit rows again, when this cache keeps them so. */
        private Tensor requantized(Tensor values) {
            if (!quantized) {
                return values;
            }
            Tensor q = Tensor.quantizedRows(values.shape(), values.dtype(), values.device());
            q.write(2, 0, values);
            return q;
        }
    }

    @Override
    public State newState(int batch) {
        return new Cache(config.layers(), batch, quantizedCache);
    }

    private volatile boolean quantizedCache;

    /**
     * Sets whether states made from now on keep their keys and values as 8-bit levels with a
     * scale per head and step: half of float16's memory (a quarter of float32's), for long
     * texts, at a small loss of accuracy. Where the backend has a kernel for it, attention
     * reads the levels directly.
     *
     * @param quantized whether to quantize the cache
     * @return this model
     */
    public Llama quantizeCache(boolean quantized) {
        this.quantizedCache = quantized;
        return this;
    }

    private volatile int slidingWindow;
    private volatile int prefillChunk = 1024;

    /**
     * Sets sliding-window attention, as Mistral and some Qwen models use it: each step sees
     * only itself and the {@code steps - 1} before it, in {@link #forward}, {@link #loss} and
     * through a state alike.
     *
     * @param steps the window, or 0 for full causal attention
     * @return this model
     */
    public Llama slidingWindow(int steps) {
        if (steps < 0) {
            throw new IllegalArgumentException("window must be at least 0, got " + steps);
        }
        this.slidingWindow = steps;
        return this;
    }

    /**
     * Sets how many new tokens a state reads at once: a longer prompt is read in pieces of
     * this many steps, so the attention scores and mask stay {@code chunk x cache} rather than
     * growing with the square of the prompt. The results are the same.
     *
     * @param steps the steps per piece, or 0 to read any prompt at once
     * @return this model
     */
    public Llama prefillChunk(int steps) {
        if (steps < 0) {
            throw new IllegalArgumentException("chunk must be at least 0, got " + steps);
        }
        this.prefillChunk = steps;
        return this;
    }

    /**
     * A state that forgets the middle of long sequences, as StreamingLLM does (Xiao et al.,
     * 2024): it keeps the first {@code sinks} steps, which attention leans on, and the last
     * {@code window}, so memory and time per token stay bounded however long the text. Keys
     * keep the positions they were read at. It cannot {@linkplain State#truncate go back}.
     *
     * @param batch the number of sequences
     * @param sinks the first steps always kept
     * @param window the latest steps kept
     * @return an empty state
     */
    public State newState(int batch, int sinks, int window) {
        if (sinks < 0 || window < 1) {
            throw new IllegalArgumentException("sinks must be at least 0 and window at least 1, got " + sinks + ", "
                    + window);
        }
        Cache c = new Cache(config.layers(), batch, quantizedCache);
        c.sinks = sinks;
        c.window = window;
        return c;
    }

    /**
     * Saves a state to a file, as llama.cpp's prompt cache does: a long document or chat read
     * once can be picked up again later without reading it again. The keys and values go in
     * as safetensors with the positions and padding in the metadata.
     *
     * @param state a state of this model
     * @param file the file to write
     */
    public void saveState(State state, Path file) {
        Cache c = (Cache) Objects.requireNonNull(state, "state");
        if (c.next.length == 0) {
            throw new IllegalArgumentException("an empty state");
        }
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        if (c.length > 0) {
            for (int l = 0; l < c.keys.length; l++) {
                tensors.put("keys." + l, plain(c.keys[l].slice(2, 0, c.length)));
                tensors.put("values." + l, plain(c.values[l].slice(2, 0, c.length)));
            }
        }
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("layers", (long) c.keys.length);
        meta.put("length", (long) c.length);
        meta.put("sinks", (long) c.sinks);
        meta.put("window", (long) c.window);
        meta.put("quantized", c.quantized);
        List<Object> next = new ArrayList<>();
        List<Object> padding = new ArrayList<>();
        for (int b = 0; b < c.next.length; b++) {
            next.add((long) c.next[b]);
            StringBuilder p = new StringBuilder();
            for (int t = 0; t < c.length; t++) {
                p.append(c.padding[b][t] ? '1' : '0');
            }
            padding.add(p.toString());
        }
        meta.put("next", next);
        meta.put("padding", padding);
        io.github.zzuegg.brainiac.io.SafeTensors.write(file, tensors,
                Map.of("brainiac.state", Json.write(meta), "model", config.architecture()));
    }

    /**
     * Reads a state {@link #saveState saved} from this model (or one of the same shape).
     *
     * @param file the file
     * @return the state, ready to read more tokens
     */
    public State loadState(Path file) {
        try (io.github.zzuegg.brainiac.io.SafeTensors st = io.github.zzuegg.brainiac.io.SafeTensors.open(file)) {
            String text = st.metadata().get("brainiac.state");
            if (text == null) {
                throw new IllegalArgumentException(file + " is not a saved state");
            }
            Map<String, Object> meta = Json.object(Json.parse(text));
            if (Json.integer(meta, "layers") != blocks.size()) {
                throw new IllegalArgumentException(file + " is a state of a model with " + meta.get("layers")
                        + " blocks, not " + blocks.size());
            }
            List<Object> next = Json.array(meta.get("next"));
            List<Object> padding = Json.array(meta.get("padding"));
            int length = (int) Json.integer(meta, "length");
            Cache c = new Cache(blocks.size(), next.size(), Boolean.TRUE.equals(meta.get("quantized")));
            c.sinks = (int) Json.integer(meta, "sinks");
            c.window = (int) Json.integer(meta, "window");
            for (int b = 0; b < next.size(); b++) {
                c.next[b] = ((Number) next.get(b)).intValue();
                String p = (String) padding.get(b);
                c.padding[b] = new boolean[length];
                for (int t = 0; t < length; t++) {
                    c.padding[b][t] = p.charAt(t) == '1';
                }
            }
            if (length > 0) {
                c.reserve(length, config.kvHeads(), config.headDim(), this::typeOf, this::deviceOf);
                for (int l = 0; l < blocks.size(); l++) {
                    if (!st.shape("keys." + l).equals(Shape.of(next.size(), config.kvHeads(), length, config.headDim()))) {
                        throw new IllegalArgumentException(file + " does not fit this model's keys");
                    }
                    c.keys[l].write(2, 0, st.tensor("keys." + l, deviceOf(l), typeOf(l)));
                    c.values[l].write(2, 0, st.tensor("values." + l, deviceOf(l), typeOf(l)));
                }
                c.length = length;
            }
            return c;
        }
    }

    /**
     * Runs the whole model over sequences, for training and scoring: logits for every step.
     *
     * @param tokens {@code [batch][steps]} token ids, all sequences the same length
     * @return logits {@code [batch, steps, vocab]} in the model's type
     */
    public Var forward(int[][] tokens) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        int[][] positions = new int[batch][steps];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                positions[b][t] = t;
            }
        }
        if (insert != null && insert.grids != null) {
            multiPositions = gridPositions(tokens, new int[batch]);
        }
        try {
            return run(tokens, positions, null, null);
        } finally {
            multiPositions = null;
        }
    }

    /**
     * As {@link #forward(int[][])}, with outside embeddings (image features from a
     * vision-language model's projector) in place of a placeholder token: each occurrence of
     * {@code placeholder}, row by row, takes the next row of {@code embeddings}.
     *
     * @param tokens {@code [batch][steps]} token ids, all sequences the same length
     * @param placeholder the token the embeddings replace
     * @param embeddings {@code [occurrences, hidden]}
     * @return logits {@code [batch, steps, vocab]} in the model's type
     */
    public Var forward(int[][] tokens, int placeholder, Tensor embeddings) {
        return inserting(placeholder, embeddings, false, () -> forward(tokens));
    }

    /**
     * As {@link #forward(int[][], int, Tensor)}, optionally with each run of consecutive
     * placeholders attending to itself both ways (as Gemma 3 reads an image).
     *
     * @param tokens {@code [batch][steps]} token ids, all sequences the same length
     * @param placeholder the token the embeddings replace
     * @param embeddings {@code [occurrences, hidden]}
     * @param bidirectional whether a run of placeholders sees all of itself
     * @return logits {@code [batch, steps, vocab]} in the model's type
     */
    public Var forward(int[][] tokens, int placeholder, Tensor embeddings, boolean bidirectional) {
        return inserting(placeholder, embeddings, bidirectional, () -> forward(tokens));
    }

    /**
     * As {@link #next}, with outside embeddings in place of a placeholder token (see
     * {@link #forward(int[][], int, Tensor)}): how a vision-language model reads a prompt
     * with images. The calls that follow read tokens as usual.
     *
     * @param state the state; updated
     * @param tokens the new tokens per sequence
     * @param placeholder the token the embeddings replace
     * @param embeddings {@code [occurrences, hidden]}, in order
     * @return the next-token logits {@code [batch, vocab]}, float32
     */
    public Tensor next(State state, int[][] tokens, int placeholder, Tensor embeddings) {
        return inserting(placeholder, embeddings, false, () -> next(state, tokens, null));
    }

    /**
     * As {@link #next(State, int[][], int, Tensor)}, optionally with each run of consecutive
     * placeholders attending to itself both ways (as Gemma 3 reads an image; a run must not
     * cross a {@linkplain #prefillChunk prefill chunk}).
     *
     * @param state the state; updated
     * @param tokens the new tokens per sequence
     * @param placeholder the token the embeddings replace
     * @param embeddings {@code [occurrences, hidden]}, in order
     * @param bidirectional whether a run of placeholders sees all of itself
     * @return the next-token logits {@code [batch, vocab]}, float32
     */
    public Tensor next(State state, int[][] tokens, int placeholder, Tensor embeddings, boolean bidirectional) {
        return inserting(placeholder, embeddings, bidirectional, () -> next(state, tokens, null));
    }

    /** Embeddings that replace a placeholder token, consumed in order across prefill chunks. */
    private static final class Insert {
        final int token;
        final Tensor rows;
        final boolean bidirectional;
        int used;

        /** Per image, its {height, width} in tokens, for 3D rotary positions; or null. */
        int[][] grids;
        int gridsUsed;

        Insert(int token, Tensor rows, boolean bidirectional) {
            this.token = token;
            this.rows = rows;
            this.bidirectional = bidirectional;
        }
    }

    private Insert insert;

    /** Qwen2-VL's rotary sections (of the frequency pairs: time, height, width), or null. */
    private int[] ropeSections;

    /** The current call's 3D positions {@code [3][batch][steps]}, or null for 1D ones. */
    private int[][][] multiPositions;

    /**
     * Turns on Qwen2-VL's 3D rotary positions: the frequency pairs split in sections that
     * rotate by a token's time, height and width positions (all equal for text).
     */
    void ropeSections(int[] sections) {
        int sum = 0;
        for (int v : sections) {
            sum += v;
        }
        if (sections.length != 3 || sum * 2 != config.rope().dim()) {
            throw new IllegalArgumentException("rotary sections " + java.util.Arrays.toString(sections)
                    + " do not split " + config.rope().dim() / 2 + " frequency pairs in three");
        }
        ropeSections = sections.clone();
    }

    /**
     * As {@link #next(State, int[][], int, Tensor)} for a model with {@linkplain
     * #ropeSections 3D positions}: each run of placeholders is an image of {@code grids[i]}
     * tokens (height, width), placed at the run's start and spanning the larger of the two.
     */
    Tensor next(State state, int[][] tokens, int placeholder, Tensor embeddings, int[][] grids) {
        return inserting(placeholder, embeddings, false, grids, () -> next(state, tokens, null));
    }

    /** As {@link #forward(int[][], int, Tensor)} with images placed in 3D (see the {@code next} above). */
    Var forward(int[][] tokens, int placeholder, Tensor embeddings, int[][] grids) {
        return inserting(placeholder, embeddings, false, grids, () -> forward(tokens));
    }

    /**
     * Qwen2-VL's positions {@code [3][batch][steps]} from each row's next position: text
     * moves all three by one; an image's tokens share its start in time and count rows and
     * columns from it, and the text after it starts past its larger side.
     */
    private int[][][] gridPositions(int[][] tokens, int[] start) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        int[][][] out = new int[3][batch][steps];
        for (int b = 0; b < batch; b++) {
            int cur = start[b];
            int t = 0;
            while (t < steps) {
                if (tokens[b][t] != insert.token) {
                    out[0][b][t] = out[1][b][t] = out[2][b][t] = cur++;
                    t++;
                    continue;
                }
                if (insert.gridsUsed >= insert.grids.length) {
                    throw new IllegalArgumentException("more images than the " + insert.grids.length + " grids given");
                }
                int[] g = insert.grids[insert.gridsUsed++];
                for (int k = 0; k < g[0] * g[1]; k++, t++) {
                    if (t >= steps || tokens[b][t] != insert.token) {
                        throw new IllegalArgumentException("an image of " + g[0] + " x " + g[1]
                                + " tokens needs as many placeholders in one run");
                    }
                    out[0][b][t] = cur;
                    out[1][b][t] = cur + k / g[1];
                    out[2][b][t] = cur + k % g[1];
                }
                cur += Math.max(g[0], g[1]);
            }
            start[b] = cur;
        }
        return out;
    }

    /**
     * Per token, the index of its run of placeholders that attends both ways, or -1; {@code
     * null} when no run does.
     */
    private int[][] runs(int[][] tokens) {
        if (insert == null || !insert.bidirectional) {
            return null;
        }
        int[][] out = new int[tokens.length][];
        boolean any = false;
        for (int b = 0; b < tokens.length; b++) {
            out[b] = new int[tokens[b].length];
            int run = -1;
            for (int t = 0; t < tokens[b].length; t++) {
                if (tokens[b][t] == insert.token) {
                    if (t == 0 || tokens[b][t - 1] != insert.token) {
                        run++;
                    }
                    out[b][t] = run;
                    any = true;
                } else {
                    out[b][t] = -1;
                }
            }
        }
        return any ? out : null;
    }

    private <T> T inserting(int placeholder, Tensor embeddings, boolean bidirectional,
            java.util.function.Supplier<T> body) {
        return inserting(placeholder, embeddings, bidirectional, null, body);
    }

    private <T> T inserting(int placeholder, Tensor embeddings, boolean bidirectional, int[][] grids,
            java.util.function.Supplier<T> body) {
        if (embeddings.shape().rank() != 2 || embeddings.shape().dim(1) != config.hidden()) {
            throw new IllegalArgumentException("embeddings must be [n, " + config.hidden() + "], got "
                    + embeddings.shape());
        }
        Insert in = new Insert(placeholder, embeddings.to(device).to(dtype), bidirectional);
        if (grids != null && ropeSections == null) {
            throw new IllegalStateException("images placed in 3D need rotary sections");
        }
        in.grids = grids;
        insert = in;
        try {
            T out = body.get();
            if (in.used != embeddings.shape().dim(0)) {
                throw new IllegalArgumentException("the tokens hold " + in.used + " placeholders for "
                        + embeddings.shape().dim(0) + " embeddings");
            }
            return out;
        } finally {
            insert = null;
        }
    }

    /**
     * The next-token cross-entropy of sequences, for training: the mean over every step but
     * the last of {@code -log p(next token)}. The output layer runs a few thousand rows at a
     * time (see {@link Functional#linearCrossEntropy}), so the {@code [tokens, vocab]} logits
     * are never stored whole.
     *
     * @param tokens {@code [batch][steps]} token ids, all sequences the same length
     * @return the loss, a scalar
     */
    public Var loss(int[][] tokens) {
        return loss(tokens, Math.max(1, (1 << 24) / config.vocabSize()));
    }

    /**
     * {@link #loss(int[][])} with the rows of logits computed at a time.
     *
     * @param tokens {@code [batch][steps]} token ids
     * @param chunk rows of logits at a time
     * @return the loss
     */
    public Var loss(int[][] tokens, int chunk) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        int[][] positions = new int[batch][steps];
        float[] next = new float[batch * steps];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                positions[b][t] = t;
                next[b * steps + t] = t + 1 < steps ? tokens[b][t + 1] : -1;   // the last step predicts nothing
            }
        }
        return loss(tokens, next, positions, null, chunk);
    }

    /**
     * The cross-entropy of instruction-tuning data: padded or packed rows where only the
     * marked tokens count (see {@link InstructionData}). Packed examples see only their own
     * earlier tokens, and each starts again at position 0.
     *
     * @param batch the rows
     * @return the mean over the tokens that count
     */
    public Var loss(InstructionData.Batch batch) {
        int rows = batch.tokens().length;
        int steps = batch.tokens()[0].length;
        int[][] positions = new int[rows][steps];
        float[] next = new float[rows * steps];
        boolean packed = false;
        for (int b = 0; b < rows; b++) {
            int[] seg = batch.segments()[b];
            for (int t = 0; t < steps; t++) {
                boolean continues = t > 0 && seg[t] == seg[t - 1];
                positions[b][t] = continues ? positions[b][t - 1] + 1 : 0;
                packed |= t > 0 && seg[t] != seg[t - 1] && seg[t] != 0;
                next[b * steps + t] = batch.targets()[b][t];
            }
        }
        Tensor[] masks = null;
        if (packed) {
            masks = new Tensor[blocks.size()];
            Map<String, Tensor> byWindow = new java.util.HashMap<>();
            for (int l = 0; l < masks.length; l++) {
                int window = windowOf(l);
                Device d = deviceOf(l);
                DType t = typeOf(l);
                masks[l] = byWindow.computeIfAbsent(window + " " + d + " " + t,
                        k -> segmentMask(batch.segments(), window, d, t));
            }
        }
        return loss(batch.tokens(), next, positions, masks, Math.max(1, (1 << 24) / config.vocabSize()));
    }

    /**
     * The log-probability of each row's marked tokens, summed: {@code sum log p(target)} over
     * the steps whose target is not -1, as preference losses such as DPO need (see
     * {@link PreferenceTraining}). Unlike {@link #loss(InstructionData.Batch)} it holds the
     * {@code [tokens, vocab]} log-probabilities, so it suits short rows.
     *
     * @param batch padded rows, one example each (see {@link InstructionData#pad})
     * @return {@code [rows]}
     */
    public Var sequenceLogProbs(InstructionData.Batch batch) {
        int rows = batch.tokens().length;
        int steps = batch.tokens()[0].length;
        int[][] positions = new int[rows][steps];
        float[] next = new float[rows * steps];
        for (int b = 0; b < rows; b++) {
            int[] seg = batch.segments()[b];
            for (int t = 0; t < steps; t++) {
                if (t > 0 && seg[t] != seg[t - 1] && seg[t] != 0) {
                    throw new IllegalArgumentException("sequenceLogProbs needs one example per row, not packed rows");
                }
                positions[b][t] = t;
                next[b * steps + t] = batch.targets()[b][t];
            }
        }
        Var h = hidden(batch.tokens(), positions, null, null).reshape(rows * steps, config.hidden());
        Var logits = softcap(head(h, false));
        Tensor ids = Tensor.of(Shape.of(rows * steps, 1), next, headDevice());
        Tensor pick = ids.equalTo(Tensor.arange(config.vocabSize()).to(headDevice())).to(logits.value().dtype());
        Var logp = logits.sub(logits.logsumexp(-1, true));
        return logp.mul(pick).sum(-1, false).reshape(rows, steps).sum(1, false);
    }

    /** Causal within each segment, optionally windowed: {@code [rows, 1, steps, steps]}. */
    private Tensor segmentMask(int[][] segments, int window, Device device, DType dtype) {
        int rows = segments.length;
        int steps = segments[0].length;
        float[] m = new float[rows * steps * steps];
        for (int b = 0; b < rows; b++) {
            for (int t = 0; t < steps; t++) {
                for (int k = 0; k < steps; k++) {
                    if (k > t || segments[b][k] != segments[b][t] || window > 0 && k <= t - window) {
                        m[(b * steps + t) * steps + k] = -1e9f;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(rows, 1, steps, steps), m, device).to(dtype);
    }

    private Var loss(int[][] tokens, float[] next, int[][] positions, Tensor[] masks, int chunk) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        Var h = hidden(tokens, positions, null, masks).reshape(batch * steps, config.hidden());
        Tensor ids = Tensor.of(Shape.of(batch * steps), next, headDevice());
        if (config.extras().finalSoftcap() > 0) {
            return Functional.crossEntropy(softcap(head(h, false)).reshape(batch * steps, config.vocabSize()), ids);
        }
        if (lmHead != null && lmHead.bias() != null) {
            return Functional.crossEntropy(lmHead.apply(h).reshape(batch * steps, config.vocabSize()), ids);
        }
        if (lmHead != null && lmHead.loraA != null) {
            return Functional.crossEntropy(lmHead.apply(h).reshape(batch * steps, config.vocabSize()), ids);
        }
        Var weight = lmHead != null ? lmHead.base()
                : tiedOnHead != null ? tiedOnHead.transpose()   // a split model's copy: it does not learn
                : base(embed).transpose();
        return Functional.linearCrossEntropy(h, weight, ids, chunk);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        return read(state, tokens, padding, false);
    }

    @Override
    public Tensor nextAll(State state, int[][] tokens) {
        return read(state, tokens, null, true);
    }

    /**
     * As {@link #next}, also giving the final normalized hidden states of the new tokens,
     * the input of the output layer: for retrieval by hidden state (kNN-LM), probes and
     * steering.
     *
     * @param state the state; updated
     * @param tokens the new tokens per sequence
     * @param padding which new tokens are padding, or {@code null}
     * @return the next-token logits {@code [batch, vocab]} and the hidden states
     *     {@code [batch, steps, hidden]}, both float32
     */
    public Tensor[] nextWithHidden(State state, int[][] tokens, boolean[][] padding) {
        Tensor[] hidden = new Tensor[1];
        Tensor logits = read(state, tokens, padding, false, hidden);
        return new Tensor[] {logits, hidden[0]};
    }

    /** How {@link #embed} turns a sequence's hidden states into one vector. */
    public enum Pooling {
        /** The last token's hidden state, as Qwen3-Embedding and other decoder embedders do. */
        LAST,
        /** The mean over the sequence's tokens. */
        MEAN
    }

    /**
     * Sentence embeddings from the final normalized hidden states, for retrieval and
     * similarity with decoder embedding models (Qwen3-Embedding, gte-Qwen2, e5-mistral).
     * Sequences may differ in length; each is padded after its end, which a causal model
     * never looks at.
     *
     * @param tokens the token ids of each sequence, none empty
     * @param pooling how the steps are pooled
     * @param normalize whether each vector is scaled to length 1, for cosine similarity by
     *     dot product
     * @return {@code [sequences][hidden]}
     */
    public float[][] embed(int[][] tokens, Pooling pooling, boolean normalize) {
        int batch = tokens.length;
        int steps = 0;
        for (int[] t : tokens) {
            if (t.length == 0) {
                throw new IllegalArgumentException("cannot embed an empty sequence");
            }
            steps = Math.max(steps, t.length);
        }
        int[][] padded = new int[batch][steps];
        boolean[][] pad = new boolean[batch][steps];
        boolean any = false;
        for (int b = 0; b < batch; b++) {
            System.arraycopy(tokens[b], 0, padded[b], 0, tokens[b].length);
            for (int t = tokens[b].length; t < steps; t++) {
                pad[b][t] = true;
                any = true;
            }
        }
        State state = newState(batch);
        float[] h;
        try {
            Tensor[] out = nextWithHidden(state, padded, any ? pad : null);
            h = out[1].toArray();
            out[0].release();
            out[1].release();
        } finally {
            state.close();
        }
        int hidden = config.hidden();
        float[][] result = new float[batch][hidden];
        for (int b = 0; b < batch; b++) {
            int n = tokens[b].length;
            float[] v = result[b];
            if (pooling == Pooling.LAST) {
                System.arraycopy(h, (b * steps + n - 1) * hidden, v, 0, hidden);
            } else {
                for (int t = 0; t < n; t++) {
                    int at = (b * steps + t) * hidden;
                    for (int i = 0; i < hidden; i++) {
                        v[i] += h[at + i] / n;
                    }
                }
            }
            if (normalize) {
                double sq = 0;
                for (float x : v) {
                    sq += (double) x * x;
                }
                float inv = (float) (1 / Math.max(Math.sqrt(sq), 1e-12));
                for (int i = 0; i < hidden; i++) {
                    v[i] *= inv;
                }
            }
        }
        return result;
    }

    private Tensor read(State state, int[][] tokens, boolean[][] padding, boolean everyStep) {
        return read(state, tokens, padding, everyStep, null);
    }

    private Tensor read(State state, int[][] tokens, boolean[][] padding, boolean everyStep, Tensor[] hiddenOut) {
        Cache cache = (Cache) Objects.requireNonNull(state, "state");
        int batch = tokens.length;
        int steps = tokens[0].length;
        int chunk = prefillChunk;
        if (chunk > 0 && steps > chunk) {
            List<Tensor> outs = new ArrayList<>();
            List<Tensor> hiddens = new ArrayList<>();
            for (int from = 0; from < steps; from += chunk) {
                int to = Math.min(steps, from + chunk);
                int[][] part = new int[batch][];
                boolean[][] pad = padding == null ? null : new boolean[batch][];
                for (int b = 0; b < batch; b++) {
                    part[b] = java.util.Arrays.copyOfRange(tokens[b], from, to);
                    if (pad != null) {
                        pad[b] = java.util.Arrays.copyOfRange(padding[b], from, to);
                    }
                }
                Tensor[] h = hiddenOut == null ? null : new Tensor[1];
                Tensor out = readAtOnce(cache, part, pad, everyStep, h);
                if (everyStep || to == steps) {
                    outs.add(out);
                }
                if (h != null) {
                    hiddens.add(h[0]);
                }
            }
            if (hiddenOut != null) {
                hiddenOut[0] = Tensor.concat(1, hiddens.toArray(Tensor[]::new));
            }
            return outs.size() == 1 ? outs.getFirst() : Tensor.concat(1, outs.toArray(Tensor[]::new));
        }
        return readAtOnce(cache, tokens, padding, everyStep, hiddenOut);
    }

    private Tensor readAtOnce(Cache cache, int[][] tokens, boolean[][] padding, boolean everyStep,
            Tensor[] hiddenOut) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        if (batch != cache.next.length) {
            throw new IllegalArgumentException("the state holds " + cache.next.length + " sequences, got " + batch);
        }
        cache.reserve(steps, config.kvHeads(), config.headDim(), this::typeOf, this::deviceOf);
        int[][] positions = new int[batch][steps];
        boolean[][] keysPad = new boolean[batch][];
        for (int b = 0; b < batch; b++) {
            keysPad[b] = java.util.Arrays.copyOf(cache.padding[b], cache.length + steps);
            for (int t = 0; t < steps; t++) {
                boolean pad = padding != null && padding[b][t];
                keysPad[b][cache.length + t] = pad;
                positions[b][t] = pad ? 1 : cache.next[b]++;
            }
        }
        // 3D positions: images move the next position by their larger side, not their size
        int[][][] multi = null;
        if (ropeSections != null) {
            int[] start = new int[batch];
            for (int b = 0; b < batch; b++) {
                start[b] = cache.next[b] - steps;
            }
            if (insert != null && insert.grids != null) {
                multi = gridPositions(tokens, start);
                for (int b = 0; b < batch; b++) {
                    cache.next[b] = start[b];
                }
            }
        }
        // causal, padding and the buffer's unused end: [batch, 1, steps, capacity]; one per window
        int[][] runs = runs(tokens);
        Tensor[] masks = new Tensor[blocks.size()];
        Map<String, Tensor> byWindow = new java.util.HashMap<>();
        for (int l = 0; l < masks.length; l++) {
            int window = windowOf(l);
            Device d = deviceOf(l);
            DType t = typeOf(l);
            masks[l] = byWindow.computeIfAbsent(window + " " + d + " " + t,
                    k -> cacheMask(cache, keysPad, steps, window, d, t, runs));
        }
        Var[] hidden = new Var[1];
        multiPositions = multi;
        Var logits;
        try {
            logits = run(tokens, positions, cache, masks, hidden);
        } finally {
            multiPositions = null;
        }
        cache.padding = keysPad;
        cache.length += steps;
        Tensor out = everyStep ? logits.value().to(DType.FLOAT32)
                : logits.value().slice(1, steps - 1, steps).reshape(batch, config.vocabSize()).to(DType.FLOAT32);
        // computed before the next step writes into the cache it read
        if (hiddenOut != null) {
            hiddenOut[0] = hidden[0].value().to(DType.FLOAT32);
            Tensor.evaluate(List.of(out, hiddenOut[0]));
        } else {
            Tensor.evaluate(List.of(out));
        }
        cache.evict();
        return out;
    }

    private Tensor cacheMask(Cache cache, boolean[][] keysPad, int steps, int window, Device device, DType dtype,
            int[][] runs) {
        int batch = keysPad.length;
        int keys = cache.capacity;
        float[] m = new float[batch * steps * keys];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                int row = (b * steps + t) * keys;
                for (int k = 0; k < keys; k++) {
                    int at = k - cache.length;
                    boolean together = runs != null && at > t && at < steps && runs[b][t] >= 0
                            && runs[b][at] == runs[b][t];
                    if (k > cache.length + t && !together || keysPad[b][Math.min(k, keysPad[b].length - 1)]
                            || window > 0 && k <= cache.length + t - window) {
                        m[row + k] = -1e9f;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(batch, 1, steps, keys), m, device).to(dtype);
    }

    private Var run(int[][] tokens, int[][] positions, Cache cache, Tensor[] masks) {
        return run(tokens, positions, cache, masks, null);
    }

    private Var run(int[][] tokens, int[][] positions, Cache cache, Tensor[] masks, Var[] hiddenOut) {
        Var x = hidden(tokens, positions, cache, masks);
        if (hiddenOut != null) {
            hiddenOut[0] = x;
        }
        return softcap(head(x, cache != null));
    }

    /** Gemma 2's {@code cap tanh(logits / cap)}, or the logits as they are. */
    private Var softcap(Var logits) {
        float cap = config.extras().finalSoftcap();
        return cap > 0 ? logits.mul(1 / cap).tanh().mul(cap) : logits;
    }

    private Var head(Var x, boolean decoding) {
        if (lmHead != null) {
            return lmHead.apply(x);
        }
        // tied: while decoding, a transposed copy of the embeddings made once; while
        // training, through the embeddings so they get the gradient
        // decoding on the CPU keeps a transposed copy of the embeddings; a GPU reads the
        // transpose in place, saving the copy's memory (272 MB for Qwen2.5-0.5B)
        if (tiedOnHead != null) {
            return x.matmul(tiedOnHead.transpose());
        }
        boolean copy = decoding && embed.value().device().kind() == io.github.zzuegg.brainiac.tensor.Device.Kind.CPU;
        return x.matmul(copy ? Var.of(tiedHead()) : base(embed).transpose());
    }

    /** The final normalized hidden states {@code [batch, steps, hidden]}. */
    private Var hidden(int[][] tokens, int[][] positions, Cache cache, Tensor[] masks) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        int[] flat = new int[batch * steps];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(tokens[b], 0, flat, b * steps, steps);
        }
        LlamaConfig.Extras e = config.extras();
        int[] lookup = flat;
        if (insert != null) {
            // the placeholder need not be in the vocabulary: it reads row 0, then is replaced
            lookup = flat.clone();
            for (int i = 0; i < lookup.length; i++) {
                if (lookup[i] == insert.token) {
                    lookup[i] = 0;
                }
            }
        }
        Var x = base(embed).gather(0, indices(lookup, device)).reshape(batch, steps, config.hidden());
        if (e.scaleEmbeddings()) {
            // as Gemma: the factor rounded to the model's type first
            x = x.mul(Tensor.full(Shape.of(), (float) Math.sqrt(config.hidden()), Device.cpu()).to(dtype).item());
        }
        if (insert != null) {
            // the placeholders read rows appended after the token embeddings
            int[] from = new int[flat.length];
            int n = flat.length;
            int count = 0;
            for (int i = 0; i < flat.length; i++) {
                from[i] = flat[i] == insert.token ? n + count++ : i;
            }
            if (count > 0) {
                if (insert.used + count > insert.rows.shape().dim(0)) {
                    throw new IllegalArgumentException("more placeholders than the " + insert.rows.shape().dim(0)
                            + " embeddings given");
                }
                Var rows = Var.of(insert.rows.slice(0, insert.used, insert.used + count));
                insert.used += count;
                x = Var.concat(0, x.reshape(n, config.hidden()), rows).gather(0, indices(from, device))
                        .reshape(batch, steps, config.hidden());
            }
        }
        int[][] runs = masks == null ? runs(tokens) : null;
        if (runs != null) {
            masks = new Tensor[blocks.size()];
            Map<String, Tensor> byWindow = new java.util.HashMap<>();
            for (int l = 0; l < blocks.size(); l++) {
                int window = windowOf(l);
                Device d = deviceOf(l);
                DType t = typeOf(l);
                masks[l] = byWindow.computeIfAbsent(window + " " + d + " " + t, k -> runMask(runs, window, d, t));
            }
        }
        if (masks == null) {
            masks = new Tensor[blocks.size()];
            Map<String, Tensor> byWindow = new java.util.HashMap<>();
            for (int l = 0; l < blocks.size(); l++) {
                int window = windowOf(l);
                if (window > 0 && steps > window) {
                    Device d = deviceOf(l);
                    DType t = typeOf(l);
                    masks[l] = byWindow.computeIfAbsent(window + " " + d + " " + t, k -> mask(steps, window, d, t));
                }
            }
        }
        boolean decoding = cache != null;
        boolean recompute = checkpointing && !decoding;
        for (int l = 0; l < blocks.size(); l++) {
            x = x.to(deviceOf(l)).to(typeOf(l));   // crossing a device split
            int layer = l;
            Tensor[] m = masks;
            if (recompute) {
                // activation checkpointing: the block runs again for its gradients
                x = io.github.zzuegg.brainiac.autograd.Recompute.apply(in -> block(layer, in, positions, null, m, false), x,
                        blockParameters(l));
            } else {
                x = block(layer, x, positions, cache, m, decoding);
            }
        }
        x = x.to(headDevice()).to(headType());
        return rmsNorm(x, norm, decoding);
    }

    /** Block {@code l} on the hidden states {@code x} {@code [batch, steps, hidden]}. */
    private Var block(int l, Var x, int[][] positions, Cache cache, Tensor[] masks, boolean decoding) {
        LlamaConfig.Extras e = config.extras();
        int batch = x.shape().dim(0);
        int steps = x.shape().dim(1);
        int hd = config.headDim();
        int rotary = config.rope().dim();
        float scale = e.queryScale() > 0 ? e.queryScale() : (float) (1 / Math.sqrt(hd));
        Block blk = blocks.get(l);
        Device layerDev = deviceOf(l);
        Var h = rmsNorm(x, blk.attnNorm, decoding);
        Fused f = fused == null ? null : fused[l];
        Var qOut;
        Var kOut;
        Var vOut;
        if (f != null && f.qkv() != null) {
            Var all = f.qkv().apply(h);
            int qs = config.heads() * hd;
            int ks = config.kvHeads() * hd;
            qOut = all.slice(-1, 0, qs);
            kOut = all.slice(-1, qs, qs + ks);
            vOut = all.slice(-1, qs + ks, qs + 2 * ks);
        } else {
            qOut = blk.q.apply(h);
            kOut = blk.k.apply(h);
            vOut = blk.v.apply(h);
        }
        Var q = qOut.reshape(batch, steps, config.heads(), hd);
        Var k = kOut.reshape(batch, steps, config.kvHeads(), hd);
        if (blk.qNorm != null) {
            q = rmsNorm(q, blk.qNorm, decoding);
            k = rmsNorm(k, blk.kNorm, decoding);
        }
        q = q.permute(0, 2, 1, 3);
        k = k.permute(0, 2, 1, 3);
        Var v = vOut.reshape(batch, steps, config.kvHeads(), hd).permute(0, 2, 1, 3);
        Rope r = localRope != null && e.local(l) ? localRope : rope;
        q = rotate(r, q, positions, rotary, hd);
        k = rotate(r, k, positions, rotary, hd);
        boolean causal = steps > 1;
        if (cache != null) {
            cache.keys[l].write(2, cache.length, k.value());
            cache.values[l].write(2, cache.length, v.value());
            k = Var.of(cache.keys[l]);
            v = Var.of(cache.values[l]);
            q = Var.of(q.value());   // decoding has no gradients: lets the fused kernel take the mask
        }
        Tensor bias = masks[l];
        if (bias == null && e.attentionSoftcap() > 0 && causal) {
            bias = Functional.causalMask(steps, k.shape().dim(2), layerDev).to(typeOf(l));
        }
        Var a = e.attentionSoftcap() > 0 ? Functional.softCappedAttention(q, k, v, bias, scale, e.attentionSoftcap())
                : bias != null ? Functional.attention(q, k, v, bias, scale, 0, null)
                : Functional.attention(q, k, v, causal, scale);
        a = a.permute(0, 2, 1, 3).reshape(batch, steps, config.heads() * hd);
        Var attn = blk.o.apply(a);
        if (blk.postAttnNorm != null) {
            attn = rmsNorm(attn, blk.postAttnNorm, decoding);
        }
        x = x.add(attn);
        h = rmsNorm(x, blk.ffnNorm, decoding);
        // both products first: the activation and the product then run as one chain
        if (blk.experts != null) {
            Var ffnOut = experts(blk.experts, h);
            if (blk.postFfnNorm != null) {
                ffnOut = rmsNorm(ffnOut, blk.postFfnNorm, decoding);
            }
            x = x.add(ffnOut);
            return x;
        }
        Var gate;
        Var up;
        if (f != null && f.gateUp() != null) {
            Var both = f.gateUp().apply(h);
            int ffn = config.intermediate();
            gate = both.slice(-1, 0, ffn);
            up = both.slice(-1, ffn, 2 * ffn);
        } else {
            gate = blk.gate.apply(h);
            up = blk.up.apply(h);
        }
        Var gated = gate.activation(e.activation(), 0).mul(up);
        Var ffnOut = blk.down.apply(gated);
        if (blk.postFfnNorm != null) {
            ffnOut = rmsNorm(ffnOut, blk.postFfnNorm, decoding);
        }
        x = x.add(ffnOut);
        return x;
    }

    /** The values block {@code l} learns: its weights unless frozen, and its adapters. */
    private List<Var> blockParameters(int l) {
        String prefix = "model.layers." + l + ".";
        List<Var> out = new ArrayList<>();
        for (Map.Entry<String, Var> en : named.entrySet()) {
            if (en.getKey().startsWith(prefix) && en.getValue().requiresGrad() && !frozen) {
                out.add(en.getValue());
            }
        }
        for (Map.Entry<String, Linear> en : linears.entrySet()) {
            if (en.getKey().startsWith(prefix) && en.getValue().loraA != null) {
                out.add(en.getValue().loraA);
                out.add(en.getValue().loraB);
            }
        }
        return out;
    }

    private volatile boolean checkpointing;

    /**
     * Sets activation checkpointing for training ({@link #forward}, {@link #loss}): each block
     * keeps only its input and runs again during the backward pass, so memory for activations
     * grows with the number of blocks instead of with everything inside them, for about a
     * third more compute. The results are the same.
     *
     * @param on whether to checkpoint
     * @return this model
     */
    public Llama checkpointActivations(boolean on) {
        this.checkpointing = on;
        return this;
    }

    /**
     * A mixture of experts over {@code h} {@code [batch, steps, hidden]}: the router's softmax
     * picks each token's experts on the host, each expert runs once on its tokens, and a
     * gather sums every token's outputs. The weights stay differentiable.
     */
    private Var experts(Experts m, Var h) {
        LlamaConfig.Moe moe = config.extras().moe();
        int[] s = h.shape().toArray();
        int n = s[0] * s[1];
        int hidden = config.hidden();
        int count = moe.experts();
        int k = moe.perToken();
        Device dev = h.value().device();
        Var flat = h.reshape(n, hidden);
        Var probs = Activation.SOFTMAX.forward(m.router().apply(flat).to(DType.FLOAT32));   // [n, experts]
        float[] p = probs.value().toArray();
        // each token's experts, largest weight first
        int[] chosen = new int[n * k];
        for (int t = 0; t < n; t++) {
            boolean[] taken = new boolean[count];
            for (int j = 0; j < k; j++) {
                int best = -1;
                for (int e = 0; e < count; e++) {
                    if (!taken[e] && (best < 0 || p[t * count + e] > p[t * count + best])) {
                        best = e;
                    }
                }
                taken[best] = true;
                chosen[t * k + j] = best;
            }
        }
        int[] picked = new int[n * k];
        for (int i = 0; i < picked.length; i++) {
            picked[i] = i / k * count + chosen[i];
        }
        Var weights = probs.reshape(n * count).gather(0, indices(picked, dev));   // [n * k]
        if (moe.normalize()) {
            Var w2 = weights.reshape(n, k);
            weights = w2.div(w2.sum(1, true)).reshape(n * k);
        }
        // the rows of each expert, in expert order; then where each (token, slot) landed
        List<Var> outs = new ArrayList<>();
        int[] place = new int[n * k];
        int rows = 0;
        for (int e = 0; e < count; e++) {
            List<Integer> slots = new ArrayList<>();
            for (int i = 0; i < chosen.length; i++) {
                if (chosen[i] == e) {
                    slots.add(i);
                }
            }
            if (slots.isEmpty()) {
                continue;
            }
            int[] tokens = new int[slots.size()];
            int[] slotIds = new int[slots.size()];
            for (int i = 0; i < tokens.length; i++) {
                slotIds[i] = slots.get(i);
                tokens[i] = slotIds[i] / k;
                place[slotIds[i]] = rows + i;
            }
            rows += tokens.length;
            Var xe = flat.gather(0, indices(tokens, dev));
            Var y = m.down()[e].apply(m.gate()[e].apply(xe).activation(config.extras().activation(), 0)
                    .mul(m.up()[e].apply(xe)));
            Var w = weights.gather(0, indices(slotIds, dev)).reshape(tokens.length, 1).to(y.dtype());
            outs.add(y.mul(w));
        }
        Var all = outs.size() == 1 ? outs.getFirst() : Var.concat(0, outs.toArray(Var[]::new));
        Var out = all.gather(0, indices(place, dev)).reshape(n, k, hidden).sum(1, false);
        if (m.sharedGate() != null) {
            Var shared = m.sharedDown().apply(m.sharedGate().apply(flat).activation(config.extras().activation(), 0)
                    .mul(m.sharedUp().apply(flat)));
            out = out.add(shared.mul(m.sharedScale().apply(flat).activation(ActivationOp.SIGMOID, 0)));
        }
        return out.reshape(s[0], s[1], hidden);
    }

    /** Rotates the first {@code rotary} features of each head; the rest pass (partial rotary, Phi). */
    private Var rotate(Rope r, Var x, int[][] positions, int rotary, int hd) {
        if (multiPositions != null) {
            if (rotary != hd) {
                throw new IllegalStateException("3D positions with partial rotation");
            }
            return r.apply(x, multiPositions, ropeSections);
        }
        if (rotary == hd) {
            return r.apply(x, positions);
        }
        return Var.concat(3, r.apply(x.slice(-1, 0, rotary), positions), x.slice(-1, rotary, hd));
    }

    /** The window layer {@code l} sees, or 0 for all of the past. */
    private int windowOf(int layer) {
        int set = slidingWindow;
        if (set > 0) {
            return set;
        }
        return config.extras().local(layer) ? config.extras().slidingWindow() : 0;
    }

    /**
     * Causal, within a window when {@code window > 0}, with each run seeing all of itself:
     * {@code [batch, 1, steps, steps]}.
     */
    private static Tensor runMask(int[][] runs, int window, Device device, DType dtype) {
        int batch = runs.length;
        int steps = runs[0].length;
        float[] m = new float[batch * steps * steps];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                for (int k = 0; k < steps; k++) {
                    boolean together = runs[b][t] >= 0 && runs[b][k] == runs[b][t];
                    if (k > t && !together || window > 0 && k <= t - window) {
                        m[(b * steps + t) * steps + k] = -1e9f;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(batch, 1, steps, steps), m, device).to(dtype);
    }

    /** A causal mask over {@code steps} steps where each sees only the last {@code window}. */
    private Tensor mask(int steps, int window, Device device, DType dtype) {
        float[] m = new float[steps * steps];
        for (int t = 0; t < steps; t++) {
            for (int k = 0; k < steps; k++) {
                if (k > t || k <= t - window) {
                    m[t * steps + k] = -1e9f;
                }
            }
        }
        return Tensor.of(Shape.of(1, 1, steps, steps), m, device).to(dtype);
    }

    private Tensor headSource;
    private Tensor head;

    /** The embeddings transposed, made again whenever they change. */
    private synchronized Tensor tiedHead() {
        if (headSource != embed.value()) {
            if (head != null) {
                head.release();
            }
            headSource = embed.value();
            head = headSource.transpose();
            Tensor.evaluate(List.of(head));
        }
        return head;
    }

    // ---------------------------------------------------------------- adapters

    /** A weight as the computation uses it: a constant once the model is frozen. */
    private Var base(Var v) {
        return frozen ? Var.of(v.value()) : v;
    }

    /**
     * Adds LoRA adapters (Hu et al., 2021) to the projections {@code target} accepts, by
     * Hugging Face module name such as {@code model.layers.3.self_attn.q_proj}: each computes
     * {@code x W + (alpha / rank) x A B}, with {@code A} random and {@code B} zero, so the model
     * starts unchanged. Every other weight becomes a constant: train the returned adapters
     * alone. On {@linkplain #quantizeWeights quantized} weights this is QLoRA.
     *
     * @param rank the adapters' inner size
     * @param alpha their scale numerator
     * @param target which projections get one; see {@link #attentionAndMlp()}
     * @param seed the seed for {@code A}
     * @return the new adapter parameters
     */
    public List<Var> addLora(int rank, float alpha, java.util.function.Predicate<String> target, long seed) {
        if (rank <= 0) {
            throw new IllegalArgumentException("rank must be positive, got " + rank);
        }
        SplittableRandom random = new SplittableRandom(seed);
        List<Var> added = new ArrayList<>();
        frozen = true;
        for (Map.Entry<String, Linear> e : linears.entrySet()) {
            Linear l = e.getValue();
            l.frozen = true;
            if (!target.test(e.getKey()) || l.loraA != null) {
                continue;
            }
            int in = l.weight.shape().dim(0);
            int out = l.weight.shape().dim(1);
            // as PEFT: A uniform in +-1/sqrt(in) (Kaiming with a = sqrt(5)), B zero
            float bound = (float) (1 / Math.sqrt(in));
            l.loraA = Var.param(Tensor.uniform(Shape.of(in, rank), -bound, bound, random).to(device).to(dtype));
            l.loraB = Var.param(Tensor.zeros(Shape.of(rank, out), dtype, device));
            l.loraScale = alpha / rank;
            added.add(l.loraA);
            added.add(l.loraB);
        }
        if (added.isEmpty()) {
            throw new IllegalArgumentException("no projection matches the target");
        }
        return added;
    }

    /** The prefix PEFT puts before a causal language model's module names. */
    private static final String PEFT_PREFIX = "base_model.model.";

    /**
     * Loads a LoRA adapter saved by Hugging Face PEFT ({@code adapter_config.json} and
     * {@code adapter_model.safetensors}), as {@link #addLora} would have made it: the base
     * weights become constants and the returned adapters can be trained further, merged with
     * {@link #mergeLora}, or used as they are.
     *
     * @param dir the adapter's directory
     * @return the adapter parameters
     */
    public List<Var> loadLora(Path dir) {
        Map<String, Object> cfg;
        try {
            cfg = Json.object(Json.read(dir.resolve("adapter_config.json")));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (!"LORA".equals(cfg.get("peft_type"))) {
            throw new IllegalArgumentException("not a LoRA adapter: peft_type " + cfg.get("peft_type"));
        }
        for (String unsupported : List.of("rank_pattern", "alpha_pattern")) {
            if (cfg.get(unsupported) instanceof Map<?, ?> m && !m.isEmpty()) {
                throw new IllegalArgumentException(unsupported + " is not supported");
            }
        }
        if (Boolean.TRUE.equals(cfg.get("use_dora"))) {
            throw new IllegalArgumentException("DoRA adapters are not supported");
        }
        int rank = (int) Json.integer(cfg, "r");
        double alpha = Json.number(cfg, "lora_alpha", rank);
        float scale = (float) (Boolean.TRUE.equals(cfg.get("use_rslora")) ? alpha / Math.sqrt(rank) : alpha / rank);
        List<Var> added = new ArrayList<>();
        try (io.github.zzuegg.brainiac.io.SafeTensors st =
                io.github.zzuegg.brainiac.io.SafeTensors.open(dir.resolve("adapter_model.safetensors"))) {
            for (String name : st.names()) {
                if (!name.endsWith(".lora_A.weight")) {
                    continue;
                }
                String module = name.substring(0, name.length() - ".lora_A.weight".length());
                if (module.startsWith(PEFT_PREFIX)) {
                    module = module.substring(PEFT_PREFIX.length());
                }
                Linear l = linears.get(module);
                if (l == null) {
                    throw new IllegalArgumentException("the adapter changes " + module + ", which this model does not have");
                }
                Device d = l.weight.value().device();
                DType t = l.weight.value().isPacked() ? dtype : l.weight.value().dtype();
                l.loraA = Var.param(inOut(st.tensor(name, d, t)));
                l.loraB = Var.param(inOut(st.tensor(name.replace(".lora_A.", ".lora_B."), d, t)));
                l.loraScale = scale;
                added.add(l.loraA);
                added.add(l.loraB);
            }
        }
        if (added.isEmpty()) {
            throw new IllegalArgumentException(dir + " holds no LoRA weights");
        }
        frozen = true;
        for (Linear l : linears.values()) {
            l.frozen = true;
        }
        return added;
    }

    /**
     * Saves the adapters as Hugging Face PEFT does, so PEFT, vLLM and llama.cpp's converter can
     * load them. Every adapter must share one rank and scale, as those made by one
     * {@link #addLora} call do.
     *
     * @param dir the directory, created if needed
     */
    public void saveLora(Path dir) {
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        java.util.Set<String> targets = new java.util.TreeSet<>();
        int rank = -1;
        float scale = Float.NaN;
        for (Map.Entry<String, Linear> e : linears.entrySet()) {
            Linear l = e.getValue();
            if (l.loraA == null) {
                continue;
            }
            int r = l.loraA.shape().dim(1);
            if (rank >= 0 && (r != rank || l.loraScale != scale)) {
                throw new IllegalStateException("adapters of different ranks or scales cannot be saved together");
            }
            rank = r;
            scale = l.loraScale;
            tensors.put(PEFT_PREFIX + e.getKey() + ".lora_A.weight", l.loraA.value().transpose());
            tensors.put(PEFT_PREFIX + e.getKey() + ".lora_B.weight", l.loraB.value().transpose());
            targets.add(e.getKey().substring(e.getKey().lastIndexOf('.') + 1));
        }
        if (tensors.isEmpty()) {
            throw new IllegalStateException("the model has no adapters");
        }
        Map<String, Object> cfg = new LinkedHashMap<>();
        cfg.put("peft_type", "LORA");
        cfg.put("task_type", "CAUSAL_LM");
        cfg.put("r", (long) rank);
        cfg.put("lora_alpha", (double) scale * rank);
        cfg.put("lora_dropout", 0.0);
        cfg.put("bias", "none");
        cfg.put("fan_in_fan_out", false);
        cfg.put("use_rslora", false);
        cfg.put("target_modules", List.copyOf(targets));
        cfg.put("base_model_name_or_path", null);
        try {
            Files.createDirectories(dir);
            Files.writeString(dir.resolve("adapter_config.json"), Json.write(cfg), StandardCharsets.UTF_8);
            io.github.zzuegg.brainiac.io.SafeTensors.write(dir.resolve("adapter_model.safetensors"), tensors,
                    Map.of("format", "pt"));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return a target for {@link #addLora}: every attention and MLP projection, not the output layer */
    public static java.util.function.Predicate<String> attentionAndMlp() {
        return name -> name.contains(".self_attn.") || name.contains(".mlp.");
    }

    /** @return the adapters by name ({@code <module>.lora_A}, {@code <module>.lora_B}), in model order */
    public Map<String, Var> loraParameters() {
        Map<String, Var> out = new LinkedHashMap<>();
        for (Map.Entry<String, Linear> e : linears.entrySet()) {
            if (e.getValue().loraA != null) {
                out.put(e.getKey() + ".lora_A", e.getValue().loraA);
                out.put(e.getKey() + ".lora_B", e.getValue().loraB);
            }
        }
        return out;
    }

    /**
     * Adds every adapter into its weight ({@code W + (alpha / rank) A B}) and removes it, so
     * the model runs as fast as without adapters. Quantized weights cannot take them.
     */
    public void mergeLora() {
        for (Map.Entry<String, Linear> e : linears.entrySet()) {
            Linear l = e.getValue();
            if (l.loraA == null) {
                continue;
            }
            if (l.weight.value().isPacked()) {
                throw new IllegalStateException(e.getKey() + " is quantized: adapters cannot be merged into it");
            }
            Tensor delta = l.loraA.value().matmul(l.loraB.value()).mul(l.loraScale);
            l.weight = Var.param(l.weight.value().add(delta.to(l.weight.value().dtype())));
            named.put(e.getKey() + ".weight", l.weight);
            l.loraA = null;
            l.loraB = null;
        }
        headSource = null;
    }

    /**
     * Joins each block's q, k and v projections into one matrix, and its gate and up
     * projections into another, so that each block runs two products where it ran five: the
     * same results, faster on a GPU, where small products cost more in overhead than in
     * arithmetic. The weights then go by the names {@code self_attn.qkv_proj} and
     * {@code mlp.gate_up_proj} (for {@link #addLora} targets and {@link #namedParameters()}), and
     * the model can no longer be saved. Packed projections (GGUF, {@link #quantizeWeights})
     * join when they share bits and group; projections with adapters stay as they are, and a
     * mixture of experts joins only q, k and v.
     *
     * @return this model
     */
    public Llama fuseProjections() {
        if (fused == null) {
            fused = new Fused[blocks.size()];
        }
        for (int i = 0; i < blocks.size(); i++) {
            Block b = blocks.get(i);
            String p = "model.layers." + i + ".";
            Fused f = fused[i] == null ? new Fused(null, null) : fused[i];
            Linear qkv = f.qkv() != null ? f.qkv()
                    : join(p + "self_attn.", List.of("q_proj", "k_proj", "v_proj"), "qkv_proj",
                            List.of(b.q, b.k, b.v));
            Linear gateUp = f.gateUp() != null || b.gate == null ? f.gateUp()   // experts have no single gate
                    : join(p + "mlp.", List.of("gate_proj", "up_proj"), "gate_up_proj", List.of(b.gate, b.up));
            fused[i] = new Fused(qkv, gateUp);
        }
        return this;
    }

    /** One projection from several with the same input, or null if they cannot be joined. */
    private Linear join(String prefix, List<String> names, String joined, List<Linear> parts) {
        boolean biased = parts.getFirst().bias != null;
        boolean packed = parts.getFirst().weight != null && parts.getFirst().weight.value().isPacked();
        for (Linear l : parts) {
            if (l.weight == null || l.weight.value().isPacked() != packed || l.loraA != null
                    || (l.bias != null) != biased) {
                return null;
            }
        }
        Tensor[] w = new Tensor[parts.size()];
        Tensor[] bias = new Tensor[parts.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = parts.get(i).weight.value();
            bias[i] = biased ? parts.get(i).bias.value() : null;
        }
        Tensor joinedWeight = Tensor.concat(1, w);
        if (packed && !joinedWeight.isPacked()) {
            joinedWeight.release();   // different formats (a k-quant mix): joined only as floats
            return null;
        }
        Var weight = Var.param(joinedWeight);
        Var b = biased ? Var.param(Tensor.concat(0, bias)) : null;
        Tensor.evaluate(b == null ? List.of(weight.value()) : List.of(weight.value(), b.value()));
        for (int i = 0; i < w.length; i++) {
            w[i].release();
            if (biased) {
                bias[i].release();
            }
            Linear old = linears.remove(prefix + names.get(i));
            old.weight = null;
            named.remove(prefix + names.get(i) + ".weight");
            named.remove(prefix + names.get(i) + ".bias");
        }
        Linear l = new Linear(weight, b);
        l.frozen = frozen;
        linears.put(prefix + joined, l);
        named.put(prefix + joined + ".weight", weight);
        if (b != null) {
            named.put(prefix + joined + ".bias", b);
        }
        return l;
    }

    /**
     * Quantizes every projection's weight in place to {@code bits} bits with one scale per
     * {@code group} input rows (see {@link Tensor#quantized}), for inference or QLoRA. The
     * model can then no longer be saved as safetensors, and those weights no longer train.
     *
     * @param bits the bits per weight, 2 to 8
     * @param group the input rows per scale; must divide every projection's input size
     * @return this model
     */
    public Llama quantizeWeights(int bits, int group) {
        for (Map.Entry<String, Linear> e : linears.entrySet()) {
            Linear l = e.getValue();
            Tensor w = l.weight.value();
            if (w.isPacked()) {
                continue;
            }
            Tensor q = w.to(DType.FLOAT32).quantized(bits, group);
            w.release();
            l.weight = Var.of(q);
            named.remove(e.getKey() + ".weight");
        }
        return this;
    }

    // ---------------------------------------------------------------- saving

    /**
     * Saves the model as a Hugging Face model directory: {@code config.json} and {@code
     * model.safetensors}, readable by {@link #load} and by transformers. Models with packed
     * GGUF matrices cannot be saved this way.
     *
     * @param dir the directory, created if needed
     */
    public void save(Path dir) {
        if (fused != null && !config.extras().fused()) {
            throw new IllegalStateException("a model with fused projections cannot be saved");
        }
        for (Linear l : linears.values()) {
            if (l.weight.value().isPacked()) {
                throw new IllegalStateException("packed weights cannot be saved as safetensors");
            }
        }
        try {
            Files.createDirectories(dir);
            Files.writeString(dir.resolve("config.json"), Json.write(config.toHuggingFace()), StandardCharsets.UTF_8);
            Map<String, Tensor> tensors = new LinkedHashMap<>();
            java.util.Set<String> matrices = new java.util.HashSet<>();
            for (String l : linearNames(config)) {
                matrices.add(l + ".weight");
            }
            for (Map.Entry<String, Var> e : named.entrySet()) {
                Tensor t = e.getValue().value();
                if (config.extras().normOffset() && e.getKey().endsWith("norm.weight")) {
                    t = t.sub(1f);   // Gemma's files keep w for a scale of 1 + w
                }
                tensors.put(e.getKey(), matrices.contains(e.getKey()) ? t.transpose() : t);
            }
            SafeTensors.write(dir.resolve("model.safetensors"), tensors, Map.of("format", "pt"));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Writes the model as a GGUF file that llama.cpp, Ollama and LM Studio can run, and
     * {@link #load} reads back: llama.cpp's tensor names and rotary layout, float16 or float32
     * matrices (packed ones unpacked), and the tokenizer's {@code tokenizer.ggml.*} metadata
     * when one is given. Llama 3 rope scaling goes in as llama.cpp's {@code rope_freqs}.
     *
     * @param file the {@code .gguf} file to write
     * @param tokenizer the model's tokenizer, or {@code null} to leave it out (llama.cpp
     *     needs one)
     * @param half whether matrices are stored as float16
     */
    public void saveGguf(Path file, PretrainedTokenizer tokenizer, boolean half) {
        GgufWriter.Type t = half ? GgufWriter.Type.F16 : GgufWriter.Type.F32;
        saveGguf(file, tokenizer, name -> t);
    }

    /**
     * Writes the model as a quantized GGUF file, as {@code llama-quantize} would: matrices in
     * {@code type}, norms and biases in float32, and the expert router in float32. With a
     * 4-bit type the output layer (the embedding, when the two are tied) goes in Q6_K, or Q8_0
     * where its rows do not fit Q6_K's blocks; with {@link GgufWriter.Type#Q4_K Q4_K} the value
     * and down projections do too, after llama.cpp's Q4_K_M mix.
     *
     * @param file the {@code .gguf} file to write
     * @param tokenizer the model's tokenizer, or {@code null}
     * @param type the storage type of the matrices
     */
    public void saveGguf(Path file, PretrainedTokenizer tokenizer, GgufWriter.Type type) {
        saveGguf(file, tokenizer, name -> {
            if (name.contains("ffn_gate_inp")) {
                return GgufWriter.Type.F32;
            }
            boolean four = type == GgufWriter.Type.Q4_0 || type == GgufWriter.Type.Q4_K;
            // the output layer (the embedding, when tied) stays at 6 bits, as llama-quantize keeps it
            boolean output = name.equals("output.weight") || lmHead == null && name.equals("token_embd.weight");
            if (four && output || type == GgufWriter.Type.Q4_K && (name.contains("attn_v.")
                    || name.contains("ffn_down"))) {
                return GgufWriter.Type.Q6_K;
            }
            return type;
        });
    }

    private void saveGguf(Path file, PretrainedTokenizer tokenizer,
            java.util.function.Function<String, GgufWriter.Type> types) {
        if (fused != null && !config.extras().fused()) {
            throw new IllegalStateException("a model with fused projections cannot be saved");
        }
        String a = switch (config.architecture()) {
            case "qwen2", "qwen3", "phi3", "gemma", "gemma2" -> config.architecture();
            case "gemma3_text" -> "gemma3";
            case "qwen2_moe" -> "qwen2moe";
            case "qwen3_moe" -> "qwen3moe";
            default -> "llama";   // Mixtral too
        };
        LlamaConfig.Extras x = config.extras();
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("general.architecture", a);
        m.put("general.name", "brainiac-" + a);
        m.put(a + ".context_length", (long) config.maxPositions());
        m.put(a + ".embedding_length", (long) config.hidden());
        m.put(a + ".block_count", (long) config.layers());
        m.put(a + ".feed_forward_length", (long) config.intermediate());
        m.put(a + ".attention.head_count", (long) config.heads());
        m.put(a + ".attention.head_count_kv", (long) config.kvHeads());
        m.put(a + ".attention.layer_norm_rms_epsilon", config.rmsEps());
        m.put(a + ".rope.freq_base", (float) config.rope().theta());
        m.put(a + ".rope.dimension_count", (long) config.rope().dim());
        m.put(a + ".attention.key_length", (long) config.headDim());
        m.put(a + ".attention.value_length", (long) config.headDim());
        m.put(a + ".vocab_size", (long) config.vocabSize());
        if (x.slidingWindow() > 0) {
            m.put(a + ".attention.sliding_window", (long) x.slidingWindow());
            if (x.globalEvery() > 0) {
                m.put(a + ".attention.sliding_window_pattern", (long) x.globalEvery());
            }
        }
        if (x.localTheta() > 0) {
            m.put(a + ".rope.freq_base_swa", (float) x.localTheta());
        }
        if (x.queryScale() > 0) {
            m.put(a + ".attention.scale", x.queryScale());
        }
        if (x.attentionSoftcap() > 0) {
            m.put(a + ".attn_logit_softcapping", x.attentionSoftcap());
        }
        if (x.finalSoftcap() > 0) {
            m.put(a + ".final_logit_softcapping", x.finalSoftcap());
        }
        LlamaConfig.Moe moe = x.moe();
        if (moe.present()) {
            m.put(a + ".expert_count", (long) moe.experts());
            m.put(a + ".expert_used_count", (long) moe.perToken());
            m.put(a + ".expert_feed_forward_length", (long) moe.ffn());
            if (moe.sharedFfn() > 0) {
                m.put(a + ".expert_shared_feed_forward_length", (long) moe.sharedFfn());
            }
            if (!a.equals("llama")) {
                m.put(a + ".expert_weights_norm", moe.normalize());
            }
        }
        Rope.Settings r = config.rope();
        Map<String, Tensor> t = new LinkedHashMap<>();
        switch (r.type()) {
            case "default" -> {
            }
            case "longrope" -> {
                m.put(a + ".rope.scaling.original_context_length", (long) r.originalMaxPositions());
                m.put(a + ".rope.scaling.attn_factor", (float) rope.magnitude());
                t.put("rope_factors_short.weight", Tensor.of(Shape.of(r.freqFactors().length), r.freqFactors()));
                t.put("rope_factors_long.weight", Tensor.of(Shape.of(r.freqFactors().length), r.freqFactors()));
            }
            case "linear", "yarn" -> {
                m.put(a + ".rope.scaling.type", r.type());
                m.put(a + ".rope.scaling.factor", (float) r.factor());
                m.put(a + ".rope.scaling.original_context_length", (long) r.originalMaxPositions());
                if (r.type().equals("yarn")) {
                    m.put(a + ".rope.scaling.yarn_beta_fast", (float) r.betaFast());
                    m.put(a + ".rope.scaling.yarn_beta_slow", (float) r.betaSlow());
                }
            }
            default -> {
                // llama3 and others: the divisors from plain frequencies to the scaled ones
                double[] plain = new Rope(new Rope.Settings(r.dim(), r.theta(), "default", 1, 1, 4,
                        r.originalMaxPositions(), null)).frequencies();
                double[] scaled = rope.frequencies();
                float[] f = new float[plain.length];
                for (int i = 0; i < f.length; i++) {
                    f[i] = (float) (plain[i] / scaled[i]);
                }
                t.put("rope_freqs.weight", Tensor.of(Shape.of(f.length), f));
            }
        }
        if (tokenizer != null) {
            m.putAll(tokenizer.ggufMetadata());
        }
        t.put("token_embd.weight", plain(embed.value()));
        t.put("output_norm.weight", plain(norm.value()));
        if (lmHead != null) {
            t.put("output.weight", plain(lmHead.weight.value()).transpose());
        }
        boolean permuted = a.equals("llama");
        Map<String, String> parts = ggufParts(config);
        for (int i = 0; i < config.layers(); i++) {
            String p = "model.layers." + i + ".";
            String g = "blk." + i + ".";
            for (Map.Entry<String, String> n : ggufNorms(config).entrySet()) {
                t.put(g + n.getValue() + ".weight", plain(named.get(p + n.getKey() + ".weight").value()));
            }
            for (Map.Entry<String, String> e : parts.entrySet()) {
                Linear l = linears.get(p + e.getKey());
                int heads = e.getValue().equals("attn_q") ? config.heads()
                        : e.getValue().equals("attn_k") ? config.kvHeads() : 0;
                Tensor w = plain(l.weight.value()).transpose();   // [out, in], as llama.cpp keeps it
                Tensor bias = l.bias == null ? null : plain(l.bias.value());
                if (permuted && heads > 0) {
                    int[] order = unpermute(heads, config.headDim());
                    int[] inverse = new int[order.length];
                    for (int k = 0; k < order.length; k++) {
                        inverse[order[k]] = k;
                    }
                    Tensor idx = indices(inverse, w.device());
                    w = w.gather(0, idx);
                    bias = bias == null ? null : bias.gather(0, idx);
                }
                t.put(g + e.getValue() + ".weight", w);
                if (bias != null) {
                    t.put(g + e.getValue() + ".bias", bias);
                }
            }
            Experts ex = blocks.get(i).experts;
            if (ex != null) {
                // as llama.cpp: the router, then each projection's experts in one [experts, out, in] stack
                t.put(g + "ffn_gate_inp.weight", plain(ex.router.weight.value()).transpose());
                Linear[][] all = {ex.gate, ex.up, ex.down};
                String[] stored = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
                for (int k = 0; k < 3; k++) {
                    Tensor[] each = new Tensor[all[k].length];
                    for (int e = 0; e < each.length; e++) {
                        Tensor w = plain(all[k][e].weight.value()).transpose();
                        each[e] = w.reshape(1, w.shape().dim(0), w.shape().dim(1));
                    }
                    t.put(g + stored[k] + ".weight", Tensor.concat(0, each));
                }
                if (ex.sharedGate != null) {
                    t.put(g + "ffn_gate_shexp.weight", plain(ex.sharedGate.weight.value()).transpose());
                    t.put(g + "ffn_up_shexp.weight", plain(ex.sharedUp.weight.value()).transpose());
                    t.put(g + "ffn_down_shexp.weight", plain(ex.sharedDown.weight.value()).transpose());
                    t.put(g + "ffn_gate_inp_shexp.weight", plain(ex.sharedScale.weight.value()).reshape(config.hidden()));
                }
            }
        }
        GgufWriter.write(file, m, t, types);
    }

    /** Values a writer can read: packed matrices unpacked. */
    private static Tensor plain(Tensor t) {
        return t.isPacked() ? t.unpacked() : t;
    }

    @Override
    public String toString() {
        return "Llama[" + config.architecture() + ", " + config.layers() + " layers, " + config.hidden() + " hidden, "
                + config.heads() + "/" + config.kvHeads() + " heads, " + dtype + " on " + device + "]";
    }
}
