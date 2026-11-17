package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.SafeTensors;
import io.github.zzuegg.brainiac.io.WeightSource;
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
        private final Var bias;
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
            Linear down, Var qNorm, Var kNorm, Var postAttnNorm, Var postFfnNorm) {
    }

    private final LlamaConfig config;
    private final Device device;
    private final DType dtype;
    private final Rope rope;
    /** The rotation of local (sliding-window) layers when it differs, or null. */
    private final Rope localRope;
    private final Var embed;
    private final Linear lmHead;
    private final Var norm;
    private final List<Block> blocks;
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
                    post ? need(named, p + "post_feedforward_layernorm.weight") : null));
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
            for (String s : c.extras().fused()
                    ? List.of("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
                    : List.of("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")) {
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
        boolean permuted = config.architecture().equals("llama");
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
            int[] shape = l.endsWith("qkv_proj") ? new int[] {qOut + 2 * kvOut, config.hidden()}
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
        void reserve(int steps, int kvHeads, int headDim, DType dtype, Device device) {
            int need = length + steps;
            if (need <= capacity) {
                return;
            }
            int grown = Math.max(need, Math.max(64, 2 * capacity));
            int batch = next.length;
            Shape shape = Shape.of(batch, kvHeads, grown, headDim);
            for (int l = 0; l < keys.length; l++) {
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
                Tensor index = indices(rows, keys[0].device());
                for (int l = 0; l < keys.length; l++) {
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
                    c.keys[l] = Tensor.concat(0, padded(keys[l], next.length, length, len, cap, any),
                            padded(o.keys[l], o.next.length, o.length, len, cap, any));
                    c.values[l] = Tensor.concat(0, padded(values[l], next.length, length, len, cap, any),
                            padded(o.values[l], o.next.length, o.length, len, cap, any));
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
        return run(tokens, positions, null, null);
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
        Var h = hidden(tokens, positions, null, null).reshape(batch * steps, config.hidden());
        Tensor ids = Tensor.of(Shape.of(batch * steps), next, device);
        if (config.extras().finalSoftcap() > 0) {
            return Functional.crossEntropy(softcap(head(h, false)).reshape(batch * steps, config.vocabSize()), ids);
        }
        if (lmHead != null && lmHead.bias() != null) {
            return Functional.crossEntropy(lmHead.apply(h).reshape(batch * steps, config.vocabSize()), ids);
        }
        if (lmHead != null && lmHead.loraA != null) {
            return Functional.crossEntropy(lmHead.apply(h).reshape(batch * steps, config.vocabSize()), ids);
        }
        Var weight = lmHead != null ? lmHead.base() : base(embed).transpose();
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
        cache.reserve(steps, config.kvHeads(), config.headDim(), dtype, device);
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
        // causal, padding and the buffer's unused end: [batch, 1, steps, capacity]; one per window
        Tensor[] masks = new Tensor[blocks.size()];
        Map<Integer, Tensor> byWindow = new java.util.HashMap<>();
        for (int l = 0; l < masks.length; l++) {
            masks[l] = byWindow.computeIfAbsent(windowOf(l), w -> cacheMask(cache, keysPad, steps, w));
        }
        Var[] hidden = new Var[1];
        Var logits = run(tokens, positions, cache, masks, hidden);
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

    private Tensor cacheMask(Cache cache, boolean[][] keysPad, int steps, int window) {
        int batch = keysPad.length;
        int keys = cache.capacity;
        float[] m = new float[batch * steps * keys];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                int row = (b * steps + t) * keys;
                for (int k = 0; k < keys; k++) {
                    if (k > cache.length + t || keysPad[b][Math.min(k, keysPad[b].length - 1)]
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
        Var x = base(embed).gather(0, indices(flat, device)).reshape(batch, steps, config.hidden());
        if (e.scaleEmbeddings()) {
            // as Gemma: the factor rounded to the model's type first
            x = x.mul(Tensor.full(Shape.of(), (float) Math.sqrt(config.hidden()), Device.cpu()).to(dtype).item());
        }
        if (masks == null) {
            masks = new Tensor[blocks.size()];
            Map<Integer, Tensor> byWindow = new java.util.HashMap<>();
            for (int l = 0; l < blocks.size(); l++) {
                int window = windowOf(l);
                if (window > 0 && steps > window) {
                    masks[l] = byWindow.computeIfAbsent(window, w -> mask(steps, w));
                }
            }
        }
        int hd = config.headDim();
        int rotary = config.rope().dim();
        float scale = e.queryScale() > 0 ? e.queryScale() : (float) (1 / Math.sqrt(hd));
        boolean decoding = cache != null;
        for (int l = 0; l < blocks.size(); l++) {
            Block blk = blocks.get(l);
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
                bias = Functional.causalMask(steps, k.shape().dim(2), device).to(dtype);
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
        }
        return rmsNorm(x, norm, decoding);
    }

    /** Rotates the first {@code rotary} features of each head; the rest pass (partial rotary, Phi). */
    private static Var rotate(Rope r, Var x, int[][] positions, int rotary, int hd) {
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

    /** A causal mask over {@code steps} steps where each sees only the last {@code window}. */
    private Tensor mask(int steps, int window) {
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
     * join when they share bits and group; projections with adapters stay as they are.
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
            Linear gateUp = f.gateUp() != null ? f.gateUp()
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
        if (fused != null && !config.extras().fused()) {
            throw new IllegalStateException("a model with fused projections cannot be saved");
        }
        String a = switch (config.architecture()) {
            case "qwen2", "qwen3", "phi3", "gemma", "gemma2" -> config.architecture();
            case "gemma3_text" -> "gemma3";
            default -> "llama";
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
        }
        io.github.zzuegg.brainiac.io.GgufWriter.write(file, m, t, half);
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
