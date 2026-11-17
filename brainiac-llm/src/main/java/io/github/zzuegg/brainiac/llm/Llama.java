package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.Json;
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

    private record Block(Var attnNorm, Linear q, Linear k, Linear v, Linear o, Var ffnNorm, Linear gate, Linear up,
            Linear down) {
    }

    private final LlamaConfig config;
    private final Device device;
    private final DType dtype;
    private final Rope rope;
    private final Var embed;
    private final Linear lmHead;
    private final Var norm;
    private final List<Block> blocks;
    private final Map<String, Var> named;
    private final Map<String, Linear> linears;
    /** Whether the embeddings and norms are constants (see {@link #addLora}). */
    private boolean frozen;

    private Llama(LlamaConfig config, Device device, DType dtype, Map<String, Var> named, Map<String, Linear> linears) {
        this.config = config;
        this.device = device;
        this.dtype = dtype;
        this.rope = new Rope(config.rope());
        this.named = named;
        this.linears = linears;
        this.embed = need(named, "model.embed_tokens.weight");
        this.norm = need(named, "model.norm.weight");
        this.lmHead = config.tieEmbeddings() ? null : linears.get("lm_head");
        this.blocks = new ArrayList<>();
        for (int i = 0; i < config.layers(); i++) {
            String p = "model.layers." + i + ".";
            blocks.add(new Block(need(named, p + "input_layernorm.weight"), linears.get(p + "self_attn.q_proj"),
                    linears.get(p + "self_attn.k_proj"), linears.get(p + "self_attn.v_proj"),
                    linears.get(p + "self_attn.o_proj"), need(named, p + "post_attention_layernorm.weight"),
                    linears.get(p + "mlp.gate_proj"), linears.get(p + "mlp.up_proj"), linears.get(p + "mlp.down_proj")));
        }
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
            for (String s : List.of("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
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
        read.apply("model.norm.weight");
        for (int i = 0; i < config.layers(); i++) {
            read.apply("model.layers." + i + ".input_layernorm.weight");
            read.apply("model.layers." + i + ".post_attention_layernorm.weight");
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
            config = new LlamaConfig(config.architecture(), config.vocabSize(), config.hidden(),
                    config.intermediate(), config.layers(), config.heads(), config.kvHeads(), config.headDim(),
                    config.rmsEps(), config.rope().withFreqFactors(f), config.tieEmbeddings(),
                    config.attentionBias(), config.maxPositions());
        }
        // llama.cpp's converter interleaves the two rotary halves of Llama's query and key rows
        boolean permuted = config.architecture().equals("llama");
        Map<String, Var> named = new LinkedHashMap<>();
        Map<String, Linear> linears = new LinkedHashMap<>();
        named.put("model.embed_tokens.weight", Var.param(gguf.tensor("token_embd.weight", device, dtype)));
        named.put("model.norm.weight", Var.param(gguf.tensor("output_norm.weight", device, dtype)));
        Map<String, String> parts = Map.of("self_attn.q_proj", "attn_q", "self_attn.k_proj", "attn_k",
                "self_attn.v_proj", "attn_v", "self_attn.o_proj", "attn_output", "mlp.gate_proj", "ffn_gate",
                "mlp.up_proj", "ffn_up", "mlp.down_proj", "ffn_down");
        for (int i = 0; i < config.layers(); i++) {
            String p = "model.layers." + i + ".";
            String g = "blk." + i + ".";
            named.put(p + "input_layernorm.weight", Var.param(gguf.tensor(g + "attn_norm.weight", device, dtype)));
            named.put(p + "post_attention_layernorm.weight",
                    Var.param(gguf.tensor(g + "ffn_norm.weight", device, dtype)));
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
            for (String n : List.of("input_layernorm", "post_attention_layernorm")) {
                named.put("model.layers." + i + "." + n + ".weight",
                        Var.param(Tensor.ones(Shape.of(config.hidden())).to(device).to(dtype)));
            }
        }
        int qOut = config.heads() * config.headDim();
        int kvOut = config.kvHeads() * config.headDim();
        // residual projections scaled down by depth, as GPT-2 does
        float residual = (float) (std / Math.sqrt(2.0 * config.layers()));
        for (String l : linearNames(config)) {
            int[] shape = l.endsWith("q_proj") ? new int[] {qOut, config.hidden()}
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

        Cache(int layers, int batch) {
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
            for (int l = 0; l < keys.length; l++) {
                Tensor k = Tensor.zeros(Shape.of(batch, kvHeads, grown, headDim), dtype, device);
                Tensor v = Tensor.zeros(Shape.of(batch, kvHeads, grown, headDim), dtype, device);
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

        @Override
        public State truncate(int keep) {
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
            Cache c = new Cache(keys.length, rows.length);
            if (capacity > 0) {
                Tensor index = indices(rows, keys[0].device());
                for (int l = 0; l < keys.length; l++) {
                    c.keys[l] = keys[l].gather(0, index);
                    c.values[l] = values[l].gather(0, index);
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
    }

    @Override
    public State newState(int batch) {
        return new Cache(config.layers(), batch);
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

    private Tensor read(State state, int[][] tokens, boolean[][] padding, boolean everyStep) {
        Cache cache = (Cache) Objects.requireNonNull(state, "state");
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
        // causal, padding and the buffer's unused end: [batch, 1, steps, capacity]
        int keys = cache.capacity;
        float[] m = new float[batch * steps * keys];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < steps; t++) {
                int row = (b * steps + t) * keys;
                for (int k = 0; k < keys; k++) {
                    if (k > cache.length + t || keysPad[b][Math.min(k, keysPad[b].length - 1)]) {
                        m[row + k] = -1e9f;
                    }
                }
            }
        }
        Tensor bias = Tensor.of(Shape.of(batch, 1, steps, keys), m, device).to(dtype);
        Var logits = run(tokens, positions, cache, bias);
        cache.padding = keysPad;
        cache.length += steps;
        Tensor out = everyStep ? logits.value().to(DType.FLOAT32)
                : logits.value().slice(1, steps - 1, steps).reshape(batch, config.vocabSize()).to(DType.FLOAT32);
        // computed before the next step writes into the cache it read
        Tensor.evaluate(List.of(out));
        return out;
    }

    private Var run(int[][] tokens, int[][] positions, Cache cache, Tensor bias) {
        Var x = hidden(tokens, positions, cache, bias);
        if (lmHead != null) {
            return lmHead.apply(x);
        }
        // tied: while decoding, a transposed copy of the embeddings made once; while
        // training, through the embeddings so they get the gradient
        return x.matmul(cache != null ? Var.of(tiedHead()) : base(embed).transpose());
    }

    /** The final normalized hidden states {@code [batch, steps, hidden]}. */
    private Var hidden(int[][] tokens, int[][] positions, Cache cache, Tensor bias) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        int[] flat = new int[batch * steps];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(tokens[b], 0, flat, b * steps, steps);
        }
        Var x = base(embed).gather(0, indices(flat, device)).reshape(batch, steps, config.hidden());
        int hd = config.headDim();
        float scale = (float) (1 / Math.sqrt(hd));
        for (int l = 0; l < blocks.size(); l++) {
            Block blk = blocks.get(l);
            Var h = rmsNorm(x, blk.attnNorm, cache != null);
            Var q = blk.q.apply(h).reshape(batch, steps, config.heads(), hd).permute(0, 2, 1, 3);
            Var k = blk.k.apply(h).reshape(batch, steps, config.kvHeads(), hd).permute(0, 2, 1, 3);
            Var v = blk.v.apply(h).reshape(batch, steps, config.kvHeads(), hd).permute(0, 2, 1, 3);
            q = rope.apply(q, positions);
            k = rope.apply(k, positions);
            boolean causal = steps > 1;
            if (cache != null) {
                cache.keys[l].write(2, cache.length, k.value());
                cache.values[l].write(2, cache.length, v.value());
                k = Var.of(cache.keys[l]);
                v = Var.of(cache.values[l]);
                q = Var.of(q.value());   // decoding has no gradients: lets the fused kernel take the mask
            }
            Var a = bias != null ? Functional.attention(q, k, v, bias, scale, 0, null)
                    : Functional.attention(q, k, v, causal, scale);
            a = a.permute(0, 2, 1, 3).reshape(batch, steps, config.heads() * hd);
            x = x.add(blk.o.apply(a));
            h = rmsNorm(x, blk.ffnNorm, cache != null);
            // both products first: the activation and the product then run as one chain
            Var gate = blk.gate.apply(h);
            Var up = blk.up.apply(h);
            Var gated = gate.activation(ActivationOp.SILU, 0).mul(up);
            x = x.add(blk.down.apply(gated));
        }
        return rmsNorm(x, norm, cache != null);
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
        int expected = 2 + 2 * config.layers() + linearNames(config).size()
                + (config.attentionBias() ? 3 * config.layers() : 0);
        if (named.size() != expected) {
            throw new IllegalStateException("packed weights cannot be saved as safetensors");
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
                tensors.put(e.getKey(), matrices.contains(e.getKey()) ? t.transpose() : t);
            }
            SafeTensors.write(dir.resolve("model.safetensors"), tensors, Map.of("format", "pt"));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public String toString() {
        return "Llama[" + config.architecture() + ", " + config.layers() + " layers, " + config.hidden() + " hidden, "
                + config.heads() + "/" + config.kvHeads() + " heads, " + dtype + " on " + device + "]";
    }
}
