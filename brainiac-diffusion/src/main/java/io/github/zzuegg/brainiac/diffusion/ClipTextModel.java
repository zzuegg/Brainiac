package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The CLIP text transformer that turns prompt tokens into the embeddings a diffusion model
 * attends to: OpenAI's CLIP ViT-L/14 (Stable Diffusion 1.x and SDXL's first encoder) or
 * OpenCLIP ViT-H and ViT-bigG (Stable Diffusion 2.x and SDXL's second encoder).
 */
public final class ClipTextModel {

    /**
     * The shape of a CLIP text model.
     *
     * @param vocabSize the number of tokens
     * @param hidden the width
     * @param layers the number of transformer layers
     * @param heads the number of attention heads
     * @param positions the longest prompt in tokens (77)
     * @param quickGelu whether the MLP uses {@code x sigmoid(1.702 x)} instead of GELU
     * @param eps the layer norm epsilon
     * @param eosId the end-of-text token; {@code 2} means "the largest id", as older configs say
     * @param projection the width of the text projection, or 0 if the model has none
     */
    public record Config(int vocabSize, int hidden, int layers, int heads, int positions, boolean quickGelu,
            float eps, int eosId, int projection) {

        /**
         * Reads a Hugging Face {@code CLIPTextConfig}.
         *
         * @param json the parsed {@code config.json}
         * @return the config
         */
        public static Config fromJson(Map<String, Object> json) {
            Map<String, Object> c = json.get("text_config") instanceof Map<?, ?> m ? Json.object(m) : json;
            return new Config((int) Json.integer(c, "vocab_size", 49408), (int) Json.integer(c, "hidden_size", 768),
                    (int) Json.integer(c, "num_hidden_layers", 12), (int) Json.integer(c, "num_attention_heads", 12),
                    (int) Json.integer(c, "max_position_embeddings", 77),
                    !"gelu".equals(Json.string(c, "hidden_act", "quick_gelu")),
                    (float) Json.number(c, "layer_norm_eps", 1e-5), (int) Json.integer(c, "eos_token_id", 2),
                    0);
        }

        Config withProjection(int width) {
            return new Config(vocabSize, hidden, layers, heads, positions, quickGelu, eps, eosId, width);
        }
    }

    /**
     * What the encoder returns for a batch of prompts.
     *
     * @param last the last layer after the final layer norm, {@code [batch, tokens, hidden]}
     * @param penultimate the second-to-last layer's output (no final norm), {@code [batch, tokens, hidden]}
     * @param pooled the final-normed state at each prompt's end token, projected if the model
     *     has a projection, {@code [batch, width]}
     */
    public record Output(Tensor last, Tensor penultimate, Tensor pooled) {
    }

    private record Layer(Nn.LayerNorm norm1, Nn.Linear q, Nn.Linear k, Nn.Linear v, Nn.Linear out,
            Nn.LayerNorm norm2, Nn.Linear fc1, Nn.Linear fc2) {
    }

    private final Config config;
    private final Tensor tokenEmbedding;
    private final Tensor positionEmbedding;
    private final List<Layer> layers;
    private final Nn.LayerNorm finalNorm;
    private final Nn.Linear projection;
    private final List<Tensor> weights;

    private ClipTextModel(Config config, Tensor tokenEmbedding, Tensor positionEmbedding, List<Layer> layers,
            Nn.LayerNorm finalNorm, Nn.Linear projection, List<Tensor> weights) {
        this.config = config;
        this.tokenEmbedding = tokenEmbedding;
        this.positionEmbedding = positionEmbedding;
        this.layers = layers;
        this.finalNorm = finalNorm;
        this.projection = projection;
        this.weights = weights;
    }

    /**
     * Loads the model in Hugging Face's naming ({@code text_model.encoder.layers.0...}).
     *
     * @param w the weights, at the model's prefix
     * @param config its config
     * @param layersNeeded how many layers to load; with fewer than the config has, the model
     *     behaves as if it had only these (Stable Diffusion 2 uses the penultimate layer this way)
     */
    static ClipTextModel load(Weights w, Config config, int layersNeeded) {
        if (layersNeeded < config.layers()) {
            config = new Config(config.vocabSize(), config.hidden(), layersNeeded, config.heads(), config.positions(),
                    config.quickGelu(), config.eps(), config.eosId(), config.projection());
        }
        Weights t = w.has("text_model.embeddings.token_embedding.weight") ? w.at("text_model.") : w;
        Tensor tokens = t.get("embeddings.token_embedding.weight");
        Tensor positions = t.get("embeddings.position_embedding.weight");
        List<Layer> layers = new ArrayList<>();
        for (int i = 0; i < config.layers(); i++) {
            Weights l = t.at("encoder.layers." + i + ".");
            layers.add(new Layer(l.layerNorm("layer_norm1", config.eps()), l.linear("self_attn.q_proj"),
                    l.linear("self_attn.k_proj"), l.linear("self_attn.v_proj"), l.linear("self_attn.out_proj"),
                    l.layerNorm("layer_norm2", config.eps()), l.linear("mlp.fc1"), l.linear("mlp.fc2")));
        }
        Nn.LayerNorm finalNorm = t.layerNorm("final_layer_norm", config.eps());
        Nn.Linear projection = null;
        Config c = config;
        if (w.has("text_projection.weight")) {
            projection = w.linear("text_projection");
            c = config.withProjection(projection.out());
        }
        return new ClipTextModel(c, tokens, positions, layers, finalNorm, projection, w.all);
    }

    /**
     * Loads an OpenCLIP text tower ({@code transformer.resblocks.0.attn.in_proj_weight}, ...),
     * as Stable Diffusion 2 and SDXL single-file checkpoints store it. Its shape is read from
     * the weights; heads are 64 wide.
     *
     * @param w the weights at the tower's prefix
     * @param skip how many of the last layers to leave out (see {@link #load}): 1 for Stable
     *     Diffusion 2, which conditions on the penultimate layer
     * @param headDim the width of an attention head (64)
     */
    static ClipTextModel loadOpenClip(Weights w, int skip, int headDim) {
        Shape embed = w.shape("token_embedding.weight");
        int hidden = embed.dim(1);
        int count = 0;
        while (w.has("transformer.resblocks." + count + ".ln_1.weight")) {
            count++;
        }
        int projection = w.has("text_projection") ? w.shape("text_projection").dim(1) : 0;
        Config config = new Config(embed.dim(0), hidden, count - skip, hidden / headDim,
                w.shape("positional_embedding").dim(0), false, 1e-5f, 2, projection);
        Tensor tokens = w.get("token_embedding.weight");
        Tensor positions = w.get("positional_embedding");
        List<Layer> layers = new ArrayList<>();
        for (int i = 0; i < config.layers(); i++) {
            Weights l = w.at("transformer.resblocks." + i + ".");
            String in = "attn.in_proj_weight";
            String inBias = "attn.in_proj_bias";
            layers.add(new Layer(l.layerNorm("ln_1", config.eps()), l.linearRows(in, inBias, 0, hidden),
                    l.linearRows(in, inBias, hidden, 2 * hidden), l.linearRows(in, inBias, 2 * hidden, 3 * hidden),
                    l.linear("attn.out_proj"), l.layerNorm("ln_2", config.eps()), l.linear("mlp.c_fc"),
                    l.linear("mlp.c_proj")));
        }
        Nn.LayerNorm finalNorm = w.layerNorm("ln_final", config.eps());
        Nn.Linear proj = projection > 0 ? new Nn.Linear(w.get("text_projection"), null) : null;
        return new ClipTextModel(config, tokens, positions, layers, finalNorm, proj, w.all);
    }

    /**
     * Returns the config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }

    /**
     * Encodes token ids.
     *
     * @param ids {@code [batch][tokens]} token ids, each row at most {@link Config#positions()} long
     * @return the hidden states and pooled embedding
     */
    public Output encode(int[][] ids) {
        int batch = ids.length;
        int n = ids[0].length;
        float[] flat = new float[batch * n];
        int[] eos = new int[batch];
        for (int b = 0; b < batch; b++) {
            int best = 0;
            for (int i = 0; i < n; i++) {
                flat[b * n + i] = ids[b][i];
                boolean end = config.eosId() == 2 ? ids[b][i] > ids[b][best] : ids[b][i] == config.eosId()
                        && ids[b][best] != config.eosId();
                if (end) {
                    best = i;
                }
            }
            eos[b] = best;
        }
        Tensor idx = Tensor.of(Shape.of(batch, n), flat, tokenEmbedding.device());
        Tensor x = Nn.scoped(weights, () -> tokenEmbedding.gather(0, idx)
                .add(positionEmbedding.slice(0, 0, n)));
        Tensor penultimate = null;
        for (int i = 0; i < layers.size(); i++) {
            Layer l = layers.get(i);
            Tensor in = x;
            x = Nn.scoped(weights, () -> block(l, in), in);
            if (in != penultimate) {
                in.release();
            }
            if (i == config.layers() - 2) {
                penultimate = x;
            }
        }
        Tensor h = x;
        Tensor last = Nn.scoped(weights, () -> finalNorm.apply(h), h);
        if (h != penultimate) {
            h.release();
        }
        Tensor pooled = Nn.scoped(weights, () -> {
            Tensor[] rows = new Tensor[batch];
            for (int b = 0; b < batch; b++) {
                rows[b] = last.slice(0, b, b + 1).slice(1, eos[b], eos[b] + 1).reshape(1, config.hidden());
            }
            Tensor p = Tensor.concat(0, rows);
            return projection == null ? p : projection.apply(p);
        }, last);
        return new Output(last, penultimate, pooled);
    }

    private Tensor block(Layer l, Tensor x) {
        Tensor h = l.norm1().apply(x);
        Tensor a = Nn.attention(l.q().apply(h), l.k().apply(h), l.v().apply(h), config.heads(), true);
        x = x.add(l.out().apply(a));
        Tensor m = l.fc1().apply(l.norm2().apply(x));
        m = config.quickGelu() ? Nn.quickGelu(m) : Nn.gelu(m);
        return x.add(l.fc2().apply(m));
    }

    /**
     * Returns the model's type.
     *
     * @return the type its weights are held in
     */
    public DType dtype() {
        return tokenEmbedding.dtype();
    }
}
