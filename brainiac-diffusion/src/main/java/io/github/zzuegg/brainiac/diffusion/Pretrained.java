package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.nn.diffusion.AutoencoderKL;
import io.github.zzuegg.brainiac.nn.diffusion.ClipTextEncoder;
import io.github.zzuegg.brainiac.nn.diffusion.FluxTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.ImageAttention;
import io.github.zzuegg.brainiac.nn.diffusion.ResnetBlock;
import io.github.zzuegg.brainiac.nn.diffusion.SpatialTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.nn.diffusion.UNet;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Builds brainiac's diffusion layers ({@link UNet}, {@link AutoencoderKL},
 * {@link ClipTextEncoder}, {@link FluxTransformer}, {@link T5Encoder}) from Hugging Face configs and fills them with checkpoint weights in
 * diffusers' and transformers' naming.
 */
final class Pretrained {

    private Pretrained() {
    }

    /** Builds a layer without spending memory on initial weights, which loading replaces. */
    private static <L> L shell(Supplier<L> make) {
        return Tensor.withoutValues(make);
    }

    // ------------------------------------------------------------------------------- configs

    /** Reads diffusers' {@code unet/config.json}. */
    static UNet.Settings unetSettings(Map<String, Object> c) {
        int[] channels = ints(c.get("block_out_channels"), 0, 0);
        int levels = channels.length;
        List<Object> types = Json.array(c.get("down_block_types"));
        boolean[] cross = new boolean[levels];
        for (int i = 0; i < levels; i++) {
            cross[i] = ((String) types.get(i)).startsWith("CrossAttn");
        }
        Object heads = c.get("num_attention_heads") != null ? c.get("num_attention_heads") : c.get("attention_head_dim");
        Object ctx = c.get("cross_attention_dim");
        int contextDim = ctx instanceof List<?> l ? ((Number) l.get(0)).intValue() : ((Number) ctx).intValue();
        boolean textTime = "text_time".equals(Json.string(c, "addition_embed_type", null));
        int addTime = textTime ? (int) Json.integer(c, "addition_time_embed_dim", 256) : 0;
        int pooled = textTime ? (int) Json.integer(c, "projection_class_embeddings_input_dim", 2816) - 6 * addTime : 0;
        return new UNet.Settings((int) Json.integer(c, "in_channels", 4), (int) Json.integer(c, "out_channels", 4),
                channels, cross, (int) Json.integer(c, "layers_per_block", 2), ints(heads, levels, 8),
                ints(c.get("transformer_layers_per_block"), levels, 1), contextDim,
                Boolean.TRUE.equals(c.get("use_linear_projection")), (int) Json.integer(c, "norm_num_groups", 32),
                (float) Json.number(c, "norm_eps", 1e-5), addTime, pooled, textTime ? 6 : 0,
                !Boolean.FALSE.equals(c.get("flip_sin_to_cos")), (int) Json.integer(c, "freq_shift", 0));
    }

    private static int[] ints(Object v, int n, int otherwise) {
        if (v == null) {
            int[] out = new int[n];
            Arrays.fill(out, otherwise);
            return out;
        }
        if (v instanceof Number num) {
            int[] out = new int[n];
            Arrays.fill(out, num.intValue());
            return out;
        }
        return Json.array(v).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
    }

    /** Reads diffusers' {@code vae/config.json}. */
    static AutoencoderKL.Settings vaeSettings(Map<String, Object> c, boolean encoder) {
        return new AutoencoderKL.Settings((int) Json.integer(c, "in_channels", 3), ints(c.get("block_out_channels"),
                0, 0), (int) Json.integer(c, "layers_per_block", 2), (int) Json.integer(c, "latent_channels", 4),
                (int) Json.integer(c, "norm_num_groups", 32), 1e-6f, (float) Json.number(c, "scaling_factor", 0.18215),
                (float) Json.number(c, "shift_factor", 0), !Boolean.FALSE.equals(c.get("use_quant_conv")), encoder);
    }

    /** Reads a transformers {@code CLIPTextConfig} (or a {@code CLIPConfig}'s text part). */
    static ClipTextEncoder.Settings clipSettings(Map<String, Object> json) {
        Map<String, Object> c = json.get("text_config") instanceof Map<?, ?> m ? Json.object(m) : json;
        int eos = (int) Json.integer(c, "eos_token_id", 2);
        int hidden = (int) Json.integer(c, "hidden_size", 768);
        return new ClipTextEncoder.Settings((int) Json.integer(c, "vocab_size", 49408), hidden,
                (int) Json.integer(c, "num_hidden_layers", 12), (int) Json.integer(c, "num_attention_heads", 12),
                (int) Json.integer(c, "intermediate_size", 4 * hidden),
                (int) Json.integer(c, "max_position_embeddings", 77),
                "gelu".equals(Json.string(c, "hidden_act", "quick_gelu")) ? Activation.GELU : Activation.QUICK_GELU,
                (float) Json.number(c, "layer_norm_eps", 1e-5), eos == 2 ? -1 : eos, 0);
    }

    // ------------------------------------------------------------------------------- UNet

    /** A UNet in diffusers' naming. */
    static UNet unet(Weights w, UNet.Settings settings) {
        UNet u = shell(() -> new UNet(settings, new SplittableRandom(0)));
        w.conv(u.convIn(), "conv_in");
        w.dense(u.timeEmbedding().linear1(), "time_embedding.linear_1");
        w.dense(u.timeEmbedding().linear2(), "time_embedding.linear_2");
        if (u.addLinear1() != null) {
            w.dense(u.addLinear1(), "add_embedding.linear_1");
            w.dense(u.addLinear2(), "add_embedding.linear_2");
        }
        List<UNet.Level> down = u.down();
        for (int i = 0; i < down.size(); i++) {
            level(w.at("down_blocks." + i + "."), down.get(i), "downsamplers.0.conv");
        }
        Weights m = w.at("mid_block.");
        resnet(m.at("resnets.0."), (ResnetBlock) u.middle().get(0));
        transformer(m.at("attentions.0."), (SpatialTransformer) u.middle().get(1));
        resnet(m.at("resnets.1."), (ResnetBlock) u.middle().get(2));
        List<UNet.Level> up = u.up();
        for (int i = 0; i < up.size(); i++) {
            level(w.at("up_blocks." + i + "."), up.get(i), "upsamplers.0.conv");
        }
        w.norm(u.normOut(), "conv_norm_out");
        w.conv(u.convOut(), "conv_out");
        w.checkFilled(u, "UNet");
        return u;
    }

    private static void level(Weights w, UNet.Level level, String resample) {
        for (int j = 0; j < level.resnets().size(); j++) {
            resnet(w.at("resnets." + j + "."), level.resnets().get(j));
            if (level.transformers().get(j) != null) {
                transformer(w.at("attentions." + j + "."), level.transformers().get(j));
            }
        }
        if (level.resample() != null) {
            w.conv(level.resample(), resample);
        }
    }

    static void resnet(Weights w, ResnetBlock r) {
        w.norm(r.norm1(), "norm1");
        w.conv(r.conv1(), "conv1");
        if (r.timeProjection() != null) {
            w.dense(r.timeProjection(), "time_emb_proj");
        }
        w.norm(r.norm2(), "norm2");
        w.conv(r.conv2(), "conv2");
        if (r.shortcut() != null) {
            w.conv(r.shortcut(), w.has("nin_shortcut.weight") ? "nin_shortcut" : "conv_shortcut");
        }
    }

    private static void transformer(Weights w, SpatialTransformer t) {
        w.norm(t.norm(), "norm");
        w.projection(t.projectIn(), "proj_in");
        List<TransformerBlock> blocks = t.blocks();
        for (int i = 0; i < blocks.size(); i++) {
            Weights b = w.at("transformer_blocks." + i + ".");
            List<Var> p = blocks.get(i).parameters();
            // TransformerBlock lists: self-attention (8), cross-attention (8), its norm (2),
            // the self-attention norm (2), the feed-forward norm (2), up, gate, down (2 each)
            b.attention(p, 0, "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0");
            b.attention(p, 8, "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0");
            b.set(p.get(16), b.get("norm2.weight"));
            b.set(p.get(17), b.get("norm2.bias"));
            b.set(p.get(18), b.get("norm1.weight"));
            b.set(p.get(19), b.get("norm1.bias"));
            b.set(p.get(20), b.get("norm3.weight"));
            b.set(p.get(21), b.get("norm3.bias"));
            // GEGLU: the first half of ff.net.0.proj is the value, the second goes through GELU;
            // TransformerBlock computes activation(up) * gate
            Tensor ff = b.get("ff.net.0.proj.weight");
            Tensor ffBias = b.get("ff.net.0.proj.bias");
            int half = ff.shape().dim(0) / 2;
            b.set(p.get(22), ff.slice(0, half, 2 * half).transpose());
            b.set(p.get(23), ffBias.slice(0, half, 2 * half));
            b.set(p.get(24), ff.slice(0, 0, half).transpose(), ff);
            b.set(p.get(25), ffBias.slice(0, 0, half), ffBias);
            b.matrix(p.get(26), "ff.net.2.weight");
            b.bias(p.get(27), "ff.net.2.bias");
        }
        w.projection(t.projectOut(), "proj_out");
    }

    // ------------------------------------------------------------------------------- VAE

    /** An autoencoder in diffusers' naming (also its older attention names). */
    static AutoencoderKL vae(Weights w, AutoencoderKL.Settings settings) {
        AutoencoderKL v = shell(() -> new AutoencoderKL(settings, new SplittableRandom(0)));
        int levels = settings.channels().length;
        List<Layer> parts = v.layers();
        int at = 0;
        for (String coder : settings.encoder() ? new String[] {"encoder", "decoder"} : new String[] {"decoder"}) {
            Weights c = w.at(coder + ".");
            boolean enc = coder.equals("encoder");
            if (!enc && settings.quantConv()) {
                w.conv((Conv) parts.get(at++), "post_quant_conv");
            }
            w.conv((Conv) parts.get(at++), coder + ".conv_in");
            if (!enc) {
                at = middle(c.at("mid_block."), parts, at);
            }
            for (int i = 0; i < levels; i++) {
                Weights b = c.at((enc ? "down_blocks." : "up_blocks.") + i + ".");
                int count = enc ? settings.layersPerBlock() : settings.layersPerBlock() + 1;
                for (int j = 0; j < count; j++) {
                    resnet(b.at("resnets." + j + "."), (ResnetBlock) parts.get(at++));
                }
                if (i < levels - 1) {
                    b.conv((Conv) parts.get(at++), enc ? "downsamplers.0.conv" : "upsamplers.0.conv");
                }
            }
            if (enc) {
                at = middle(c.at("mid_block."), parts, at);
            }
            c.norm(parts.get(at++), "conv_norm_out");
            c.conv((Conv) parts.get(at++), "conv_out");
            if (enc && settings.quantConv()) {
                w.conv((Conv) parts.get(at++), "quant_conv");
            }
        }
        w.checkFilled(v, "VAE");
        return v;
    }

    private static int middle(Weights w, List<Layer> parts, int at) {
        resnet(w.at("resnets.0."), (ResnetBlock) parts.get(at++));
        Weights a = w.at("attentions.0.");
        ImageAttention attention = (ImageAttention) parts.get(at++);
        List<Var> p = attention.parameters();
        a.set(p.get(0), a.get("group_norm.weight"));
        a.set(p.get(1), a.get("group_norm.bias"));
        if (a.has("query.weight")) {   // diffusers' older names
            a.attention(p, 2, "query", "key", "value", "proj_attn");
        } else {
            a.attention(p, 2, "to_q", "to_k", "to_v", "to_out.0");
        }
        resnet(w.at("resnets.1."), (ResnetBlock) parts.get(at++));
        return at;
    }

    // ------------------------------------------------------------------------------- CLIP

    /**
     * A CLIP text model in transformers' naming ({@code text_model.encoder.layers.0...}).
     *
     * @param layers how many layers to load; fewer than the model has behaves as if it ended there
     */
    static ClipTextEncoder clip(Weights w, ClipTextEncoder.Settings settings, int layers) {
        int projection = w.has("text_projection.weight") ? w.shape("text_projection.weight").dim(0) : 0;
        ClipTextEncoder.Settings s = new ClipTextEncoder.Settings(settings.vocabulary(), settings.features(), layers,
                settings.heads(), settings.ffnSize(), settings.positions(), settings.activation(), settings.epsilon(),
                settings.endToken(), projection);
        ClipTextEncoder e = shell(() -> new ClipTextEncoder(s, new SplittableRandom(0)));
        Weights t = w.has("text_model.embeddings.token_embedding.weight") ? w.at("text_model.") : w;
        t.set(e.tokenEmbedding(), t.get("embeddings.token_embedding.weight"));
        t.set(e.positionEmbedding(), t.get("embeddings.position_embedding.weight"));
        List<TransformerBlock> blocks = e.blocks();
        for (int i = 0; i < blocks.size(); i++) {
            clipLayer(t.at("encoder.layers." + i + "."), blocks.get(i));
        }
        t.norm(e.finalNorm(), "final_layer_norm");
        if (e.projection() != null) {
            Dense d = e.projection();
            w.matrix(d.weight(), "text_projection.weight");
            w.zero(d.bias());
        }
        w.checkFilled(e, "text encoder");
        return e;
    }

    /** One CLIP encoder layer in transformers' naming ({@code self_attn.q_proj}, {@code mlp.fc1}, ...). */
    static void clipLayer(Weights l, TransformerBlock block) {
        List<Var> p = block.parameters();
        // self-attention (8), its norm (2), the feed-forward norm (2), up, down (2 each)
        l.attention(p, 0, "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj");
        l.set(p.get(8), l.get("layer_norm1.weight"));
        l.set(p.get(9), l.get("layer_norm1.bias"));
        l.set(p.get(10), l.get("layer_norm2.weight"));
        l.set(p.get(11), l.get("layer_norm2.bias"));
        l.matrix(p.get(12), "mlp.fc1.weight");
        l.bias(p.get(13), "mlp.fc1.bias");
        l.matrix(p.get(14), "mlp.fc2.weight");
        l.bias(p.get(15), "mlp.fc2.bias");
    }

    /**
     * An OpenCLIP text tower ({@code transformer.resblocks.0.attn.in_proj_weight}, ...), as
     * Stable Diffusion 2 and XL single-file checkpoints store it. Its shape is read from the
     * weights.
     *
     * @param skip how many of the last layers to leave out: 1 for Stable Diffusion 2, which
     *     conditions on the penultimate layer
     * @param headDim the width of an attention head (64)
     */
    static ClipTextEncoder openClip(Weights w, int skip, int headDim) {
        Shape embed = w.shape("token_embedding.weight");
        int hidden = embed.dim(1);
        int count = 0;
        while (w.has("transformer.resblocks." + count + ".ln_1.weight")) {
            count++;
        }
        int projection = w.has("text_projection") ? w.shape("text_projection").dim(1) : 0;
        ClipTextEncoder.Settings s = new ClipTextEncoder.Settings(embed.dim(0), hidden, count - skip,
                hidden / headDim, w.shape("transformer.resblocks.0.mlp.c_fc.weight").dim(0),
                w.shape("positional_embedding").dim(0), Activation.GELU, 1e-5f, -1, projection);
        ClipTextEncoder e = shell(() -> new ClipTextEncoder(s, new SplittableRandom(0)));
        w.set(e.tokenEmbedding(), w.get("token_embedding.weight"));
        w.set(e.positionEmbedding(), w.get("positional_embedding"));
        List<TransformerBlock> blocks = e.blocks();
        for (int i = 0; i < blocks.size(); i++) {
            Weights l = w.at("transformer.resblocks." + i + ".");
            List<Var> p = blocks.get(i).parameters();
            for (int j = 0; j < 3; j++) {
                l.matrixRows(p.get(j), "attn.in_proj_weight", j * hidden, (j + 1) * hidden);
                l.biasRows(p.get(4 + j), "attn.in_proj_bias", j * hidden, (j + 1) * hidden);
            }
            l.matrix(p.get(3), "attn.out_proj.weight");
            l.bias(p.get(7), "attn.out_proj.bias");
            l.set(p.get(8), l.get("ln_1.weight"));
            l.set(p.get(9), l.get("ln_1.bias"));
            l.set(p.get(10), l.get("ln_2.weight"));
            l.set(p.get(11), l.get("ln_2.bias"));
            l.matrix(p.get(12), "mlp.c_fc.weight");
            l.bias(p.get(13), "mlp.c_fc.bias");
            l.matrix(p.get(14), "mlp.c_proj.weight");
            l.bias(p.get(15), "mlp.c_proj.bias");
        }
        w.norm(e.finalNorm(), "ln_final");
        if (e.projection() != null) {
            // stored as [hidden, projection] already
            w.set(e.projection().weight(), w.get("text_projection"));
            w.zero(e.projection().bias());
        }
        w.checkFilled(e, "text encoder");
        return e;
    }

    // ---------------------------------------------------------------------------------- Flux

    /** Reads diffusers' {@code transformer/config.json} of a {@code FluxTransformer2DModel}. */
    static FluxTransformer.Settings fluxSettings(Map<String, Object> c) {
        int packedPatch = (int) Json.integer(c, "patch_size", 1);
        if (packedPatch != 1) {
            throw new IllegalArgumentException("Flux transformers with patch_size " + packedPatch + " are not supported");
        }
        int in = (int) Json.integer(c, "in_channels", 64);
        int heads = (int) Json.integer(c, "num_attention_heads", 24);
        int headSize = (int) Json.integer(c, "attention_head_dim", 128);
        int[] axes = ints(c.get("axes_dims_rope"), 3, 0);
        if (axes[0] == 0) {
            axes = new int[] {16, 56, 56};
        }
        return new FluxTransformer.Settings(in / 4, 2, heads, headSize, (int) Json.integer(c, "num_layers", 19),
                (int) Json.integer(c, "num_single_layers", 38), (int) Json.integer(c, "joint_attention_dim", 4096),
                (int) Json.integer(c, "pooled_projection_dim", 768),
                Boolean.TRUE.equals(c.get("guidance_embeds")), axes, 10000, 4);
    }

    /** Builds a Flux transformer and fills it from diffusers' names. */
    static FluxTransformer flux(Weights w, FluxTransformer.Settings settings) {
        FluxTransformer f = shell(() -> new FluxTransformer(settings, new SplittableRandom(0)));
        w.dense(f.imageIn(), "x_embedder");
        w.dense(f.contextIn(), "context_embedder");
        w.dense(f.timeEmbedding().linear1(), "time_text_embed.timestep_embedder.linear_1");
        w.dense(f.timeEmbedding().linear2(), "time_text_embed.timestep_embedder.linear_2");
        if (f.guidanceEmbedding() != null) {
            w.dense(f.guidanceEmbedding().linear1(), "time_text_embed.guidance_embedder.linear_1");
            w.dense(f.guidanceEmbedding().linear2(), "time_text_embed.guidance_embedder.linear_2");
        }
        w.dense(f.pooled1(), "time_text_embed.text_embedder.linear_1");
        w.dense(f.pooled2(), "time_text_embed.text_embedder.linear_2");
        List<FluxTransformer.JointBlock> joint = f.jointBlocks();
        for (int i = 0; i < joint.size(); i++) {
            Weights b = w.at("transformer_blocks." + i + ".");
            stream(b, joint.get(i).image(), "norm1", "attn.to_q", "attn.to_k", "attn.to_v", "attn.norm_q",
                    "attn.norm_k", "attn.to_out.0", "ff");
            stream(b, joint.get(i).text(), "norm1_context", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
                    "attn.norm_added_q", "attn.norm_added_k", "attn.to_add_out", "ff_context");
        }
        List<FluxTransformer.SingleBlock> single = f.singleBlocks();
        for (int i = 0; i < single.size(); i++) {
            Weights b = w.at("single_transformer_blocks." + i + ".");
            FluxTransformer.SingleBlock s = single.get(i);
            b.dense(s.modulation(), "norm.linear");
            b.dense(s.q(), "attn.to_q");
            b.dense(s.k(), "attn.to_k");
            b.dense(s.v(), "attn.to_v");
            rms(b, s.qNorm(), "attn.norm_q");
            rms(b, s.kNorm(), "attn.norm_k");
            b.dense(s.mlp(), "proj_mlp");
            b.dense(s.out(), "proj_out");
        }
        w.dense(f.outModulation(), "norm_out.linear");
        w.dense(f.out(), "proj_out");
        w.checkFilled(f, "Flux transformer");
        return f;
    }

    private static void stream(Weights b, FluxTransformer.JointBlock.Stream s, String modulation, String q, String k,
            String v, String qNorm, String kNorm, String out, String ff) {
        b.dense(s.modulation(), modulation + ".linear");
        b.dense(s.q(), q);
        b.dense(s.k(), k);
        b.dense(s.v(), v);
        rms(b, s.qNorm(), qNorm);
        rms(b, s.kNorm(), kNorm);
        b.dense(s.attentionOut(), out);
        b.dense(s.up(), ff + ".net.0.proj");
        b.dense(s.down(), ff + ".net.2");
    }

    private static void rms(Weights w, LayerNorm norm, String name) {
        w.set(norm.parameters().get(0), w.get(name + ".weight"));
    }

    // ------------------------------------------------------------------------------------ T5

    /** Reads transformers' {@code config.json} of a T5 encoder. */
    static T5Encoder.Settings t5Settings(Map<String, Object> c) {
        String ffn = Json.string(c, "feed_forward_proj", "relu");
        boolean gated = ffn.startsWith("gated-");
        if (!ffn.equals("relu") && !ffn.equals("gated-gelu")) {
            throw new IllegalArgumentException("T5 feed-forward " + ffn + " is not supported");
        }
        int d = (int) Json.integer(c, "d_model", 512);
        return new T5Encoder.Settings((int) Json.integer(c, "vocab_size", 32128), d,
                (int) Json.integer(c, "num_layers", 6), (int) Json.integer(c, "num_heads", 8),
                (int) Json.integer(c, "d_kv", 64), (int) Json.integer(c, "d_ff", 2048),
                (int) Json.integer(c, "relative_attention_num_buckets", 32),
                (int) Json.integer(c, "relative_attention_max_distance", 128),
                (float) Json.number(c, "layer_norm_epsilon", 1e-6), gated);
    }

    /** Builds a T5 encoder and fills it from transformers' names. */
    static T5Encoder t5(Weights w, T5Encoder.Settings settings) {
        T5Encoder e = shell(() -> new T5Encoder(settings, new SplittableRandom(0)));
        w.set(e.embedding(), w.get(w.has("shared.weight") ? "shared.weight" : "encoder.embed_tokens.weight"));
        Weights enc = w.at("encoder.");
        enc.set(e.relativeBias(), enc.get("block.0.layer.0.SelfAttention.relative_attention_bias.weight"));
        List<T5Encoder.Block> blocks = e.blocks();
        for (int i = 0; i < blocks.size(); i++) {
            Weights b = enc.at("block." + i + ".layer.");
            T5Encoder.Block k = blocks.get(i);
            rms(b, k.attentionNorm(), "0.layer_norm");
            b.matrix(k.q(), "0.SelfAttention.q.weight");
            b.matrix(k.k(), "0.SelfAttention.k.weight");
            b.matrix(k.v(), "0.SelfAttention.v.weight");
            b.matrix(k.o(), "0.SelfAttention.o.weight");
            rms(b, k.ffnNorm(), "1.layer_norm");
            if (k.gate() != null) {
                b.matrix(k.up(), "1.DenseReluDense.wi_0.weight");
                b.matrix(k.gate(), "1.DenseReluDense.wi_1.weight");
            } else {
                b.matrix(k.up(), "1.DenseReluDense.wi.weight");
            }
            b.matrix(k.down(), "1.DenseReluDense.wo.weight");
        }
        rms(enc, e.finalNorm(), "final_layer_norm");
        w.checkFilled(e, "T5 encoder");
        return e;
    }
}
