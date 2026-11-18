package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.nn.diffusion.WanTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.WanVae;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Builds Wan's networks ({@link WanTransformer}, {@link WanVae}, umT5 as a {@link T5Encoder})
 * and fills them from checkpoints in either of the layouts they are published in: diffusers'
 * ({@code Wan-AI/*-Diffusers}) or the original one of Wan's own repositories, which ComfyUI's
 * single files use too (sometimes under {@code model.diffusion_model.}). The shapes are read
 * from the weights, so single files need no config.
 */
final class WanPretrained {

    private WanPretrained() {
    }

    /** The features per attention head of every published Wan model. */
    static final int HEAD_SIZE = 128;

    // ------------------------------------------------------------------------------- transformer

    /** Names of the original layout that lead to diffusers' names, in order. */
    private static final String[][] TRANSFORMER_NAMES = {
            {"condition_embedder.time_embedder.linear_1", "time_embedding.0"},
            {"condition_embedder.time_embedder.linear_2", "time_embedding.2"},
            {"condition_embedder.text_embedder.linear_1", "text_embedding.0"},
            {"condition_embedder.text_embedder.linear_2", "text_embedding.2"},
            {"condition_embedder.time_proj", "time_projection.1"},
            {"condition_embedder.image_embedder.norm1", "img_emb.proj.0"},
            {"condition_embedder.image_embedder.ff.net.0.proj", "img_emb.proj.1"},
            {"condition_embedder.image_embedder.ff.net.2", "img_emb.proj.3"},
            {"condition_embedder.image_embedder.norm2", "img_emb.proj.4"},
            {".attn1.", ".self_attn."},
            {".attn2.", ".cross_attn."},
            {".to_out.0.", ".o."},
            {".to_q.", ".q."},
            {".to_k.", ".k."},
            {".to_v.", ".v."},
            {".add_k_proj.", ".k_img."},
            {".add_v_proj.", ".v_img."},
            {".norm_added_k.", ".norm_k_img."},
            {"ffn.net.0.proj", "ffn.0"},
            {"ffn.net.2", "ffn.2"},
            {"proj_out", "head.head"},
    };

    private static final Pattern BLOCK_TABLE = Pattern.compile("^blocks\\.(\\d+)\\.scale_shift_table$");
    private static final Pattern BLOCK_NORM = Pattern.compile("^blocks\\.(\\d+)\\.norm2\\.(weight|bias)$");

    /** Translates a diffusers transformer name to the original layout, under {@code prefix}. */
    static UnaryOperator<String> originalTransformerNames(String prefix) {
        return name -> {
            if (name.equals("scale_shift_table")) {
                return prefix + "head.modulation";
            }
            Matcher t = BLOCK_TABLE.matcher(name);
            if (t.matches()) {
                return prefix + "blocks." + t.group(1) + ".modulation";
            }
            Matcher n = BLOCK_NORM.matcher(name);
            if (n.matches()) {
                // diffusers' norm2 (before cross-attention) is the original norm3
                return prefix + "blocks." + n.group(1) + ".norm3." + n.group(2);
            }
            String out = name;
            for (String[] r : TRANSFORMER_NAMES) {
                out = out.replace(r[0], r[1]);
            }
            return prefix + out;
        };
    }

    /**
     * The names of a transformer checkpoint as diffusers' names: identity for diffusers files,
     * a translation for the original layout.
     */
    static UnaryOperator<String> transformerNames(WeightSource source) {
        if (source.contains("condition_embedder.time_proj.weight")) {
            return UnaryOperator.identity();
        }
        for (String prefix : new String[] {"", "model.diffusion_model.", "diffusion_model.", "model."}) {
            if (source.contains(prefix + "time_projection.1.weight")) {
                return originalTransformerNames(prefix);
            }
        }
        throw new IllegalArgumentException("not a Wan transformer checkpoint (no time projection weight)");
    }

    /**
     * The transformer's shape, from a diffusers config if there is one, otherwise from the
     * weights (heads of {@link #HEAD_SIZE} features).
     */
    static WanTransformer.Settings transformerSettings(WeightSource source, UnaryOperator<String> names,
            Map<String, Object> config) {
        Shape patch = source.shape(names.apply("patch_embedding.weight"));
        int d = patch.dim(0);
        int in = patch.dim(1);
        int[] p = {patch.dim(2), patch.dim(3), patch.dim(4)};
        int out = source.shape(names.apply("proj_out.weight")).dim(0) / (p[0] * p[1] * p[2]);
        int text = source.shape(names.apply("condition_embedder.text_embedder.linear_1.weight")).dim(1);
        int freq = source.shape(names.apply("condition_embedder.time_embedder.linear_1.weight")).dim(1);
        int ffn = source.shape(names.apply("blocks.0.ffn.net.0.proj.weight")).dim(0);
        int layers = 0;
        while (source.contains(names.apply("blocks." + layers + ".attn1.to_q.weight"))) {
            layers++;
        }
        String image = names.apply("condition_embedder.image_embedder.norm1.weight");
        int imageFeatures = source.contains(image) ? source.shape(image).dim(0) : 0;
        int heads = config != null && config.get("num_attention_heads") != null
                ? (int) Json.integer(config, "num_attention_heads", d / HEAD_SIZE) : d / HEAD_SIZE;
        float eps = config != null ? (float) Json.number(config, "eps", 1e-6) : 1e-6f;
        boolean crossNorm = source.contains(names.apply("blocks.0.norm2.weight"));
        return new WanTransformer.Settings(in, out, p, heads, d / heads, layers, ffn, text, freq, eps, crossNorm,
                imageFeatures, 10000);
    }

    /** Builds a transformer and fills it; {@code w} sees the checkpoint under diffusers' names. */
    static WanTransformer transformer(Weights w, WanTransformer.Settings s) {
        WanTransformer t = Tensor.withoutValues(() -> new WanTransformer(s, new SplittableRandom(0)));
        // the patch embedding is a strided 3-d convolution: [d, c, pt, ph, pw] as a projection
        Tensor pw = w.get("patch_embedding.weight");
        int d = pw.shape().dim(0);
        w.set(t.patchIn().weight(), pw.reshape(d, Math.toIntExact(pw.size() / d)).transpose(), pw);
        w.bias(t.patchIn().bias(), "patch_embedding.bias");
        w.dense(t.timeEmbedding().linear1(), "condition_embedder.time_embedder.linear_1");
        w.dense(t.timeEmbedding().linear2(), "condition_embedder.time_embedder.linear_2");
        w.dense(t.timeProjection(), "condition_embedder.time_proj");
        w.dense(t.text1(), "condition_embedder.text_embedder.linear_1");
        w.dense(t.text2(), "condition_embedder.text_embedder.linear_2");
        if (t.image1() != null) {
            w.norm(t.imageNorm1(), "condition_embedder.image_embedder.norm1");
            w.dense(t.image1(), "condition_embedder.image_embedder.ff.net.0.proj");
            w.dense(t.image2(), "condition_embedder.image_embedder.ff.net.2");
            w.norm(t.imageNorm2(), "condition_embedder.image_embedder.norm2");
        }
        List<WanTransformer.Block> blocks = t.blocks();
        for (int i = 0; i < blocks.size(); i++) {
            Weights b = w.at("blocks." + i + ".");
            WanTransformer.Block k = blocks.get(i);
            attention(b.at("attn1."), k.selfAttention());
            if (k.crossNorm() != null) {
                b.norm(k.crossNorm(), "norm2");
            }
            attention(b.at("attn2."), k.crossAttention());
            b.dense(k.up(), "ffn.net.0.proj");
            b.dense(k.down(), "ffn.net.2");
            table(b, k.table(), "scale_shift_table");
        }
        table(w, t.outTable(), "scale_shift_table");
        w.dense(t.out(), "proj_out");
        w.checkFilled(t, "Wan transformer");
        return t;
    }

    private static void attention(Weights w, WanTransformer.Attention a) {
        w.dense(a.q(), "to_q");
        w.dense(a.k(), "to_k");
        w.dense(a.v(), "to_v");
        w.dense(a.o(), "to_out.0");
        rms(w, a.qNorm(), "norm_q");
        rms(w, a.kNorm(), "norm_k");
        if (a.imageK() != null) {
            w.dense(a.imageK(), "add_k_proj");
            w.dense(a.imageV(), "add_v_proj");
            rms(w, a.imageKNorm(), "norm_added_k");
        }
    }

    /** A modulation table stored {@code [1, n, d]} into {@code [n, d]}. */
    private static void table(Weights w, Var parameter, String name) {
        Tensor t = w.get(name);
        w.set(parameter, t.reshape(parameter.shape()));
    }

    private static void rms(Weights w, LayerNorm norm, String name) {
        w.set(norm.parameters().get(0), w.get(name + ".weight"));
    }

    // ------------------------------------------------------------------------------- VAE

    /** Whether a VAE checkpoint uses diffusers' names. */
    static boolean diffusersVae(WeightSource source) {
        return source.contains("decoder.conv_in.weight");
    }

    /** The prefix of an original-layout VAE checkpoint ({@code ""}, or ComfyUI's {@code first_stage_model.}). */
    static String vaePrefix(WeightSource source) {
        for (String prefix : new String[] {"", "first_stage_model.", "vae."}) {
            if (source.contains(prefix + "decoder.conv_in.weight") || source.contains(prefix + "decoder.conv1.weight")) {
                return prefix;
            }
        }
        throw new IllegalArgumentException("not a Wan VAE checkpoint (no decoder input convolution)");
    }

    /**
     * The autoencoder's shape: diffusers' config if given, otherwise Wan 2.1's, with the base
     * width and latent channels read from the decoder's input convolution.
     */
    static WanVae.Settings vaeSettings(WeightSource source, String prefix, Map<String, Object> config) {
        if (config != null && config.get("base_dim") != null) {
            if (Boolean.TRUE.equals(config.get("is_residual"))) {
                throw new IllegalArgumentException("the Wan 2.2 autoencoder (residual blocks) is not supported yet");
            }
            int z = (int) Json.integer(config, "z_dim", 16);
            return new WanVae.Settings((int) Json.integer(config, "base_dim", 96), z,
                    Json.array(config.get("dim_mult")).stream().mapToInt(o -> ((Number) o).intValue()).toArray(),
                    (int) Json.integer(config, "num_res_blocks", 2), booleans(config.get("temperal_downsample")),
                    floats(config.get("latents_mean"), z), floats(config.get("latents_std"), z));
        }
        String in = source.contains(prefix + "decoder.conv_in.weight") ? "decoder.conv_in.weight"
                : "decoder.conv1.weight";
        Shape s = source.shape(prefix + in);
        WanVae.Settings w = WanVae.Settings.wan21();
        if (s.dim(1) != w.latentChannels() || s.dim(0) != w.base() * 4) {
            throw new IllegalArgumentException("a VAE with " + s.dim(1) + " latent channels is not Wan 2.1's; give "
                    + "its diffusers config");
        }
        return w;
    }

    private static boolean[] booleans(Object v) {
        List<Object> l = Json.array(v);
        boolean[] out = new boolean[l.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = Boolean.TRUE.equals(l.get(i));
        }
        return out;
    }

    private static float[] floats(Object v, int n) {
        if (v == null) {
            throw new IllegalArgumentException("the VAE config has no latent mean and std");
        }
        List<Object> l = Json.array(v);
        float[] out = new float[n];
        for (int i = 0; i < n; i++) {
            out[i] = ((Number) l.get(i)).floatValue();
        }
        return out;
    }

    /** Builds an autoencoder and fills it from either layout; {@code w} is at the checkpoint's prefix. */
    static WanVae vae(Weights w, WanVae.Settings s, boolean diffusers, boolean encoder) {
        WanVae v = Tensor.withoutValues(() -> new WanVae(s, encoder, new SplittableRandom(0)));
        if (encoder) {
            coder(w.at("encoder."), v.encoder(), diffusers, s.resBlocks());
            conv(w, v.quant(), diffusers ? "quant_conv" : "conv1");
        }
        conv(w, v.postQuant(), diffusers ? "post_quant_conv" : "conv2");
        coder(w.at("decoder."), v.decoder(), diffusers, s.resBlocks());
        w.checkFilled(v, "Wan VAE");
        return v;
    }

    private static void coder(Weights w, WanVae.Coder c, boolean diffusers, int resBlocks) {
        conv(w, c.in(), diffusers ? "conv_in" : "conv1");
        List<WanVae.Stage> stages = c.stages();
        int perLevel = resBlocks + 2;   // the decoder's residual blocks and a resampler
        for (int i = 0; i < stages.size(); i++) {
            String name;
            if (!diffusers) {
                name = (c.isEncoder() ? "downsamples." : "upsamples.") + i + ".";
            } else if (c.isEncoder()) {
                name = "down_blocks." + i + ".";
            } else {
                int level = i / perLevel;
                int j = i % perLevel;
                name = "up_blocks." + level + (j <= resBlocks ? ".resnets." + j + "." : ".upsamplers.0.");
            }
            stage(w.at(name), stages.get(i), diffusers);
        }
        List<WanVae.Stage> middle = c.middle();
        String[] mid = diffusers ? new String[] {"mid_block.resnets.0.", "mid_block.attentions.0.", "mid_block.resnets.1."}
                : new String[] {"middle.0.", "middle.1.", "middle.2."};
        for (int i = 0; i < 3; i++) {
            stage(w.at(mid[i]), middle.get(i), diffusers);
        }
        w.set(c.headNorm().gamma(), gamma(w, diffusers ? "norm_out.gamma" : "head.0.gamma"));
        conv(w, c.head(), diffusers ? "conv_out" : "head.2");
    }

    private static void stage(Weights w, WanVae.Stage stage, boolean diffusers) {
        switch (stage) {
            case WanVae.ResBlock r -> {
                w.set(r.norm1().gamma(), gamma(w, diffusers ? "norm1.gamma" : "residual.0.gamma"));
                conv(w, r.conv1(), diffusers ? "conv1" : "residual.2");
                w.set(r.norm2().gamma(), gamma(w, diffusers ? "norm2.gamma" : "residual.3.gamma"));
                conv(w, r.conv2(), diffusers ? "conv2" : "residual.6");
                if (r.shortcut() != null) {
                    conv(w, r.shortcut(), diffusers ? "conv_shortcut" : "shortcut");
                }
            }
            case WanVae.AttentionBlock a -> {
                w.set(a.norm().gamma(), gamma(w, "norm.gamma"));
                w.matrix(a.qkv(), "to_qkv.weight");
                w.bias(a.qkvBias(), "to_qkv.bias");
                w.matrix(a.proj(), "proj.weight");
                w.bias(a.projBias(), "proj.bias");
            }
            case WanVae.Resample r -> {
                w.set(r.weight(), w.get("resample.1.weight"));
                w.bias(r.bias(), "resample.1.bias");
                if (r.time() != null) {
                    conv(w, r.time(), "time_conv");
                }
            }
        }
    }

    private static Tensor gamma(Weights w, String name) {
        Tensor g = w.get(name);
        return g.reshape(Math.toIntExact(g.size()));
    }

    private static void conv(Weights w, WanVae.CausalConv c, String name) {
        w.set(c.weight(), w.get(name + ".weight"));
        w.bias(c.bias(), name + ".bias");
    }

    // ------------------------------------------------------------------------------- umT5

    /** Whether a text encoder checkpoint is in the layout of Wan's own repository ({@code models_t5_umt5-xxl-enc-bf16.pth}). */
    static boolean originalT5(WeightSource source) {
        return source.contains("token_embedding.weight");
    }

    /** The umT5 encoder's shape: transformers' config if given, otherwise read from the weights. */
    static T5Encoder.Settings t5Settings(WeightSource source, Map<String, Object> config) {
        if (config != null && config.get("d_model") != null) {
            T5Encoder.Settings s = Pretrained.t5Settings(config);
            boolean perLayer = "umt5".equals(config.get("model_type"))
                    || source.contains("encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight");
            return new T5Encoder.Settings(s.vocabulary(), s.features(), s.layers(), s.heads(), s.headSize(),
                    s.ffnSize(), s.buckets(), s.maxDistance(), s.epsilon(), s.gated(), perLayer);
        }
        boolean original = originalT5(source);
        Shape embedding = source.shape(original ? "token_embedding.weight"
                : source.contains("shared.weight") ? "shared.weight" : "encoder.embed_tokens.weight");
        int layers = 0;
        while (source.contains(original ? "blocks." + layers + ".attn.q.weight"
                : "encoder.block." + layers + ".layer.0.SelfAttention.q.weight")) {
            layers++;
        }
        Shape bias = source.shape(original ? "blocks.0.pos_embedding.embedding.weight"
                : "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight");
        Shape q = source.shape(original ? "blocks.0.attn.q.weight" : "encoder.block.0.layer.0.SelfAttention.q.weight");
        Shape ffn = source.shape(original ? "blocks.0.ffn.fc1.weight"
                : "encoder.block.0.layer.1.DenseReluDense.wi_0.weight");
        int heads = bias.dim(1);
        boolean perLayer = original || source.contains("encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight");
        return new T5Encoder.Settings(embedding.dim(0), embedding.dim(1), layers, heads, q.dim(0) / heads,
                ffn.dim(0), bias.dim(0), 128, 1e-6f, true, perLayer);
    }

    /** Builds umT5 and fills it from transformers' names or Wan's own. */
    static T5Encoder t5(Weights w, T5Encoder.Settings settings, boolean original) {
        if (!original) {
            T5Encoder e = Tensor.withoutValues(() -> new T5Encoder(settings, new SplittableRandom(0)));
            w.set(e.embedding(), w.get(w.has("shared.weight") ? "shared.weight" : "encoder.embed_tokens.weight"));
            Weights enc = w.at("encoder.");
            List<T5Encoder.Block> blocks = e.blocks();
            if (e.relativeBias() != null) {
                enc.set(e.relativeBias(), enc.get("block.0.layer.0.SelfAttention.relative_attention_bias.weight"));
            }
            for (int i = 0; i < blocks.size(); i++) {
                Weights b = enc.at("block." + i + ".layer.");
                T5Encoder.Block k = blocks.get(i);
                if (k.relativeBias() != null) {
                    b.set(k.relativeBias(), b.get("0.SelfAttention.relative_attention_bias.weight"));
                }
                rms(b, k.attentionNorm(), "0.layer_norm");
                b.matrix(k.q(), "0.SelfAttention.q.weight");
                b.matrix(k.k(), "0.SelfAttention.k.weight");
                b.matrix(k.v(), "0.SelfAttention.v.weight");
                b.matrix(k.o(), "0.SelfAttention.o.weight");
                rms(b, k.ffnNorm(), "1.layer_norm");
                b.matrix(k.up(), "1.DenseReluDense.wi_0.weight");
                b.matrix(k.gate(), "1.DenseReluDense.wi_1.weight");
                b.matrix(k.down(), "1.DenseReluDense.wo.weight");
            }
            rms(enc, e.finalNorm(), "final_layer_norm");
            w.checkFilled(e, "umT5 encoder");
            return e;
        }
        T5Encoder e = Tensor.withoutValues(() -> new T5Encoder(settings, new SplittableRandom(0)));
        w.set(e.embedding(), w.get("token_embedding.weight"));
        List<T5Encoder.Block> blocks = e.blocks();
        for (int i = 0; i < blocks.size(); i++) {
            Weights b = w.at("blocks." + i + ".");
            T5Encoder.Block k = blocks.get(i);
            b.set(k.relativeBias(), b.get("pos_embedding.embedding.weight"));
            rms(b, k.attentionNorm(), "norm1");
            b.matrix(k.q(), "attn.q.weight");
            b.matrix(k.k(), "attn.k.weight");
            b.matrix(k.v(), "attn.v.weight");
            b.matrix(k.o(), "attn.o.weight");
            rms(b, k.ffnNorm(), "norm2");
            // Wan's feed-forward: fc1(x) * gelu(gate(x)), then fc2
            b.matrix(k.up(), "ffn.gate.0.weight");
            b.matrix(k.gate(), "ffn.fc1.weight");
            b.matrix(k.down(), "ffn.fc2.weight");
        }
        rms(w, e.finalNorm(), "norm");
        w.checkFilled(e, "umT5 encoder");
        return e;
    }

}
