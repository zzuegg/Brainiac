package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The variational autoencoder ({@code AutoencoderKL}) that maps between images and the
 * 8x smaller latents Stable Diffusion denoises.
 */
public final class Vae {

    /**
     * The shape of the autoencoder.
     *
     * @param channels the width of each resolution level, from the image side
     * @param layersPerBlock residual blocks per encoder level (the decoder has one more)
     * @param latentChannels latent channels
     * @param groups group-norm groups
     * @param scalingFactor what latents are multiplied by before diffusion
     * @param shiftFactor what is subtracted from latents before scaling (0 for Stable Diffusion)
     */
    public record Config(int[] channels, int layersPerBlock, int latentChannels, int groups, float scalingFactor,
            float shiftFactor) {

        /**
         * Reads diffusers' {@code vae/config.json}.
         *
         * @param c the parsed config
         * @return the config
         */
        public static Config fromJson(Map<String, Object> c) {
            int[] channels = Json.array(c.get("block_out_channels")).stream().mapToInt(o -> ((Number) o).intValue())
                    .toArray();
            return new Config(channels, (int) Json.integer(c, "layers_per_block", 2),
                    (int) Json.integer(c, "latent_channels", 4), (int) Json.integer(c, "norm_num_groups", 32),
                    (float) Json.number(c, "scaling_factor", 0.18215), (float) Json.number(c, "shift_factor", 0));
        }
    }

    private record Resnet(Nn.GroupNorm norm1, Nn.Conv2d conv1, Nn.GroupNorm norm2, Nn.Conv2d conv2,
            Nn.Conv2d shortcut) {
    }

    private record Attention(Nn.GroupNorm norm, Nn.Linear q, Nn.Linear k, Nn.Linear v, Nn.Linear out) {
    }

    private record Level(List<Resnet> resnets, Nn.Conv2d resample) {
    }

    private record Coder(Nn.Conv2d convIn, List<Level> levels, Resnet mid1, Attention attention, Resnet mid2,
            Nn.GroupNorm normOut, Nn.Conv2d convOut, Nn.Conv2d quant) {
    }

    private static final float EPS = 1e-6f;

    private final Config config;
    private final List<Tensor> weights;
    private final Coder decoder;
    private final Coder encoder;

    private Vae(Config config, Weights w, boolean withEncoder) {
        this.config = config;
        this.weights = w.all;
        int levels = config.channels().length;
        Weights d = w.at("decoder.");
        List<Level> up = new ArrayList<>();
        for (int i = 0; i < levels; i++) {
            Weights b = d.at("up_blocks." + i + ".");
            List<Resnet> resnets = new ArrayList<>();
            for (int j = 0; j <= config.layersPerBlock(); j++) {
                resnets.add(resnet(b.at("resnets." + j + ".")));
            }
            up.add(new Level(resnets, i < levels - 1 ? b.conv("upsamplers.0.conv") : null));
        }
        decoder = new Coder(d.conv("conv_in"), up, resnet(d.at("mid_block.resnets.0.")),
                attention(d.at("mid_block.attentions.0.")), resnet(d.at("mid_block.resnets.1.")),
                d.groupNorm("conv_norm_out", config.groups(), EPS), d.conv("conv_out"),
                w.has("post_quant_conv.weight") ? w.conv("post_quant_conv") : null);
        if (withEncoder) {
            Weights e = w.at("encoder.");
            List<Level> downs = new ArrayList<>();
            for (int i = 0; i < levels; i++) {
                Weights b = e.at("down_blocks." + i + ".");
                List<Resnet> resnets = new ArrayList<>();
                for (int j = 0; j < config.layersPerBlock(); j++) {
                    resnets.add(resnet(b.at("resnets." + j + ".")));
                }
                downs.add(new Level(resnets, i < levels - 1 ? b.conv("downsamplers.0.conv", 2, 0, 1) : null));
            }
            encoder = new Coder(e.conv("conv_in"), downs, resnet(e.at("mid_block.resnets.0.")),
                    attention(e.at("mid_block.attentions.0.")), resnet(e.at("mid_block.resnets.1.")),
                    e.groupNorm("conv_norm_out", config.groups(), EPS), e.conv("conv_out"),
                    w.has("quant_conv.weight") ? w.conv("quant_conv") : null);
        } else {
            encoder = null;
        }
    }

    /**
     * Loads the autoencoder in diffusers' naming.
     *
     * @param w the weights at its prefix
     * @param config its config
     * @param withEncoder whether to load the encoder too (only image-to-image needs it)
     * @return the model
     */
    static Vae load(Weights w, Config config, boolean withEncoder) {
        return new Vae(config, w, withEncoder);
    }

    private Resnet resnet(Weights w) {
        return new Resnet(w.groupNorm("norm1", config.groups(), EPS), w.conv("conv1"),
                w.groupNorm("norm2", config.groups(), EPS), w.conv("conv2"),
                w.has("conv_shortcut.weight") ? w.conv("conv_shortcut")
                        : w.has("nin_shortcut.weight") ? w.conv("nin_shortcut") : null);
    }

    private Attention attention(Weights w) {
        if (w.has("query.weight")) {   // diffusers' older names
            return new Attention(w.groupNorm("group_norm", config.groups(), EPS), w.linear("query"), w.linear("key"),
                    w.linear("value"), w.linear("proj_attn"));
        }
        return new Attention(w.groupNorm("group_norm", config.groups(), EPS), w.linear("to_q"), w.linear("to_k"),
                w.linear("to_v"), w.linear("to_out.0"));
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
     * Returns whether the encoder was loaded.
     *
     * @return whether {@link #encode} works
     */
    public boolean hasEncoder() {
        return encoder != null;
    }

    /**
     * Decodes diffusion latents (already divided by nothing: as the UNet sees them) into images.
     *
     * @param latents {@code [batch, latentChannels, h, w]}
     * @return images {@code [batch, 3, 8h, 8w]} in {@code [-1, 1]} (not clamped), float32
     */
    public Tensor decode(Tensor latents) {
        DType type = decoder.convIn().weight().dtype();
        Tensor z = Nn.scoped(weights, () -> {
            Tensor s = latents.to(DType.FLOAT32).mul(1 / config.scalingFactor()).add(config.shiftFactor()).to(type);
            return decoder.quant() == null ? s : decoder.quant().apply(s);
        }, latents);
        Tensor h = step(() -> decoder.convIn().apply(z), z);
        h = middle(decoder, h);
        for (Level level : decoder.levels()) {
            for (Resnet r : level.resnets()) {
                Tensor in = h;
                h = step(() -> resnet(r, in), in);
            }
            if (level.resample() != null) {
                Tensor in = h;
                h = step(() -> level.resample().apply(Nn.upsample2x(in)), in);
            }
        }
        Tensor last = h;
        return step(() -> decoder.convOut().apply(Nn.silu(decoder.normOut().apply(last))).to(DType.FLOAT32), last);
    }

    /**
     * Encodes images into latents scaled for diffusion: the mean of the latent distribution.
     *
     * @param images {@code [batch, 3, h, w]} in {@code [-1, 1]}, h and w multiples of 8
     * @return {@code [batch, latentChannels, h / 8, w / 8]}, float32
     * @throws IllegalStateException if the encoder was not loaded
     */
    public Tensor encode(Tensor images) {
        if (encoder == null) {
            throw new IllegalStateException("the VAE was loaded without its encoder");
        }
        DType type = encoder.convIn().weight().dtype();
        Tensor h = Nn.scoped(weights, () -> encoder.convIn().apply(images.to(type)), images);
        for (Level level : encoder.levels()) {
            for (Resnet r : level.resnets()) {
                Tensor in = h;
                h = step(() -> resnet(r, in), in);
            }
            if (level.resample() != null) {
                Tensor in = h;
                h = step(() -> level.resample().apply(in), in);
            }
        }
        h = middle(encoder, h);
        Tensor last = h;
        return step(() -> {
            Tensor moments = encoder.convOut().apply(Nn.silu(encoder.normOut().apply(last)));
            if (encoder.quant() != null) {
                moments = encoder.quant().apply(moments);
            }
            Tensor mean = moments.slice(1, 0, config.latentChannels()).to(DType.FLOAT32);
            return mean.sub(config.shiftFactor()).mul(config.scalingFactor());
        }, last);
    }

    /** Runs a step and frees its input. */
    private Tensor step(java.util.function.Supplier<Tensor> f, Tensor input) {
        Tensor out = Nn.scoped(weights, f, input);
        input.release();
        return out;
    }

    private Tensor middle(Coder c, Tensor h) {
        Tensor a = h;
        h = step(() -> resnet(c.mid1(), a), a);
        Tensor b = h;
        h = step(() -> attention(c.attention(), b), b);
        Tensor d = h;
        return step(() -> resnet(c.mid2(), d), d);
    }

    private static Tensor resnet(Resnet r, Tensor x) {
        Tensor h = r.conv1().apply(Nn.silu(r.norm1().apply(x)));
        h = r.conv2().apply(Nn.silu(r.norm2().apply(h)));
        return (r.shortcut() == null ? x : r.shortcut().apply(x)).add(h);
    }

    private static Tensor attention(Attention a, Tensor x) {
        int h = x.shape().dim(2);
        int w = x.shape().dim(3);
        Tensor t = Nn.tokens(a.norm().apply(x));
        Tensor o = Nn.attention(a.q().apply(t), a.k().apply(t), a.v().apply(t), 1, false);
        return Nn.image(a.out().apply(o), h, w).add(x);
    }
}
