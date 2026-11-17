package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Map;

/**
 * Stable Diffusion's denoising UNet ({@code UNet2DConditionModel} in diffusers): residual
 * blocks conditioned on the timestep, and transformer blocks that cross-attend to the text
 * embeddings, on a down path, a middle and an up path joined by skip connections. Covers
 * Stable Diffusion 1.x, 2.x and XL (with its extra size and pooled-text conditioning).
 */
public final class UNet {

    /**
     * The shape of a UNet, as diffusers' {@code unet/config.json} describes it.
     *
     * @param inChannels latent channels in
     * @param outChannels latent channels out
     * @param channels the width of each resolution level
     * @param crossAttention whether each level's down block (and its mirrored up block) has transformers
     * @param layersPerBlock residual blocks per down block (up blocks have one more)
     * @param heads attention heads per level
     * @param transformerDepth transformer blocks per transformer, per level
     * @param contextDim the width of the text embeddings
     * @param linearProjection whether transformers project in and out with linear layers instead of 1x1 convolutions
     * @param groups group-norm groups
     * @param eps the residual blocks' group-norm epsilon
     * @param addTimeDim SDXL's per-number size embedding width, or 0 without size conditioning
     * @param flipSinToCos whether the timestep embedding puts cosines first
     * @param freqShift the timestep embedding's frequency shift
     */
    public record Config(int inChannels, int outChannels, int[] channels, boolean[] crossAttention,
            int layersPerBlock, int[] heads, int[] transformerDepth, int contextDim, boolean linearProjection,
            int groups, float eps, int addTimeDim, boolean flipSinToCos, int freqShift) {

        /**
         * Reads diffusers' UNet config.
         *
         * @param c the parsed {@code config.json}
         * @return the config
         */
        public static Config fromJson(Map<String, Object> c) {
            int[] channels = ints(c.get("block_out_channels"), 0, null);
            int levels = channels.length;
            List<Object> types = Json.array(c.get("down_block_types"));
            boolean[] cross = new boolean[levels];
            for (int i = 0; i < levels; i++) {
                cross[i] = ((String) types.get(i)).startsWith("CrossAttn");
            }
            Object heads = c.get("num_attention_heads") != null ? c.get("num_attention_heads") : c.get("attention_head_dim");
            String addType = Json.string(c, "addition_embed_type", null);
            Object ctx = c.get("cross_attention_dim");
            int contextDim = ctx instanceof List<?> l ? ((Number) l.get(0)).intValue() : ((Number) ctx).intValue();
            return new Config((int) Json.integer(c, "in_channels", 4), (int) Json.integer(c, "out_channels", 4),
                    channels, cross, (int) Json.integer(c, "layers_per_block", 2), ints(heads, levels, 8),
                    ints(c.get("transformer_layers_per_block"), levels, 1), contextDim,
                    Boolean.TRUE.equals(c.get("use_linear_projection")), (int) Json.integer(c, "norm_num_groups", 32),
                    (float) Json.number(c, "norm_eps", 1e-5),
                    "text_time".equals(addType) ? (int) Json.integer(c, "addition_time_embed_dim", 256) : 0,
                    !Boolean.FALSE.equals(c.get("flip_sin_to_cos")), (int) Json.integer(c, "freq_shift", 0));
        }

        private static int[] ints(Object v, int n, Integer otherwise) {
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

        /**
         * Returns the width of the time embedding.
         *
         * @return four times the first level's width
         */
        public int timeDim() {
            return channels[0] * 4;
        }
    }

    private record Resnet(Nn.GroupNorm norm1, Nn.Conv2d conv1, Nn.Linear time, Nn.GroupNorm norm2, Nn.Conv2d conv2,
            Nn.Conv2d shortcut) {
    }

    private record Block(Nn.LayerNorm norm1, Nn.Linear q1, Nn.Linear k1, Nn.Linear v1, Nn.Linear out1,
            Nn.LayerNorm norm2, Nn.Linear q2, Nn.Linear k2, Nn.Linear v2, Nn.Linear out2, Nn.LayerNorm norm3,
            Nn.Linear ffValue, Nn.Linear ffGate, Nn.Linear ffOut) {
    }

    private record Transformer(Nn.GroupNorm norm, Nn.Linear projIn, Nn.Conv2d convIn, List<Block> blocks,
            Nn.Linear projOut, Nn.Conv2d convOut, int heads) {
    }

    /** A residual block, optionally followed by a transformer. */
    private record Layer(Resnet resnet, Transformer transformer) {
    }

    private record Level(List<Layer> layers, Nn.Conv2d resample) {
    }

    private final Config config;
    private final List<Tensor> weights;
    private final Nn.Conv2d convIn;
    private final Nn.Linear time1;
    private final Nn.Linear time2;
    private final Nn.Linear add1;
    private final Nn.Linear add2;
    private final List<Level> down;
    private final List<Layer> mid;
    private final List<Level> up;
    private final Nn.GroupNorm normOut;
    private final Nn.Conv2d convOut;

    private UNet(Config config, Weights w) {
        this.config = config;
        this.weights = w.all;
        convIn = w.conv("conv_in");
        time1 = w.linear("time_embedding.linear_1");
        time2 = w.linear("time_embedding.linear_2");
        add1 = config.addTimeDim() > 0 ? w.linear("add_embedding.linear_1") : null;
        add2 = config.addTimeDim() > 0 ? w.linear("add_embedding.linear_2") : null;
        int levels = config.channels().length;
        down = new ArrayList<>();
        for (int i = 0; i < levels; i++) {
            Weights b = w.at("down_blocks." + i + ".");
            List<Layer> layers = new ArrayList<>();
            for (int j = 0; j < config.layersPerBlock(); j++) {
                layers.add(new Layer(resnet(b.at("resnets." + j + ".")),
                        config.crossAttention()[i] ? transformer(b.at("attentions." + j + "."), config.heads()[i]) : null));
            }
            Nn.Conv2d downsample = i < levels - 1 ? b.conv("downsamplers.0.conv", 2, 1) : null;
            down.add(new Level(layers, downsample));
        }
        Weights m = w.at("mid_block.");
        mid = List.of(new Layer(resnet(m.at("resnets.0.")), transformer(m.at("attentions.0."), config.heads()[levels - 1])),
                new Layer(resnet(m.at("resnets.1.")), null));
        up = new ArrayList<>();
        for (int i = 0; i < levels; i++) {
            int level = levels - 1 - i;
            Weights b = w.at("up_blocks." + i + ".");
            List<Layer> layers = new ArrayList<>();
            for (int j = 0; j <= config.layersPerBlock(); j++) {
                layers.add(new Layer(resnet(b.at("resnets." + j + ".")), config.crossAttention()[level]
                        ? transformer(b.at("attentions." + j + "."), config.heads()[level]) : null));
            }
            Nn.Conv2d upsample = i < levels - 1 ? b.conv("upsamplers.0.conv") : null;
            up.add(new Level(layers, upsample));
        }
        normOut = w.groupNorm("conv_norm_out", config.groups(), config.eps());
        convOut = w.conv("conv_out");
    }

    /**
     * Loads a UNet in diffusers' naming.
     *
     * @param w the weights at the UNet's prefix
     * @param config its config
     * @return the model
     */
    static UNet load(Weights w, Config config) {
        return new UNet(config, w);
    }

    private Resnet resnet(Weights w) {
        return new Resnet(w.groupNorm("norm1", config.groups(), config.eps()), w.conv("conv1"),
                w.linear("time_emb_proj"), w.groupNorm("norm2", config.groups(), config.eps()), w.conv("conv2"),
                w.has("conv_shortcut.weight") ? w.conv("conv_shortcut") : null);
    }

    private Transformer transformer(Weights w, int heads) {
        List<Block> blocks = new ArrayList<>();
        for (int i = 0; w.has("transformer_blocks." + i + ".norm1.weight"); i++) {
            Weights b = w.at("transformer_blocks." + i + ".");
            Nn.Linear[] ff = b.linearHalves("ff.net.0.proj");
            blocks.add(new Block(b.layerNorm("norm1", 1e-5f), b.linear("attn1.to_q"), b.linear("attn1.to_k"),
                    b.linear("attn1.to_v"), b.linear("attn1.to_out.0"), b.layerNorm("norm2", 1e-5f),
                    b.linear("attn2.to_q"), b.linear("attn2.to_k"), b.linear("attn2.to_v"), b.linear("attn2.to_out.0"),
                    b.layerNorm("norm3", 1e-5f), ff[0], ff[1], b.linear("ff.net.2")));
        }
        boolean linear = w.shape("proj_in.weight").rank() == 2;
        return new Transformer(w.groupNorm("norm", config.groups(), 1e-6f),
                linear ? w.linear("proj_in") : null, linear ? null : w.conv("proj_in"), blocks,
                linear ? w.linear("proj_out") : null, linear ? null : w.conv("proj_out"), heads);
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
     * Returns the type the weights are held in.
     *
     * @return the type
     */
    public DType dtype() {
        return convIn.weight().dtype();
    }

    /**
     * Predicts the noise (or velocity) in a batch of latents.
     *
     * @param x the noisy latents {@code [batch, inChannels, h, w]}, h and w multiples of 2^(levels - 1)
     * @param timesteps each latent's timestep, {@code 0..999}
     * @param context the text embeddings {@code [batch, tokens, contextDim]}
     * @param pooled SDXL's pooled text embedding {@code [batch, dim]}, or null
     * @param timeIds SDXL's six size numbers per latent {@code [batch][6]}, or null
     * @return the prediction, shaped like {@code x}
     */
    public Tensor forward(Tensor x, float[] timesteps, Tensor context, Tensor pooled, float[][] timeIds) {
        DType type = dtype();
        Tensor t = Nn.scoped(weights, () -> {
            Tensor e = sinusoids(timesteps, config.channels()[0], x).to(type);
            e = time2.apply(Nn.silu(time1.apply(e)));
            if (add1 != null) {
                int n = timeIds[0].length;
                float[] flat = new float[timeIds.length * n];
                for (int b = 0; b < timeIds.length; b++) {
                    System.arraycopy(timeIds[b], 0, flat, b * n, n);
                }
                Tensor ids = sinusoids(flat, config.addTimeDim(), x).reshape(timeIds.length, n * config.addTimeDim());
                Tensor a = Tensor.concat(1, pooled.to(DType.FLOAT32), ids).to(type);
                e = e.add(add2.apply(Nn.silu(add1.apply(a))));
            }
            // every residual block takes silu(embedding)
            return Nn.silu(e);
        }, x, context, pooled);
        Tensor ctx = context.to(type);
        Tensor h0 = x.to(type);
        Tensor h = Nn.scoped(weights, () -> convIn.apply(h0), h0, x);
        if (h0 != x) {
            h0.release();
        }
        Deque<Tensor> skips = new ArrayDeque<>();
        skips.push(h);
        for (Level level : down) {
            for (Layer layer : level.layers()) {
                h = layer(layer, h, t, ctx, null);
                skips.push(h);
            }
            if (level.resample() != null) {
                Tensor in = h;
                h = Nn.scoped(weights, () -> level.resample().apply(in), in);
                skips.push(h);
            }
        }
        Tensor downOut = h;   // also the first skip of the up path
        for (Layer layer : mid) {
            Tensor in = h;
            h = layer(layer, h, t, ctx, null);
            if (in != downOut) {
                in.release();
            }
        }
        for (Level level : up) {
            for (Layer layer : level.layers()) {
                Tensor skip = skips.pop();
                Tensor in = h;
                h = layer(layer, h, t, ctx, skip);
                in.release();
                skip.release();
            }
            if (level.resample() != null) {
                Tensor in = h;
                h = Nn.scoped(weights, () -> level.resample().apply(Nn.upsample2x(in)), in);
                in.release();
            }
        }
        Tensor last = h;
        Tensor out = Nn.scoped(weights, () -> convOut.apply(Nn.silu(normOut.apply(last))), last);
        last.release();
        t.release();
        if (ctx != context) {
            ctx.release();
        }
        return out;
    }

    /** A residual block (on {@code h} joined with {@code skip}, on the up path) and its transformer. */
    private Tensor layer(Layer layer, Tensor h, Tensor t, Tensor ctx, Tensor skip) {
        Tensor r = Nn.scoped(weights, () -> resnet(layer.resnet(), skip == null ? h : Tensor.concat(1, h, skip), t),
                h, t, skip);
        if (layer.transformer() == null) {
            return r;
        }
        Transformer tr = layer.transformer();
        Shape s = r.shape();
        int height = s.dim(2);
        int width = s.dim(3);
        Tensor x = Nn.scoped(weights, () -> {
            Tensor n = tr.norm().apply(r);
            return tr.projIn() != null ? tr.projIn().apply(Nn.tokens(n)) : Nn.tokens(tr.convIn().apply(n));
        }, r);
        for (Block b : tr.blocks()) {
            Tensor in = x;
            x = Nn.scoped(weights, () -> block(b, in, ctx, tr.heads()), in, ctx);
            in.release();
        }
        Tensor tokens = x;
        Tensor out = Nn.scoped(weights, () -> {
            Tensor y = tr.projOut() != null ? Nn.image(tr.projOut().apply(tokens), height, width)
                    : tr.convOut().apply(Nn.image(tokens, height, width));
            return y.add(r);
        }, tokens, r);
        tokens.release();
        r.release();
        return out;
    }

    private static Tensor resnet(Resnet r, Tensor x, Tensor t) {
        Tensor h = r.conv1().apply(Nn.silu(r.norm1().apply(x)));
        Tensor e = r.time().apply(t);
        h = h.add(e.reshape(e.shape().dim(0), e.shape().dim(1), 1, 1));
        h = r.conv2().apply(Nn.silu(r.norm2().apply(h)));
        return (r.shortcut() == null ? x : r.shortcut().apply(x)).add(h);
    }

    private static Tensor block(Block b, Tensor x, Tensor ctx, int heads) {
        Tensor n = b.norm1().apply(x);
        x = x.add(b.out1().apply(Nn.attention(b.q1().apply(n), b.k1().apply(n), b.v1().apply(n), heads, false)));
        n = b.norm2().apply(x);
        x = x.add(b.out2().apply(Nn.attention(b.q2().apply(n), b.k2().apply(ctx), b.v2().apply(ctx), heads, false)));
        n = b.norm3().apply(x);
        Tensor ff = b.ffValue().apply(n).mul(Nn.gelu(b.ffGate().apply(n)));
        return x.add(b.ffOut().apply(ff));
    }

    /** The sinusoidal embedding of each value, {@code [values, dim]}, in float32 on the device of {@code like}. */
    private Tensor sinusoids(float[] values, int dim, Tensor like) {
        int half = dim / 2;
        float[] out = new float[values.length * dim];
        for (int i = 0; i < values.length; i++) {
            for (int j = 0; j < half; j++) {
                double freq = Math.exp(-Math.log(10000) * j / (half - config.freqShift()));
                double a = values[i] * freq;
                int sin = config.flipSinToCos() ? half + j : j;
                int cos = config.flipSinToCos() ? j : half + j;
                out[i * dim + sin] = (float) Math.sin(a);
                out[i * dim + cos] = (float) Math.cos(a);
            }
        }
        return Tensor.of(Shape.of(values.length, dim), out, like.device());
    }
}
