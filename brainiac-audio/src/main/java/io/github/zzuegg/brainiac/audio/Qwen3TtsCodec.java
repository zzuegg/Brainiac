package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The Qwen3-TTS 12 Hz speech tokenizer ({@code Qwen/Qwen3-TTS-Tokenizer-12Hz}, the {@code
 * speech_tokenizer} folder of every Qwen3-TTS-12Hz model): 24 kHz audio as 12.5 frames a
 * second of 16 codes each, the first a semantic code and the rest acoustic refinements.
 *
 * <p>{@link #decode} turns codes into audio as the reference does: the split residual
 * quantizer's vectors, a causal convolution, a causal transformer attending to the last 72
 * frames, two ConvNeXt upsampling stages and a DAC-style decoder with SnakeBeta activations,
 * run over chunks of 300 frames with 25 frames of context. {@link #encode} turns audio into
 * codes with the Mimi encoder (Kyutai, 2024) the tokenizer was trained with: a causal SEANet,
 * a causal transformer, a strided convolution and nearest-codeword quantization. Qwen3-TTS
 * encodes a reference voice this way to clone it.
 */
public final class Qwen3TtsCodec implements AutoCloseable {

    /** Frames decoded at a time, and the frames of context before each chunk, as the reference. */
    static final int CHUNK = 300;
    static final int CONTEXT = 25;

    private final Weights weights;
    private final Device device;
    private final DType dtype;
    private final int sampleRate;
    private final int hop;
    private final int codebooks;
    private final Decoder decoder;
    private final Encoder encoder;
    private final Map<String, Var> cache = new java.util.HashMap<>();

    /** A weight as stored, read once. */
    private Var w(String name) {
        return cache.computeIfAbsent(name, weights::read);
    }

    /** A linear layer's weight as {@code [in, out]}, read once. */
    private Var lin(String name) {
        return cache.computeIfAbsent("linear:" + name, n -> weights.linear(name));
    }

    private Qwen3TtsCodec(Map<String, Object> config, WeightSource source, Device device, DType dtype) {
        this.device = device;
        this.dtype = dtype;
        this.weights = new Weights(source, device, dtype);
        try {
            sampleRate = (int) Json.integer(config, "output_sample_rate", 24_000);
            hop = (int) Json.integer(config, "decode_upsample_rate", 1920);
            Map<String, Object> dc = Json.object(config.get("decoder_config"));
            codebooks = (int) Json.integer(dc, "num_quantizers", 16);
            decoder = new Decoder(dc);
            encoder = config.get("encoder_config") instanceof Map<?, ?> ec && weights.contains(
                    "encoder.encoder.layers.0.conv.weight")
                    ? new Encoder(Json.object(ec), (int) Json.integer(config, "encoder_valid_num_quantizers", 16))
                    : null;
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    /**
     * Loads a speech tokenizer folder ({@code config.json} with {@code model_type}
     * {@code qwen3_tts_tokenizer_12hz}, and {@code model.safetensors}).
     *
     * @param dir the folder
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the codec
     */
    public static Qwen3TtsCodec load(Path dir, Device device, DType dtype) {
        if (!isCodec(dir)) {
            throw new IllegalArgumentException(dir + " is not a Qwen3-TTS 12 Hz speech tokenizer (config.json with"
                    + " model_type qwen3_tts_tokenizer_12hz)");
        }
        try (WeightSource source = WeightSource.open(dir)) {
            return new Qwen3TtsCodec(Json.object(Json.read(dir.resolve("config.json"))), source, device, dtype);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Whether a folder holds the 12 Hz speech tokenizer.
     *
     * @param dir a folder
     * @return whether {@link #load} reads it
     */
    public static boolean isCodec(Path dir) {
        Path c = dir.resolve("config.json");
        try {
            return Files.isRegularFile(c)
                    && "qwen3_tts_tokenizer_12hz".equals(Json.object(Json.read(c)).get("model_type"));
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /** @return the sample rate of the audio */
    public int sampleRate() {
        return sampleRate;
    }

    /** @return audio samples per frame */
    public int hop() {
        return hop;
    }

    /** @return codes per frame */
    public int codebooks() {
        return codebooks;
    }

    /** @return whether {@link #encode} works: the folder has the encoder's weights */
    public boolean canEncode() {
        return encoder != null;
    }

    /**
     * Decodes frames to audio.
     *
     * @param codes {@code [frames][codebooks]}
     * @return {@code frames * hop} samples in [-1, 1]
     */
    public float[] decode(int[][] codes) {
        int frames = codes.length;
        if (frames == 0) {
            return new float[0];
        }
        float[] out = new float[frames * hop];
        int at = 0;
        for (int start = 0; start < frames; start += CHUNK) {
            int end = Math.min(start + CHUNK, frames);
            int context = start - CONTEXT > 0 ? CONTEXT : start;
            int[][] part = java.util.Arrays.copyOfRange(codes, start - context, end);
            float[] wave = Var.noGrad(() -> decoder.forward(part)).value().to(DType.FLOAT32).toArray();
            int n = (end - start) * hop;
            System.arraycopy(wave, context * hop, out, at, n);
            at += n;
        }
        return out;
    }

    /**
     * Encodes audio to frames.
     *
     * @param audio samples at {@link #sampleRate()}
     * @return {@code [ceil(samples / hop)][codebooks]}
     * @throws IllegalStateException if the folder has no encoder weights
     */
    public int[][] encode(float[] audio) {
        if (encoder == null) {
            throw new IllegalStateException("this speech tokenizer has no encoder weights");
        }
        if (audio.length == 0) {
            return new int[0][];
        }
        return Var.noGrad(() -> encoder.encode(audio));
    }

    @Override
    public void close() {
        weights.close();
    }

    // ---------------------------------------------------------------- shared pieces

    /** A convolution padded on the left so each output sees only the past, as both halves pad it. */
    private final class CausalConv {
        final Var w;
        final Var b;
        final int stride;
        final int dilation;
        final int kernel;
        final boolean replicate;

        CausalConv(String name, int stride, int dilation, boolean replicate) {
            this.w = weights.read(name + ".weight");
            this.b = weights.optional(name + ".bias");
            this.stride = stride;
            this.dilation = dilation;
            this.kernel = (w.shape().dim(2) - 1) * dilation + 1;
            this.replicate = replicate;
        }

        Var apply(Var x) {
            // depthwise convolutions see one input channel per group
            int groups = x.shape().dim(1) / w.shape().dim(1);
            int length = x.shape().dim(2);
            int pad = kernel - stride;
            int frames = (int) Math.ceil((double) (length - kernel + pad) / stride + 1) - 1;
            int extra = frames * stride + kernel - pad - length;
            Var in = x;
            if (replicate) {
                in = Functional.padAxis(x, 2, pad, extra, io.github.zzuegg.brainiac.nn.PadMode.REPLICATE, 0);
                return Functional.conv(in, w, b, new int[] {stride}, new int[] {0}, new int[] {0},
                        new int[] {dilation}, groups);
            }
            return Functional.conv(in, w, b, new int[] {stride}, new int[] {pad}, new int[] {extra},
                    new int[] {dilation}, groups);
        }
    }

    /** A transposed convolution with the extra samples at its end cut, as the reference's causal one. */
    private Var causalTranspose(Var x, Var w, Var b, int stride) {
        int k = w.shape().dim(2);
        return Functional.convTranspose(x, w, b, new int[] {stride}, new int[] {0}, new int[] {k - stride},
                new int[] {0}, new int[] {1}, 1);
    }

    private static Var rmsNorm(Var x, Var weight, float eps) {
        int last = x.shape().rank() - 1;
        return x.div(x.square().mean(last, true).add(eps).sqrt()).mul(weight);
    }

    private static Var layerNorm(Var x, Var w, Var b, float eps) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        return centred.div(centred.square().mean(last, true).add(eps).sqrt()).mul(w).add(b);
    }

    /** {@code [1, 1, t, t]}: 0 where query {@code i} sees key {@code j} ({@code 0 <= i - j < window}), else masked. */
    private Tensor windowMask(int t, int window) {
        float[] m = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                m[i * t + j] = j <= i && i - j < window ? 0 : -1e9f;
            }
        }
        return Tensor.of(Shape.of(1, 1, t, t), m, device).to(dtype);
    }

    /** Multi-head attention over {@code [1, t, hidden]} with rotary positions and a mask. */
    private Var attention(Var x, String p, int heads, int kvHeads, int headDim, float theta, Tensor mask) {
        int t = x.shape().dim(1);
        Var q = x.matmul(lin(p + "q_proj.weight"));
        Var k = x.matmul(lin(p + "k_proj.weight"));
        Var v = x.matmul(lin(p + "v_proj.weight"));
        q = Functional.rotary(q.reshape(1, t, heads, headDim).permute(0, 2, 1, 3), theta, 0);
        k = Functional.rotary(k.reshape(1, t, kvHeads, headDim).permute(0, 2, 1, 3), theta, 0);
        v = v.reshape(1, t, kvHeads, headDim).permute(0, 2, 1, 3);
        Var o = Functional.attention(q, k, v, mask, (float) (1 / Math.sqrt(headDim)), 0, null);
        return o.permute(0, 2, 1, 3).reshape(1, t, heads * headDim).matmul(lin(p + "o_proj.weight"));
    }

    // ---------------------------------------------------------------- decoder

    private final class SnakeBeta {
        final Var alpha;
        final Var inverseBeta;

        SnakeBeta(String name) {
            float[] a = weights.floats(name + ".alpha");
            float[] b = weights.floats(name + ".beta");
            for (int i = 0; i < a.length; i++) {
                a[i] = (float) Math.exp(a[i]);
                b[i] = (float) (1 / (Math.exp(b[i]) + 1e-9));
            }
            alpha = Var.of(Tensor.of(Shape.of(1, a.length, 1), a, device).to(dtype));
            inverseBeta = Var.of(Tensor.of(Shape.of(1, b.length, 1), b, device).to(dtype));
        }

        Var apply(Var x) {
            return x.add(x.mul(alpha).sin().square().mul(inverseBeta));
        }
    }

    private record Unit(SnakeBeta act1, CausalConv conv1, SnakeBeta act2, CausalConv conv2) {
        Var apply(Var x) {
            return x.add(conv2.apply(act2.apply(conv1.apply(act1.apply(x)))));
        }
    }

    private record Block(SnakeBeta act, Var up, Var upB, int stride, List<Unit> units) {
    }

    private record ConvNeXt(CausalConv dw, Var normW, Var normB, Var pw1, Var pw1b, Var pw2, Var pw2b, Var gamma) {
    }

    private final class Decoder {
        final Var[] codebooks;                  // per quantizer, [size, dim]
        final Var firstOut;
        final Var restOut;
        final CausalConv preConv;
        final int heads;
        final int kvHeads;
        final int headDim;
        final int window;
        final float theta;
        final float eps;
        final int layers;
        final List<Var> upsample = new ArrayList<>();
        final List<ConvNeXt> convNexts = new ArrayList<>();
        final List<Integer> upFactors = new ArrayList<>();
        final CausalConv first;
        final List<Block> blocks = new ArrayList<>();
        final SnakeBeta lastAct;
        final CausalConv last;

        Decoder(Map<String, Object> c) {
            int quantizers = (int) Json.integer(c, "num_quantizers", 16);
            codebooks = new Var[quantizers];
            for (int q = 0; q < quantizers; q++) {
                String p = q == 0 ? "decoder.quantizer.rvq_first.vq.layers.0._codebook."
                        : "decoder.quantizer.rvq_rest.vq.layers." + (q - 1) + "._codebook.";
                codebooks[q] = codebook(p + "embedding_sum", p + "cluster_usage");
            }
            firstOut = conv1x1("decoder.quantizer.rvq_first.output_proj.weight");
            restOut = conv1x1("decoder.quantizer.rvq_rest.output_proj.weight");
            preConv = new CausalConv("decoder.pre_conv.conv", 1, 1, false);
            int hidden = (int) Json.integer(c, "hidden_size", 1024);
            heads = (int) Json.integer(c, "num_attention_heads", 16);
            kvHeads = (int) Json.integer(c, "num_key_value_heads", heads);
            headDim = c.get("head_dim") instanceof Number n ? n.intValue() : hidden / heads;
            window = (int) Json.integer(c, "sliding_window", 72);
            theta = ((Number) c.getOrDefault("rope_theta", 10_000)).floatValue();
            eps = ((Number) c.getOrDefault("rms_norm_eps", 1e-5)).floatValue();
            layers = (int) Json.integer(c, "num_hidden_layers", 8);
            List<Object> ratios = Json.array(c.get("upsampling_ratios"));
            for (int i = 0; i < ratios.size(); i++) {
                String p = "decoder.upsample." + i + ".";
                upsample.add(weights.read(p + "0.conv.weight"));
                upsample.add(weights.read(p + "0.conv.bias"));
                upFactors.add(((Number) ratios.get(i)).intValue());
                String n = p + "1.";
                convNexts.add(new ConvNeXt(new CausalConv(n + "dwconv.conv", 1, 1, false), weights.read(n + "norm.weight"),
                        weights.read(n + "norm.bias"), weights.linear(n + "pwconv1.weight"), weights.read(n + "pwconv1.bias"),
                        weights.linear(n + "pwconv2.weight"), weights.read(n + "pwconv2.bias"), weights.read(n + "gamma")));
            }
            first = new CausalConv("decoder.decoder.0.conv", 1, 1, false);
            List<Object> rates = Json.array(c.get("upsample_rates"));
            for (int i = 0; i < rates.size(); i++) {
                String p = "decoder.decoder." + (i + 1) + ".block.";
                int r = ((Number) rates.get(i)).intValue();
                List<Unit> units = new ArrayList<>();
                int[] dilations = {1, 3, 9};
                for (int u = 0; u < 3; u++) {
                    String up = p + (u + 2) + ".";
                    units.add(new Unit(new SnakeBeta(up + "act1"), new CausalConv(up + "conv1.conv", 1, dilations[u], false),
                            new SnakeBeta(up + "act2"), new CausalConv(up + "conv2.conv", 1, 1, false)));
                }
                blocks.add(new Block(new SnakeBeta(p + "0"), weights.read(p + "1.conv.weight"),
                        weights.read(p + "1.conv.bias"), r, units));
            }
            lastAct = new SnakeBeta("decoder.decoder." + (rates.size() + 1));
            last = new CausalConv("decoder.decoder." + (rates.size() + 2) + ".conv", 1, 1, false);
            String p = "decoder.pre_transformer.";
            for (String n : List.of("input_proj.weight", "output_proj.weight")) {
                lin(p + n);
            }
            for (String n : List.of("input_proj.bias", "output_proj.bias", "norm.weight")) {
                w(p + n);
            }
            for (int l = 0; l < layers; l++) {
                String lp = p + "layers." + l + ".";
                for (String n : List.of("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")) {
                    lin(lp + n + ".weight");
                }
                for (String n : List.of("input_layernorm.weight", "post_attention_layernorm.weight",
                        "self_attn_layer_scale.scale", "mlp_layer_scale.scale")) {
                    w(lp + n);
                }
            }
        }

        Var forward(int[][] codes) {
            int t = codes.length;
            Var semantic = null;
            Var rest = null;
            for (int q = 0; q < codebooks.length; q++) {
                float[] idx = new float[t];
                int size = codebooks[q].shape().dim(0);
                for (int f = 0; f < t; f++) {
                    int c = Math.max(0, codes[f][q]);
                    if (c >= size) {
                        throw new IllegalArgumentException("code " + c + " is outside codebook " + q + " of " + size);
                    }
                    idx[f] = c;
                }
                Var e = codebooks[q].gather(0, Tensor.of(Shape.of(t), idx, device));     // [t, dim]
                if (q == 0) {
                    semantic = e;
                } else {
                    rest = rest == null ? e : rest.add(e);
                }
            }
            Var z = semantic.matmul(firstOut);
            if (rest != null) {
                z = z.add(rest.matmul(restOut));
            }
            Var x = preConv.apply(z.permute(1, 0).reshape(1, z.shape().dim(1), t));     // [1, latent, t]
            x = Vocos.settle(transformer(x.permute(0, 2, 1))).permute(0, 2, 1);
            for (int i = 0; i < convNexts.size(); i++) {
                x = causalTranspose(x, upsample.get(2 * i), upsample.get(2 * i + 1), upFactors.get(i));
                ConvNeXt n = convNexts.get(i);
                Var h = n.dw().apply(x).permute(0, 2, 1);
                h = layerNorm(h, n.normW(), n.normB(), 1e-6f).matmul(n.pw1()).add(n.pw1b()).gelu()
                        .matmul(n.pw2()).add(n.pw2b()).mul(n.gamma());
                x = Vocos.settle(x.add(h.permute(0, 2, 1)));
            }
            x = first.apply(x);
            for (Block b : blocks) {
                x = causalTranspose(b.act().apply(x), b.up(), b.upB(), b.stride());
                for (Unit u : b.units()) {
                    x = Vocos.settle(u.apply(x));
                }
            }
            return last.apply(lastAct.apply(x)).clamp(-1, 1).reshape(x.shape().dim(2));
        }

        private Var transformer(Var x) {
            int t = x.shape().dim(1);
            String p = "decoder.pre_transformer.";
            Tensor mask = windowMask(t, window);
            Var h = x.matmul(lin(p + "input_proj.weight")).add(w(p + "input_proj.bias"));
            for (int l = 0; l < layers; l++) {
                String lp = p + "layers." + l + ".";
                Var a = attention(rmsNorm(h, w(lp + "input_layernorm.weight"), eps), lp + "self_attn.",
                        heads, kvHeads, headDim, theta, mask);
                h = h.add(a.mul(w(lp + "self_attn_layer_scale.scale")));
                Var m = rmsNorm(h, w(lp + "post_attention_layernorm.weight"), eps);
                m = m.matmul(lin(lp + "mlp.gate_proj.weight")).silu()
                        .mul(m.matmul(lin(lp + "mlp.up_proj.weight")))
                        .matmul(lin(lp + "mlp.down_proj.weight"));
                h = Vocos.settle(h.add(m.mul(w(lp + "mlp_layer_scale.scale"))));
            }
            h = rmsNorm(h, w(p + "norm.weight"), eps);
            return h.matmul(lin(p + "output_proj.weight")).add(w(p + "output_proj.bias"));
        }
    }

    /** A Mimi-style codebook: its running sums divided by how often each code was used. */
    private Var codebook(String sum, String usage) {
        Shape s = weights.shape(sum);
        float[] e = weights.floats(sum);
        float[] u = weights.floats(usage);
        int dim = s.dim(1);
        for (int i = 0; i < s.dim(0); i++) {
            float d = Math.max(u[i], 1e-5f);
            for (int j = 0; j < dim; j++) {
                e[i * dim + j] /= d;
            }
        }
        return Var.of(Tensor.of(s, e, device).to(dtype));
    }

    /** A bias-free 1x1 convolution {@code [out, in, 1]} as an {@code [in, out]} matrix. */
    private Var conv1x1(String name) {
        Shape s = weights.shape(name);
        float[] w = weights.floats(name);
        int out = s.dim(0);
        int in = s.dim(1);
        float[] t = new float[w.length];
        for (int o = 0; o < out; o++) {
            for (int i = 0; i < in; i++) {
                t[i * out + o] = w[o * in + i];
            }
        }
        return Var.of(Tensor.of(Shape.of(in, out), t, device).to(dtype));
    }

    // ---------------------------------------------------------------- encoder

    private record Resnet(CausalConv a, CausalConv b) {
    }

    private final class Encoder {
        final List<Object> seanet = new ArrayList<>();      // CausalConv, Resnet or "elu"
        final CausalConv downsample;
        final int layers;
        final int heads;
        final int kvHeads;
        final int headDim;
        final float theta;
        final float eps;
        final Var semanticIn;
        final Var acousticIn;
        final Var[] semantic;
        final Var[] acoustic;
        final int valid;

        Encoder(Map<String, Object> c, int valid) {
            this.valid = valid;
            int residual = (int) Json.integer(c, "num_residual_layers", 1);
            int growth = (int) Json.integer(c, "dilation_growth_rate", 2);
            List<Object> ratios = Json.array(c.get("upsampling_ratios"));
            int at = 0;
            String p = "encoder.encoder.layers.";
            seanet.add(new CausalConv(p + at++ + ".conv", 1, 1, false));
            for (int r = ratios.size() - 1; r >= 0; r--) {
                for (int j = 0; j < residual; j++) {
                    String rp = p + at++ + ".block.";
                    seanet.add(new Resnet(new CausalConv(rp + "1.conv", 1, (int) Math.pow(growth, j), false),
                            new CausalConv(rp + "3.conv", 1, 1, false)));
                }
                seanet.add("elu");
                at++;
                seanet.add(new CausalConv(p + at++ + ".conv", ((Number) ratios.get(r)).intValue(), 1, false));
            }
            seanet.add("elu");
            at++;
            seanet.add(new CausalConv(p + at + ".conv", 1, 1, false));
            int hidden = (int) Json.integer(c, "hidden_size", 512);
            heads = (int) Json.integer(c, "num_attention_heads", 8);
            kvHeads = (int) Json.integer(c, "num_key_value_heads", heads);
            headDim = c.get("head_dim") instanceof Number n ? n.intValue() : hidden / heads;
            theta = ((Number) c.getOrDefault("rope_theta", 10_000)).floatValue();
            eps = ((Number) c.getOrDefault("norm_eps", 1e-5)).floatValue();
            layers = (int) Json.integer(c, "num_hidden_layers", 8);
            int stride = 2;
            downsample = new CausalConv("encoder.downsample.conv", stride, 1, true);
            String q = "encoder.quantizer.";
            String sem = q + "semantic_residual_vector_quantizer.";
            String aco = q + "acoustic_residual_vector_quantizer.";
            semanticIn = weights.contains(sem + "input_proj.weight") ? conv1x1(sem + "input_proj.weight") : null;
            acousticIn = weights.contains(aco + "input_proj.weight") ? conv1x1(aco + "input_proj.weight") : null;
            int semanticCount = (int) Json.integer(c, "num_semantic_quantizers", 1);
            semantic = new Var[semanticCount];
            for (int i = 0; i < semanticCount; i++) {
                semantic[i] = codebook(sem + "layers." + i + ".codebook.embed_sum", sem + "layers." + i + ".codebook.cluster_usage");
            }
            acoustic = new Var[Math.max(0, valid - semanticCount)];
            for (int i = 0; i < acoustic.length; i++) {
                acoustic[i] = codebook(aco + "layers." + i + ".codebook.embed_sum", aco + "layers." + i + ".codebook.cluster_usage");
            }
            for (int l = 0; l < layers; l++) {
                String lp = "encoder.encoder_transformer.layers." + l + ".";
                for (String n : List.of("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                        "mlp.fc1", "mlp.fc2")) {
                    lin(lp + n + ".weight");
                }
                for (String n : List.of("input_layernorm.weight", "input_layernorm.bias", "post_attention_layernorm.weight",
                        "post_attention_layernorm.bias", "self_attn_layer_scale.scale", "mlp_layer_scale.scale")) {
                    w(lp + n);
                }
            }
        }

        int[][] encode(float[] audio) {
            Var x = Var.of(Tensor.of(Shape.of(1, 1, audio.length), audio, device).to(dtype));
            for (Object o : seanet) {
                if (o instanceof CausalConv c) {
                    x = c.apply(x);
                } else if (o instanceof Resnet r) {
                    x = x.add(r.b().apply(elu(r.a().apply(elu(x)))));
                } else {
                    x = elu(x);
                }
                x = Vocos.settle(x);
            }
            Var h = transformer(x.permute(0, 2, 1));
            Var z = downsample.apply(h.permute(0, 2, 1));                // [1, hidden, frames]
            int frames = z.shape().dim(2);
            Var e = z.reshape(z.shape().dim(1), frames).permute(1, 0).to(DType.FLOAT32);   // [frames, hidden]
            int keep = Math.min(frames, (audio.length + hop - 1) / hop);
            int[][] codes = new int[keep][semantic.length + acoustic.length];
            quantize(semanticIn == null ? e : e.matmul(semanticIn.to(DType.FLOAT32)), semantic, codes, 0, keep);
            quantize(acousticIn == null ? e : e.matmul(acousticIn.to(DType.FLOAT32)), acoustic, codes, semantic.length,
                    keep);
            return codes;
        }

        /** Residual quantization: per frame, the nearest codeword, then the rest of the vector. */
        private void quantize(Var x, Var[] books, int[][] codes, int column, int keep) {
            Var residual = x;
            for (int b = 0; b < books.length; b++) {
                Var book = books[b].to(DType.FLOAT32);                  // [size, dim]
                // |r - c|^2 = |r|^2 - 2 r.c + |c|^2; |r|^2 does not change the nearest
                Var score = residual.matmul(book.permute(1, 0)).mul(-2f)
                        .add(book.square().sum(1, false).reshape(1, book.shape().dim(0)));
                float[] s = score.value().toArray();
                int size = book.shape().dim(0);
                int frames = residual.shape().dim(0);
                float[] idx = new float[frames];
                for (int f = 0; f < frames; f++) {
                    int best = 0;
                    for (int k = 1; k < size; k++) {
                        if (s[f * size + k] < s[f * size + best]) {
                            best = k;
                        }
                    }
                    idx[f] = best;
                    if (f < keep) {
                        codes[f][column + b] = best;
                    }
                }
                residual = Vocos.settle(residual.sub(book.gather(0, Tensor.of(Shape.of(frames), idx, device))));
            }
        }

        private Var transformer(Var x) {
            int t = x.shape().dim(1);
            String p = "encoder.encoder_transformer.layers.";
            Tensor mask = windowMask(t, Integer.MAX_VALUE);
            Var h = x;
            for (int l = 0; l < layers; l++) {
                String lp = p + l + ".";
                Var n = layerNorm(h, w(lp + "input_layernorm.weight"), w(lp + "input_layernorm.bias"), eps);
                h = h.add(attention(n, lp + "self_attn.", heads, kvHeads, headDim, theta, mask)
                        .mul(w(lp + "self_attn_layer_scale.scale")));
                Var m = layerNorm(h, w(lp + "post_attention_layernorm.weight"),
                        w(lp + "post_attention_layernorm.bias"), eps);
                m = m.matmul(lin(lp + "mlp.fc1.weight")).gelu().matmul(lin(lp + "mlp.fc2.weight"));
                h = Vocos.settle(h.add(m.mul(w(lp + "mlp_layer_scale.scale"))));
            }
            return h;
        }

        private Var elu(Var x) {
            return x.activation(ActivationOp.ELU, 1f);
        }
    }
}
