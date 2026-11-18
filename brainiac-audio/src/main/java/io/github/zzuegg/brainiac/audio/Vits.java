package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.HifiGanBlock;
import io.github.zzuegg.brainiac.nn.RelativeAttention;
import io.github.zzuegg.brainiac.nn.WaveNet;
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
import java.util.Random;

/**
 * VITS (Kim et al., 2021), the network behind Meta's MMS-TTS voices: a transformer text
 * encoder with relative positions, a (stochastic) duration predictor, normalizing flows
 * from text to acoustic latents and a HiFi-GAN vocoder from latents to waveform, all run
 * as Hugging Face's {@code VitsModel} does at inference.
 *
 * <p>It loads Hugging Face folders ({@code facebook/mms-tts-eng} and the other MMS-TTS
 * languages, and VITS models converted to that layout). Weight-normalized convolutions are
 * folded when loading. Text goes in as character ids from {@link VitsTokenizer}; for
 * ready-to-use speech see {@link VitsSpeech}.
 */
public final class Vits {

    private record Conv(Tensor weight, Tensor bias, int dilation, int padBefore, int padAfter, int groups) {
    }

    private record Norm(Tensor weight, Tensor bias) {
    }

    private record Linear(Tensor weight, Tensor bias) {
        Var apply(Var x) {
            Var y = x.matmul(weight);
            return bias == null ? y : y.add(bias);
        }
    }

    private record EncoderLayer(RelativeAttention attention, Norm norm, Conv ffn1, Conv ffn2, Norm finalNorm) {
    }

    private record Coupling(Conv pre, WaveNet wavenet, Conv post) {
    }

    private record Dds(Conv[] dilated, Conv[] pointwise, Norm[] norms1, Norm[] norms2) {
    }

    private record ConvFlow(Conv pre, Dds dds, Conv proj) {
    }


    private final VitsConfig config;
    private final Device device;
    private final DType dtype;
    private final List<Tensor> weights = new ArrayList<>();
    private final WeightSource source;

    private final Tensor embedTokens;
    private final List<EncoderLayer> encoder = new ArrayList<>();
    private final Conv project;
    private final Tensor speakerEmbeddings;
    // duration predictor (stochastic)
    private Conv dpPre;
    private Conv dpProj;
    private Dds dpDds;
    private Conv dpCond;
    private float[] dpAffineTranslate;
    private float[] dpAffineLogScale;
    private final List<ConvFlow> dpFlows = new ArrayList<>();
    // duration predictor (deterministic)
    private Conv dConv1;
    private Norm dNorm1;
    private Conv dConv2;
    private Norm dNorm2;
    private Conv dProj;
    private Conv dCond;
    // flow and vocoder
    private final List<Coupling> flows = new ArrayList<>();
    private final Conv decPre;
    private final Conv decCond;
    private final List<Conv> upsamplers = new ArrayList<>();
    private final List<HifiGanBlock> resblocks = new ArrayList<>();
    private final Conv decPost;

    private Vits(VitsConfig c, Device device, DType dtype, WeightSource w) {
        this.config = c;
        this.device = device;
        this.dtype = dtype;
        this.source = w;
        embedTokens = read("text_encoder.embed_tokens.weight");
        int window = c.window();
        for (int i = 0; i < c.layers(); i++) {
            String p = "text_encoder.encoder.layers." + i + ".";
            String a = p + "attention.";
            if (window < 1) {
                throw new IllegalArgumentException("VITS without relative attention (window_size 0) is not supported");
            }
            Linear q = linear(a + "q_proj");
            Linear k = linear(a + "k_proj");
            Linear v = linear(a + "v_proj");
            Linear o = linear(a + "out_proj");
            RelativeAttention attention = new RelativeAttention(c.heads(), window, Var.of(q.weight), opt(q.bias),
                    Var.of(k.weight), opt(k.bias), Var.of(v.weight), opt(v.bias), Var.of(o.weight), opt(o.bias),
                    Var.of(read(a + "emb_rel_k").reshape(2 * window + 1, c.hidden() / c.heads())),
                    Var.of(read(a + "emb_rel_v").reshape(2 * window + 1, c.hidden() / c.heads())));
            int pl = (c.ffnKernel() - 1) / 2;
            int pr = c.ffnKernel() / 2;
            encoder.add(new EncoderLayer(attention, norm(p + "layer_norm"),
                    conv(p + "feed_forward.conv_1", 1, pl, pr, 1), conv(p + "feed_forward.conv_2", 1, pl, pr, 1),
                    norm(p + "final_layer_norm")));
        }
        project = conv("text_encoder.project", 1, 0, 0, 1);
        speakerEmbeddings = c.speakers() > 1 && c.speakerEmbedding() > 0 ? read("embed_speaker.weight") : null;
        boolean cond = c.speakerEmbedding() > 0;
        if (c.stochasticDuration()) {
            String p = "duration_predictor.";
            dpPre = conv(p + "conv_pre", 1, 0, 0, 1);
            dpProj = conv(p + "conv_proj", 1, 0, 0, 1);
            dpDds = dds(p + "conv_dds");
            dpCond = cond ? conv(p + "cond", 1, 0, 0, 1) : null;
            dpAffineTranslate = w.tensor(p + "flows.0.translate", Device.cpu(), DType.FLOAT32).toArray();
            dpAffineLogScale = w.tensor(p + "flows.0.log_scale", Device.cpu(), DType.FLOAT32).toArray();
            for (int i = 1; i <= c.durationFlows(); i++) {
                String f = p + "flows." + i + ".";
                dpFlows.add(new ConvFlow(conv(f + "conv_pre", 1, 0, 0, 1), dds(f + "conv_dds"),
                        conv(f + "conv_proj", 1, 0, 0, 1)));
            }
        } else {
            String p = "duration_predictor.";
            int k = c.durationKernel();
            dConv1 = conv(p + "conv_1", 1, k / 2, k / 2, 1);
            dNorm1 = norm(p + "norm_1");
            dConv2 = conv(p + "conv_2", 1, k / 2, k / 2, 1);
            dNorm2 = norm(p + "norm_2");
            dProj = conv(p + "proj", 1, 0, 0, 1);
            dCond = cond ? conv(p + "cond", 1, 0, 0, 1) : null;
        }
        for (int i = 0; i < c.priorFlows(); i++) {
            String p = "flow.flows." + i + ".";
            flows.add(new Coupling(conv(p + "conv_pre", 1, 0, 0, 1), wavenet(p + "wavenet", c.priorWavenetLayers()),
                    conv(p + "conv_post", 1, 0, 0, 1)));
        }
        decPre = conv("decoder.conv_pre", 1, 3, 3, 1);
        decCond = cond ? conv("decoder.cond", 1, 0, 0, 1) : null;
        int[] rates = c.upsampleRates();
        for (int i = 0; i < rates.length; i++) {
            int pad = (c.upsampleKernels()[i] - rates[i]) / 2;
            upsamplers.add(conv("decoder.upsampler." + i, 1, pad, pad, 1));
        }
        int r = 0;
        for (int i = 0; i < rates.length; i++) {
            for (int j = 0; j < c.resblockKernels().length; j++) {
                int k = c.resblockKernels()[j];
                int[] dil = c.resblockDilations()[j];
                Var[] w1 = new Var[dil.length];
                Var[] b1 = new Var[dil.length];
                Var[] w2 = new Var[dil.length];
                Var[] b2 = new Var[dil.length];
                String p = "decoder.resblocks." + r + ".";
                for (int d = 0; d < dil.length; d++) {
                    Conv c1 = conv(p + "convs1." + d, dil[d], 0, 0, 1);
                    Conv c2 = conv(p + "convs2." + d, 1, 0, 0, 1);
                    w1[d] = Var.of(c1.weight);
                    b1[d] = Var.of(c1.bias);
                    w2[d] = Var.of(c2.weight);
                    b2[d] = Var.of(c2.bias);
                }
                resblocks.add(new HifiGanBlock(k, dil, c.leakyReluSlope(), w1, b1, w2, b2));
                r++;
            }
        }
        decPost = conv("decoder.conv_post", 1, 3, 3, 1);
    }

    // ---------------------------------------------------------------- loading

    private Tensor read(String name) {
        Tensor t = source.tensor(name, device, dtype);
        weights.add(t);
        return t;
    }

    private Linear linear(String name) {
        Tensor w = source.tensor(name + ".weight", device, dtype);
        Tensor t = w.transpose();
        Tensor.evaluate(List.of(t));
        w.release();
        weights.add(t);
        return new Linear(t, source.contains(name + ".bias") ? read(name + ".bias") : null);
    }

    private Norm norm(String name) {
        return new Norm(read(name + ".weight"), read(name + ".bias"));
    }

    /**
     * A convolution's weight, folding weight normalization ({@code g * v / |v|} per output
     * channel) in either of PyTorch's spellings.
     */
    private Conv conv(String name, int dilation, int padBefore, int padAfter, int groups) {
        Tensor weight;
        if (source.contains(name + ".weight")) {
            weight = read(name + ".weight");
        } else {
            String g = source.contains(name + ".weight_g") ? name + ".weight_g"
                    : name + ".parametrizations.weight.original0";
            String v = source.contains(name + ".weight_v") ? name + ".weight_v"
                    : name + ".parametrizations.weight.original1";
            if (!source.contains(v)) {
                throw new IllegalArgumentException("the VITS weights have no " + name + ".weight");
            }
            Shape shape = source.shape(v);
            float[] gv = source.tensor(g, Device.cpu(), DType.FLOAT32).toArray();
            float[] vv = source.tensor(v, Device.cpu(), DType.FLOAT32).toArray();
            int rows = shape.dim(0);
            int per = vv.length / rows;
            for (int o = 0; o < rows; o++) {
                double sum = 0;
                for (int j = 0; j < per; j++) {
                    sum += (double) vv[o * per + j] * vv[o * per + j];
                }
                float s = (float) (gv[o] / Math.sqrt(sum));
                for (int j = 0; j < per; j++) {
                    vv[o * per + j] *= s;
                }
            }
            weight = Tensor.of(shape, vv, device).to(dtype);
            weights.add(weight);
        }
        Tensor bias = source.contains(name + ".bias") ? read(name + ".bias") : null;
        return new Conv(weight, bias, dilation, padBefore, padAfter, groups);
    }

    private Dds dds(String p) {
        int n = config.dsLayers();
        int k = config.durationKernel();
        Conv[] dil = new Conv[n];
        Conv[] pw = new Conv[n];
        Norm[] n1 = new Norm[n];
        Norm[] n2 = new Norm[n];
        for (int i = 0; i < n; i++) {
            int d = (int) Math.pow(k, i);
            int pad = (k * d - d) / 2;
            dil[i] = conv(p + ".convs_dilated." + i, d, pad, pad, config.hidden());
            pw[i] = conv(p + ".convs_pointwise." + i, 1, 0, 0, 1);
            n1[i] = norm(p + ".norms_1." + i);
            n2[i] = norm(p + ".norms_2." + i);
        }
        return new Dds(dil, pw, n1, n2);
    }

    private WaveNet wavenet(String p, int layers) {
        Var[] inW = new Var[layers];
        Var[] inB = new Var[layers];
        Var[] rsW = new Var[layers];
        Var[] rsB = new Var[layers];
        for (int i = 0; i < layers; i++) {
            Conv in = conv(p + ".in_layers." + i, 1, 0, 0, 1);
            Conv rs = conv(p + ".res_skip_layers." + i, 1, 0, 0, 1);
            inW[i] = Var.of(in.weight);
            inB[i] = Var.of(in.bias);
            rsW[i] = Var.of(rs.weight);
            rsB[i] = Var.of(rs.bias);
        }
        Conv cond = config.speakerEmbedding() > 0 ? conv(p + ".cond_layer", 1, 0, 0, 1) : null;
        return new WaveNet(config.wavenetKernel(), config.wavenetDilationRate(), inW, inB, rsW, rsB,
                cond == null ? null : Var.of(cond.weight), cond == null ? null : Var.of(cond.bias));
    }

    private static Var opt(Tensor t) {
        return t == null ? null : Var.of(t);
    }

    /**
     * Loads a Hugging Face VITS folder.
     *
     * @param dir the folder with {@code config.json} and safetensors weights
     * @param device where to put the weights
     * @param dtype the type of the weights and activations
     * @return the model
     * @throws IllegalArgumentException if the folder holds no VITS model
     */
    public static Vits load(Path dir, Device device, DType dtype) {
        Path config = dir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            throw new IllegalArgumentException(dir + " has no config.json; expected a Hugging Face VITS folder such"
                    + " as facebook/mms-tts-eng");
        }
        try {
            VitsConfig c = VitsConfig.fromHuggingFace(Json.object(Json.read(config)));
            try (WeightSource w = WeightSource.open(dir)) {
                return new Vits(c, device, dtype, w);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + dir + ": " + e.getMessage(), e);
        }
    }

    /**
     * Loads a model from weights named as in Hugging Face files.
     *
     * @param weights the weights
     * @param config the shape
     * @param device where to put them
     * @param dtype their type
     * @return the model
     */
    public static Vits load(WeightSource weights, VitsConfig config, Device device, DType dtype) {
        return new Vits(config, device, dtype, weights);
    }

    /**
     * Returns whether a folder holds a Hugging Face VITS model.
     *
     * @param dir the folder
     * @return {@code true} if its {@code config.json} says {@code model_type: vits}
     */
    public static boolean isVits(Path dir) {
        return SpeechModels.modelType(dir).equals("vits");
    }

    /**
     * Returns the model's shape.
     *
     * @return the config
     */
    public VitsConfig config() {
        return config;
    }

    /** Frees the weights; the model cannot be used afterwards. */
    public void close() {
        for (Tensor t : weights) {
            t.release();
        }
        weights.clear();
    }

    // ---------------------------------------------------------------- inference

    /**
     * How to sample: the noise of the latents and of the durations, the speaking rate
     * and the random source of both noises.
     *
     * @param noiseScale the latent noise (the config's is 0.667); 0 is deterministic
     * @param noiseScaleDuration the duration noise (0.8); 0 is deterministic
     * @param speakingRate 1 for normal, 2 for twice as fast
     * @param speaker the speaker, for models with several
     * @param seed the seed of the noise
     */
    public record Sampling(float noiseScale, float noiseScaleDuration, float speakingRate, int speaker, long seed) {

        /**
         * The model's own settings.
         *
         * @param c the config
         * @return the sampling settings
         */
        public static Sampling of(VitsConfig c) {
            return new Sampling(c.noiseScale(), c.noiseScaleDuration(), c.speakingRate(), 0, 0);
        }
    }

    /**
     * Speaks character ids.
     *
     * @param ids the ids of {@link VitsTokenizer#encode}
     * @param s the sampling settings
     * @return the waveform at {@link VitsConfig#sampleRate()}
     */
    public float[] synthesize(int[] ids, Sampling s) {
        if (ids.length == 0) {
            return new float[0];
        }
        Random random = new Random(s.seed());
        int len = ids.length;
        Var g = null;
        if (config.speakerEmbedding() > 0) {
            if (speakerEmbeddings == null) {
                throw new IllegalStateException("the model conditions on speakers but has no speaker embeddings");
            }
            if (s.speaker() < 0 || s.speaker() >= config.speakers()) {
                throw new IllegalArgumentException("speaker " + s.speaker() + " out of range 0.." + (config.speakers() - 1));
            }
            g = Var.of(speakerEmbeddings.slice(0, s.speaker(), s.speaker() + 1).reshape(1, config.speakerEmbedding(), 1));
        }
        // text encoder: [1, len, hidden]
        float[] idx = new float[len];
        for (int i = 0; i < len; i++) {
            idx[i] = ids[i];
        }
        Var x = Var.of(embedTokens).gather(0, Tensor.of(Shape.of(len), idx, device)).mul((float) Math.sqrt(config.hidden()))
                .reshape(1, len, config.hidden());
        for (EncoderLayer l : encoder) {
            x = layerNorm(x.add(l.attention.forward(x)), l.norm, config.layerNormEps());
            Var h = conv(x.permute(0, 2, 1), l.ffn1).relu();
            h = conv(h, l.ffn2).permute(0, 2, 1);
            x = layerNorm(x.add(h), l.finalNorm, config.layerNormEps());
        }
        Var hidden = x.permute(0, 2, 1);   // [1, hidden, len]
        Var stats = conv(hidden, project);
        Var means = stats.slice(1, 0, config.flowSize());
        Var logVars = stats.slice(1, config.flowSize(), 2 * config.flowSize());
        // durations
        float[] logDuration = config.stochasticDuration() ? stochasticDuration(hidden, g, s.noiseScaleDuration(), random)
                : conv(durationDeterministic(hidden, g), dProj).to(DType.FLOAT32).value().toArray();
        float lengthScale = 1 / s.speakingRate();
        int[] durations = new int[len];
        int frames = 0;
        for (int i = 0; i < len; i++) {
            durations[i] = (int) Math.ceil(Math.exp(logDuration[i]) * lengthScale);
            frames += durations[i];
        }
        frames = Math.max(frames, 1);
        float[] expand = new float[frames];
        int t = 0;
        for (int i = 0; i < len; i++) {
            for (int d = 0; d < durations[i] && t < frames; d++) {
                expand[t++] = i;
            }
        }
        Tensor at = Tensor.of(Shape.of(frames), expand, device);
        Var m = means.gather(2, at);
        Var logs = logVars.gather(2, at);
        Var z = m;
        if (s.noiseScale() != 0) {
            float[] noise = new float[config.flowSize() * frames];
            for (int i = 0; i < noise.length; i++) {
                noise[i] = (float) random.nextGaussian() * s.noiseScale();
            }
            z = m.add(Var.of(Tensor.of(Shape.of(1, config.flowSize(), frames), noise, device).to(dtype)).mul(logs.exp()));
        }
        // flows, reversed
        for (int i = flows.size() - 1; i >= 0; i--) {
            z = z.flip(1);
            Coupling c = flows.get(i);
            int half = config.flowSize() / 2;
            Var first = z.slice(1, 0, half);
            Var second = z.slice(1, half, 2 * half);
            Var h = c.wavenet.forward(conv(first, c.pre), g);
            Var mean = conv(h, c.post);
            z = Var.concat(1, first, second.sub(mean));
        }
        z = settle(z);
        return vocode(z, g, frames);
    }

    /** Latent frames per vocoder pass; longer speech is vocoded in overlapping pieces. */
    int vocoderChunk = 200;
    /** Latent frames of context on each side of a piece, beyond the vocoder's reach. */
    int vocoderContext = 24;

    /** The vocoder over {@code [1, flow, frames]} latents, a piece at a time to bound memory. */
    private float[] vocode(Var z, Var g, int frames) {
        int hop = config.hop();
        float[] out = new float[frames * hop];
        for (int start = 0; start < frames; start += vocoderChunk) {
            int end = Math.min(frames, start + vocoderChunk);
            int from = Math.max(0, start - vocoderContext);
            int to = Math.min(frames, end + vocoderContext);
            Var piece = from == 0 && to == frames ? z : z.slice(2, from, to);
            Tensor wave = vocoder(piece, g).value().to(DType.FLOAT32);
            float[] w = wave.toArray();
            wave.release();
            System.arraycopy(w, (start - from) * hop, out, start * hop, (end - start) * hop);
        }
        return out;
    }

    /** Computes a value now and cuts it from the graph, so its inputs can be freed. */
    private static Var settle(Var x) {
        Tensor t = x.value();
        Tensor.evaluate(List.of(t));
        return Var.of(t);
    }

    private Var conv(Var x, Conv c) {
        return Functional.conv(x, Var.of(c.weight), c.bias == null ? null : Var.of(c.bias), new int[] {1},
                new int[] {c.padBefore}, new int[] {c.padAfter}, new int[] {c.dilation}, c.groups);
    }

    private static Var layerNorm(Var x, Norm n, float eps) {
        Var x32 = x.to(DType.FLOAT32);
        int last = x.shape().rank() - 1;
        Var centred = x32.sub(x32.mean(last, true));
        Var y = centred.div(centred.square().mean(last, true).add(eps).sqrt());
        return y.to(x.dtype()).mul(n.weight).add(n.bias);
    }

    /** Layer norm over the channels of {@code [1, channels, length]}. */
    private Var channelNorm(Var x, Norm n) {
        return layerNorm(x.permute(0, 2, 1), n, config.layerNormEps()).permute(0, 2, 1);
    }

    private Var dds(Dds d, Var x, Var g) {
        if (g != null) {
            x = x.add(g);
        }
        for (int i = 0; i < d.dilated.length; i++) {
            Var h = conv(x, d.dilated[i]);
            h = channelNorm(h, d.norms1[i]).activation(ActivationOp.GELU, 0);
            h = conv(h, d.pointwise[i]);
            h = channelNorm(h, d.norms2[i]).activation(ActivationOp.GELU, 0);
            x = x.add(h);
        }
        return x;
    }

    private Var durationDeterministic(Var hidden, Var g) {
        Var x = hidden;
        if (g != null) {
            x = x.add(conv(g, dCond));
        }
        x = channelNorm(conv(x, dConv1).relu(), dNorm1);
        return channelNorm(conv(x, dConv2).relu(), dNorm2);
    }

    /** The stochastic duration predictor run backwards from noise: log durations per token. */
    private float[] stochasticDuration(Var hidden, Var g, float noiseScale, Random random) {
        int len = hidden.shape().dim(2);
        Var x = conv(hidden, dpPre);
        if (g != null) {
            x = x.add(conv(g, dpCond));
        }
        x = conv(dds(dpDds, x, null), dpProj);
        float[] noise = new float[2 * len];
        for (int i = 0; i < noise.length; i++) {
            noise[i] = (float) random.nextGaussian() * noiseScale;
        }
        Var z = Var.of(Tensor.of(Shape.of(1, 2, len), noise, device).to(dtype));
        // the flows reversed, without the first conv flow (unused at inference)
        for (int f = dpFlows.size() - 1; f >= 1; f--) {
            z = z.flip(1);
            z = convFlowReverse(dpFlows.get(f), z, x);
        }
        z = z.flip(1);
        Tensor translate = Tensor.of(Shape.of(1, 2, 1), dpAffineTranslate, device).to(dtype);
        Tensor logScale = Tensor.of(Shape.of(1, 2, 1), dpAffineLogScale, device).to(dtype);
        z = z.sub(translate).mul(Var.of(logScale).neg().exp());
        return z.slice(1, 0, 1).to(DType.FLOAT32).value().toArray();
    }

    private Var convFlowReverse(ConvFlow f, Var z, Var cond) {
        int len = z.shape().dim(2);
        int bins = config.flowBins();
        Var first = z.slice(1, 0, 1);
        Var second = z.slice(1, 1, 2);
        Var h = dds(f.dds, conv(first, f.pre), cond);
        Var p = conv(h, f.proj).reshape(1, 1, 3 * bins - 1, len).permute(0, 1, 3, 2);   // [1, 1, len, params]
        float norm = (float) (1 / Math.sqrt(config.hidden()));
        Var widths = p.slice(3, 0, bins).mul(norm);
        Var heights = p.slice(3, bins, 2 * bins).mul(norm);
        Var derivatives = p.slice(3, 2 * bins, 3 * bins - 1);
        Var y = Flows.rationalQuadraticSpline(second, widths, heights, derivatives, config.tailBound(), true).output();
        return Var.concat(1, first, y);
    }

    private Var vocoder(Var z, Var g) {
        Var x = conv(z, decPre);
        if (g != null) {
            x = x.add(conv(g, decCond));
        }
        int kernels = config.resblockKernels().length;
        int[] rates = config.upsampleRates();
        for (int i = 0; i < rates.length; i++) {
            x = x.activation(ActivationOp.LEAKY_RELU, config.leakyReluSlope());
            Conv u = upsamplers.get(i);
            x = Functional.convTranspose(x, Var.of(u.weight), Var.of(u.bias), new int[] {rates[i]},
                    new int[] {u.padBefore}, new int[] {u.padAfter}, new int[] {0}, new int[] {1}, 1);
            x = settle(x);
            Var sum = null;
            for (int j = 0; j < kernels; j++) {
                Var r = resblocks.get(i * kernels + j).forward(x);
                sum = settle(sum == null ? r : sum.add(r));
            }
            x = settle(sum.mul(1f / kernels));
        }
        x = x.activation(ActivationOp.LEAKY_RELU, 0.01f);
        return conv(x, decPost).tanh();
    }

    @Override
    public String toString() {
        return "Vits[hidden " + config.hidden() + ", " + config.layers() + " layers, " + config.sampleRate() + " Hz]";
    }
}
