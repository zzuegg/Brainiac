package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.AdainResBlock;
import io.github.zzuegg.brainiac.nn.AdainSnakeBlock;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.nn.StyleNorm;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * Kokoro (hexgrad/Kokoro-82M): StyleTTS2's inference path with an iSTFTNet vocoder. Phonemes
 * go through a PL-BERT (ALBERT) encoder and a style-conditioned prosody predictor that
 * chooses each phoneme's duration, pitch (F0) and energy; a text encoder and a decoder of
 * AdaIN residual blocks then turn the aligned phonemes into a spectrogram's magnitude and
 * phase, driven by a harmonic-plus-noise source at the predicted pitch, and an inverse STFT
 * makes 24 kHz audio. A 256-wide style vector (half for the decoder, half for prosody) sets
 * the voice; voices are packs of style vectors indexed by the phoneme count.
 */
public final class Kokoro implements AutoCloseable {

    /** The output sample rate. */
    public static final int SAMPLE_RATE = 24_000;

    /**
     * The model's settings, from {@code config.json}.
     *
     * @param hidden the hidden width
     * @param style the style width (each half of a voice's vector)
     * @param layers the prosody and text encoder depth
     * @param textKernel the text encoder's kernel
     * @param bertHeads the ALBERT attention heads
     * @param bertLayers the ALBERT layer repetitions
     * @param resblockKernels the vocoder's residual block kernels
     * @param resblockDilations their dilations
     * @param upsampleRates the vocoder's upsampling factors
     * @param upsampleKernels their kernels
     * @param nFft the inverse STFT's frame length
     * @param hop its hop
     * @param vocab phoneme to token id
     */
    public record Config(int hidden, int style, int layers, int textKernel, int bertHeads, int bertLayers,
            int[] resblockKernels, int[][] resblockDilations, int[] upsampleRates, int[] upsampleKernels, int nFft,
            int hop, Map<String, Integer> vocab) {

        /**
         * Reads {@code config.json}.
         *
         * @param file the file
         * @return the settings
         */
        public static Config read(Path file) {
            Map<String, Object> c;
            try {
                c = Json.object(Json.read(file));
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
            Map<String, Object> bert = Json.object(c.get("plbert"));
            Map<String, Object> ist = Json.object(c.get("istftnet"));
            Map<String, Integer> vocab = new HashMap<>();
            Json.object(c.get("vocab")).forEach((k, v) -> vocab.put(k, ((Number) v).intValue()));
            List<Object> dil = Json.array(ist.get("resblock_dilation_sizes"));
            int[][] dilations = new int[dil.size()][];
            for (int i = 0; i < dilations.length; i++) {
                dilations[i] = ints(dil.get(i));
            }
            return new Config(num(c, "hidden_dim"), num(c, "style_dim"), num(c, "n_layer"),
                    num(c, "text_encoder_kernel_size"), num(bert, "num_attention_heads"),
                    num(bert, "num_hidden_layers"), ints(ist.get("resblock_kernel_sizes")), dilations,
                    ints(ist.get("upsample_rates")), ints(ist.get("upsample_kernel_sizes")),
                    num(ist, "gen_istft_n_fft"), num(ist, "gen_istft_hop_size"), Map.copyOf(vocab));
        }

        private static int num(Map<String, Object> m, String key) {
            Object v = m.get(key);
            if (!(v instanceof Number n)) {
                throw new IllegalArgumentException("Kokoro config.json has no " + key);
            }
            return n.intValue();
        }

        private static int[] ints(Object o) {
            return Json.array(o).stream().mapToInt(v -> ((Number) v).intValue()).toArray();
        }
    }

    /**
     * Synthesized audio and the frames (of {@code 2 * hop * prod(upsampleRates)} samples)
     * each token got, the padding tokens at both ends included.
     *
     * @param audio 24 kHz samples
     * @param durations frames per token
     */
    public record Output(float[] audio, int[] durations) {
    }

    private record Linear(Var w, Var b) {
        Var apply(Var x) {
            Var y = x.matmul(w);
            return b == null ? y : y.add(b);
        }
    }

    private record Norm(Var w, Var b) {
    }

    private record Conv(Var w, Var b) {
    }

    private record BertLayer(Linear q, Linear k, Linear v, Linear o, Norm attnNorm, Linear ffn, Linear ffnOut,
            Norm ffnNorm) {
    }

    private final Config config;
    private final Weights weights;
    private final Device device;
    private final DType dtype;

    // PL-BERT
    private final Var wordEmb;
    private final Var posEmb;
    private final Var typeEmb;
    private final Norm embNorm;
    private final Linear embMap;
    private final BertLayer bert;
    private final Linear bertEncoder;
    // prosody predictor
    private final Recurrent[] durLstms;
    private final StyleNorm[] durNorms;
    private final Recurrent durLstm;
    private final Linear durProj;
    private final Recurrent shared;
    private final AdainResBlock[] f0Blocks;
    private final AdainResBlock[] nBlocks;
    private final Conv f0Proj;
    private final Conv nProj;
    // text encoder
    private final Var textEmb;
    private final Conv[] textConvs;
    private final Norm[] textNorms;
    private final Recurrent textLstm;
    // decoder
    private final AdainResBlock encode;
    private final AdainResBlock[] decode;
    private final Conv f0Conv;
    private final Conv nConv;
    private final Conv asrRes;
    // generator
    private final float[] sourceW;
    private final float sourceB;
    private final Conv[] noiseConvs;
    private final AdainSnakeBlock[] noiseRes;
    private final Conv[] ups;
    private final AdainSnakeBlock[] resblocks;
    private final Conv post;

    private Kokoro(Config config, WeightSource source, Device device, DType dtype) {
        this.config = config;
        this.device = device;
        this.dtype = dtype;
        this.weights = new Weights(source, device, dtype);
        Weights w = weights;
        try {
            String b = "bert.module.";
            wordEmb = w.read(b + "embeddings.word_embeddings.weight");
            posEmb = w.read(b + "embeddings.position_embeddings.weight");
            typeEmb = w.read(b + "embeddings.token_type_embeddings.weight");
            embNorm = norm(b + "embeddings.LayerNorm");
            embMap = linear(b + "encoder.embedding_hidden_mapping_in");
            String l = b + "encoder.albert_layer_groups.0.albert_layers.0.";
            bert = new BertLayer(linear(l + "attention.query"), linear(l + "attention.key"),
                    linear(l + "attention.value"), linear(l + "attention.dense"), norm(l + "attention.LayerNorm"),
                    linear(l + "ffn"), linear(l + "ffn_output"), norm(l + "full_layer_layer_norm"));
            bertEncoder = linear("bert_encoder.module");

            String p = "predictor.module.";
            durLstms = new Recurrent[config.layers()];
            durNorms = new StyleNorm[config.layers()];
            for (int i = 0; i < config.layers(); i++) {
                durLstms[i] = w.lstm(p + "text_encoder.lstms." + (2 * i));
                durNorms[i] = w.styleNorm(p + "text_encoder.lstms." + (2 * i + 1), StyleNorm.Kind.LAYER);
            }
            durLstm = w.lstm(p + "lstm");
            durProj = linear(p + "duration_proj.linear_layer");
            shared = w.lstm(p + "shared");
            f0Blocks = blocks(p + "F0.", 3);
            nBlocks = blocks(p + "N.", 3);
            f0Proj = conv(p + "F0_proj");
            nProj = conv(p + "N_proj");

            String t = "text_encoder.module.";
            textEmb = w.read(t + "embedding.weight");
            textConvs = new Conv[config.layers()];
            textNorms = new Norm[config.layers()];
            for (int i = 0; i < config.layers(); i++) {
                textConvs[i] = conv(t + "cnn." + i + ".0");
                textNorms[i] = new Norm(w.read(t + "cnn." + i + ".1.gamma"), w.read(t + "cnn." + i + ".1.beta"));
            }
            textLstm = w.lstm(t + "lstm");

            String d = "decoder.module.";
            encode = block(d + "encode");
            decode = blocks(d + "decode.", 4);
            f0Conv = conv(d + "F0_conv");
            nConv = conv(d + "N_conv");
            asrRes = conv(d + "asr_res.0");

            String g = d + "generator.";
            sourceW = w.floats(g + "m_source.l_linear.weight");
            sourceB = w.floats(g + "m_source.l_linear.bias")[0];
            int ups = config.upsampleRates().length;
            int kernels = config.resblockKernels().length;
            noiseConvs = new Conv[ups];
            noiseRes = new AdainSnakeBlock[ups];
            this.ups = new Conv[ups];
            resblocks = new AdainSnakeBlock[ups * kernels];
            for (int i = 0; i < ups; i++) {
                noiseConvs[i] = conv(g + "noise_convs." + i);
                noiseRes[i] = snake(g + "noise_res." + i, new int[] {1, 3, 5});
                this.ups[i] = conv(g + "ups." + i);
                for (int j = 0; j < kernels; j++) {
                    resblocks[i * kernels + j] = snake(g + "resblocks." + (i * kernels + j),
                            config.resblockDilations()[j]);
                }
            }
            post = conv(g + "conv_post");
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    private Linear linear(String name) {
        return new Linear(weights.linear(name + ".weight"), weights.optional(name + ".bias"));
    }

    private Norm norm(String name) {
        return new Norm(weights.read(name + ".weight"), weights.read(name + ".bias"));
    }

    private Conv conv(String name) {
        return new Conv(weights.conv(name), weights.optional(name + ".bias"));
    }

    private AdainResBlock block(String p) {
        Var pool = weights.contains(p + ".pool.weight_v") || weights.contains(p + ".pool.weight")
                ? weights.conv(p + ".pool") : null;
        Var shortcut = weights.contains(p + ".conv1x1.weight_v") || weights.contains(p + ".conv1x1.weight")
                ? weights.conv(p + ".conv1x1") : null;
        return new AdainResBlock(weights.styleNorm(p + ".norm1", StyleNorm.Kind.INSTANCE),
                weights.styleNorm(p + ".norm2", StyleNorm.Kind.INSTANCE), weights.conv(p + ".conv1"),
                weights.read(p + ".conv1.bias"), weights.conv(p + ".conv2"), weights.read(p + ".conv2.bias"), shortcut,
                pool, pool == null ? null : weights.read(p + ".pool.bias"), 0.2f);
    }

    private AdainResBlock[] blocks(String prefix, int n) {
        AdainResBlock[] out = new AdainResBlock[n];
        for (int i = 0; i < n; i++) {
            out[i] = block(prefix + i);
        }
        return out;
    }

    private AdainSnakeBlock snake(String p, int[] dilations) {
        int n = dilations.length;
        StyleNorm[] n1 = new StyleNorm[n];
        StyleNorm[] n2 = new StyleNorm[n];
        Var[] a1 = new Var[n];
        Var[] a2 = new Var[n];
        Var[] w1 = new Var[n];
        Var[] b1 = new Var[n];
        Var[] w2 = new Var[n];
        Var[] b2 = new Var[n];
        for (int i = 0; i < n; i++) {
            n1[i] = weights.styleNorm(p + ".adain1." + i, StyleNorm.Kind.INSTANCE);
            n2[i] = weights.styleNorm(p + ".adain2." + i, StyleNorm.Kind.INSTANCE);
            int c = weights.shape(p + ".alpha1." + i).dim(1);
            a1[i] = weights.read(p + ".alpha1." + i, c);
            a2[i] = weights.read(p + ".alpha2." + i, c);
            w1[i] = weights.conv(p + ".convs1." + i);
            b1[i] = weights.read(p + ".convs1." + i + ".bias");
            w2[i] = weights.conv(p + ".convs2." + i);
            b2[i] = weights.read(p + ".convs2." + i + ".bias");
        }
        return new AdainSnakeBlock(dilations, n1, n2, a1, a2, w1, b1, w2, b2);
    }

    /**
     * Loads a Kokoro folder: {@code config.json} and a {@code .pth} checkpoint
     * ({@code kokoro-v1_0.pth}).
     *
     * @param dir the folder
     * @param device where to run
     * @param dtype the weights' type
     * @return the model
     */
    public static Kokoro load(Path dir, Device device, DType dtype) {
        Path config = dir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            throw new IllegalArgumentException(dir + " has no config.json; expected a Kokoro folder such as"
                    + " hexgrad/Kokoro-82M");
        }
        Path pth = checkpoint(dir);
        WeightSource source = WeightSource.open(pth);
        try {
            return new Kokoro(Config.read(config), source, device, dtype);
        } finally {
            source.close();
        }
    }

    private static Path checkpoint(Path dir) {
        try (var files = Files.list(dir)) {
            return files.filter(f -> f.getFileName().toString().endsWith(".pth")).sorted().findFirst()
                    .orElseThrow(() -> new IllegalArgumentException(dir + " has no .pth checkpoint"));
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        }
    }

    /**
     * Whether a folder looks like a Kokoro model: a {@code config.json} with {@code istftnet}
     * and {@code plbert} settings.
     *
     * @param dir a folder
     * @return whether Kokoro can load it
     */
    public static boolean isKokoro(Path dir) {
        Path config = dir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            return false;
        }
        try {
            Map<String, Object> c = Json.object(Json.read(config));
            return c.containsKey("istftnet") && c.containsKey("plbert") && c.containsKey("vocab");
        } catch (RuntimeException | java.io.IOException e) {
            return false;
        }
    }

    /** @return the settings */
    public Config config() {
        return config;
    }

    /**
     * Turns phonemes into token ids, skipping symbols the vocabulary lacks.
     *
     * @param phonemes IPA phonemes, as the G2P makes them
     * @return the ids, without the padding tokens
     */
    public int[] tokenize(String phonemes) {
        return phonemes.codePoints().mapToObj(Character::toString).map(config.vocab()::get)
                .filter(java.util.Objects::nonNull).mapToInt(Integer::intValue).toArray();
    }

    /**
     * Synthesizes speech.
     *
     * @param ids phoneme token ids (without the padding tokens)
     * @param style the voice's style vector {@code [2 * style]}: the decoder's half, then the
     *     prosody predictor's
     * @param speed the speaking rate, 1 for normal
     * @param noise the source of the vocoder's unvoiced noise, or {@code null} for none
     * @return the audio and the durations
     */
    public Output synthesize(int[] ids, float[] style, float speed, RandomGenerator noise) {
        int s = config.style();
        if (style.length != 2 * s) {
            throw new IllegalArgumentException("a Kokoro style vector has " + 2 * s + " values, got " + style.length);
        }
        int maxTokens = posEmb.shape().dim(0);
        if (ids.length + 2 > maxTokens) {
            throw new IllegalArgumentException("Kokoro reads at most " + (maxTokens - 2) + " phonemes at once, got "
                    + ids.length);
        }
        int n = ids.length + 2;
        float[] tokens = new float[n];
        for (int i = 0; i < ids.length; i++) {
            tokens[i + 1] = ids[i];
        }
        Tensor tok = Tensor.of(Shape.of(n), tokens, device);
        Var sDec = Var.of(Tensor.of(Shape.of(1, s), java.util.Arrays.copyOfRange(style, 0, s), device).to(dtype));
        Var sPro = Var.of(Tensor.of(Shape.of(1, s), java.util.Arrays.copyOfRange(style, s, 2 * s), device).to(dtype));

        // durations
        Var dEn = bertEncoder.apply(bert(tok, n));                       // [1, n, hidden]
        Var sSeq = sPro.reshape(1, 1, s).expand(Shape.of(1, n, s));
        Var x = Var.concat(-1, dEn, sSeq);
        for (int i = 0; i < config.layers(); i++) {
            x = durLstms[i].forward(x);                                  // [1, n, hidden]
            x = durNorms[i].forward(x.permute(0, 2, 1), sPro).permute(0, 2, 1);
            x = Var.concat(-1, x, sSeq);
        }
        Var d = settle(x);                                               // [1, n, hidden + style]
        Var logits = durProj.apply(durLstm.forward(d));
        float[] dur = logits.sigmoid().sum(2, false).value().to(DType.FLOAT32).toArray();
        int[] frames = new int[n];
        int total = 0;
        for (int i = 0; i < n; i++) {
            frames[i] = Math.max(1, (int) Math.rint(dur[i] / speed));
            total += frames[i];
        }
        float[] idx = new float[total];
        for (int i = 0, f = 0; i < n; i++) {
            for (int k = 0; k < frames[i]; k++) {
                idx[f++] = i;
            }
        }
        Tensor align = Tensor.of(Shape.of(total), idx, device);

        // pitch and energy
        Var en = d.gather(1, align);                                     // [1, frames, hidden + style]
        Var sh = shared.forward(en).permute(0, 2, 1);                    // [1, hidden, frames]
        Var f0 = sh;
        for (AdainResBlock blk : f0Blocks) {
            f0 = blk.forward(f0, sPro);
        }
        f0 = settle(conv1d(f0, f0Proj, 1, 0, 0));                        // [1, 1, 2 frames]
        Var nn = sh;
        for (AdainResBlock blk : nBlocks) {
            nn = blk.forward(nn, sPro);
        }
        nn = settle(conv1d(nn, nProj, 1, 0, 0));

        // text encoder
        Var t = textEmb.gather(0, tok).reshape(1, n, config.hidden()).permute(0, 2, 1);
        for (int i = 0; i < config.layers(); i++) {
            int k = textConvs[i].w().shape().dim(2);
            t = conv1d(t, textConvs[i], 1, (k - 1) / 2, (k - 1) / 2);
            t = channelNorm(t, textNorms[i], 1e-5f).activation(ActivationOp.LEAKY_RELU, 0.2f);
        }
        t = textLstm.forward(t.permute(0, 2, 1));                        // [1, n, hidden]
        Var asr = settle(t.gather(1, align).permute(0, 2, 1));           // [1, hidden, frames]

        // decoder
        Var f0d = conv1d(f0, f0Conv, 2, 1, 1);
        Var nd = conv1d(nn, nConv, 2, 1, 1);
        Var h = encode.forward(Var.concat(1, asr, f0d, nd), sDec);
        Var res = conv1d(asr, asrRes, 1, 0, 0);
        boolean residual = true;
        for (AdainResBlock blk : decode) {
            if (residual) {
                h = Var.concat(1, h, res, f0d, nd);
            }
            h = blk.forward(h, sDec);
            if (blk.upsamples()) {
                residual = false;
            }
        }
        h = settle(h);
        float[] audio = generate(h, sDec, f0.value().to(DType.FLOAT32).toArray(), noise);
        return new Output(audio, frames);
    }

    /** The PL-BERT (ALBERT) encoder with every position attended. */
    private Var bert(Tensor tok, int n) {
        int e = wordEmb.shape().dim(1);
        float[] pos = new float[n];
        for (int i = 0; i < n; i++) {
            pos[i] = i;
        }
        Var x = wordEmb.gather(0, tok).add(posEmb.gather(0, Tensor.of(Shape.of(n), pos, device)))
                .add(typeEmb.slice(0, 0, 1)).reshape(1, n, e);
        x = embMap.apply(layerNorm(x, embNorm, 1e-12f));
        int hidden = x.shape().dim(2);
        int heads = config.bertHeads();
        int dk = hidden / heads;
        for (int l = 0; l < config.bertLayers(); l++) {
            Var q = split(bert.q().apply(x), heads);
            Var k = split(bert.k().apply(x), heads);
            Var v = split(bert.v().apply(x), heads);
            Var a = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(dk)));
            a = a.permute(0, 2, 1, 3).reshape(1, n, hidden);
            x = layerNorm(x.add(bert.o().apply(a)), bert.attnNorm(), 1e-12f);
            Var f = bert.ffnOut().apply(bert.ffn().apply(x).activation(ActivationOp.GELU_TANH, 0));
            x = settle(layerNorm(f.add(x), bert.ffnNorm(), 1e-12f));
        }
        return x;
    }

    private static Var split(Var x, int heads) {
        Shape s = x.shape();
        return x.reshape(s.dim(0), s.dim(1), heads, s.dim(2) / heads).permute(0, 2, 1, 3);
    }

    /** The iSTFTNet generator. */
    private float[] generate(Var x, Var style, float[] f0, RandomGenerator noise) {
        int upsample = config.hop();
        for (int r : config.upsampleRates()) {
            upsample *= r;
        }
        float[] source = harmonicSource(f0, upsample, noise);
        int nFft = config.nFft();
        int hop = config.hop();
        Var spec = Signal.stft(Var.of(Tensor.of(Shape.of(1, source.length), source, device).to(dtype)), nFft, hop);
        Var re = spec.slice(3, 0, 1);
        int bins = nFft / 2 + 1;
        // the DC and Nyquist bins are real: their imaginary part is +0, so their phase is 0 or pi as in PyTorch
        float[] real = new float[bins];
        java.util.Arrays.fill(real, 1, bins - 1, 1f);
        Var im = spec.slice(3, 1, 2).mul(Var.of(Tensor.of(Shape.of(bins, 1), real, device).to(dtype))).add(0f);
        Var mag = re.square().add(im.square()).sqrt();
        Var phase = im.atan2(re);
        int frames = spec.shape().dim(1);
        Var har = settle(Var.concat(2, mag.reshape(1, frames, bins), phase.reshape(1, frames, bins))
                .permute(0, 2, 1));                                      // [1, 2 bins, frames]
        int stages = config.upsampleRates().length;
        int kernels = config.resblockKernels().length;
        for (int i = 0; i < stages; i++) {
            x = x.activation(ActivationOp.LEAKY_RELU, 0.1f);
            Var xs;
            if (i + 1 < stages) {
                int stride = 1;
                for (int j = i + 1; j < stages; j++) {
                    stride *= config.upsampleRates()[j];
                }
                int pad = (stride + 1) / 2;
                xs = conv1d(har, noiseConvs[i], stride, pad, pad);
            } else {
                xs = conv1d(har, noiseConvs[i], 1, 0, 0);
            }
            xs = settle(noiseRes[i].forward(xs, style));
            int u = config.upsampleRates()[i];
            int k = config.upsampleKernels()[i];
            int p = (k - u) / 2;
            x = Functional.convTranspose(x, ups[i].w(), ups[i].b(), new int[] {u}, new int[] {p}, new int[] {p},
                    new int[] {0}, new int[] {1}, 1);
            if (i == stages - 1) {
                x = Functional.padAxis(x, 2, 1, 0, PadMode.REFLECT, 0);
            }
            x = settle(x.add(xs));
            Var sum = null;
            for (int j = 0; j < kernels; j++) {
                Var r = settle(resblocks[i * kernels + j].forward(x, style));
                sum = sum == null ? r : settle(sum.add(r));
            }
            x = settle(sum.mul(1f / kernels));
        }
        x = x.activation(ActivationOp.LEAKY_RELU, 0.01f);
        x = conv1d(x, post, 1, 3, 3);                                    // [1, nFft + 2, frames]
        int outFrames = x.shape().dim(2);
        Var logMag = x.slice(1, 0, bins);
        Var ph = x.slice(1, bins, 2 * bins).sin();
        Var m = logMag.exp();
        Var cplx = Var.concat(3, m.mul(ph.cos()).permute(0, 2, 1).reshape(1, outFrames, bins, 1),
                m.mul(ph.sin()).permute(0, 2, 1).reshape(1, outFrames, bins, 1));
        Var audio = Signal.istft(cplx, nFft, hop, hop * (outFrames - 1));
        return audio.value().to(DType.FLOAT32).toArray();
    }

    /**
     * The neural source-filter excitation (SineGen with 8 harmonics and a learned merge): sines
     * at the pitch and its harmonics, phase-accumulated at the frame rate and interpolated,
     * silent below 10 Hz, plus noise, merged by a linear map and {@code tanh}.
     */
    private float[] harmonicSource(float[] f0, int upsample, RandomGenerator noise) {
        int harmonics = sourceW.length;
        int frames = f0.length;
        int samples = frames * upsample;
        double[][] phase = new double[harmonics][frames];
        for (int h = 0; h < harmonics; h++) {
            float acc = 0;
            for (int t = 0; t < frames; t++) {
                float rad = f0[t] * (h + 1) / SAMPLE_RATE;
                rad = rad - (float) Math.floor(rad);
                acc += rad;
                phase[h][t] = acc * 2 * Math.PI * upsample;
            }
        }
        float[] out = new float[samples];
        for (int i = 0; i < samples; i++) {
            double src = Math.max((i + 0.5) / upsample - 0.5, 0);
            int i0 = Math.min((int) Math.floor(src), frames - 1);
            int i1 = Math.min(i0 + 1, frames - 1);
            double lambda = src - i0;
            float f = f0[i / upsample];
            boolean voiced = f > 10;
            double amp = voiced ? 0.003 : 0.1 / 3;
            double sum = sourceB;
            for (int h = 0; h < harmonics; h++) {
                double ph = phase[h][i0] * (1 - lambda) + phase[h][i1] * lambda;
                double sine = voiced ? Math.sin(ph) * 0.1 : 0;
                if (noise != null) {
                    sine += amp * noise.nextGaussian();
                }
                sum += sourceW[h] * sine;
            }
            out[i] = (float) Math.tanh(sum);
        }
        return out;
    }

    private static Var conv1d(Var x, Conv c, int stride, int padBefore, int padAfter) {
        return Functional.conv(x, c.w(), c.b(), new int[] {stride}, new int[] {padBefore}, new int[] {padAfter},
                new int[] {1}, 1);
    }

    private static Var layerNorm(Var x, Norm n, float eps) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        Var y = centred.div(centred.square().mean(last, true).add(eps).sqrt());
        return y.mul(n.w()).add(n.b());
    }

    private static Var channelNorm(Var x, Norm n, float eps) {
        return layerNorm(x.permute(0, 2, 1), n, eps).permute(0, 2, 1);
    }

    private static Var settle(Var x) {
        Tensor t = x.value();
        Tensor.evaluate(List.of(t));
        return Var.of(t);
    }

    @Override
    public void close() {
        weights.close();
    }
}
