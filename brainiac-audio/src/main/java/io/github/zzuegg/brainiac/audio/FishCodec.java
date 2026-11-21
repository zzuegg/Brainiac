package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Fish Audio's codec ({@code codec.pth} of {@code fishaudio/s2-pro}): a causal DAC whose 44.1 kHz
 * audio becomes about 21.5 frames a second of 10 codes, one semantic code from a 4096-entry
 * codebook and nine residual ones. The encoder is DAC's (Snake activations, strided causal
 * convolutions) with a windowed causal transformer at its end; the quantizer halves the frame
 * rate twice with ConvNeXt blocks around a windowed causal transformer, and quantizes the
 * semantic part and then the residual by nearest cosine codewords; the decoder is DAC's.
 *
 * <p>The sizes come from the weights; the attention windows (512 frames in the encoder, 128
 * around the quantizer) and dilations (1, 3, 9) are the reference's.
 */
final class FishCodec implements AutoCloseable {

    static final int SAMPLE_RATE = 44_100;
    private static final int HEAD_DIM = 64;
    private static final int ENCODER_WINDOW = 512;
    private static final int QUANTIZER_WINDOW = 128;
    private static final int[] DILATIONS = {1, 3, 9};
    /** Frames decoded at a time, and the frames of context before each chunk. */
    static final int CHUNK = 400;
    static final int CONTEXT = 64;
    /** Latent steps the waveform decoder renders at once (32 frames). */
    static final int TILE = 128;

    private final Weights w;
    private final Device device;
    private final DType dtype;

    private record Snake(Var alpha, Var inverse) {
        Var apply(Var x) {
            return x.add(x.mul(alpha).sin().square().mul(inverse));
        }
    }

    /** A causal convolution: {@code (k - 1) * dilation + 1 - stride} zeros before, as many after as the last frame needs. */
    private record CausalConv(Var w, Var b, int stride, int dilation) {
        Var apply(Var x) {
            int kernel = (w.shape().dim(2) - 1) * dilation + 1;
            int length = x.shape().dim(2);
            int pad = kernel - stride;
            double frames = (double) (length - kernel + pad) / stride + 1;
            int extra = (int) ((Math.ceil(frames) - 1) * stride + (kernel - pad) - length);
            int groups = x.shape().dim(1) / w.shape().dim(1);
            return Functional.conv(x, w, b, new int[] {stride}, new int[] {pad}, new int[] {Math.max(0, extra)},
                    new int[] {dilation}, groups);
        }
    }

    /** A transposed convolution with its last {@code k - stride} samples cut. */
    private record CausalTranspose(Var w, Var b, int stride) {
        Var apply(Var x) {
            int k = w.shape().dim(2);
            return Functional.convTranspose(x, w, b, new int[] {stride}, new int[] {0}, new int[] {k - stride},
                    new int[] {0}, new int[] {1}, 1);
        }
    }

    private record Unit(Snake a1, CausalConv c1, Snake a2, CausalConv c2) {
        Var apply(Var x) {
            return x.add(c2.apply(a2.apply(c1.apply(a1.apply(x)))));
        }
    }

    private record EncoderBlock(List<Unit> units, Snake act, CausalConv down, Transformer transformer) {
    }

    private record DecoderBlock(Snake act, CausalTranspose up, List<Unit> units) {
    }

    private record ConvNeXt(CausalConv dw, Var normW, Var normB, Var pw1, Var pw1b, Var pw2, Var pw2b, Var gamma) {
        Var apply(Var x) {
            Var h = dw.apply(x).permute(0, 2, 1);
            int last = h.shape().rank() - 1;
            Var c = h.sub(h.mean(last, true));
            h = c.div(c.square().mean(last, true).add(1e-6f).sqrt()).mul(normW).add(normB);
            h = h.matmul(pw1).add(pw1b).gelu().matmul(pw2).add(pw2b).mul(gamma);
            return x.add(h.permute(0, 2, 1));
        }
    }

    private record Layer(Var wq, Var wk, Var wv, Var wo, Var w1, Var w2, Var w3, Var attnNorm, Var ffnNorm,
            Var attnScale, Var ffnScale) {
    }

    private record Vq(Var inW, Var inB, Var outW, Var outB, float[] book, Var bookV, int dim) {
    }

    /** A windowed causal transformer over {@code [1, channels, t]}, with Fish's bfloat16 rotary tables. */
    private final class Transformer {
        final List<Layer> layers = new ArrayList<>();
        final Var norm;
        final int window;
        final int heads;

        Transformer(String p, int window) {
            this.window = window;
            for (int l = 0; w.contains(p + "layers." + l + ".attention.wqkv.weight"); l++) {
                String lp = p + "layers." + l + ".";
                Var[] qkv = qkv(lp + "attention.wqkv.weight");
                layers.add(new Layer(qkv[0], qkv[1], qkv[2], w.linear(lp + "attention.wo.weight"),
                        w.linear(lp + "feed_forward.w1.weight"), w.linear(lp + "feed_forward.w2.weight"),
                        w.linear(lp + "feed_forward.w3.weight"), w.read(lp + "attention_norm.weight"),
                        w.read(lp + "ffn_norm.weight"), w.read(lp + "attention_layer_scale.gamma"),
                        w.read(lp + "ffn_layer_scale.gamma")));
            }
            norm = w.read(p + "norm.weight");
            heads = norm.shape().dim(0) / HEAD_DIM;
        }

        Var apply(Var x) {
            int t = x.shape().dim(2);
            int d = x.shape().dim(1);
            Var h = x.permute(0, 2, 1);
            float[] m = new float[t * t];
            for (int i = 0; i < t; i++) {
                for (int j = 0; j < t; j++) {
                    m[i * t + j] = j <= i && i - j < window ? 0 : Float.NEGATIVE_INFINITY;
                }
            }
            Tensor mask = Tensor.of(Shape.of(1, 1, t, t), m, device).to(h.dtype());
            Tensor[] table = FishSpeech.rotaryTable(t, HEAD_DIM, 10_000, device, h.dtype());
            for (Layer l : layers) {
                Var a = rms(h, l.attnNorm());
                Var q = FishSpeech.rotate(a.matmul(l.wq()).reshape(1, t, heads, HEAD_DIM).permute(0, 2, 1, 3), table);
                Var k = FishSpeech.rotate(a.matmul(l.wk()).reshape(1, t, heads, HEAD_DIM).permute(0, 2, 1, 3), table);
                Var v = a.matmul(l.wv()).reshape(1, t, heads, HEAD_DIM).permute(0, 2, 1, 3);
                Var o = Functional.attention(q, k, v, mask, (float) (1 / Math.sqrt(HEAD_DIM)), 0, null);
                o = o.permute(0, 2, 1, 3).reshape(1, t, d).matmul(l.wo());
                h = h.add(o.mul(l.attnScale()));
                Var f = rms(h, l.ffnNorm());
                f = f.matmul(l.w1()).silu().mul(f.matmul(l.w3())).matmul(l.w2());
                h = Vocos.settle(h.add(f.mul(l.ffnScale())));
            }
            return rms(h, norm).permute(0, 2, 1);
        }
    }

    private final CausalConv encIn;
    private final List<EncoderBlock> encBlocks = new ArrayList<>();
    private final Snake encAct;
    private final CausalConv encOut;
    private final List<CausalConv> downs = new ArrayList<>();
    private final List<ConvNeXt> downNexts = new ArrayList<>();
    private final List<CausalTranspose> ups = new ArrayList<>();
    private final List<ConvNeXt> upNexts = new ArrayList<>();
    private final Transformer pre;
    private final Transformer post;
    private final List<Vq> quantizers = new ArrayList<>();
    private final CausalConv decIn;
    private final List<DecoderBlock> decBlocks = new ArrayList<>();
    private final Snake decAct;
    private final CausalConv decOut;
    private final int hop;
    private final int frame;
    /** Latent steps per waveform tile; tests change it. */
    int tile = TILE;

    FishCodec(WeightSource source, Device device, DType dtype) {
        this.device = device;
        this.dtype = dtype;
        this.w = new Weights(source, device, dtype);
        try {
            encIn = conv("encoder.block.0.conv", 1, 1);
            int n = 1;
            int hopProduct = 1;
            for (; w.contains("encoder.block." + n + ".block.4.conv.bias"); n++) {
                String p = "encoder.block." + n + ".block.";
                CausalConv down = conv(p + "4.conv", -1, 1);
                hopProduct *= down.stride();
                encBlocks.add(new EncoderBlock(units(p, 0), snake(p + "3.alpha"), down,
                        w.contains(p + "5.layers.0.attention.wqkv.weight") ? new Transformer(p + "5.", ENCODER_WINDOW)
                                : null));
            }
            encAct = snake("encoder.block." + n + ".alpha");
            encOut = conv("encoder.block." + (n + 1) + ".conv", 1, 1);
            int factor = 1;
            for (int i = 0; w.contains("quantizer.downsample." + i + ".0.conv.weight"); i++) {
                String p = "quantizer.downsample." + i + ".";
                Var dw = w.read(p + "0.conv.weight");
                downs.add(new CausalConv(dw, w.read(p + "0.conv.bias"), dw.shape().dim(2), 1));
                factor *= dw.shape().dim(2);
                downNexts.add(convNeXt(p + "1."));
                String u = "quantizer.upsample." + i + ".";
                Var uw = w.read(u + "0.conv.weight");
                ups.add(new CausalTranspose(uw, w.read(u + "0.conv.bias"), uw.shape().dim(2)));
                upNexts.add(convNeXt(u + "1."));
            }
            pre = new Transformer("quantizer.pre_module.", QUANTIZER_WINDOW);
            post = new Transformer("quantizer.post_module.", QUANTIZER_WINDOW);
            quantizers.add(vq("quantizer.semantic_quantizer.quantizers.0."));
            for (int i = 0; w.contains("quantizer.quantizer.quantizers." + i + ".codebook.weight"); i++) {
                quantizers.add(vq("quantizer.quantizer.quantizers." + i + "."));
            }
            decIn = conv("decoder.model.0.conv", 1, 1);
            int m = 1;
            for (; w.contains("decoder.model." + m + ".block.0.alpha"); m++) {
                String p = "decoder.model." + m + ".block.";
                Var up = w.conv(p + "1.conv");
                decBlocks.add(new DecoderBlock(snake(p + "0.alpha"),
                        new CausalTranspose(up, w.read(p + "1.conv.bias"), up.shape().dim(2) / 2), units(p, 2)));
            }
            decAct = snake("decoder.model." + m + ".alpha");
            decOut = conv("decoder.model." + (m + 1) + ".conv", 1, 1);
            hop = hopProduct;
            frame = hopProduct * factor;
        } catch (RuntimeException e) {
            w.close();
            throw e;
        }
    }

    /**
     * Loads a {@code codec.pth}.
     *
     * @param file the checkpoint
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the codec
     */
    static FishCodec load(Path file, Device device, DType dtype) {
        try (WeightSource source = WeightSource.open(file)) {
            return new FishCodec(source, device, dtype);
        }
    }

    /** @return audio samples per frame (2048) */
    int frame() {
        return frame;
    }

    /** @return codes per frame (10) */
    int codebooks() {
        return quantizers.size();
    }

    /** @return the size of each codebook, the semantic one first */
    int codebookSize(int q) {
        return quantizers.get(q).book().length / quantizers.get(q).dim();
    }

    // ---------------------------------------------------------------- pieces

    private Snake snake(String name) {
        float[] a = w.floats(name);
        float[] inv = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            inv[i] = 1 / (a[i] + 1e-9f);
        }
        return new Snake(Var.of(Tensor.of(Shape.of(1, a.length, 1), a, device).to(dtype)),
                Var.of(Tensor.of(Shape.of(1, a.length, 1), inv, device).to(dtype)));
    }

    /** A weight-normalized causal convolution; a stride of -1 is half the kernel. */
    private CausalConv conv(String name, int stride, int dilation) {
        Var k = w.conv(name);
        return new CausalConv(k, w.optional(name + ".bias"), stride < 0 ? k.shape().dim(2) / 2 : stride, dilation);
    }

    /** Three residual units at {@code p + (first + u)}, dilated 1, 3 and 9. */
    private List<Unit> units(String p, int first) {
        List<Unit> out = new ArrayList<>();
        for (int u = 0; u < 3; u++) {
            String b = p + (first + u) + ".block.";
            out.add(new Unit(snake(b + "0.alpha"), conv(b + "1.conv", 1, DILATIONS[u]), snake(b + "2.alpha"),
                    conv(b + "3.conv", 1, 1)));
        }
        return out;
    }

    private ConvNeXt convNeXt(String p) {
        Var dw = w.read(p + "dwconv.conv.weight");
        return new ConvNeXt(new CausalConv(dw, w.read(p + "dwconv.conv.bias"), 1, 1), w.read(p + "norm.weight"),
                w.read(p + "norm.bias"), w.linear(p + "pwconv1.weight"), w.read(p + "pwconv1.bias"),
                w.linear(p + "pwconv2.weight"), w.read(p + "pwconv2.bias"), w.read(p + "gamma"));
    }

    /** q, k and v of a fused projection as {@code [in, out]}, q and k rows reordered to rotate halves. */
    private Var[] qkv(String name) {
        Shape s = w.shape(name);
        float[] all = w.floats(name);
        int in = s.dim(1);
        int d = s.dim(0) / 3;
        Var[] out = new Var[3];
        for (int part = 0; part < 3; part++) {
            float[] rows = java.util.Arrays.copyOfRange(all, part * d * in, (part + 1) * d * in);
            if (part < 2) {
                rows = FishSpeech.halves(rows, d / HEAD_DIM, HEAD_DIM, in);
            }
            float[] t = new float[rows.length];
            for (int o = 0; o < d; o++) {
                for (int i = 0; i < in; i++) {
                    t[i * d + o] = rows[o * in + i];
                }
            }
            out[part] = Var.of(Tensor.of(Shape.of(in, d), t, device).to(dtype));
        }
        return out;
    }

    private Vq vq(String p) {
        Var inW = w.conv(p + "in_proj");
        Var outW = w.conv(p + "out_proj");
        int cdim = inW.shape().dim(0);
        int dim = inW.shape().dim(1);
        float[] book = w.floats(p + "codebook.weight");
        return new Vq(inW.reshape(cdim, dim).permute(1, 0), w.read(p + "in_proj.bias"),
                outW.reshape(dim, cdim).permute(1, 0), w.read(p + "out_proj.bias"), book,
                Var.of(Tensor.of(Shape.of(book.length / cdim, cdim), book, device).to(dtype)), cdim);
    }

    private static Var rms(Var x, Var weight) {
        int last = x.shape().rank() - 1;
        return x.div(x.square().mean(last, true).add(1e-5f).sqrt()).mul(weight);
    }

    // ---------------------------------------------------------------- encoding

    /**
     * Encodes audio to frames.
     *
     * @param audio 44.1 kHz samples
     * @return {@code [codebooks][ceil(samples / frame)]}
     */
    int[][] encode(float[] audio) {
        int frames = (audio.length + frame - 1) / frame;
        float[] padded = java.util.Arrays.copyOf(audio, frames * frame);
        return Var.noGrad(() -> {
            Var x = Var.of(Tensor.of(Shape.of(1, 1, padded.length), padded, device).to(dtype));
            x = encIn.apply(x);
            for (EncoderBlock b : encBlocks) {
                for (Unit u : b.units()) {
                    x = Vocos.settle(u.apply(x));
                }
                x = Vocos.settle(b.down().apply(b.act().apply(x)));
                if (b.transformer() != null) {
                    x = b.transformer().apply(x);
                }
            }
            x = Vocos.settle(encOut.apply(encAct.apply(x)));
            for (int i = 0; i < downs.size(); i++) {
                x = Vocos.settle(downNexts.get(i).apply(downs.get(i).apply(x)));
            }
            x = pre.apply(x);                                               // [1, dim, t]
            int t = x.shape().dim(2);
            int[][] codes = new int[quantizers.size()][t];
            Var residual = x.reshape(x.shape().dim(1), t).permute(1, 0);     // [t, dim]
            for (int q = 0; q < quantizers.size(); q++) {
                Vq v = quantizers.get(q);
                float[] e = residual.matmul(v.inW()).add(v.inB()).to(DType.FLOAT32).value().toArray();
                codes[q] = nearest(e, v.book(), v.dim());
                Var zq = lookup(v, codes[q]).matmul(v.outW()).add(v.outB());
                residual = Vocos.settle(residual.sub(zq));
            }
            int keep = Math.min(t, frames);
            for (int q = 0; q < codes.length; q++) {
                codes[q] = java.util.Arrays.copyOf(codes[q], keep);
            }
            return codes;
        });
    }

    private Var lookup(Vq v, int[] codes) {
        float[] idx = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            idx[i] = codes[i];
        }
        return v.bookV().gather(0, Tensor.of(Shape.of(codes.length), idx, device));
    }

    /** The codeword nearest each row by cosine, as DAC's distance between normalized vectors. */
    private static int[] nearest(float[] e, float[] book, int dim) {
        int t = e.length / dim;
        int size = book.length / dim;
        float[] nb = new float[book.length];
        float[] bb = new float[size];
        for (int c = 0; c < size; c++) {
            normalize(book, c * dim, nb, dim);
            float s = 0;
            for (int j = 0; j < dim; j++) {
                s += nb[c * dim + j] * nb[c * dim + j];
            }
            bb[c] = s;
        }
        float[] ne = new float[e.length];
        int[] out = new int[t];
        for (int i = 0; i < t; i++) {
            normalize(e, i * dim, ne, dim);
            float ee = 0;
            for (int j = 0; j < dim; j++) {
                ee += ne[i * dim + j] * ne[i * dim + j];
            }
            int best = 0;
            float bestD = Float.POSITIVE_INFINITY;
            for (int c = 0; c < size; c++) {
                float dot = 0;
                for (int j = 0; j < dim; j++) {
                    dot += ne[i * dim + j] * nb[c * dim + j];
                }
                float dist = ee - 2 * dot + bb[c];
                if (dist < bestD) {
                    bestD = dist;
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    private static void normalize(float[] src, int at, float[] dst, int dim) {
        double s = 0;
        for (int j = 0; j < dim; j++) {
            s += (double) src[at + j] * src[at + j];
        }
        float n = (float) Math.max(Math.sqrt(s), 1e-12);
        for (int j = 0; j < dim; j++) {
            dst[at + j] = src[at + j] / n;
        }
    }

    // ---------------------------------------------------------------- decoding

    /**
     * Decodes frames to audio, in chunks of {@link #CHUNK} frames each with {@link #CONTEXT}
     * frames of the previous one as context (the whole at once when it fits in one).
     *
     * @param codes {@code [codebooks][frames]}
     * @return {@code frames * frame()} samples in [-1, 1]
     */
    float[] decode(int[][] codes) {
        int frames = codes[0].length;
        float[] out = new float[frames * frame];
        int at = 0;
        for (int start = 0; start < frames; start += CHUNK) {
            int end = Math.min(frames, start + CHUNK);
            int context = Math.min(start, CONTEXT);
            int[][] part = new int[codes.length][];
            for (int q = 0; q < codes.length; q++) {
                part[q] = java.util.Arrays.copyOfRange(codes[q], start - context, end);
            }
            float[] wave = Var.noGrad(() -> decodeAll(part));
            int n = (end - start) * frame;
            System.arraycopy(wave, context * frame, out, at, n);
            at += n;
        }
        return out;
    }

    private float[] decodeAll(int[][] codes) {
        int t = codes[0].length;
        Var z = null;
        for (int q = 0; q < quantizers.size(); q++) {
            Vq v = quantizers.get(q);
            int size = codebookSize(q);
            int[] c = new int[t];
            for (int i = 0; i < t; i++) {
                c[i] = Math.min(Math.max(codes[q][i], 0), size - 1);
            }
            Var e = lookup(v, c).matmul(v.outW()).add(v.outB());          // [t, dim]
            z = z == null ? e : z.add(e);
        }
        Var x = Vocos.settle(z.permute(1, 0).reshape(1, z.shape().dim(1), t));
        x = post.apply(x);
        for (int i = 0; i < ups.size(); i++) {
            x = Vocos.settle(upNexts.get(i).apply(ups.get(i).apply(x)));
        }
        return waveform(Vocos.settle(x));
    }

    /**
     * The waveform decoder over {@code [1, channels, steps]}, in tiles of {@link #TILE} steps,
     * each after as many steps before it as the causal decoder looks back: its activations at
     * 44.1 kHz would otherwise take gigabytes for a long piece.
     */
    private float[] waveform(Var x) {
        int steps = x.shape().dim(2);
        int up = 1;
        for (DecoderBlock b : decBlocks) {
            up *= b.up().stride();
        }
        int context = lookBack();
        float[] out = new float[steps * up];
        for (int start = 0; start < steps; start += tile) {
            int end = Math.min(steps, start + tile);
            int from = Math.max(0, start - context);
            List<Tensor> made = new ArrayList<>();
            Var tile = from == 0 && end == steps ? x : Vocos.settle(x.slice(2, from, end));
            made.add(tile.value());
            Var y = Vocos.settle(decIn.apply(tile));
            made.add(y.value());
            for (DecoderBlock b : decBlocks) {
                y = Vocos.settle(b.up().apply(b.act().apply(y)));
                made.add(y.value());
                for (Unit u : b.units()) {
                    y = Vocos.settle(u.apply(y));
                    made.add(y.value());
                }
            }
            Tensor wave = decOut.apply(decAct.apply(y)).tanh().to(DType.FLOAT32).value();
            System.arraycopy(wave.toArray(), (start - from) * up, out, start * up, (end - start) * up);
            made.add(wave);
            // the tiles' activations go at once, not when the collector finds them
            Tensor.release(made, List.of(x.value()));
        }
        return out;
    }

    /** The steps before a sample that the waveform decoder reads, counted back from its output. */
    private int lookBack() {
        long rf = reach(decOut);
        for (int i = decBlocks.size() - 1; i >= 0; i--) {
            DecoderBlock b = decBlocks.get(i);
            for (Unit u : b.units()) {
                rf += reach(u.c1()) + reach(u.c2());
            }
            int k = b.up().w().shape().dim(2);
            int s = b.up().stride();
            rf = (rf + k - 1 + s - 1) / s;
        }
        return (int) (rf + reach(decIn) + 1);
    }

    private static long reach(CausalConv c) {
        return (long) (c.w().shape().dim(2) - 1) * c.dilation();
    }

    @Override
    public void close() {
        w.close();
    }
}
