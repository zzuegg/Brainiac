package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Snake;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.random.RandomGenerator;

/**
 * The decoder of SNAC, the Multi-Scale Neural Audio Codec (Siuzdak et al., 2024): audio
 * from codes at three (or four) time scales, coarse codes covering more samples. Codes pick
 * vectors from each scale's codebook, which are projected, stretched to the finest rate and
 * summed, then a DAC-style decoder (Snake activations, transposed convolutions, dilated
 * residual units, learned noise) makes the waveform. Loads {@code hubertsiuzdak/snac_24khz}
 * ({@code config.json}, {@code pytorch_model.bin}); Orpheus speaks through it.
 */
public final class Snac implements AutoCloseable {

    /**
     * The settings, from {@code config.json}.
     *
     * @param sampleRate the audio's sample rate
     * @param rates the decoder's upsampling factors
     * @param strides each code level's step in latent frames, coarse first
     * @param window the local attention window, or 0 for none
     * @param noise whether the decoder adds learned noise
     */
    public record Config(int sampleRate, int[] rates, int[] strides, int window, boolean noise) {

        /**
         * Reads {@code config.json}.
         *
         * @param file the file
         * @return the settings
         */
        public static Config read(Path file) {
            try {
                Map<String, Object> c = Json.object(Json.read(file));
                Object w = c.get("attn_window_size");
                return new Config(((Number) c.getOrDefault("sampling_rate", 24_000)).intValue(),
                        ints(c.get("decoder_rates")), ints(c.get("vq_strides")),
                        w instanceof Number n ? n.intValue() : 0, !Boolean.FALSE.equals(c.get("noise")));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        private static int[] ints(Object o) {
            return Json.array(o).stream().mapToInt(v -> ((Number) v).intValue()).toArray();
        }

        /** @return audio samples per finest-level code */
        public int hop() {
            int h = 1;
            for (int r : rates) {
                h *= r;
            }
            return h;
        }
    }

    private record Conv(Var w, Var b, int dilation, int groups) {
        Var apply(Var x) {
            int k = w.shape().dim(2);
            int pad = (k - 1) * dilation / 2;
            return Functional.conv(x, w, b, new int[] {1}, new int[] {pad}, new int[] {pad}, new int[] {dilation},
                    groups);
        }
    }

    private record Unit(Var alpha1, Conv conv1, Var alpha2, Conv conv2) {
        Var apply(Var x) {
            return x.add(conv2.apply(Snake.apply(conv1.apply(Snake.apply(x, alpha1)), alpha2)));
        }
    }

    private record Block(Var alpha, Var up, Var upB, int stride, Var noise, List<Unit> units) {
    }

    private record Level(Var codebook, Conv out, int stride) {
    }

    private final Config config;
    private final Weights weights;
    private final Device device;
    private final DType dtype;
    private final List<Level> levels = new ArrayList<>();
    private final List<Conv> input = new ArrayList<>();
    private Var normW;
    private Var normB;
    private Var qkv;
    private Var attnOut;
    private final List<Block> blocks = new ArrayList<>();
    private final Var finalAlpha;
    private final Conv finalConv;

    private Snac(Config config, WeightSource source, Device device, DType dtype) {
        this.config = config;
        this.device = device;
        this.dtype = dtype;
        this.weights = new Weights(source, device, dtype);
        try {
            Weights w = weights;
            for (int i = 0; i < config.strides().length; i++) {
                String q = "quantizer.quantizers." + i + ".";
                levels.add(new Level(w.read(q + "codebook.weight"), conv(q + "out_proj", 1), config.strides()[i]));
            }
            int at = 0;
            input.add(conv("decoder.model." + at++, 1));
            if (w.contains("decoder.model.1.bias") && !w.contains("decoder.model.1.block.0.alpha")) {
                input.add(conv("decoder.model." + at++, 1));
            }
            if (config.window() > 0) {
                String a = "decoder.model." + at++ + ".";
                normW = w.read(a + "norm.weight");
                normB = w.read(a + "norm.bias");
                qkv = w.linear(a + "to_qkv.weight");
                attnOut = w.linear(a + "to_out.weight");
            }
            for (int stride : config.rates()) {
                String b = "decoder.model." + at++ + ".block.";
                int sub = 0;
                Var alpha = alpha(b + sub++ + ".alpha");
                String up = b + sub++;
                Var upW = w.conv(up);
                Var upB = w.read(up + ".bias");
                Var noise = null;
                if (config.noise()) {
                    noise = w.conv(b + sub++ + ".linear");
                }
                List<Unit> units = new ArrayList<>();
                for (int dilation : new int[] {1, 3, 9}) {
                    String u = b + sub++ + ".block.";
                    units.add(new Unit(alpha(u + "0.alpha"), conv(u + "1", dilation), alpha(u + "2.alpha"),
                            conv(u + "3", 1)));
                }
                blocks.add(new Block(alpha, upW, upB, stride, noise, units));
            }
            finalAlpha = alpha("decoder.model." + at++ + ".alpha");
            finalConv = conv("decoder.model." + at, 1);
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    private Var alpha(String name) {
        int c = weights.shape(name).dim(1);
        return weights.read(name, c);
    }

    private Conv conv(String name, int dilation) {
        Var w = weights.conv(name);
        int channels = weights.shape(name + ".bias").dim(0);
        int perGroup = w.shape().dim(1);
        // depthwise convolutions: as many groups as output channels when each sees one input
        int groups = perGroup == 1 && channels > 1 && w.shape().dim(0) == channels ? channels : 1;
        return new Conv(w, weights.read(name + ".bias"), dilation, groups);
    }

    /**
     * Loads a SNAC folder.
     *
     * @param dir a folder with {@code config.json} and {@code pytorch_model.bin}
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the decoder
     */
    public static Snac load(Path dir, Device device, DType dtype) {
        if (!isSnac(dir)) {
            throw new IllegalArgumentException(dir + " is not a SNAC folder (config.json with vq_strides and"
                    + " pytorch_model.bin), such as hubertsiuzdak/snac_24khz");
        }
        try (WeightSource source = WeightSource.open(dir.resolve("pytorch_model.bin"))) {
            return new Snac(Config.read(dir.resolve("config.json")), source, device, dtype);
        }
    }

    /**
     * Whether a folder holds a SNAC codec.
     *
     * @param dir a folder
     * @return whether {@link #load} can read it
     */
    public static boolean isSnac(Path dir) {
        Path c = dir.resolve("config.json");
        try {
            return Files.isRegularFile(c) && Files.isRegularFile(dir.resolve("pytorch_model.bin"))
                    && Json.object(Json.read(c)).containsKey("vq_strides");
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /** @return the settings */
    public Config config() {
        return config;
    }

    /**
     * Decodes codes to audio.
     *
     * @param codes one array per level, coarse first; level {@code i} has
     *     {@code frames * strides[last] / strides[i]} codes
     * @param noise the decoder's noise for each upsampling block ({@code nextGaussian})
     * @return the samples, {@code hop} per finest code
     */
    public float[] decode(int[][] codes, IntFunction<RandomGenerator> noise) {
        if (codes.length != levels.size()) {
            throw new IllegalArgumentException("SNAC needs " + levels.size() + " code levels, got " + codes.length);
        }
        int frames = codes[0].length * levels.getFirst().stride();
        Var z = null;
        for (int i = 0; i < levels.size(); i++) {
            Level l = levels.get(i);
            if (codes[i].length * l.stride() != frames) {
                throw new IllegalArgumentException("code level " + i + " has " + codes[i].length + " codes; expected "
                        + frames / l.stride());
            }
            float[] idx = new float[codes[i].length];
            int size = l.codebook().shape().dim(0);
            for (int t = 0; t < idx.length; t++) {
                if (codes[i][t] < 0 || codes[i][t] >= size) {
                    throw new IllegalArgumentException("code " + codes[i][t] + " is outside the codebook of " + size);
                }
                idx[t] = codes[i][t];
            }
            Var e = l.codebook().gather(0, Tensor.of(Shape.of(idx.length), idx, device)).permute(1, 0)
                    .reshape(1, l.codebook().shape().dim(1), idx.length);
            Var zi = l.out().apply(e);
            if (l.stride() > 1) {
                zi = zi.repeatInterleave(l.stride(), 2);
            }
            z = z == null ? zi : z.add(zi);
        }
        Var x = z;
        for (Conv c : input) {
            x = c.apply(x);
        }
        if (qkv != null) {
            x = localAttention(Vocos.settle(x));
        }
        for (int bi = 0; bi < blocks.size(); bi++) {
            Block b = blocks.get(bi);
            int s = b.stride();
            x = Snake.apply(x, b.alpha());
            x = Functional.convTranspose(x, b.up(), b.upB(), new int[] {s}, new int[] {(s + 1) / 2},
                    new int[] {(s + 1) / 2}, new int[] {s % 2}, new int[] {1}, 1);
            if (b.noise() != null) {
                int len = x.shape().dim(2);
                RandomGenerator r = noise.apply(bi);
                float[] n = new float[len];
                for (int i = 0; i < len; i++) {
                    n[i] = (float) r.nextGaussian();
                }
                Var h = Functional.conv(x, b.noise(), null, new int[] {1}, new int[] {0}, new int[] {0},
                        new int[] {1}, 1);
                x = x.add(h.mul(Var.of(Tensor.of(Shape.of(1, 1, len), n, device).to(dtype))));
            }
            for (Unit u : b.units()) {
                x = Vocos.settle(u.apply(x));
            }
        }
        x = finalConv.apply(Snake.apply(x, finalAlpha)).tanh();
        return x.to(DType.FLOAT32).value().toArray();
    }

    /** Attention within windows of the sequence, with rotary positions, plus the input. */
    private Var localAttention(Var x) {
        int c = x.shape().dim(1);
        int t = x.shape().dim(2);
        int w = config.window();
        if (t % w != 0) {
            throw new IllegalArgumentException("SNAC's local attention needs a multiple of " + w + " frames, got " + t);
        }
        int heads = c / 64;
        int d = c / heads;
        int windows = t / w;
        Var h = layerNorm(x.permute(0, 2, 1)).mul(normW).add(normB).matmul(qkv);      // [1, t, 3c]
        Var[] qkvs = new Var[3];
        for (int i = 0; i < 3; i++) {
            qkvs[i] = h.slice(2, i * c, (i + 1) * c).reshape(windows, w, heads, d).permute(2, 0, 1, 3)
                    .reshape(1, heads * windows, w, d);
        }
        Var q = Functional.rotary(qkvs[0], 10_000f, 0);
        Var k = Functional.rotary(qkvs[1], 10_000f, 0);
        Var o = Functional.attention(q, k, qkvs[2], false, (float) (1 / Math.sqrt(d)));
        o = o.reshape(heads, windows, w, d).permute(1, 2, 0, 3).reshape(1, t, c).matmul(attnOut);
        return o.permute(0, 2, 1).add(x);
    }

    private static Var layerNorm(Var x) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        return centred.div(centred.square().mean(last, true).add(1e-5f).sqrt());
    }

    @Override
    public void close() {
        weights.close();
    }
}
