package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.ConvNeXt1d;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.MelSpectrogram;
import io.github.zzuegg.brainiac.nn.Signal;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Vocos vocoder (Siuzdak, 2023), mel version: a ConvNeXt backbone over mel frames and a
 * head that predicts each frame's STFT magnitude and phase, turned into audio by an inverse
 * STFT, which is why it is fast. Loads {@code charactr/vocos-mel-24khz} ({@code config.yaml},
 * {@code pytorch_model.bin}); F5-TTS speaks through it.
 */
public final class Vocos implements AutoCloseable {

    /**
     * The settings, from {@code config.yaml}.
     *
     * @param sampleRate the audio's sample rate
     * @param featureFft the mel features' frame length
     * @param mels the mel bands
     * @param nFft the head's inverse STFT frame length
     * @param hop samples per frame
     */
    public record Config(int sampleRate, int featureFft, int mels, int nFft, int hop) {

        /**
         * Reads {@code config.yaml} (only the numbers it needs, without a YAML library).
         *
         * @param file the file
         * @return the settings
         */
        public static Config read(Path file) {
            Map<String, String> v = new HashMap<>();
            String section = "";
            Pattern kv = Pattern.compile("^(\\s*)([A-Za-z_]+):\\s*(.*?)\\s*$");
            try {
                for (String line : Files.readAllLines(file)) {
                    Matcher m = kv.matcher(line);
                    if (!m.matches()) {
                        continue;
                    }
                    if (m.group(1).isEmpty()) {
                        section = m.group(2);
                    } else if (!m.group(3).isEmpty()) {
                        v.put(section + "." + m.group(2), m.group(3));
                    }
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            int sr = num(v, "feature_extractor.sample_rate", 24_000);
            int nFft = num(v, "head.n_fft", 1024);
            int hop = num(v, "head.hop_length", num(v, "feature_extractor.hop_length", 256));
            if (v.containsKey("head.padding") && !v.get("head.padding").equals("same")) {
                throw new IllegalArgumentException("only Vocos heads with \"same\" padding are supported, got "
                        + v.get("head.padding"));
            }
            return new Config(sr, num(v, "feature_extractor.n_fft", nFft), num(v, "feature_extractor.n_mels", 100),
                    nFft, hop);
        }

        private static int num(Map<String, String> v, String key, int fallback) {
            String s = v.get(key);
            return s == null ? fallback : Integer.parseInt(s.replace("'", "").replace("\"", ""));
        }
    }

    private final Config config;
    private final Weights weights;
    private final Var embedW;
    private final Var embedB;
    private final Var normW;
    private final Var normB;
    private final List<ConvNeXt1d> blocks = new ArrayList<>();
    private final Var finalW;
    private final Var finalB;
    private final Var outW;
    private final Var outB;

    private Vocos(Config config, WeightSource source, Device device, DType dtype) {
        this.config = config;
        this.weights = new Weights(source, device, dtype);
        try {
            Weights w = weights;
            embedW = w.read("backbone.embed.weight");
            embedB = w.read("backbone.embed.bias");
            normW = w.read("backbone.norm.weight");
            normB = w.read("backbone.norm.bias");
            for (int i = 0; w.contains("backbone.convnext." + i + ".dwconv.weight"); i++) {
                String p = "backbone.convnext." + i + ".";
                blocks.add(new ConvNeXt1d(1e-6f, w.read(p + "dwconv.weight"), w.read(p + "dwconv.bias"),
                        w.read(p + "norm.weight"), w.read(p + "norm.bias"), w.linear(p + "pwconv1.weight"),
                        w.read(p + "pwconv1.bias"), null, null, w.linear(p + "pwconv2.weight"),
                        w.read(p + "pwconv2.bias"), w.contains(p + "gamma") ? w.read(p + "gamma") : null));
            }
            finalW = w.read("backbone.final_layer_norm.weight");
            finalB = w.read("backbone.final_layer_norm.bias");
            outW = w.linear("head.out.weight");
            outB = w.read("head.out.bias");
            if (outB.shape().dim(0) != config.nFft() + 2 || embedW.shape().dim(1) != config.mels()) {
                throw new IllegalArgumentException("the Vocos weights do not match config.yaml: head "
                        + outB.shape() + ", embedding " + embedW.shape() + ", " + config);
            }
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    /**
     * Loads a Vocos folder.
     *
     * @param dir a folder with {@code config.yaml} and {@code pytorch_model.bin} (or {@code .safetensors})
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the vocoder
     */
    public static Vocos load(Path dir, Device device, DType dtype) {
        Path config = dir.resolve("config.yaml");
        if (!Files.isRegularFile(config)) {
            throw new IllegalArgumentException(dir + " has no config.yaml; expected a Vocos folder such as"
                    + " charactr/vocos-mel-24khz");
        }
        Path weightsFile = Files.isRegularFile(dir.resolve("pytorch_model.bin")) ? dir.resolve("pytorch_model.bin")
                : dir.resolve("model.safetensors");
        try (WeightSource source = WeightSource.open(weightsFile)) {
            return new Vocos(Config.read(config), source, device, dtype);
        }
    }

    /**
     * Whether a folder looks like a mel Vocos.
     *
     * @param dir a folder
     * @return whether {@link #load} can read it
     */
    public static boolean isVocos(Path dir) {
        Path config = dir.resolve("config.yaml");
        try {
            return Files.isRegularFile(config) && Files.readString(config).contains("VocosBackbone")
                    && Files.readString(config).contains("MelSpectrogramFeatures");
        } catch (IOException e) {
            return false;
        }
    }

    /** @return the settings */
    public Config config() {
        return config;
    }

    /** @return the sample rate */
    public int sampleRate() {
        return config.sampleRate();
    }

    /**
     * The features this vocoder reads, as F5-TTS computes them from reference audio.
     *
     * @return a log-amplitude mel spectrogram layer
     */
    public MelSpectrogram features() {
        return MelSpectrogram.vocos(config.sampleRate(), config.featureFft(), config.hop(), config.mels());
    }

    /**
     * Turns mel frames into audio.
     *
     * @param mel {@code [batch, mels, frames]}
     * @return {@code [batch, frames * hop]}
     */
    public Var decode(Var mel) {
        int n = mel.shape().dim(0);
        int frames = mel.shape().dim(2);
        Var x = Functional.conv(mel.to(embedW.dtype()), embedW, embedB, new int[] {1}, new int[] {3}, new int[] {3},
                new int[] {1}, 1);
        x = layerNorm(x.permute(0, 2, 1), normW, normB).permute(0, 2, 1);
        for (ConvNeXt1d b : blocks) {
            x = settle(b.forward(x));
        }
        x = layerNorm(x.permute(0, 2, 1), finalW, finalB);              // [batch, frames, dim]
        Var h = x.matmul(outW).add(outB).to(DType.FLOAT32);
        int bins = config.nFft() / 2 + 1;
        Var mag = h.slice(2, 0, bins).exp().minimum(100f).reshape(n, frames, bins, 1);
        Var phase = h.slice(2, bins, 2 * bins).reshape(n, frames, bins, 1);
        Var spec = Var.concat(3, mag.mul(phase.cos()), mag.mul(phase.sin()));
        return settle(Signal.istftSame(spec, config.nFft(), config.hop()));
    }

    /**
     * Turns mel frames into audio.
     *
     * @param mel {@code [mels][frames]}
     * @return the samples
     */
    public float[] decode(float[][] mel) {
        int m = mel.length;
        int frames = mel[0].length;
        float[] flat = new float[m * frames];
        for (int i = 0; i < m; i++) {
            System.arraycopy(mel[i], 0, flat, i * frames, frames);
        }
        Device device = embedW.value().device();
        Tensor t = Tensor.of(Shape.of(1, m, frames), flat, device);
        return decode(Var.of(t)).value().toArray();
    }

    static Var layerNorm(Var x, Var w, Var b) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        Var y = centred.div(centred.square().mean(last, true).add(1e-6f).sqrt());
        return w == null ? y : y.mul(w).add(b);
    }

    static Var settle(Var x) {
        Tensor t = x.value();
        Tensor.evaluate(List.of(t));
        return Var.of(t);
    }

    @Override
    public void close() {
        weights.close();
    }
}
