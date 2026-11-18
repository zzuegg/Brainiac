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

/**
 * OpenAI's Whisper (Radford et al., 2022), the network: an audio encoder (two
 * convolutions, then pre-norm transformer blocks over 1500 positions of a 30-second
 * window) and a text decoder that attends to its own tokens and to the encoded audio.
 *
 * <p>It loads Hugging Face folders ({@code openai/whisper-tiny} up to {@code large-v3},
 * {@code .en} models and fine-tunes; {@code config.json} with safetensors weights). For
 * transcription use {@link WhisperTranscriber}, which adds the tokenizer, the decoding
 * rules and long audio.
 *
 * <pre>{@code
 * Whisper model = Whisper.load(Path.of("whisper-base"), Device.cpu(), DType.FLOAT32);
 * Tensor audio = model.encode(WhisperFeatures.window(WhisperFeatures.padded(samples, 80), 0));
 * try (Whisper.Decoder d = model.decoder(audio)) {
 *     float[] logits = d.next(new int[] {50258, 50259, 50359, 50363});
 * }
 * }</pre>
 */
public final class Whisper {

    /** {@code x W + b} with {@code W} stored {@code [in, out]}. */
    private record Linear(Tensor weight, Tensor bias) {
        Var apply(Var x) {
            Var y = x.matmul(weight);
            return bias == null ? y : y.add(bias);
        }
    }

    private record Norm(Tensor weight, Tensor bias) {
    }

    private record Attention(Linear q, Linear k, Linear v, Linear out) {
    }

    private record Block(Norm attnNorm, Attention self, Norm crossNorm, Attention cross, Norm ffnNorm, Linear fc1,
            Linear fc2) {
    }

    private final WhisperConfig config;
    private final Device device;
    private final DType dtype;
    private final Tensor conv1Weight;
    private final Tensor conv1Bias;
    private final Tensor conv2Weight;
    private final Tensor conv2Bias;
    private final Tensor audioPositions;
    private final List<Block> encoder;
    private final Norm encoderNorm;
    private final Tensor tokens;
    /** The token embeddings transposed, {@code [width, vocab]}: the output layer. */
    private final Tensor head;
    private final Tensor textPositions;
    private final List<Block> decoder;
    private final Norm decoderNorm;
    /** Every weight, to free them. */
    private final List<Tensor> weights = new ArrayList<>();

    private Whisper(WhisperConfig config, Device device, DType dtype, WeightSource w) {
        this.config = config;
        this.device = device;
        this.dtype = dtype;
        String e = "model.encoder.";
        String d = "model.decoder.";
        conv1Weight = read(w, e + "conv1.weight");
        conv1Bias = read(w, e + "conv1.bias");
        conv2Weight = read(w, e + "conv2.weight");
        conv2Bias = read(w, e + "conv2.bias");
        audioPositions = read(w, e + "embed_positions.weight");
        encoder = new ArrayList<>();
        for (int i = 0; i < config.encoderLayers(); i++) {
            encoder.add(block(w, e + "layers." + i + ".", false));
        }
        encoderNorm = norm(w, e + "layer_norm");
        tokens = read(w, d + "embed_tokens.weight");
        head = keep(inOut(w.tensor(d + "embed_tokens.weight", device, dtype)));
        textPositions = read(w, d + "embed_positions.weight");
        decoder = new ArrayList<>();
        for (int i = 0; i < config.decoderLayers(); i++) {
            decoder.add(block(w, d + "layers." + i + ".", true));
        }
        decoderNorm = norm(w, d + "layer_norm");
    }

    private Block block(WeightSource w, String p, boolean cross) {
        return new Block(norm(w, p + "self_attn_layer_norm"), attention(w, p + "self_attn."),
                cross ? norm(w, p + "encoder_attn_layer_norm") : null, cross ? attention(w, p + "encoder_attn.") : null,
                norm(w, p + "final_layer_norm"), linear(w, p + "fc1"), linear(w, p + "fc2"));
    }

    private Attention attention(WeightSource w, String p) {
        return new Attention(linear(w, p + "q_proj"), linear(w, p + "k_proj"), linear(w, p + "v_proj"),
                linear(w, p + "out_proj"));
    }

    private Linear linear(WeightSource w, String name) {
        Tensor weight = keep(inOut(w.tensor(name + ".weight", device, dtype)));
        Tensor bias = w.contains(name + ".bias") ? read(w, name + ".bias") : null;
        return new Linear(weight, bias);
    }

    private Norm norm(WeightSource w, String name) {
        return new Norm(read(w, name + ".weight"), read(w, name + ".bias"));
    }

    private Tensor read(WeightSource w, String name) {
        return keep(w.tensor(name, device, dtype));
    }

    private Tensor keep(Tensor t) {
        weights.add(t);
        return t;
    }

    /** Hugging Face's {@code [out, in]} matrix as {@code [in, out]}, freeing the original. */
    private static Tensor inOut(Tensor outIn) {
        Tensor t = outIn.transpose();
        Tensor.evaluate(List.of(t));
        outIn.release();
        return t;
    }

    /**
     * Loads a Hugging Face Whisper folder.
     *
     * @param dir the folder with {@code config.json} and safetensors weights
     * @param device where to put the weights
     * @param dtype the type of the weights and activations
     * @return the model
     * @throws IllegalArgumentException if the folder holds no Whisper model
     */
    public static Whisper load(Path dir, Device device, DType dtype) {
        Path config = dir.resolve("config.json");
        if (!Files.isRegularFile(config)) {
            throw new IllegalArgumentException(dir + " has no config.json; expected a Hugging Face Whisper folder"
                    + " such as openai/whisper-base");
        }
        try {
            WhisperConfig c = WhisperConfig.fromHuggingFace(Json.object(Json.read(config)));
            try (WeightSource w = WeightSource.open(dir)) {
                return load(w, c, device, dtype);
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
    public static Whisper load(WeightSource weights, WhisperConfig config, Device device, DType dtype) {
        // some exports name the weights without the leading "model."
        WeightSource w = weights.contains("model.encoder.conv1.weight") ? weights
                : weights.renamed(n -> n.startsWith("model.") ? n.substring(6) : n);
        return new Whisper(config, device, dtype, w);
    }

    /**
     * Returns whether a folder holds a Hugging Face Whisper model.
     *
     * @param dir the folder
     * @return {@code true} if its {@code config.json} says {@code model_type: whisper}
     */
    public static boolean isWhisper(Path dir) {
        return SpeechModels.modelType(dir).equals("whisper");
    }

    /**
     * Returns the model's shape.
     *
     * @return the config
     */
    public WhisperConfig config() {
        return config;
    }

    /**
     * Returns where the weights are.
     *
     * @return the device
     */
    public Device device() {
        return device;
    }

    /**
     * Returns the type of the weights.
     *
     * @return the type
     */
    public DType dtype() {
        return dtype;
    }

    // ---------------------------------------------------------------- encoder

    /**
     * Encodes one 30-second window of log-mel features.
     *
     * @param mel {@code [mels][3000]}, as {@link WhisperFeatures#window}
     * @return the audio features {@code [1, 1500, width]}
     */
    public Tensor encode(float[][] mel) {
        int mels = mel.length;
        if (mels != config.mels()) {
            throw new IllegalArgumentException("this model takes " + config.mels() + " mel bands, got " + mels);
        }
        int frames = mel[0].length;
        float[] flat = new float[mels * frames];
        for (int m = 0; m < mels; m++) {
            System.arraycopy(mel[m], 0, flat, m * frames, frames);
        }
        Var x = Var.of(Tensor.of(Shape.of(1, mels, frames), flat, device).to(dtype));
        int[] one = {1};
        x = Functional.conv(x, Var.of(conv1Weight), Var.of(conv1Bias), one, one, one, one, 1).activation(ActivationOp.GELU, 0);
        x = Functional.conv(x, Var.of(conv2Weight), Var.of(conv2Bias), new int[] {2}, one, one, one, 1)
                .activation(ActivationOp.GELU, 0);
        x = x.permute(0, 2, 1);   // [1, positions, width]
        int positions = x.shape().dim(1);
        if (positions > config.audioPositions()) {
            throw new IllegalArgumentException("at most " + config.audioPositions() * 2 + " frames, got " + frames);
        }
        x = x.add(Var.of(audioPositions.slice(0, 0, positions)));
        for (Block b : encoder) {
            Var h = layerNorm(x, b.attnNorm);
            x = x.add(selfAttention(b.self, h, config.encoderHeads(), null, false));
            x = x.add(ffn(b, layerNorm(x, b.ffnNorm)));
        }
        Tensor out = layerNorm(x, encoderNorm).value();
        Tensor.evaluate(List.of(out));
        return out;
    }

    private Var ffn(Block b, Var h) {
        return b.fc2.apply(b.fc1.apply(h).activation(ActivationOp.GELU, 0));
    }

    private static Var layerNorm(Var x, Norm n) {
        Var x32 = x.to(DType.FLOAT32);
        int last = x.shape().rank() - 1;
        Var centred = x32.sub(x32.mean(last, true));
        Var y = centred.div(centred.square().mean(last, true).add(1e-5f).sqrt());
        return y.to(x.dtype()).mul(n.weight).add(n.bias);
    }

    /** {@code [batch, steps, width]} to {@code [batch, heads, steps, headDim]}. */
    private static Var heads(Var x, int heads) {
        int b = x.shape().dim(0);
        int s = x.shape().dim(1);
        int w = x.shape().dim(2);
        return x.reshape(b, s, heads, w / heads).permute(0, 2, 1, 3);
    }

    private static Var merge(Var x) {
        int b = x.shape().dim(0);
        int h = x.shape().dim(1);
        int s = x.shape().dim(2);
        int d = x.shape().dim(3);
        return x.permute(0, 2, 1, 3).reshape(b, s, h * d);
    }

    /**
     * Self-attention of {@code h}; with a cache, the new keys and values are appended to
     * {@code cache[0..1]} and attention covers all of them, causally for the new steps.
     */
    private Var selfAttention(Attention a, Var h, int heads, Tensor[] cache, boolean causal) {
        int width = h.shape().dim(2);
        float scale = (float) (1 / Math.sqrt(width / heads));
        Var q = heads(a.q.apply(h), heads);
        Var k = heads(a.k.apply(h), heads);
        Var v = heads(a.v.apply(h), heads);
        if (cache != null) {
            if (cache[0] != null) {
                k = Var.of(Tensor.concat(2, cache[0], k.value()));
                v = Var.of(Tensor.concat(2, cache[1], v.value()));
            }
            Tensor.evaluate(List.of(k.value(), v.value()));
            if (cache[0] != null) {
                cache[0].release();
                cache[1].release();
            }
            cache[0] = k.value();
            cache[1] = v.value();
        }
        int steps = q.shape().dim(2);
        Var o;
        if (causal && steps > 1) {
            o = Functional.attention(q, k, v, Functional.causalMask(steps, k.shape().dim(2), device).to(dtype), scale,
                    0, null);
        } else {
            o = Functional.attention(q, k, v, false, scale);
        }
        return a.out.apply(merge(o));
    }

    // ---------------------------------------------------------------- decoder

    /**
     * Starts decoding text for encoded audio.
     *
     * @param audio the audio features of {@link #encode}, {@code [1, 1500, width]}
     * @return a decoder holding the attention caches; close it to free them
     */
    public Decoder decoder(Tensor audio) {
        return new Decoder(audio);
    }

    /**
     * Decodes tokens one call at a time, caching the keys and values of earlier tokens
     * and of the audio.
     */
    public final class Decoder implements AutoCloseable {
        private final Tensor[][] selfCache;
        private final Tensor[][] crossCache;
        private int length;

        private Decoder(Tensor audio) {
            selfCache = new Tensor[decoder.size()][2];
            crossCache = new Tensor[decoder.size()][2];
            Var a = Var.of(audio.to(device).to(dtype));
            int heads = config.decoderHeads();
            for (int l = 0; l < decoder.size(); l++) {
                Attention c = decoder.get(l).cross;
                Tensor k = heads(c.k.apply(a), heads).value();
                Tensor v = heads(c.v.apply(a), heads).value();
                Tensor.evaluate(List.of(k, v));
                crossCache[l][0] = k;
                crossCache[l][1] = v;
            }
        }

        /**
         * Returns how many tokens have been decoded.
         *
         * @return the length
         */
        public int length() {
            return length;
        }

        /**
         * Feeds tokens and returns the logits after the last one.
         *
         * @param ids the next tokens
         * @return {@code [vocab]} logits
         */
        public float[] next(int... ids) {
            float[][] all = run(ids, false);
            return all[0];
        }

        /**
         * Feeds tokens and returns the logits after each of them.
         *
         * @param ids the next tokens
         * @return {@code [ids.length][vocab]} logits
         */
        public float[][] nextAll(int... ids) {
            return run(ids, true);
        }

        private float[][] run(int[] ids, boolean every) {
            int steps = ids.length;
            if (length + steps > config.textPositions()) {
                throw new IllegalStateException("Whisper decodes at most " + config.textPositions() + " tokens");
            }
            float[] idx = new float[steps];
            for (int i = 0; i < steps; i++) {
                idx[i] = ids[i];
            }
            Var x = Var.of(tokens).gather(0, Tensor.of(Shape.of(steps), idx, device))
                    .add(Var.of(textPositions.slice(0, length, length + steps))).reshape(1, steps, config.width());
            int heads = config.decoderHeads();
            float scale = (float) (1 / Math.sqrt(config.width() / heads));
            for (int l = 0; l < decoder.size(); l++) {
                Block b = decoder.get(l);
                x = x.add(selfAttention(b.self, layerNorm(x, b.attnNorm), heads, selfCache[l], true));
                Var q = heads(b.cross.q.apply(layerNorm(x, b.crossNorm)), heads);
                Var o = Functional.attention(q, Var.of(crossCache[l][0]), Var.of(crossCache[l][1]), false, scale);
                x = x.add(b.cross.out.apply(merge(o)));
                x = x.add(ffn(b, layerNorm(x, b.ffnNorm)));
            }
            x = layerNorm(x, decoderNorm);
            if (!every) {
                x = x.slice(1, steps - 1, steps);
            }
            Tensor logits = x.matmul(Var.of(head)).value().to(DType.FLOAT32);
            float[] flat = logits.toArray();
            length += steps;
            int rows = every ? steps : 1;
            int vocab = flat.length / rows;
            float[][] out = new float[rows][];
            for (int r = 0; r < rows; r++) {
                out[r] = java.util.Arrays.copyOfRange(flat, r * vocab, (r + 1) * vocab);
            }
            logits.release();
            return out;
        }

        /**
         * Forgets the decoded tokens, keeping the audio.
         */
        public void reset() {
            for (Tensor[] c : selfCache) {
                if (c[0] != null) {
                    c[0].release();
                    c[1].release();
                }
                c[0] = null;
                c[1] = null;
            }
            length = 0;
        }

        @Override
        public void close() {
            reset();
            for (Tensor[] c : crossCache) {
                c[0].release();
                c[1].release();
            }
        }
    }

    /**
     * Frees the weights; the model cannot be used afterwards.
     */
    public void close() {
        for (Tensor t : weights) {
            t.release();
        }
        weights.clear();
    }

    @Override
    public String toString() {
        return "Whisper" + config;
    }
}
