package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.diffusion.AutoencoderKL;
import io.github.zzuegg.brainiac.nn.diffusion.ClipTextEncoder;
import io.github.zzuegg.brainiac.nn.diffusion.FlowSampler;
import io.github.zzuegg.brainiac.nn.diffusion.FluxTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;

/**
 * Text-to-image generation with Black Forest Labs' Flux.1 models (dev and schnell), from a
 * Hugging Face diffusers folder: a {@link FluxTransformer} sampled by flow matching
 * ({@link FlowSampler}), conditioned on T5-XXL token states and a pooled CLIP embedding, with
 * a 16-channel {@link AutoencoderKL}. Results match diffusers' {@code FluxPipeline}.
 *
 * <pre>{@code
 * try (Flux flux = Flux.load(Path.of("FLUX.1-schnell"), Device.best(), DType.BFLOAT16, Flux.Options.bits(8, 8))) {
 *     GeneratedImage image = flux.generate(ImageRequest.of("a red fox in fresh snow").steps(4).build(),
 *             Progress.NONE).get(0);
 * }
 * }</pre>
 *
 * <p>Flux has no negative prompt and samples with Euler steps only; dev takes the guidance
 * scale as an input (no classifier-free guidance, so one pass per step), schnell ignores it.
 * In full precision the transformer takes about 24 GB and T5-XXL about 9.5 GB;
 * {@link Options#bits} stores their weights in 8 or 4 bits, and {@link Options#fitting} picks the
 * most precise that fit a device ({@link ImageGenerator#load(Path, Device)} uses it).
 */
public final class Flux implements ImageGenerator {

    /** The model variants. */
    public enum Kind {
        /** Guidance-distilled, 20 to 50 steps, takes a guidance scale (FLUX.1-dev). */
        DEV,
        /** Timestep-distilled, 1 to 4 steps (FLUX.1-schnell). */
        SCHNELL
    }

    /**
     * How to load the weights.
     *
     * @param transformerBits bits per projection weight of the transformer: 0 keeps the
     *     load type, 8 halves bfloat16's memory, 4 quarters it
     * @param textEncoderBits the same for T5
     * @param textTokens the T5 prompt length: 512, or 256 as schnell was trained with
     */
    public record Options(int transformerBits, int textEncoderBits, int textTokens) {

        /** Checks the values. */
        public Options {
            for (int b : new int[] {transformerBits, textEncoderBits}) {
                if (b != 0 && (b < 2 || b > 8)) {
                    throw new IllegalArgumentException("bits must be 0 or 2 to 8, got " + b);
                }
            }
            if (textTokens < 1 || textTokens > 512) {
                throw new IllegalArgumentException("text tokens must be 1 to 512, got " + textTokens);
            }
        }

        /** @return weights as loaded, 512 text tokens */
        public static Options full() {
            return new Options(0, 0, 512);
        }

        /**
         * Quantized weights, 512 text tokens.
         *
         * @param transformer the transformer's bits per weight
         * @param textEncoder T5's bits per weight
         * @return the options
         */
        public static Options bits(int transformer, int textEncoder) {
            return new Options(transformer, textEncoder, 512);
        }

        /**
         * The most precise weights that leave room to generate at 1024 pixels on a device,
         * estimated from the files' headers: full, then T5 in 8 bits, both in 8 bits, the
         * transformer in 8 and T5 in 4, the transformer in 4 and T5 in 8, both in 4. On a
         * device that does not count its memory (the CPU), full.
         *
         * @param path a diffusers Flux folder
         * @param device where it will run
         * @param dtype the load type
         * @return the options, 512 text tokens
         * @throws IOException if the headers cannot be read
         * @throws IllegalArgumentException if even 4-bit weights do not fit
         */
        public static Options fitting(Path path, Device device, DType dtype) throws IOException {
            if (MemoryFit.free(device).isEmpty()) {
                return full();
            }
            Sizes sizes = Sizes.of(path);
            int[] b = choose(sizes, dtype, MemoryFit.free(device).getAsLong());
            if (b == null) {
                // short of the comfortable room, still try the smallest weights if they leave the minimum
                MemoryFit.require("Flux (" + path.getFileName() + ")", sizes.bytes(dtype, 4, 4), device,
                        "even with both networks in 4 bits");
                return bits(4, 4);
            }
            return bits(b[0], b[1]);
        }
    }

    /** The most precise {transformer, T5} bits that leave the working room in {@code free} bytes, or null. */
    private static int[] choose(Sizes sizes, DType dtype, long free) {
        for (int[] b : CHOICES) {
            if (sizes.bytes(dtype, b[0], b[1]) + WORKING <= free) {
                return b;
            }
        }
        return null;
    }

    /**
     * The device memory a Flux folder will take when nothing else is loaded: its weights at the
     * bits {@link Options#fitting} would pick on an empty device, and the least room to generate.
     *
     * @param path a diffusers Flux folder
     * @param device where it will run
     * @param dtype the load type
     * @return the bytes, or 0 on a device that does not count its memory
     * @throws IOException if the headers cannot be read
     */
    static long memoryNeed(Path path, Device device, DType dtype) throws IOException {
        java.util.Optional<Device.Memory> m = device.memory();
        if (m.isEmpty()) {
            return 0;
        }
        Sizes sizes = Sizes.of(path);
        int[] b = choose(sizes, dtype, m.get().budget());
        int[] bits = b == null ? new int[] {4, 4} : b;
        return sizes.bytes(dtype, bits[0], bits[1]) + MemoryFit.MIN_WORKING;
    }

    /** The parameters of a Flux folder's networks, from the headers. */
    private record Sizes(MemoryFit.Params transformer, MemoryFit.Params t5, MemoryFit.Params clip,
            MemoryFit.Params vae) {

        static Sizes of(Path path) throws IOException {
            return new Sizes(MemoryFit.params(path.resolve("transformer"), "diffusion_pytorch_model", false),
                    MemoryFit.params(path.resolve("text_encoder_2"), "model", false),
                    MemoryFit.params(path.resolve("text_encoder"), "model", false),
                    MemoryFit.params(path.resolve("vae"), "diffusion_pytorch_model", false));
        }

        /** @return the device bytes of all the weights */
        long bytes(DType dtype, int transformerBits, int t5Bits) {
            return transformer.bytes(dtype, transformerBits) + t5.bytes(dtype, t5Bits) + clip.bytes(dtype, 0)
                    + vae.bytes(dtype, 0);
        }
    }

    /** Weight precisions from best to smallest: {transformer bits, T5 bits}. */
    private static final int[][] CHOICES = {{0, 0}, {0, 8}, {8, 8}, {8, 4}, {4, 8}, {4, 4}};

    /** Room for activations at 1024 pixels: the transformer's 4608 tokens, T5's states and the decoder. */
    static final long WORKING = 3L << 30;

    private final Kind kind;
    private final Path path;
    private final Device device;
    private final DType dtype;
    private final List<WeightSource> sources;
    private final List<Tensor> weights;
    private final ClipTokenizer clipTokenizer;
    private final ClipTextEncoder clip;
    private final PretrainedTokenizer t5Tokenizer;
    /** T5, or null while it is released to leave its memory to the transformer ({@link #encode}). */
    private T5Encoder t5;
    private final List<WeightSource> t5Sources;
    private final List<Tensor> t5Weights;
    /** The last prompt encoded and its conditioning, kept for the next draw of the same prompt. */
    private String encodedPrompt;
    private Conditioning encoded;
    private final FluxTransformer transformer;
    private final AutoencoderKL vae;
    private final Map<String, Object> scheduler;
    private final Options options;

    private Flux(Kind kind, Path path, Device device, DType dtype, List<WeightSource> sources, List<Tensor> weights,
            ClipTokenizer clipTokenizer, ClipTextEncoder clip, PretrainedTokenizer t5Tokenizer, T5Encoder t5,
            List<WeightSource> t5Sources, List<Tensor> t5Weights, FluxTransformer transformer, AutoencoderKL vae,
            Map<String, Object> scheduler, Options options) {
        this.kind = kind;
        this.path = path;
        this.device = device;
        this.dtype = dtype;
        this.sources = sources;
        this.weights = weights;
        this.clipTokenizer = clipTokenizer;
        this.clip = clip;
        this.t5Tokenizer = t5Tokenizer;
        this.t5 = t5;
        this.t5Sources = t5Sources;
        this.t5Weights = t5Weights;
        this.transformer = transformer;
        this.vae = vae;
        this.scheduler = scheduler;
        this.options = options;
    }

    // ------------------------------------------------------------------------------- loading

    /**
     * Says whether a folder holds a diffusers Flux.1 model, and which, without loading it.
     *
     * @param path a folder
     * @return the kind, or empty
     */
    public static Optional<Kind> detect(Path path) {
        try {
            Path config = path.resolve("transformer/config.json");
            if (!Files.isDirectory(path) || !Files.isRegularFile(config)) {
                return Optional.empty();
            }
            Map<String, Object> c = Json.object(Json.read(config));
            if (!"FluxTransformer2DModel".equals(c.get("_class_name"))) {
                return Optional.empty();
            }
            return Optional.of(Boolean.TRUE.equals(c.get("guidance_embeds")) ? Kind.DEV : Kind.SCHNELL);
        } catch (IOException | RuntimeException e) {
            return Optional.empty();
        }
    }

    /**
     * Loads a model with weights as stored.
     *
     * @param path a diffusers folder ({@code transformer}, {@code text_encoder},
     *     {@code text_encoder_2}, {@code tokenizer}, {@code tokenizer_2}, {@code vae}, {@code scheduler})
     * @param device where to run it
     * @param dtype the type of the networks: {@link DType#BFLOAT16} on a GPU (Flux overflows
     *     in float16), {@link DType#FLOAT32} on the CPU
     * @return the model
     * @throws IOException if the files cannot be read
     */
    public static Flux load(Path path, Device device, DType dtype) throws IOException {
        return load(path, device, dtype, Options.full());
    }

    /**
     * Loads a model.
     *
     * @param path a diffusers folder
     * @param device where to run it
     * @param dtype the type of the networks
     * @param options how to store the weights
     * @return the model
     * @throws IOException if the files cannot be read
     * @throws IllegalArgumentException if the folder is not a Flux model
     */
    public static Flux load(Path path, Device device, DType dtype, Options options) throws IOException {
        Kind kind = detect(path).orElseThrow(() -> new IllegalArgumentException(path
                + " is not a diffusers Flux folder (transformer/config.json of a FluxTransformer2DModel)"));
        if (!device.supports(dtype)) {
            throw new IllegalArgumentException(device + " cannot hold " + dtype + " values");
        }
        MemoryFit.require("Flux (" + path.getFileName() + ")", Sizes.of(path).bytes(dtype, options.transformerBits(),
                options.textEncoderBits()), device, "Options.fitting picks weights that fit");
        List<WeightSource> sources = new ArrayList<>();
        List<Tensor> all = new ArrayList<>();
        List<WeightSource> t5Sources = new ArrayList<>();
        List<Tensor> t5All = new ArrayList<>();
        try {
            ClipTextEncoder.Settings cs = Pretrained.clipSettings(config(path.resolve("text_encoder")));
            ClipTextEncoder clip = Pretrained.clip(weights(path.resolve("text_encoder"), "model", sources, device,
                    dtype, all), cs, cs.layers());
            ClipTokenizer clipTokenizer = ClipTokenizer.load(path.resolve("tokenizer"), cs.positions());
            T5Encoder t5 = loadT5(path, device, dtype, options, t5Sources, t5All);
            PretrainedTokenizer t5Tokenizer = PretrainedTokenizer.load(path.resolve("tokenizer_2"));
            FluxTransformer transformer = Pretrained.flux(weights(path.resolve("transformer"),
                    "diffusion_pytorch_model", sources, device, dtype, all).quantized(options.transformerBits()),
                    Pretrained.fluxSettings(config(path.resolve("transformer"))));
            AutoencoderKL vae = Pretrained.vae(weights(path.resolve("vae"), "diffusion_pytorch_model", sources, device,
                    dtype, all), Pretrained.vaeSettings(config(path.resolve("vae")), false));
            Path sc = path.resolve("scheduler/scheduler_config.json");
            Map<String, Object> scheduler = Files.isRegularFile(sc) ? Json.object(Json.read(sc)) : Map.of();
            return new Flux(kind, path, device, dtype, sources, all, clipTokenizer, clip, t5Tokenizer, t5, t5Sources,
                    t5All, transformer, vae, scheduler, options);
        } catch (IOException | RuntimeException e) {
            release(t5All, t5Sources);
            release(all, sources);
            throw e;
        }
    }

    private static T5Encoder loadT5(Path path, Device device, DType dtype, Options options,
            List<WeightSource> sources, List<Tensor> all) throws IOException {
        return Pretrained.t5(weights(path.resolve("text_encoder_2"), "model", sources, device, dtype, all)
                .quantized(options.textEncoderBits()), Pretrained.t5Settings(config(path.resolve("text_encoder_2"))));
    }

    private static Map<String, Object> config(Path dir) throws IOException {
        return Json.object(Json.read(dir.resolve("config.json")));
    }

    /** Opens a component's weights: one file, a sharded index, or the first {@code .safetensors}. */
    private static Weights weights(Path dir, String base, List<WeightSource> sources, Device device, DType dtype,
            List<Tensor> all) throws IOException {
        WeightSource source = WeightSource.open(MemoryFit.weightFile(dir, base, false));
        sources.add(source);
        return new Weights(source, "", device, dtype, all);
    }

    private static void release(List<Tensor> all, List<WeightSource> sources) {
        for (Tensor t : all) {
            t.release();
        }
        all.clear();
        for (WeightSource s : sources) {
            try {
                s.close();
            } catch (Exception ignored) {
                // closing a memory map fails only if it is already closed
            }
        }
        sources.clear();
    }

    // ------------------------------------------------------------------------------- accessors

    /** @return the variant */
    public Kind kind() {
        return kind;
    }

    @Override
    public String family() {
        return "FLUX_" + kind;
    }

    /** Dev: 28 steps, guidance 3.5; schnell: 4 steps. Neither uses a negative prompt or a sampler choice. */
    @Override
    public Defaults defaults() {
        return kind == Kind.DEV ? new Defaults(28, 3.5f, false, false) : new Defaults(4, 0f, false, false);
    }

    @Override
    public Device device() {
        return device;
    }

    /** @return 1024 */
    @Override
    public int nativeSize() {
        return 1024;
    }

    /** @return the transformer */
    public FluxTransformer transformer() {
        return transformer;
    }

    /** @return the options the model was loaded with */
    public Options options() {
        return options;
    }

    // ------------------------------------------------------------------------------- generation

    /** The text conditioning: T5 states {@code [1, tokens, 4096]} and the pooled CLIP embedding {@code [1, 768]}. */
    record Conditioning(Tensor context, Tensor pooled) {
    }

    @Override
    public synchronized List<GeneratedImage> generate(ImageRequest request, Progress progress) {
        int width = request.width() == 0 ? nativeSize() : request.width();
        int height = request.height() == 0 ? nativeSize() : request.height();
        Conditioning cond = encode(request.prompt());
        List<GeneratedImage> images = new ArrayList<>();
        int[] done = {0};
        int total = request.steps() * request.count();
        int channels = transformer.settings().latentChannels();
        for (int i = 0; i < request.count(); i++) {
            long seed = request.seed() + i;
            Tensor noise = StableDiffusion.gaussian(Shape.of(1, channels, height / 8, width / 8),
                    new SplittableRandom(seed), device);
            Tensor latents = denoise(noise, cond, request.steps(), request.guidance(), (step, steps, x0) -> {
                done[0]++;
                GeneratedImage preview = null;
                int every = progress.previewEvery();
                if (every > 0 && (step % every == 0 || step == steps)) {
                    preview = preview(x0);
                }
                return progress.step(done[0], total, preview);
            });
            noise.release();
            Tensor image = decode(latents);
            latents.release();
            images.add(StableDiffusion.toImage(image, seed));
            image.release();
        }
        return images;
    }

    /** The noise levels for {@code steps} steps on an image of {@code imageTokens} patches, as diffusers sets them. */
    double[] sigmas(int steps, int imageTokens) {
        if (Boolean.TRUE.equals(scheduler.get("use_dynamic_shifting"))) {
            double mu = FlowSampler.mu(imageTokens, (int) Json.integer(scheduler, "base_image_seq_len", 256),
                    (int) Json.integer(scheduler, "max_image_seq_len", 4096), Json.number(scheduler, "base_shift", 0.5),
                    Json.number(scheduler, "max_shift", 1.15));
            return FlowSampler.dynamicSigmas(steps, mu);
        }
        return FlowSampler.sigmas(steps, Json.number(scheduler, "shift", 1.0));
    }

    /**
     * Samples clean latents from noise.
     *
     * @param noise standard normal latents {@code [1, 16, h, w]}
     * @param cond the text conditioning
     * @param steps the steps
     * @param guidance the embedded guidance scale (dev only)
     * @param listener hears each step with the data estimate; can stop the run
     * @return the latents, float32
     */
    Tensor denoise(Tensor noise, Conditioning cond, int steps, float guidance,
            io.github.zzuegg.brainiac.nn.diffusion.SigmaSampler.Listener listener) {
        int tokens = (noise.shape().dim(2) / 2) * (noise.shape().dim(3) / 2);
        double[] sigmas = sigmas(steps, tokens);
        Tensor g = kind == Kind.DEV ? Tensor.of(Shape.of(1), new float[] {guidance}, device) : null;
        DType type = transformer.dtype();
        FlowSampler.Model model = (x, sigma, step) -> Var.noGrad(() -> transformer.forward(Var.of(x.to(type)),
                Tensor.of(Shape.of(1), new float[] {(float) sigma}, device), Var.of(cond.context()),
                Var.of(cond.pooled()), g, false).value());
        Tensor start = noise.to(DType.FLOAT32);
        try {
            return FlowSampler.sample(model, start, sigmas, listener);
        } finally {
            if (start != noise) {
                start.release();
            }
            if (g != null) {
                g.release();
            }
        }
    }

    /** Decodes latents into images {@code [batch, 3, h, w]} with values in {@code [0, 1]}. */
    Tensor decode(Tensor latents) {
        return StableDiffusion.decode(vae, latents);
    }

    /**
     * Encodes a prompt with CLIP (pooled) and T5 (token states), or returns the last prompt's
     * conditioning again. T5 is needed only here, so when the weights leave less than
     * {@link #WORKING} free on the device, its weights are released afterwards and read again
     * for the next new prompt: on a 16 GB GPU that is what keeps the transformer's steps in
     * device memory instead of spilling to system memory, at a few seconds per new prompt.
     * The conditioning stays valid until the next new prompt or {@link #close}.
     */
    Conditioning encode(String prompt) {
        if (encoded != null && prompt.equals(encodedPrompt)) {
            return encoded;
        }
        if (encoded != null) {
            encoded.context().release();
            encoded.pooled().release();
            encoded = null;
            encodedPrompt = null;
        }
        if (t5 == null) {
            try {
                t5 = loadT5(path, device, dtype, options, t5Sources, t5Weights);
            } catch (IOException e) {
                release(t5Weights, t5Sources);
                throw new java.io.UncheckedIOException("cannot read T5 again from " + path, e);
            }
        }
        Conditioning cond = encodeWithT5(prompt);
        OptionalLong free = MemoryFit.free(device);
        if (free.isPresent() && free.getAsLong() < WORKING) {
            releaseTextEncoder();
        }
        encoded = cond;
        encodedPrompt = prompt;
        return cond;
    }

    /** Releases T5's weights, giving their memory back to the device; the next new prompt reads them again. */
    void releaseTextEncoder() {
        t5 = null;
        release(t5Weights, t5Sources);
        device.trim();
    }

    /** @return whether T5's weights are on the device now */
    boolean textEncoderLoaded() {
        return t5 != null;
    }

    private Conditioning encodeWithT5(String prompt) {
        return Var.noGrad(() -> {
            DType type = transformer.dtype();
            Tensor clipIds = StableDiffusion.idTensor(new int[][] {clipTokenizer.encode(prompt)}, device);
            ClipTextEncoder.Output o = clip.encode(Var.of(clipIds), false);
            Tensor pooled = o.pooled().value().to(type);
            Tensor.evaluate(List.of(pooled));
            for (Var v : new Var[] {o.last(), o.penultimate(), o.pooled()}) {
                if (v != null && v.value() != pooled) {
                    v.value().release();
                }
            }
            clipIds.release();
            Tensor t5Ids = StableDiffusion.idTensor(new int[][] {t5Ids(prompt)}, device);
            Var states = t5.encode(Var.of(t5Ids), false);
            Tensor context = states.value().to(type);
            Tensor.evaluate(List.of(context));
            if (context != states.value()) {
                states.value().release();
            }
            t5Ids.release();
            return new Conditioning(context, pooled);
        });
    }

    /** T5 ids of a prompt: truncated so the end token fits, then padded to {@link Options#textTokens()}. */
    int[] t5Ids(String prompt) {
        int n = options.textTokens();
        int[] text = t5Tokenizer.encode(prompt, false);
        int eos = t5Tokenizer.eos().orElse(1);
        int pad = t5Tokenizer.id("<pad>").orElse(0);
        int[] ids = new int[n];
        Arrays.fill(ids, pad);
        int keep = Math.min(text.length, n - 1);
        System.arraycopy(text, 0, ids, 0, keep);
        ids[keep] = eos;
        return ids;
    }

    /** Latent-to-RGB factors for quick previews: approximate linear fits of the Flux VAE decoder. */
    private static final float[][] PREVIEW = {
            {-0.0346f, 0.0244f, 0.0681f}, {0.0034f, 0.0210f, 0.0687f}, {0.0275f, -0.0668f, -0.0433f},
            {-0.0174f, 0.0160f, 0.0617f}, {0.0859f, 0.0721f, 0.0329f}, {0.0004f, 0.0383f, 0.0115f},
            {0.0405f, 0.0861f, 0.0915f}, {-0.0236f, -0.0185f, -0.0259f}, {-0.0245f, 0.0250f, 0.1180f},
            {0.1008f, 0.0755f, -0.0421f}, {-0.0515f, 0.0201f, 0.0011f}, {0.0428f, -0.0012f, -0.0036f},
            {0.0817f, 0.0765f, 0.0749f}, {-0.1264f, -0.0522f, -0.1103f}, {-0.0280f, -0.0881f, -0.0499f},
            {-0.1262f, -0.0982f, -0.0778f}};
    private static final float[] PREVIEW_BIAS = {-0.0329f, -0.0718f, -0.0851f};

    private GeneratedImage preview(Tensor latents) {
        Shape s = latents.shape();
        int c = s.dim(1);
        int h = s.dim(2);
        int w = s.dim(3);
        float[] v = latents.toArray();
        byte[] rgb = new byte[w * h * 3];
        int plane = w * h;
        for (int p = 0; p < plane; p++) {
            for (int k = 0; k < 3; k++) {
                float f = PREVIEW_BIAS[k];
                for (int ch = 0; ch < Math.min(c, PREVIEW.length); ch++) {
                    f += v[ch * plane + p] * PREVIEW[ch][k];
                }
                rgb[p * 3 + k] = (byte) Math.round(Math.max(0, Math.min(1, (f + 1) / 2)) * 255);
            }
        }
        return new GeneratedImage(w, h, rgb, -1);
    }

    @Override
    public synchronized void close() {
        if (encoded != null) {
            encoded.context().release();
            encoded.pooled().release();
            encoded = null;
        }
        t5 = null;
        release(t5Weights, t5Sources);
        release(weights, sources);
    }

    @Override
    public String toString() {
        return "Flux[" + kind + ", " + transformer.dtype() + " on " + device + "]";
    }
}
