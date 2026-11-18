package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.diffusion.FlowSampler;
import io.github.zzuegg.brainiac.nn.diffusion.SigmaSampler;
import io.github.zzuegg.brainiac.nn.diffusion.T5Encoder;
import io.github.zzuegg.brainiac.nn.diffusion.WanTransformer;
import io.github.zzuegg.brainiac.nn.diffusion.WanVae;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;
import java.util.stream.Stream;

/**
 * Text-to-video generation with Alibaba's Wan 2.1 (1.3B and 14B) and Wan 2.2's two-expert
 * A14B models: a {@link WanTransformer} sampled by flow matching with UniPC
 * ({@link FlowSampler#sampleUniPc}), conditioned on umT5-XXL token states (a {@link T5Encoder}
 * with a position bias per layer), with classifier-free guidance against a negative prompt,
 * and the causal video autoencoder {@link WanVae}. Results match diffusers' {@code WanPipeline}.
 *
 * <p>It loads a diffusers folder ({@code Wan-AI/Wan2.1-T2V-1.3B-Diffusers}) or a folder of
 * single files as ComfyUI and Wan's own repositories publish them (transformer, VAE and umT5
 * {@code .safetensors} or {@code .pth}, plus the umT5 tokenizer); {@link Files#find} tells which
 * file is which by its weights.
 *
 * <pre>{@code
 * try (Wan wan = Wan.load(Path.of("Wan2.1-T2V-1.3B"), Device.best(), DType.BFLOAT16, Wan.Options.bits(0, 8))) {
 *     GeneratedVideo video = wan.generate(VideoRequest.of("a red fox runs through fresh snow"), Progress.NONE);
 *     Files.write(Path.of("fox.png"), video.apng());
 * }
 * }</pre>
 *
 * <p>Wan 2.1 1.3B takes about 2.8 GB in 16-bit and umT5-XXL about 11 GB, or 6 GB in 8 bits.
 */
public final class Wan implements AutoCloseable {

    /** Wan's default negative prompt (in Chinese, as it was trained): garish colours, overexposure, stillness, blur, subtitles, low quality, deformities, extra fingers, crowded backgrounds. */
    public static final String DEFAULT_NEGATIVE = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
            + "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
            + "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

    /** Frames per second of Wan 2.1's (and Wan 2.2 14B's) training videos. */
    public static final int FPS = 16;

    /** Frames per second of Wan 2.2 TI2V-5B, the model with the 16x autoencoder. */
    public static final int FPS_5B = 24;

    /**
     * How to load the weights.
     *
     * @param transformerBits bits per projection weight of the transformer(s): 0 keeps the
     *     load type, 8 halves 16-bit memory, 4 quarters it
     * @param textEncoderBits the same for umT5
     * @param textTokens the length text states are padded to: 512, as Wan was trained
     */
    public record Options(int transformerBits, int textEncoderBits, int textTokens) {

        /** Checks the values. */
        public Options {
            for (int b : new int[] {transformerBits, textEncoderBits}) {
                if (b != 0 && (b < 2 || b > 8)) {
                    throw new IllegalArgumentException("bits must be 0 or 2 to 8, got " + b);
                }
            }
            if (textTokens < 1) {
                throw new IllegalArgumentException("text tokens must be positive");
            }
        }

        /** @return weights as loaded, 512 text tokens */
        public static Options full() {
            return new Options(0, 0, 512);
        }

        /**
         * @param transformer the transformer's bits per weight
         * @param textEncoder umT5's bits per weight
         * @return quantized weights, 512 text tokens
         */
        public static Options bits(int transformer, int textEncoder) {
            return new Options(transformer, textEncoder, 512);
        }

        /**
         * The most precise weights that leave room to generate on a device, from the files'
         * headers: full, then umT5 in 8 bits, both in 8 bits, umT5 in 4, both in 4. On a device
         * that does not count its memory (the CPU), full.
         *
         * @param files the model's files
         * @param device where it will run
         * @param dtype the load type
         * @return the options
         */
        public static Options fitting(Files files, Device device, DType dtype) {
            if (MemoryFit.free(device).isEmpty()) {
                return full();
            }
            long free = MemoryFit.free(device).getAsLong();
            MemoryFit.Params t = MemoryFit.Params.NONE;
            for (Path p : files.transformers()) {
                MemoryFit.Params q = MemoryFit.params(p);
                t = new MemoryFit.Params(t.matrices() + q.matrices(), t.other() + q.other());
            }
            MemoryFit.Params text = MemoryFit.params(files.textEncoder());
            long vae = MemoryFit.params(files.vae()).bytes(dtype, 0);
            int[][] choices = {{0, 0}, {0, 8}, {8, 8}, {8, 4}, {4, 4}};
            for (int[] c : choices) {
                if (t.bytes(dtype, c[0]) + text.bytes(dtype, c[1]) + vae + WORKING <= free) {
                    return bits(c[0], c[1]);
                }
            }
            return bits(4, 4);
        }
    }

    /** Room for activations: a few thousand video patches, the text states and one decoded chunk. */
    static final long WORKING = 3L << 30;

    /**
     * Where a model's parts are.
     *
     * @param transformers the transformer, or Wan 2.2's high-noise then low-noise experts
     * @param vae the autoencoder's weights
     * @param textEncoder umT5's weights
     * @param tokenizer a tokenizer folder or file, or {@code null} if the text encoder file
     *     carries it ({@code spiece_model})
     * @param transformerConfig diffusers' transformer config, or {@code null}
     * @param vaeConfig diffusers' VAE config, or {@code null}
     * @param textConfig transformers' umT5 config, or {@code null}
     * @param scheduler diffusers' scheduler config, or empty
     * @param boundary for two experts, the share of the noise range (from the top) the first
     *     one handles (Wan 2.2 text-to-video: 0.875); otherwise 0
     */
    public record Files(List<Path> transformers, Path vae, Path textEncoder, Path tokenizer,
            Map<String, Object> transformerConfig, Map<String, Object> vaeConfig, Map<String, Object> textConfig,
            Map<String, Object> scheduler, double boundary) {

        /** Checks the parts. */
        public Files {
            transformers = List.copyOf(transformers);
            if (transformers.isEmpty() || transformers.size() > 2 || vae == null || textEncoder == null) {
                throw new IllegalArgumentException("a Wan model needs one or two transformers, a VAE and umT5");
            }
            scheduler = scheduler == null ? Map.of() : scheduler;
        }

        /**
         * Finds a model's parts in a diffusers folder ({@code model_index.json}), or classifies
         * the {@code .safetensors} and {@code .pth} files of a folder by their weights: a
         * transformer (two for Wan 2.2, told apart by {@code high_noise} and {@code low_noise}
         * in their names), a VAE, umT5, and a tokenizer ({@code tokenizer.json},
         * {@code spiece.model} or a {@code tokenizer} folder).
         *
         * @param folder the folder
         * @return the parts
         * @throws IOException if the folder cannot be read
         * @throws IllegalArgumentException if a part is missing
         */
        public static Files find(Path folder) throws IOException {
            if (java.nio.file.Files.isRegularFile(folder.resolve("model_index.json"))) {
                return diffusers(folder);
            }
            List<Path> transformers = new ArrayList<>();
            Path vae = null;
            Path text = null;
            Path tokenizer = null;
            try (Stream<Path> list = java.nio.file.Files.list(folder)) {
                for (Path p : list.sorted().toList()) {
                    String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                    if (java.nio.file.Files.isDirectory(p)) {
                        if (name.contains("tokenizer") && hasTokenizer(p)) {
                            tokenizer = p;
                        }
                        continue;
                    }
                    if (name.equals("tokenizer.json") || name.equals("spiece.model")) {
                        tokenizer = p;
                        continue;
                    }
                    if (!name.endsWith(".safetensors") && !name.endsWith(".pth") && !name.endsWith(".pt")
                            && !name.endsWith(".gguf")) {
                        continue;
                    }
                    switch (kind(p)) {
                        case "transformer" -> transformers.add(p);
                        case "vae" -> vae = p;
                        case "text" -> text = p;
                        default -> { }
                    }
                }
            }
            if (transformers.size() == 2) {
                // high noise first
                transformers.sort((a, b) -> Boolean.compare(!a.getFileName().toString().contains("high"),
                        !b.getFileName().toString().contains("high")));
            }
            List<String> missing = new ArrayList<>();
            if (transformers.isEmpty()) {
                missing.add("a transformer");
            }
            if (vae == null) {
                missing.add("the VAE");
            }
            if (text == null) {
                missing.add("the umT5 text encoder");
            }
            if (!missing.isEmpty()) {
                throw new IllegalArgumentException(folder + " lacks " + String.join(", ", missing)
                        + " (weights in .safetensors or .pth)");
            }
            if (tokenizer == null && !carriesTokenizer(text)) {
                throw new IllegalArgumentException(folder + " has no umT5 tokenizer: put google/umt5-xxl's"
                        + " tokenizer.json (or spiece.model) next to the weights");
            }
            return new Files(transformers, vae, text, tokenizer, null, null, null, Map.of(),
                    transformers.size() == 2 ? 0.875 : 0);
        }

        private static boolean hasTokenizer(Path dir) {
            return java.nio.file.Files.isRegularFile(dir.resolve("tokenizer.json"))
                    || java.nio.file.Files.isRegularFile(dir.resolve("spiece.model"))
                    || java.nio.file.Files.isRegularFile(dir.resolve("tokenizer.model"));
        }

        private static boolean carriesTokenizer(Path text) {
            try (WeightSource s = WeightSource.open(text)) {
                return s.contains("spiece_model");
            }
        }

        /** "transformer", "vae", "text" or "" by the weights a file holds. */
        static String kind(Path file) {
            try (WeightSource s = WeightSource.open(file)) {
                for (String n : s.names()) {
                    if (n.endsWith("time_projection.1.weight") || n.endsWith("condition_embedder.time_proj.weight")) {
                        return "transformer";
                    }
                    if (n.endsWith("decoder.conv1.weight") || n.endsWith("decoder.conv_in.weight")) {
                        return "vae";
                    }
                    if (n.equals("token_embedding.weight") || n.equals("shared.weight")
                            || n.equals("encoder.embed_tokens.weight")) {
                        return "text";
                    }
                }
                return "";
            } catch (RuntimeException e) {
                return "";
            }
        }

        private static Files diffusers(Path folder) throws IOException {
            Map<String, Object> index = Json.object(Json.read(folder.resolve("model_index.json")));
            String cls = Json.string(index, "_class_name", "");
            if (!cls.equals("WanPipeline")) {
                throw new IllegalArgumentException(folder + " holds a " + cls + ", not a Wan text-to-video pipeline");
            }
            List<Path> transformers = new ArrayList<>();
            for (String t : new String[] {"transformer", "transformer_2"}) {
                if (index.get(t) instanceof List<?> l && l.get(0) != null
                        && java.nio.file.Files.isDirectory(folder.resolve(t))) {
                    transformers.add(MemoryFit.weightFile(folder.resolve(t), "diffusion_pytorch_model", false));
                }
            }
            Path sc = folder.resolve("scheduler/scheduler_config.json");
            return new Files(transformers,
                    MemoryFit.weightFile(folder.resolve("vae"), "diffusion_pytorch_model", false),
                    MemoryFit.weightFile(folder.resolve("text_encoder"), "model", false), folder.resolve("tokenizer"),
                    config(folder.resolve("transformer")), config(folder.resolve("vae")),
                    config(folder.resolve("text_encoder")),
                    java.nio.file.Files.isRegularFile(sc) ? Json.object(Json.read(sc)) : Map.of(),
                    index.get("boundary_ratio") instanceof Number n ? n.doubleValue() : 0);
        }

        private static Map<String, Object> config(Path dir) throws IOException {
            Path c = dir.resolve("config.json");
            return java.nio.file.Files.isRegularFile(c) ? Json.object(Json.read(c)) : null;
        }
    }

    private final Device device;
    private final List<WeightSource> sources;
    private final List<Tensor> weights;
    private final PretrainedTokenizer tokenizer;
    private final T5Encoder t5;
    private final List<WanTransformer> transformers;
    private final WanVae vae;
    private final Files files;
    private final Options options;

    private Wan(Device device, List<WeightSource> sources, List<Tensor> weights, PretrainedTokenizer tokenizer,
            T5Encoder t5, List<WanTransformer> transformers, WanVae vae, Files files, Options options) {
        this.device = device;
        this.sources = sources;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.t5 = t5;
        this.transformers = transformers;
        this.vae = vae;
        this.files = files;
        this.options = options;
    }

    // ------------------------------------------------------------------------------- loading

    /**
     * Says whether a folder holds a Wan text-to-video model, without loading it.
     *
     * @param path a folder
     * @return whether {@link Files#find} finds its parts
     */
    public static boolean detect(Path path) {
        try {
            return java.nio.file.Files.isDirectory(path) && Files.find(path) != null;
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Loads a model with the most precise weights that fit the device.
     *
     * @param path a diffusers folder or a folder of single files
     * @param device where to run it
     * @param dtype the type of the networks: {@link DType#BFLOAT16} on a GPU that has it,
     *     {@link DType#FLOAT32} on the CPU
     * @return the model
     * @throws IOException if the files cannot be read
     */
    public static Wan load(Path path, Device device, DType dtype) throws IOException {
        Files files = Files.find(path);
        return load(files, device, dtype, Options.fitting(files, device, dtype));
    }

    /**
     * Loads a model.
     *
     * @param path a diffusers folder or a folder of single files
     * @param device where to run it
     * @param dtype the type of the networks
     * @param options how to store the weights
     * @return the model
     * @throws IOException if the files cannot be read
     */
    public static Wan load(Path path, Device device, DType dtype, Options options) throws IOException {
        return load(Files.find(path), device, dtype, options);
    }

    /**
     * Loads a model from its parts.
     *
     * @param files where the parts are
     * @param device where to run it
     * @param dtype the type of the networks
     * @param options how to store the weights
     * @return the model
     * @throws IOException if the files cannot be read
     */
    public static Wan load(Files files, Device device, DType dtype, Options options) throws IOException {
        if (!device.supports(dtype)) {
            throw new IllegalArgumentException(device + " cannot hold " + dtype + " values");
        }
        List<WeightSource> sources = new ArrayList<>();
        List<Tensor> all = new ArrayList<>();
        try {
            // umT5 overflows float16; it runs in bfloat16 where it can, else float32
            DType textType = dtype == DType.FLOAT16
                    ? (device.supports(DType.BFLOAT16) ? DType.BFLOAT16 : DType.FLOAT32) : dtype;
            WeightSource text = open(files.textEncoder(), sources);
            boolean original = WanPretrained.originalT5(text);
            T5Encoder t5 = WanPretrained.t5(new Weights(text, "", device, textType, all)
                    .quantized(options.textEncoderBits()), WanPretrained.t5Settings(text, files.textConfig()), original);
            PretrainedTokenizer tokenizer = tokenizer(files, text);
            List<WanTransformer> transformers = new ArrayList<>();
            for (Path p : files.transformers()) {
                WeightSource source = open(p, sources);
                UnaryOperator<String> names = WanPretrained.transformerNames(source);
                WanTransformer.Settings s = WanPretrained.transformerSettings(source, names, files.transformerConfig());
                if (s.imageFeatures() > 0 || s.inChannels() != s.outChannels()) {
                    throw new IllegalArgumentException(p.getFileName() + " is an image-to-video transformer;"
                            + " this pipeline makes videos from text");
                }
                transformers.add(WanPretrained.transformer(new Weights(source, "", device, dtype, all).renamed(names)
                        .quantized(options.transformerBits()), s));
            }
            WeightSource vs = open(files.vae(), sources);
            boolean diffusersVae = WanPretrained.diffusersVae(vs);
            String prefix = diffusersVae ? "" : WanPretrained.vaePrefix(vs);
            WanVae vae = WanPretrained.vae(new Weights(vs, prefix, device, dtype, all),
                    WanPretrained.vaeSettings(vs, prefix, files.vaeConfig()), diffusersVae, false);
            return new Wan(device, sources, all, tokenizer, t5, transformers, vae, files, options);
        } catch (IOException | RuntimeException e) {
            release(all, sources);
            throw e;
        }
    }

    private static WeightSource open(Path p, List<WeightSource> sources) {
        WeightSource s = WeightSource.open(p);
        sources.add(s);
        return s;
    }

    private static PretrainedTokenizer tokenizer(Files files, WeightSource text) throws IOException {
        if (files.tokenizer() != null) {
            return PretrainedTokenizer.load(files.tokenizer());
        }
        // ComfyUI's umT5 files carry the SentencePiece model as bytes
        float[] bytes = text.tensor("spiece_model", Device.cpu(), DType.FLOAT32).toArray();
        byte[] model = new byte[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            model[i] = (byte) (int) bytes[i];
        }
        return PretrainedTokenizer.fromSentencePiece(model, Map.of());
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

    /** @return the device it runs on */
    public Device device() {
        return device;
    }

    /** @return the transformer, or Wan 2.2's high-noise expert */
    public WanTransformer transformer() {
        return transformers.get(0);
    }

    /** @return the autoencoder */
    public WanVae vae() {
        return vae;
    }

    /** @return the options it was loaded with */
    public Options options() {
        return options;
    }

    /** @return the schedule's shift when a request gives none: the scheduler config's, else 5 */
    public float defaultShift() {
        return (float) Json.number(files.scheduler(), "flow_shift", 5.0);
    }

    /** @return the frame rate the model was trained at: {@link #FPS_5B} with Wan 2.2's 16x autoencoder, else {@link #FPS} */
    public int fps() {
        return vae.settings().residual() ? FPS_5B : FPS;
    }

    // ------------------------------------------------------------------------------- generation

    /**
     * Generates a video.
     *
     * @param request what to generate
     * @param progress hears every step and can stop the run
     * @return the video at {@link #fps()} frames per second
     * @throws java.util.concurrent.CancellationException if the progress stopped it
     */
    public synchronized GeneratedVideo generate(VideoRequest request, Progress progress) {
        int z = vae.settings().latentChannels();
        int tf = vae.settings().temporalFactor();
        int sf = vae.settings().spatialFactor();
        int side = sf * transformers.get(0).settings().patch()[1];
        if (request.width() % side != 0 || request.height() % side != 0) {
            throw new IllegalArgumentException(this + " makes videos whose sides are multiples of " + side + ", not "
                    + request.width() + "x" + request.height());
        }
        Shape shape = Shape.of(1, z, (request.frames() - 1) / tf + 1, request.height() / sf, request.width() / sf);
        Tensor noise = StableDiffusion.gaussian(shape, new SplittableRandom(request.seed()), device);
        try {
            return generate(request, noise, progress);
        } finally {
            noise.release();
        }
    }

    /** Generates from given starting noise {@code [1, z, latentFrames, h / 8, w / 8]}. */
    GeneratedVideo generate(VideoRequest request, Tensor noise, Progress progress) {
        boolean guided = request.guidance() > 1;
        Tensor cond = encode(request.prompt());
        Tensor uncond = guided ? encode(request.negativePrompt() == null ? DEFAULT_NEGATIVE : request.negativePrompt())
                : null;
        Tensor latents;
        try {
            float shift = request.shift() > 0 ? request.shift() : defaultShift();
            latents = denoise(noise, cond, uncond, request.steps(), request.guidance(), shift,
                    (step, steps, x0) -> progress.step(step, steps, null));
        } finally {
            cond.release();
            if (uncond != null) {
                uncond.release();
            }
        }
        try {
            return new GeneratedVideo(decode(latents), fps(), request.seed());
        } finally {
            latents.release();
        }
    }

    /**
     * Samples clean latents from noise.
     *
     * @param noise standard normal latents
     * @param cond the prompt's text states
     * @param uncond the negative prompt's, or {@code null} without guidance
     * @param steps the steps
     * @param guidance the guidance scale
     * @param shift the schedule's shift
     * @param listener hears each step
     * @return the latents, float32
     */
    Tensor denoise(Tensor noise, Tensor cond, Tensor uncond, int steps, float guidance, float shift,
            SigmaSampler.Listener listener) {
        double[] sigmas = FlowSampler.uniPcSigmas(steps, shift);
        double boundary = files.boundary() * 1000;
        FlowSampler.Model model = (x, sigma, step) -> Var.noGrad(() -> {
            // diffusers hands the model whole timesteps
            float t = (float) Math.floor(sigma * 1000);
            WanTransformer dit = transformers.size() == 2 && t < boundary ? transformers.get(1) : transformers.get(0);
            DType type = dit.dtype();
            Tensor time = Tensor.of(Shape.of(1), new float[] {t}, device);
            Tensor in = x.to(type);
            Tensor c = dit.forward(Var.of(in), time, Var.of(cond), null, false).value().to(DType.FLOAT32);
            if (uncond == null) {
                return c;
            }
            Tensor u = dit.forward(Var.of(in), time, Var.of(uncond), null, false).value().to(DType.FLOAT32);
            return u.add(c.sub(u).mul(guidance));
        });
        Tensor start = noise.to(DType.FLOAT32);
        try {
            return FlowSampler.sampleUniPc(model, start, sigmas, listener);
        } finally {
            if (start != noise) {
                start.release();
            }
        }
    }

    /** Decodes latents into frames, one latent frame at a time. */
    List<GeneratedImage> decode(Tensor latents) {
        List<GeneratedImage> frames = new ArrayList<>();
        vae.decode(latents, chunk -> {
            Tensor unit = chunk.add(1f).mul(0.5f);   // the decoder clamps to [-1, 1]
            int n = unit.shape().dim(0);
            for (int i = 0; i < n; i++) {
                Tensor f = unit.slice(0, i, i + 1);
                frames.add(StableDiffusion.toImage(f, -1));
                f.release();
            }
            unit.release();
            chunk.release();
        });
        return frames;
    }

    /**
     * Encodes a prompt as Wan does: whitespace collapsed, umT5 over its tokens and the end
     * token, then zeros up to {@link Options#textTokens()}.
     *
     * @param prompt the text
     * @return {@code [1, textTokens, features]} in the transformer's type
     */
    Tensor encode(String prompt) {
        int[] ids = ids(prompt);
        return Var.noGrad(() -> {
            Tensor idTensor = StableDiffusion.idTensor(new int[][] {ids}, device);
            Var states = t5.encode(Var.of(idTensor), false);
            idTensor.release();
            int n = ids.length;
            int total = options.textTokens();
            int d = states.shape().dim(2);
            DType type = transformers.get(0).dtype();
            Tensor s = states.value().to(type);
            Tensor out = n == total ? s
                    : Tensor.concat(1, s, Tensor.zeros(Shape.of(1, total - n, d), type, device));
            Tensor.evaluate(List.of(out));
            if (s != out) {
                s.release();
            }
            if (states.value() != s) {
                states.value().release();
            }
            return out;
        });
    }

    /** The prompt's token ids: cleaned, with the end token, cut to {@link Options#textTokens()}. */
    int[] ids(String prompt) {
        String clean = prompt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
                .replace("&#39;", "'").replaceAll("\\s+", " ").strip();
        int[] text = tokenizer.encode(clean, false);
        int eos = tokenizer.eos().orElse(1);
        int keep = Math.min(text.length, options.textTokens() - 1);
        int[] ids = new int[keep + 1];
        System.arraycopy(text, 0, ids, 0, keep);
        ids[keep] = eos;
        return ids;
    }

    @Override
    public synchronized void close() {
        release(weights, sources);
    }

    @Override
    public String toString() {
        return "Wan[" + transformers.get(0) + (transformers.size() == 2 ? " + low-noise expert" : "") + ", "
                + transformers.get(0).dtype() + " on " + device + "]";
    }
}
