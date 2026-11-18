package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.diffusion.ClipVisionEncoder;
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
 * <p>Image-to-video models start from a picture ({@link #generate(VideoRequest, GeneratedImage,
 * Progress)}; {@link #input()} says what a model takes), as diffusers'
 * {@code WanImageToVideoPipeline}: Wan 2.2 TI2V-5B puts the picture's latents in place of the
 * first latent frame and holds that frame at noise level 0; Wan 2.2 I2V-A14B and Wan 2.1 I2V
 * take a mask and the latents of the picture followed by black frames as extra input channels,
 * and Wan 2.1 also attends to the picture's CLIP features (OpenCLIP ViT-H/14, all but the last
 * layer: ComfyUI's {@code clip_vision_h.safetensors} next to the other files).
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
     *     one handles (Wan 2.2 text-to-video: 0.875, image-to-video: 0.9); otherwise 0
     * @param imageEncoder the CLIP vision model Wan 2.1's image-to-video models need, or {@code null}
     * @param imageEncoderConfig transformers' config of it, or {@code null}
     */
    public record Files(List<Path> transformers, Path vae, Path textEncoder, Path tokenizer,
            Map<String, Object> transformerConfig, Map<String, Object> vaeConfig, Map<String, Object> textConfig,
            Map<String, Object> scheduler, double boundary, Path imageEncoder,
            Map<String, Object> imageEncoderConfig) {

        /**
         * Parts without an image encoder.
         *
         * @param transformers the transformer, or the high-noise then low-noise experts
         * @param vae the autoencoder's weights
         * @param textEncoder umT5's weights
         * @param tokenizer the tokenizer, or {@code null}
         * @param transformerConfig the transformer config, or {@code null}
         * @param vaeConfig the VAE config, or {@code null}
         * @param textConfig the umT5 config, or {@code null}
         * @param scheduler the scheduler config, or empty
         * @param boundary the experts' boundary, or 0
         */
        public Files(List<Path> transformers, Path vae, Path textEncoder, Path tokenizer,
                Map<String, Object> transformerConfig, Map<String, Object> vaeConfig, Map<String, Object> textConfig,
                Map<String, Object> scheduler, double boundary) {
            this(transformers, vae, textEncoder, tokenizer, transformerConfig, vaeConfig, textConfig, scheduler,
                    boundary, null, null);
        }

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
            Path clip = null;
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
                        case "clip" -> clip = p;
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
            double boundary = transformers.size() < 2 ? 0 : imageToVideo(transformers.get(0)) ? 0.9 : 0.875;
            return new Files(transformers, vae, text, tokenizer, null, null, null, Map.of(), boundary, clip, null);
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

        /** Whether a transformer file takes more input channels than it puts out (image-to-video). */
        private static boolean imageToVideo(Path transformer) {
            try (WeightSource s = WeightSource.open(transformer)) {
                UnaryOperator<String> names = WanPretrained.transformerNames(s);
                WanTransformer.Settings t = WanPretrained.transformerSettings(s, names, null);
                return t.inChannels() != t.outChannels();
            } catch (RuntimeException e) {
                return false;
            }
        }

        /** "transformer", "vae", "text", "clip" or "" by the weights a file holds. */
        static String kind(Path file) {
            try (WeightSource s = WeightSource.open(file)) {
                for (String n : s.names()) {
                    if (n.endsWith("time_projection.1.weight") || n.endsWith("condition_embedder.time_proj.weight")) {
                        return "transformer";
                    }
                    if (n.endsWith("decoder.conv1.weight") || n.endsWith("decoder.conv_in.weight")) {
                        return "vae";
                    }
                    if (n.endsWith("embeddings.patch_embedding.weight")) {
                        return "clip";
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
            if (!cls.equals("WanPipeline") && !cls.equals("WanImageToVideoPipeline")) {
                throw new IllegalArgumentException(folder + " holds a " + cls + ", not a Wan pipeline");
            }
            List<Path> transformers = new ArrayList<>();
            for (String t : new String[] {"transformer", "transformer_2"}) {
                if (index.get(t) instanceof List<?> l && l.get(0) != null
                        && java.nio.file.Files.isDirectory(folder.resolve(t))) {
                    transformers.add(MemoryFit.weightFile(folder.resolve(t), "diffusion_pytorch_model", false));
                }
            }
            Path sc = folder.resolve("scheduler/scheduler_config.json");
            boolean clip = index.get("image_encoder") instanceof List<?> ie && ie.get(0) != null
                    && java.nio.file.Files.isDirectory(folder.resolve("image_encoder"));
            return new Files(transformers,
                    MemoryFit.weightFile(folder.resolve("vae"), "diffusion_pytorch_model", false),
                    MemoryFit.weightFile(folder.resolve("text_encoder"), "model", false), folder.resolve("tokenizer"),
                    config(folder.resolve("transformer")), config(folder.resolve("vae")),
                    config(folder.resolve("text_encoder")),
                    java.nio.file.Files.isRegularFile(sc) ? Json.object(Json.read(sc)) : Map.of(),
                    index.get("boundary_ratio") instanceof Number n ? n.doubleValue() : 0,
                    clip ? MemoryFit.weightFile(folder.resolve("image_encoder"), "model", false) : null,
                    clip ? config(folder.resolve("image_encoder")) : null);
        }

        private static Map<String, Object> config(Path dir) throws IOException {
            Path c = dir.resolve("config.json");
            return java.nio.file.Files.isRegularFile(c) ? Json.object(Json.read(c)) : null;
        }
    }

    /**
     * Which device holds which part. Experts on different GPUs let Wan 2.2's two 14B experts
     * run on two smaller cards, and umT5 (11 GB in 16 bits) can sit on another GPU or the
     * CPU, since it runs once per video. Latents and text states move between devices
     * through the host, which costs little next to a step.
     *
     * @param text umT5 and the CLIP image encoder
     * @param transformers the transformer, or one device per expert (high noise first); the
     *     first one also holds the sampler's latents
     * @param vae the autoencoder
     */
    public record Placement(Device text, List<Device> transformers, Device vae) {

        /** Checks the parts. */
        public Placement {
            java.util.Objects.requireNonNull(text, "text");
            java.util.Objects.requireNonNull(vae, "vae");
            transformers = List.copyOf(transformers);
            if (transformers.isEmpty()) {
                throw new IllegalArgumentException("a placement needs a device for the transformer");
            }
        }

        /**
         * @param device the device
         * @return everything on one device
         */
        public static Placement single(Device device) {
            return new Placement(device, List.of(device), device);
        }

        /**
         * @param i the expert, 0 for the transformer or the high-noise expert
         * @return its device; experts beyond the list share the last one
         */
        public Device transformer(int i) {
            return transformers.get(Math.min(i, transformers.size() - 1));
        }

        /** @return the distinct devices, in the order text, transformers, autoencoder */
        public List<Device> devices() {
            java.util.LinkedHashSet<Device> all = new java.util.LinkedHashSet<>();
            all.add(text);
            all.addAll(transformers);
            all.add(vae);
            return List.copyOf(all);
        }

        @Override
        public String toString() {
            return "text on " + text + ", transformer" + (transformers.size() > 1 ? "s" : "") + " on "
                    + transformers.stream().map(String::valueOf).collect(java.util.stream.Collectors.joining(" + "))
                    + ", autoencoder on " + vae;
        }
    }

    /**
     * Where a model's parts can go and in what precision, for one video size, estimated from
     * the weight files' headers and the devices' free memory before anything loads.
     *
     * @param placement which device holds which part
     * @param options the precision
     * @param need what each device would hold at its peak, in bytes
     * @param free what each device has free, in bytes ({@link Long#MAX_VALUE} for a device that
     *     does not count its memory)
     * @param fits whether every device has room; when not, the plan is the smallest found
     */
    public record Plan(Placement placement, Options options, Map<Device, Long> need, Map<Device, Long> free,
            boolean fits) {

        /** @return one line per device: what it holds and how much room it has */
        public String describe() {
            StringBuilder b = new StringBuilder(fits ? "fits: " : "does not fit: ")
                    .append(placement).append(", ").append(options).append('\n');
            for (Map.Entry<Device, Long> e : need.entrySet()) {
                long f = free.get(e.getKey());
                b.append("  ").append(e.getKey()).append(": needs ").append(MemoryFit.gb(e.getValue()))
                        .append(f == Long.MAX_VALUE ? "" : " of " + MemoryFit.gb(f) + " free").append('\n');
            }
            return b.toString();
        }
    }

    /**
     * Plans where a model's parts go for videos of a size: the most precise options (as
     * {@link Options#fitting}) for which the transformers, then umT5 and the CLIP encoder,
     * then the autoencoder each fit on the device with the most room left, spreading Wan
     * 2.2's experts over different devices when there are several. Each transformer's device
     * keeps room for a step's activations at that size and the autoencoder's for decoding;
     * both are estimates, on the safe side.
     *
     * @param files the model's parts
     * @param devices the devices to use, at least one
     * @param dtype the load type
     * @param request the largest video it will make
     * @return the plan
     * @throws IOException if a header cannot be read
     */
    public static Plan plan(Files files, List<Device> devices, DType dtype, VideoRequest request) throws IOException {
        if (devices.isEmpty()) {
            throw new IllegalArgumentException("no devices");
        }
        List<Device> pool = List.copyOf(new java.util.LinkedHashSet<>(devices));
        List<MemoryFit.Params> experts = new ArrayList<>();
        WanTransformer.Settings shape;
        try (WeightSource s = WeightSource.open(files.transformers().get(0))) {
            shape = WanPretrained.transformerSettings(s, WanPretrained.transformerNames(s), files.transformerConfig());
        }
        for (Path p : files.transformers()) {
            experts.add(MemoryFit.params(p));
        }
        WanVae.Settings vaeShape;
        try (WeightSource s = WeightSource.open(files.vae())) {
            String prefix = WanPretrained.diffusersVae(s) ? "" : WanPretrained.vaePrefix(s);
            vaeShape = WanPretrained.vaeSettings(s, prefix, files.vaeConfig());
        }
        MemoryFit.Params text = MemoryFit.params(files.textEncoder());
        long clipBytes = files.imageEncoder() == null ? 0 : MemoryFit.params(files.imageEncoder()).bytes(dtype, 0);
        long vaeBytes = MemoryFit.params(files.vae()).bytes(dtype, 0);
        long stepWork = stepBytes(shape, vaeShape, dtype, request);
        long decodeWork = decodeBytes(vaeShape, dtype, request);
        Map<Device, Long> free = new java.util.LinkedHashMap<>();
        for (Device d : pool) {
            free.put(d, MemoryFit.free(d).orElse(Long.MAX_VALUE));
        }
        int[][] choices = {{0, 0}, {0, 8}, {8, 8}, {8, 4}, {4, 4}};
        Plan best = null;
        for (int[] c : choices) {
            Map<Device, Long> need = new java.util.LinkedHashMap<>();
            Map<Device, Long> work = new java.util.HashMap<>();
            List<Device> placed = new ArrayList<>();
            for (MemoryFit.Params e : experts) {
                // a device no other expert uses yet, the roomiest
                Device d = roomiest(pool, free, need, placed);
                placed.add(d);
                need.merge(d, pooled(e.bytes(dtype, c[0])), Long::sum);
                work.merge(d, stepWork, Math::max);
            }
            Device textDevice = roomiest(pool, free, sum(need, work), List.of());
            need.merge(textDevice, pooled(text.bytes(dtype, c[1]) + clipBytes), Long::sum);
            Device vaeDevice = roomiest(pool, free, sum(need, work), List.of());
            need.merge(vaeDevice, pooled(vaeBytes), Long::sum);
            work.merge(vaeDevice, decodeWork, Math::max);
            Map<Device, Long> total = sum(need, work);
            boolean fits = total.entrySet().stream().allMatch(en -> en.getValue() <= free.get(en.getKey()));
            Plan plan = new Plan(new Placement(textDevice, placed, vaeDevice), new Options(c[0], c[1], 512), total,
                    free, fits);
            if (fits) {
                return plan;
            }
            best = plan;
        }
        return best;
    }

    /** Weights as a GPU pool holds them: rounded up to its block sizes, about a tenth more (measured on CUDA). */
    private static long pooled(long bytes) {
        return bytes + bytes / 10;
    }

    private static Map<Device, Long> sum(Map<Device, Long> a, Map<Device, Long> b) {
        Map<Device, Long> s = new java.util.LinkedHashMap<>(a);
        b.forEach((k, v) -> s.merge(k, v, Long::sum));
        return s;
    }

    /** The device with the most room left, preferring ones not in {@code avoid}. */
    private static Device roomiest(List<Device> pool, Map<Device, Long> free, Map<Device, Long> used,
            List<Device> avoid) {
        Device best = null;
        long room = Long.MIN_VALUE;
        boolean bestAvoided = true;
        for (Device d : pool) {
            long f = free.get(d);
            long r = f == Long.MAX_VALUE ? Long.MAX_VALUE : f - used.getOrDefault(d, 0L);
            boolean avoided = avoid.contains(d);
            if (best == null || (bestAvoided && !avoided) || (bestAvoided == avoided && r > room)) {
                best = d;
                room = r;
                bestAvoided = avoided;
            }
        }
        return best;
    }

    /**
     * A transformer step's working memory, from runs of Wan 2.1 1.3B (832x480, 33 frames) and
     * 14B (832x480, 9 frames) on CUDA: per video patch the residual stream and a block part's
     * temporaries, about 64 floats per feature; a slice of self-attention's scores in
     * float32 and their softmax; the sampler's latents; and half the transformer's weights
     * again, for memory the CUDA backend held over the 14B's steps that is not yet
     * explained. With 1 GB to spare.
     */
    static long stepBytes(WanTransformer.Settings t, WanVae.Settings vae, DType dtype, VideoRequest request) {
        int sf = vae.spatialFactor();
        int[] p = t.patch();
        long frames = (request.frames() - 1) / vae.temporalFactor() + 1;
        long patches = frames / p[0] * (request.height() / sf / p[1]) * (request.width() / sf / p[2]);
        long stream = patches * t.features() * 64L * 4;
        long scores = Math.min((long) t.heads() * patches * patches, WanTransformer.attentionWork()) * 20;
        long latents = frames * (request.height() / sf) * (request.width() / sf) * vae.latentChannels() * 4L * 8;
        long d = t.features();
        long weights = t.layers() * (8 * d * d + 2 * d * t.ffnSize()) * dtype.bytes();
        return stream + scores + latents + weights / 2 + (1L << 30);
    }

    /**
     * Decoding one latent frame: the widest full-resolution level over the frames of a chunk
     * and the causal caches, twenty times over for the convolutions' temporaries (measured:
     * 17 GB at 832x480 on CUDA and 2.7 GB at 416x240 on Vulkan with Wan 2.1's decoder), with
     * 1 GB to spare.
     */
    static long decodeBytes(WanVae.Settings vae, DType dtype, VideoRequest request) {
        // Wan 2.2's decoder ends at half resolution and unfolds 2x2 patches
        long pixels = (long) request.width() * request.height() / ((long) vae.patch() * vae.patch());
        long width = vae.decoderBase() * (long) vae.multipliers()[0];
        return pixels * width * (vae.temporalFactor() + 2) * Math.max(4, dtype.bytes()) * 20 + (1L << 30);
    }

    private final Placement placement;
    /** Where sampling happens: the (first) transformer's device. */
    private final Device device;
    private final List<WeightSource> sources;
    private final List<Tensor> weights;
    private final PretrainedTokenizer tokenizer;
    private final T5Encoder t5;
    private final List<WanTransformer> transformers;
    private final WanVae vae;
    private final Files files;
    private final Options options;
    private final ClipVisionEncoder clip;
    private final Input input;

    /** What a model generates from. */
    public enum Input {
        /** A prompt alone. */
        TEXT,
        /** A prompt, and optionally a first frame (Wan 2.2 TI2V-5B). */
        TEXT_OR_IMAGE,
        /** A prompt and a first frame (Wan 2.1 I2V, Wan 2.2 I2V-A14B). */
        IMAGE
    }

    private Wan(Placement placement, List<WeightSource> sources, List<Tensor> weights, PretrainedTokenizer tokenizer,
            T5Encoder t5, List<WanTransformer> transformers, WanVae vae, Files files, Options options,
            ClipVisionEncoder clip, Input input) {
        this.placement = placement;
        this.device = placement.transformer(0);
        this.sources = sources;
        this.weights = weights;
        this.tokenizer = tokenizer;
        this.t5 = t5;
        this.transformers = transformers;
        this.vae = vae;
        this.files = files;
        this.options = options;
        this.clip = clip;
        this.input = input;
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
        return load(files, Placement.single(device), dtype, options);
    }

    /**
     * Loads a model with its parts on given devices, such as Wan 2.2's two experts on two
     * GPUs; {@link #plan} picks a placement that fits.
     *
     * @param files where the parts are
     * @param placement which device holds which part
     * @param dtype the type of the networks
     * @param options how to store the weights
     * @return the model
     * @throws IOException if the files cannot be read
     */
    public static Wan load(Files files, Placement placement, DType dtype, Options options) throws IOException {
        for (Device d : placement.devices()) {
            if (!d.supports(dtype)) {
                throw new IllegalArgumentException(d + " cannot hold " + dtype + " values");
            }
        }
        Device device = placement.text();
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
            WeightSource vs = open(files.vae(), sources);
            boolean diffusersVae = WanPretrained.diffusersVae(vs);
            String prefix = diffusersVae ? "" : WanPretrained.vaePrefix(vs);
            WanVae.Settings vaeSettings = WanPretrained.vaeSettings(vs, prefix, files.vaeConfig());
            List<WanTransformer.Settings> shapes = new ArrayList<>();
            List<WeightSource> transformerSources = new ArrayList<>();
            List<UnaryOperator<String>> transformerNames = new ArrayList<>();
            for (Path p : files.transformers()) {
                WeightSource source = open(p, sources);
                UnaryOperator<String> names = WanPretrained.transformerNames(source);
                shapes.add(WanPretrained.transformerSettings(source, names, files.transformerConfig()));
                transformerSources.add(source);
                transformerNames.add(names);
            }
            Input input = input(shapes.get(0), vaeSettings, files);
            List<WanTransformer> transformers = new ArrayList<>();
            for (int i = 0; i < shapes.size(); i++) {
                transformers.add(WanPretrained.transformer(new Weights(transformerSources.get(i), "",
                        placement.transformer(i), dtype, all)
                        .renamed(transformerNames.get(i)).quantized(options.transformerBits()), shapes.get(i)));
            }
            WanVae vae = WanPretrained.vae(new Weights(vs, prefix, placement.vae(), dtype, all), vaeSettings, diffusersVae,
                    input != Input.TEXT);
            ClipVisionEncoder clip = null;
            if (shapes.get(0).imageFeatures() > 0) {
                WeightSource cs = open(files.imageEncoder(), sources);
                String cp = WanPretrained.clipPrefix(cs);
                ClipVisionEncoder.Settings c = WanPretrained.clipSettings(cs, cp, files.imageEncoderConfig());
                // Wan conditions on the penultimate layer's states
                clip = WanPretrained.clipVision(new Weights(cs, cp, device, dtype, all), c.withLayers(c.layers() - 1));
            }
            return new Wan(placement, sources, all, tokenizer, t5, transformers, vae, files, options, clip, input);
        } catch (IOException | RuntimeException e) {
            release(all, sources);
            throw e;
        }
    }

    /**
     * Says what a model generates from, from its files' headers, without loading it.
     *
     * @param files the model's parts
     * @return {@link Input#IMAGE} for an image-to-video transformer, {@link Input#TEXT_OR_IMAGE}
     *     with Wan 2.2's 16x autoencoder, else {@link Input#TEXT}
     */
    public static Input inputOf(Files files) {
        WanTransformer.Settings t;
        try (WeightSource s = WeightSource.open(files.transformers().get(0))) {
            t = WanPretrained.transformerSettings(s, WanPretrained.transformerNames(s), files.transformerConfig());
        }
        if (t.inChannels() != t.outChannels()) {
            return Input.IMAGE;
        }
        try (WeightSource s = WeightSource.open(files.vae())) {
            String prefix = WanPretrained.diffusersVae(s) ? "" : WanPretrained.vaePrefix(s);
            return WanPretrained.vaeSettings(s, prefix, files.vaeConfig()).residual() ? Input.TEXT_OR_IMAGE
                    : Input.TEXT;
        }
    }

    /** What a transformer of this shape generates from, checking that its parts are there. */
    private static Input input(WanTransformer.Settings t, WanVae.Settings vae, Files files) {
        int z = vae.latentChannels();
        if (t.inChannels() == t.outChannels()) {
            if (t.imageFeatures() > 0) {
                throw new IllegalArgumentException("a Wan transformer with image features takes a mask and image"
                        + " latents too, but it has " + t.inChannels() + " input channels");
            }
            return vae.residual() ? Input.TEXT_OR_IMAGE : Input.TEXT;
        }
        if (t.inChannels() != t.outChannels() + vae.temporalFactor() + z) {
            throw new IllegalArgumentException("the transformer takes " + t.inChannels() + " channels, which fits"
                    + " neither text-to-video (" + z + ") nor image-to-video (" + (2 * z + vae.temporalFactor())
                    + ") with this autoencoder");
        }
        if (t.imageFeatures() > 0 && files.imageEncoder() == null) {
            throw new IllegalArgumentException("this Wan 2.1 image-to-video model needs the CLIP image encoder:"
                    + " put clip_vision_h.safetensors (OpenCLIP ViT-H/14) next to the other files");
        }
        return Input.IMAGE;
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

    /** @return the device it samples on: the (first) transformer's */
    public Device device() {
        return device;
    }

    /** @return which device holds which part */
    public Placement placement() {
        return placement;
    }

    /** @return the transformer, or Wan 2.2's high-noise expert */
    public WanTransformer transformer() {
        return transformers.get(0);
    }

    /** @return the autoencoder */
    public WanVae vae() {
        return vae;
    }

    /** @return what the model generates from */
    public Input input() {
        return input;
    }

    /** @return the CLIP image encoder of a Wan 2.1 image-to-video model, or {@code null} */
    public ClipVisionEncoder imageEncoder() {
        return clip;
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
     * Generates a video from text.
     *
     * @param request what to generate
     * @param progress hears every step and can stop the run
     * @return the video at {@link #fps()} frames per second
     * @throws java.util.concurrent.CancellationException if the progress stopped it
     * @throws IllegalArgumentException if the model needs a first frame ({@link Input#IMAGE})
     */
    public GeneratedVideo generate(VideoRequest request, Progress progress) {
        return generate(request, (GeneratedImage) null, progress);
    }

    /**
     * Generates a video that starts from a picture, or from text alone.
     *
     * @param request what to generate
     * @param start the first frame, scaled to cover the request's size and cut to it, or
     *     {@code null} for text alone
     * @param progress hears every step and can stop the run
     * @return the video at {@link #fps()} frames per second
     * @throws java.util.concurrent.CancellationException if the progress stopped it
     * @throws IllegalArgumentException if the model cannot take the picture, or needs one
     */
    public synchronized GeneratedVideo generate(VideoRequest request, GeneratedImage start, Progress progress) {
        int z = vae.settings().latentChannels();
        int tf = vae.settings().temporalFactor();
        int sf = vae.settings().spatialFactor();
        int side = sf * transformers.get(0).settings().patch()[1];
        if (request.width() % side != 0 || request.height() % side != 0) {
            throw new IllegalArgumentException(this + " makes videos whose sides are multiples of " + side + ", not "
                    + request.width() + "x" + request.height());
        }
        checkRoom(request);
        Shape shape = Shape.of(1, z, (request.frames() - 1) / tf + 1, request.height() / sf, request.width() / sf);
        Tensor noise = StableDiffusion.gaussian(shape, new SplittableRandom(request.seed()), device);
        try {
            return generate(request, start, noise, progress);
        } finally {
            noise.release();
        }
    }

    /**
     * Fails with a clear message when a transformer's device lacks the room a step at this
     * size needs, instead of letting the driver run out (or reset the GPU) mid-step.
     */
    private void checkRoom(VideoRequest request) {
        long step = stepBytes(transformers.get(0).settings(), vae.settings(), transformers.get(0).dtype(), request);
        for (Device d : new java.util.LinkedHashSet<>(placement.transformers())) {
            java.util.OptionalLong free = MemoryFit.free(d);
            if (free.isPresent() && free.getAsLong() < step) {
                throw new IllegalStateException(String.format(Locale.ROOT, "a %dx%d video of %d frames needs about"
                        + " %s free on %s for a step, but only %s is free: close other programs that use the GPU,"
                        + " or make it smaller or shorter", request.width(), request.height(), request.frames(),
                        MemoryFit.gb(step), d, MemoryFit.gb(free.getAsLong())));
            }
        }
    }

    /** Generates from text and given starting noise. */
    GeneratedVideo generate(VideoRequest request, Tensor noise, Progress progress) {
        return generate(request, null, noise, progress);
    }

    /** Generates from given starting noise {@code [1, z, latentFrames, h / f, w / f]}. */
    GeneratedVideo generate(VideoRequest request, GeneratedImage start, Tensor noise, Progress progress) {
        if (start == null && input == Input.IMAGE) {
            throw new IllegalArgumentException(this + " is an image-to-video model: it needs a first frame");
        }
        if (start != null && input == Input.TEXT) {
            throw new IllegalArgumentException(this + " makes videos from text only");
        }
        boolean guided = request.guidance() > 1 || request.lowNoiseGuidance() > 1;
        Conditioning image = start == null ? Conditioning.NONE : conditioning(request, start);
        Tensor cond = null;
        Tensor uncond = null;
        Tensor latents;
        try {
            cond = encode(request.prompt());
            uncond = guided ? encode(request.negativePrompt() == null ? DEFAULT_NEGATIVE : request.negativePrompt())
                    : null;
            float shift = request.shift() > 0 ? request.shift() : defaultShift();
            latents = denoise(noise, cond, uncond, image, request.steps(), request.guidance(),
                    request.lowNoiseGuidance(), shift, (step, steps, x0) -> progress.step(step, steps, null));
            if (image.firstFrame() != null) {
                Tensor done = withFirstFrame(latents, image.firstFrame());
                latents.release();
                latents = done;
            }
        } finally {
            if (cond != null) {
                cond.release();
            }
            if (uncond != null) {
                uncond.release();
            }
            image.release();
        }
        try {
            return new GeneratedVideo(decode(latents), fps(), request.seed());
        } finally {
            latents.release();
        }
    }

    /**
     * What a first frame adds to each step.
     *
     * @param firstFrame Wan 2.2 TI2V: the picture's latents {@code [1, z, 1, h, w]}, which
     *     replace the first latent frame
     * @param channels Wan 2.1 / 2.2 I2V: the mask and the latents of the picture and black
     *     frames, {@code [1, 4 + z, frames, h, w]}, appended to the input channels
     * @param features Wan 2.1 I2V: the picture's CLIP states {@code [1, tokens, features]}
     */
    record Conditioning(Tensor firstFrame, Tensor channels, Tensor features) {

        static final Conditioning NONE = new Conditioning(null, null, null);

        void release() {
            for (Tensor t : new Tensor[] {firstFrame, channels, features}) {
                if (t != null) {
                    t.release();
                }
            }
        }
    }

    /** Encodes the first frame for this model, as diffusers' {@code WanImageToVideoPipeline} prepares it. */
    Conditioning conditioning(VideoRequest request, GeneratedImage start) {
        GeneratedImage frame = Pixels.cover(start, request.width(), request.height(), Pixels.Filter.LANCZOS);
        return Var.noGrad(() -> {
            Tensor features = clip == null ? null : imageFeatures(start);
            Tensor picture = picture(frame);
            try {
                if (input == Input.TEXT_OR_IMAGE) {
                    Tensor latents = vae.encode(picture);
                    Tensor here = latents.to(device);
                    if (here != latents) {
                        latents.release();
                    }
                    return new Conditioning(here, null, null);
                }
                int tf = vae.settings().temporalFactor();
                Tensor video = request.frames() == 1 ? picture : Tensor.concat(2, picture, Tensor.zeros(
                        Shape.of(1, 3, request.frames() - 1, request.height(), request.width()), picture.dtype(),
                        placement.vae()));
                Tensor latents = vae.encode(video);
                if (video != picture) {
                    video.release();
                }
                Shape s = latents.shape();
                // the mask: tf channels per latent frame, ones for the given first frame only
                Tensor ones = Tensor.full(Shape.of(1, tf, 1, s.dim(3), s.dim(4)), 1f, DType.FLOAT32, placement.vae());
                Tensor mask = s.dim(2) == 1 ? ones : Tensor.concat(2, ones,
                        Tensor.zeros(Shape.of(1, tf, s.dim(2) - 1, s.dim(3), s.dim(4)), DType.FLOAT32, placement.vae()));
                Tensor channels = Tensor.concat(1, mask, latents).to(transformers.get(0).dtype());
                Tensor.evaluate(List.of(channels));
                for (Tensor t : new Tensor[] {ones, mask, latents}) {
                    if (t != channels) {
                        t.release();
                    }
                }
                return new Conditioning(null, channels, features);
            } finally {
                picture.release();
            }
        });
    }

    /** An image as one video frame {@code [1, 3, 1, h, w]} in {@code [-1, 1]}, in the autoencoder's type. */
    private Tensor picture(GeneratedImage image) {
        int w = image.width();
        int h = image.height();
        int plane = w * h;
        float[] v = new float[3 * plane];
        byte[] rgb = image.rgb();
        for (int p = 0; p < plane; p++) {
            for (int c = 0; c < 3; c++) {
                v[c * plane + p] = (rgb[p * 3 + c] & 0xff) / 255f * 2 - 1;
            }
        }
        Tensor f = Tensor.of(Shape.of(1, 3, 1, h, w), v, placement.vae());
        Tensor t = f.to(vae.dtype());
        if (t != f) {
            f.release();
        }
        return t;
    }

    /**
     * The CLIP states of a picture, preprocessed as transformers' {@code CLIPImageProcessor}:
     * the short side scaled to the encoder's size (bicubic), the middle square, CLIP's mean and
     * deviation.
     *
     * @return {@code [1, tokens, features]} in the transformer's type
     */
    Tensor imageFeatures(GeneratedImage image) {
        int size = clip.settings().imageSize();
        int w = image.width();
        int h = image.height();
        int nw = w <= h ? size : (int) ((long) size * w / h);
        int nh = w <= h ? (int) ((long) size * h / w) : size;
        GeneratedImage scaled = Pixels.resize(image, nw, nh, Pixels.Filter.BICUBIC);
        GeneratedImage square = Pixels.crop(scaled, (nw - size) / 2, (nh - size) / 2, size, size);
        int plane = size * size;
        float[] v = new float[3 * plane];
        byte[] rgb = square.rgb();
        for (int p = 0; p < plane; p++) {
            for (int c = 0; c < 3; c++) {
                v[c * plane + p] = ((rgb[p * 3 + c] & 0xff) / 255f - ClipVisionEncoder.MEAN[c])
                        / ClipVisionEncoder.STD[c];
            }
        }
        return Var.noGrad(() -> {
            Tensor f = Tensor.of(Shape.of(1, 3, size, size), v, placement.text());
            Tensor in = f.to(clip.dtype());
            Var out = clip.forward(Var.of(in), false);
            Tensor t = out.value().to(transformers.get(0).dtype());
            Tensor.evaluate(List.of(t));
            for (Tensor x : new Tensor[] {f, in, out.value()}) {
                if (x != t) {
                    x.release();
                }
            }
            return t;
        });
    }

    /** Latents with their first frame replaced, in float32. */
    private static Tensor withFirstFrame(Tensor latents, Tensor first) {
        int frames = latents.shape().dim(2);
        Tensor f = first.to(DType.FLOAT32);
        if (frames == 1) {
            return f == first ? f.add(0f) : f;
        }
        Tensor rest = latents.slice(2, 1, frames);
        Tensor out = Tensor.concat(2, f, rest);
        Tensor.evaluate(List.of(out));
        rest.release();
        if (f != first) {
            f.release();
        }
        return out;
    }

    /**
     * Samples clean latents from noise.
     *
     * @param noise standard normal latents
     * @param cond the prompt's text states
     * @param uncond the negative prompt's, or {@code null} without guidance
     * @param image what a first frame adds, or {@link Conditioning#NONE}
     * @param steps the steps
     * @param guidance the guidance scale
     * @param lowNoiseGuidance the low-noise expert's scale, or 0 for {@code guidance}
     * @param shift the schedule's shift
     * @param listener hears each step
     * @return the latents, float32
     */
    Tensor denoise(Tensor noise, Tensor cond, Tensor uncond, Conditioning image, int steps, float guidance,
            float lowNoiseGuidance, float shift, SigmaSampler.Listener listener) {
        double[] sigmas = FlowSampler.uniPcSigmas(steps, shift);
        double boundary = files.boundary() * 1000;
        // each expert's copy of what conditions it, on its device
        int experts = transformers.size();
        Tensor[][] copies = new Tensor[experts][];
        Tensor[] given = {cond, uncond, image.channels(), image.features()};
        for (int i = 0; i < experts; i++) {
            copies[i] = new Tensor[given.length];
            for (int j = 0; j < given.length; j++) {
                copies[i][j] = given[j] == null ? null : given[j].to(placement.transformer(i));
            }
        }
        FlowSampler.Model model = (sample, sigma, step) -> Var.noGrad(() -> {
            // diffusers hands the model whole timesteps
            float t = (float) Math.floor(sigma * 1000);
            boolean low = experts == 2 && t < boundary;
            int e = low ? 1 : 0;
            WanTransformer dit = transformers.get(e);
            Device here = placement.transformer(e);
            Tensor[] mine = copies[e];
            Var features = mine[3] == null ? null : Var.of(mine[3]);
            float scale = low && lowNoiseGuidance > 0 ? lowNoiseGuidance : guidance;
            DType type = dit.dtype();
            Tensor x = sample.to(here);
            Tensor time;
            Tensor in;
            if (image.firstFrame() != null) {
                // the given frame sits at noise level 0 in every step
                Tensor mixed = withFirstFrame(x, image.firstFrame());
                in = mixed.to(type);
                if (in != mixed) {
                    mixed.release();
                }
                time = firstFrameTimesteps(x.shape(), dit.settings().patch(), t, here);
            } else {
                Tensor typed = x.to(type);
                in = mine[2] == null ? typed : Tensor.concat(1, typed, mine[2]);
                if (in != typed && typed != x) {
                    typed.release();
                }
                time = Tensor.of(Shape.of(1), new float[] {t}, here);
            }
            Tensor c = dit.forward(Var.of(in), time, Var.of(mine[0]), features, false).value().to(DType.FLOAT32);
            Tensor v = c;
            if (mine[1] != null && scale > 1) {
                Tensor u = dit.forward(Var.of(in), time, Var.of(mine[1]), features, false).value().to(DType.FLOAT32);
                v = u.add(c.sub(u).mul(scale));
            }
            return v.to(device);
        });
        Tensor start = noise.to(DType.FLOAT32);
        try {
            return FlowSampler.sampleUniPc(model, start, sigmas, listener);
        } finally {
            if (start != noise) {
                start.release();
            }
            for (Tensor[] c : copies) {
                for (int j = 0; j < given.length; j++) {
                    if (c[j] != null && c[j] != given[j]) {
                        c[j].release();
                    }
                }
            }
        }
    }

    /** Per-patch timesteps {@code [1, patches]}: 0 for the first latent frame's patches, {@code t} for the rest. */
    private static Tensor firstFrameTimesteps(Shape latents, int[] patch, float t, Device device) {
        int perFrame = (latents.dim(3) / patch[1]) * (latents.dim(4) / patch[2]);
        int n = latents.dim(2) / patch[0] * perFrame;
        float[] v = new float[n];
        java.util.Arrays.fill(v, perFrame, n, t);
        return Tensor.of(Shape.of(1, n), v, device);
    }

    /** Decodes latents into frames, one latent frame at a time. */
    List<GeneratedImage> decode(Tensor latents) {
        List<GeneratedImage> frames = new ArrayList<>();
        Tensor there = latents.to(placement.vae());
        vae.decode(there, chunk -> {
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
        if (there != latents) {
            there.release();
        }
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
            Tensor idTensor = StableDiffusion.idTensor(new int[][] {ids}, placement.text());
            Var states = t5.encode(Var.of(idTensor), false);
            idTensor.release();
            int n = ids.length;
            int total = options.textTokens();
            int d = states.shape().dim(2);
            DType type = transformers.get(0).dtype();
            Tensor s = states.value().to(type);
            Tensor out = n == total ? s
                    : Tensor.concat(1, s, Tensor.zeros(Shape.of(1, total - n, d), type, placement.text()));
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
